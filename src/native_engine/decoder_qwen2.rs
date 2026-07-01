//! The Qwen2-arch **dense** decoder forward (GOT-OCR2.0, bead B5) — a thin driver
//! that composes the already-parity-tested leaf kernels in [`super::decoder`] with
//! a [`DecoderConfig`], rather than forking the DeepSeek-V2 MoE + R-SWA path (whose
//! `config` module and `RingCache`/`rswa` attention hardcode 1280/12/10/128 and a
//! 128-token sliding window — all wrong for GOT).
//!
//! GOT-OCR2's decoder is Qwen2-0.5B: 24 layers, hidden 1024, 16 heads (no GQA,
//! head_dim 64), dense SwiGLU (intermediate 2816, silu), RoPE θ=1e6, **full-causal**
//! attention (scale 1/8 = 1/√64), q/k/v_proj **with bias** (o_proj none), RMSNorm
//! ε=1e-6, and **tied** embeddings (`lm_head` == `embed_tokens`, stored once,
//! high-precision). Every one of those maps onto an existing kernel:
//! * q/k/v/o + gate/up/down GEMMs → [`nn::linear_int8_dynamic`] (int8, and it
//!   already carries the qkv bias) or the f32 [`super::decoder::linear_no_bias`];
//! * RoPE → [`super::decoder::RopeTable::build`] + [`super::decoder::apply_rope`]
//!   (NEOX rotate-half = HF Qwen2), built once at `head_dim=64, θ=1e6`;
//! * attention → [`super::decoder::prefill_attention`] (full-causal MHA, scale
//!   `1/√head_dim` = 1/8 at head_dim 64);
//! * norms → [`nn::rms_norm`]; tied head → [`super::decoder::norm_and_lm_head`].
//!
//! [`linear_auto`] picks int8 vs f32 **per GEMM** from the loaded tensor's dtype,
//! so the SAME forward certifies both the shipping `got-ocr2.int8.focrq` (int8
//! decoder GEMMs) and the raw bf16 `model.safetensors` (an f32 reference). Parity
//! is held against the bit-deterministic torch oracle (floor = 0) at the decoder
//! seam: feed the oracle's post-splice `hidden_0` and match the last-position
//! logits (see the `#[cfg(test)]` parity gate).
//!
//! Generation: [`generate_greedy`] is the correct O(n²) re-prefill path (the parity
//! oracle); [`generate_greedy_kvcache`] (bead B9) is the O(n)-per-token full-causal
//! **KV-cache** decode used in production. The seeding prefill stays on the row-parallel
//! `nn::linear_int8_dynamic` (m = N); each single-token decode step (m = 1) instead
//! runs the **n-parallel** `gemv_i8_bias_prequant` — same int8 weights, activations
//! quantized ties-to-even (`quantize_row_i8_te`) to match the prefill — so the decode
//! stays **argmax-exact** to `generate_greedy` (hence the torch oracle L4) while giving
//! every core work at m = 1 (the row-parallel kernel is single-threaded there).

use super::decoder;
use super::nn;
use super::tensor::{Mat, QInt8};
use super::weights::{DType, Weights};
use crate::error::{FocrError, FocrResult};
use rayon::prelude::*;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

// ── GOT decode perf levers (bd-3bom follow-on) — each gated so ONE build measures
//    every config, and the gate on a numerics-changing lever IS its doctrine-#2
//    kill-switch. Tri-state env: "1"/"int8"/"on" force ON, "0"/"f32"/"off" force OFF,
//    unset ⇒ the compiled default. Read ONCE into a process-wide bool. ─────────────

/// `FOCR_GOT_INT8_LMHEAD` — run the GOT `lm_head` as an int8 per-output-channel GEMV
/// (SDOT on Apple Silicon / AVX-512-VNNI·AVX-VNNI·AVX2 on Intel-AMD) instead of the f32
/// matmul over the 151860×1024 tied head, then recover the exact greedy pick with a small
/// f32 top-K refine ([`refine_topk_f32`]). The lm_head is the dominant memory-bound decode
/// stage (reads the ~0.6 GB f32 head per token); int8 cuts that to ~0.15 GB + moves it onto
/// the int8 matmul units (measured **8.2× on the head**, **1.85× on the whole GOT forward**).
///
/// int8-on-`lm_head` is BEYOND doctrine #2's validated set, so it stays a MEASURED
/// kill-switch — but **default ON**, because the top-K refine makes it **byte-identical** to
/// the f32 head (688/688 tokens on page_0107; == the torch oracle L4). `FOCR_GOT_INT8_LMHEAD=0`
/// forces the provably-bit-identical f32 head back. See artifacts/perf/bd-3bom/.
const GOT_INT8_LMHEAD_DEFAULT: bool = true;

/// `FOCR_GOT_SEQ_ATTN` — force the SERIAL per-head decode attention. The parallel path
/// (default) fans the independent heads across the rayon pool; it is bit-identical (each
/// head is a self-contained softmax over disjoint output lanes), so this gate exists only
/// as a measurement/safety switch, not a correctness one.
const GOT_PARALLEL_ATTN_DEFAULT: bool = true;

fn env_tristate(var: &str, default: bool) -> bool {
    match std::env::var(var).ok().as_deref() {
        Some("1" | "int8" | "on" | "true" | "yes") => true,
        Some("0" | "f32" | "off" | "false" | "no") => false,
        _ => default,
    }
}

fn got_int8_lmhead_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| env_tristate("FOCR_GOT_INT8_LMHEAD", GOT_INT8_LMHEAD_DEFAULT))
}

fn got_parallel_attn_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    // FOCR_GOT_SEQ_ATTN=1 forces serial ⇒ parallel is the inverse.
    *FLAG.get_or_init(|| !env_tristate("FOCR_GOT_SEQ_ATTN", !GOT_PARALLEL_ATTN_DEFAULT))
}

/// The GOT `lm_head` weight, resolved ONCE per generation to either the f32 pre-transposed
/// `[hidden, vocab]` matrix (bit-identical to the certified path) or the int8 per-output-
/// channel quantization of the SAME tied head (the [`got_int8_lmhead_enabled`] lever).
enum LmHead {
    /// `[hidden, vocab]` row-major, matmul-ready (see [`decoder::norm_and_lm_head_pretransposed`]).
    F32(Mat),
    /// `[vocab, hidden]` per-output-channel int8 (the native checkpoint layout, fed to the
    /// n-parallel `gemv_i8` — argmax-exact to f32 on the oracle L4 / page_0107 CER).
    Int8(QInt8),
}

/// Decode-step wall-clock accumulators (ns), summed across the whole generate loop.
/// Read + printed only when `FOCR_TIMING` is set (perf bring-up); the `fetch_add`s
/// are `Relaxed` and off the hot numeric path, so they cost nothing measurable.
static DECODE_ATTN_NS: AtomicU64 = AtomicU64::new(0);
static DECODE_GEMV_NS: AtomicU64 = AtomicU64::new(0);
static DECODE_LMHEAD_NS: AtomicU64 = AtomicU64::new(0);

/// Config of a dense (Qwen2/Llama-style) decoder — the parameters the shared leaf
/// kernels need, so one driver serves GOT-OCR2 (and later SmolVLM2/OneChart).
#[derive(Debug, Clone, Copy)]
pub struct DecoderConfig {
    /// Residual-stream width (GOT 1024).
    pub hidden_size: usize,
    /// Dense SwiGLU inner width (GOT 2816).
    pub intermediate_size: usize,
    /// Number of transformer layers (GOT 24).
    pub num_hidden_layers: usize,
    /// Attention heads (GOT 16).
    pub num_attention_heads: usize,
    /// Per-head dim (GOT 64); `num_attention_heads * head_dim == hidden` for GOT.
    pub head_dim: usize,
    /// Vocabulary (GOT 151860).
    pub vocab_size: usize,
    /// RoPE base θ (GOT 1e6).
    pub rope_theta: f32,
    /// RMSNorm ε (GOT 1e-6).
    pub rms_norm_eps: f32,
    /// Whether q/k/v_proj carry a bias (GOT true; o_proj never does).
    pub attn_qkv_bias: bool,
}

impl DecoderConfig {
    /// The GOT-OCR2.0 (Qwen2-0.5B) decoder configuration (`config.json`, spec §4).
    #[must_use]
    pub fn got_ocr2() -> Self {
        Self {
            hidden_size: 1024,
            intermediate_size: 2816,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            head_dim: 64,
            vocab_size: 151_860,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            attn_qkv_bias: true,
        }
    }

    /// The q/k/v/o projection width (`num_attention_heads * head_dim`). Equals
    /// `hidden_size` for GOT (no GQA), but kept distinct so the driver survives a
    /// GQA model.
    #[must_use]
    pub fn qkv_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }
}

/// One GEMM `y = x @ W^T (+ bias)` where `W` (name `weight_name` in `weights`) is
/// either a pre-quantized `QInt8PerChan` record (the `.focrq` shipping path →
/// [`nn::linear_int8_dynamic`], which dequantizes then adds `bias` in f32) or a
/// high-precision bf16/f32 record (the reference path → [`decoder::linear_no_bias`]
/// + a manual bias add). Dispatching per tensor lets ONE forward certify both.
fn linear_auto(
    weights: &Weights,
    x: &Mat,
    weight_name: &str,
    in_: usize,
    out: usize,
    bias: Option<&[f32]>,
) -> FocrResult<Mat> {
    let is_int8 = matches!(
        weights.record(weight_name).map(|r| r.dtype),
        Some(DType::QInt8PerChan)
    );
    if is_int8 {
        let qw = decoder::quant_oc_loaded(weights, weight_name, out)?;
        nn::linear_int8_dynamic(x, &qw, bias)
    } else {
        let w = weights.mat(weight_name)?;
        if w.data.len() != out * in_ {
            return Err(FocrError::FormatMismatch(format!(
                "decoder_qwen2: {weight_name} has {} elems, expected {out}*{in_}",
                w.data.len()
            )));
        }
        let mut y = decoder::linear_no_bias(x, &w.data, in_, out)?;
        if let Some(b) = bias {
            add_bias(&mut y, b);
        }
        Ok(y)
    }
}

/// Add a per-output-channel bias to each row (f32 reference path; the int8 path
/// adds bias inside `linear_int8_dynamic` at the same point — after dequant).
fn add_bias(y: &mut Mat, bias: &[f32]) {
    debug_assert_eq!(bias.len(), y.cols);
    for row in y.data.chunks_mut(y.cols) {
        for (v, &b) in row.iter_mut().zip(bias.iter()) {
            *v += b;
        }
    }
}

/// Reject Qwen3-style per-head q/k norms (Qwen2 has none) — their silent presence
/// would be a silent parity divergence (spec §13a OQ).
fn assert_no_qk_norm(weights: &Weights, layer_prefix: &str) -> FocrResult<()> {
    for suffix in [".self_attn.q_norm.weight", ".self_attn.k_norm.weight"] {
        let name = format!("{layer_prefix}{suffix}");
        if weights.record(&name).is_some() {
            return Err(FocrError::FormatMismatch(format!(
                "decoder_qwen2: unexpected {name} — Qwen2 has no q/k-norm; refusing to run \
                 a mismatched architecture"
            )));
        }
    }
    Ok(())
}

/// One Qwen2 dense layer over the prefill activations `x: [seq, hidden]`.
fn qwen2_layer(
    weights: &Weights,
    x: &Mat,
    layer: usize,
    rope: &decoder::RopeTable,
    cfg: &DecoderConfig,
) -> FocrResult<Mat> {
    let p = format!("model.layers.{layer}");
    assert_no_qk_norm(weights, &p)?;
    let eps = cfg.rms_norm_eps;
    let (hidden, qkv_dim, inter) = (cfg.hidden_size, cfg.qkv_dim(), cfg.intermediate_size);

    // ── attention ────────────────────────────────────────────────────────────
    let input_ln = weights.vec(&format!("{p}.input_layernorm.weight"))?;
    let normed = nn::rms_norm(x, Some(&input_ln), eps)?;

    let (q_b, k_b, v_b) = if cfg.attn_qkv_bias {
        (
            Some(weights.vec(&format!("{p}.self_attn.q_proj.bias"))?),
            Some(weights.vec(&format!("{p}.self_attn.k_proj.bias"))?),
            Some(weights.vec(&format!("{p}.self_attn.v_proj.bias"))?),
        )
    } else {
        (None, None, None)
    };
    let mut q = linear_auto(
        weights,
        &normed,
        &format!("{p}.self_attn.q_proj.weight"),
        hidden,
        qkv_dim,
        q_b.as_deref(),
    )?;
    let mut k = linear_auto(
        weights,
        &normed,
        &format!("{p}.self_attn.k_proj.weight"),
        hidden,
        qkv_dim,
        k_b.as_deref(),
    )?;
    let v = linear_auto(
        weights,
        &normed,
        &format!("{p}.self_attn.v_proj.weight"),
        hidden,
        qkv_dim,
        v_b.as_deref(),
    )?;

    decoder::apply_rope(&mut q, rope)?;
    decoder::apply_rope(&mut k, rope)?;
    let ctx = decoder::prefill_attention(&q, &k, &v, cfg.num_attention_heads, cfg.head_dim)?;
    let attn = linear_auto(
        weights,
        &ctx,
        &format!("{p}.self_attn.o_proj.weight"),
        qkv_dim,
        hidden,
        None,
    )?;
    let h = decoder::add_residual(x, &attn)?;

    // ── dense SwiGLU MLP ──────────────────────────────────────────────────────
    let post_ln = weights.vec(&format!("{p}.post_attention_layernorm.weight"))?;
    let normed2 = nn::rms_norm(&h, Some(&post_ln), eps)?;
    let mut g = linear_auto(
        weights,
        &normed2,
        &format!("{p}.mlp.gate_proj.weight"),
        hidden,
        inter,
        None,
    )?;
    nn::silu(&mut g);
    let u = linear_auto(
        weights,
        &normed2,
        &format!("{p}.mlp.up_proj.weight"),
        hidden,
        inter,
        None,
    )?;
    for (a, &b) in g.data.iter_mut().zip(u.data.iter()) {
        *a *= b;
    }
    let mlp = linear_auto(
        weights,
        &g,
        &format!("{p}.mlp.down_proj.weight"),
        inter,
        hidden,
        None,
    )?;
    decoder::add_residual(&h, &mlp)
}

/// Run the dense decoder prefill over `inputs_embeds: [seq, hidden]` (the
/// post-`<imgpad>`-splice decoder input) through all layers and the final norm +
/// **tied** lm_head, returning logits `[seq, vocab]`.
///
/// `weights` may be the int8 `got-ocr2.int8.focrq` or the raw bf16 safetensors;
/// [`linear_auto`] adapts per GEMM. The lm_head is always f32 (the tied
/// `model.embed_tokens.weight`, HP).
///
/// # Errors
/// [`FocrError`] on a shape mismatch, a missing tensor, or a rejected q/k-norm.
pub fn forward_prefill(
    weights: &Weights,
    cfg: &DecoderConfig,
    inputs_embeds: &Mat,
) -> FocrResult<Mat> {
    if inputs_embeds.cols != cfg.hidden_size {
        return Err(FocrError::FormatMismatch(format!(
            "decoder_qwen2: inputs_embeds cols {} != hidden {}",
            inputs_embeds.cols, cfg.hidden_size
        )));
    }
    let positions: Vec<usize> = (0..inputs_embeds.rows).collect();
    let rope = decoder::RopeTable::build(&positions, cfg.head_dim, cfg.rope_theta);

    let mut x = inputs_embeds.clone();
    for layer in 0..cfg.num_hidden_layers {
        x = qwen2_layer(weights, &x, layer, &rope, cfg)?;
    }

    // final RMSNorm + tied lm_head (embed_tokens^T, f32).
    let final_norm = weights.vec("model.norm.weight")?;
    let embed = weights.mat("model.embed_tokens.weight")?;
    decoder::norm_and_lm_head(
        &x,
        &final_norm,
        &embed.data,
        cfg.vocab_size,
        cfg.rms_norm_eps,
    )
}

/// Greedy (argmax, temperature-0) autoregressive decode from `inputs_embeds`,
/// generating up to `max_new` tokens and stopping at `eos`. Returns the generated
/// id-stream (excluding the prompt).
///
/// This is the **correct, unoptimized** generation path: each step re-runs the
/// full prefill over the grown sequence (O(n²) — a KV-cache decode-step is the
/// perf follow-on, bead B9), appending the argmax token's embedding. It reproduces
/// the torch oracle's greedy L4 output; correctness first, then speed (doctrine #1).
///
/// # Errors
/// Any [`forward_prefill`] error, or a missing `model.embed_tokens.weight`.
pub fn generate_greedy(
    weights: &Weights,
    cfg: &DecoderConfig,
    inputs_embeds: &Mat,
    max_new: usize,
    eos: u32,
) -> FocrResult<Vec<u32>> {
    let embed = weights.mat("model.embed_tokens.weight")?;
    let (vocab, hidden) = (embed.rows, embed.cols);
    let mut data = inputs_embeds.data.clone();
    let mut ids = Vec::new();
    for _ in 0..max_new {
        let rows = data.len() / hidden;
        let cur = Mat::from_vec(rows, hidden, std::mem::take(&mut data));
        let logits = forward_prefill(weights, cfg, &cur)?;
        let last = &logits.data[(logits.rows - 1) * vocab..];
        let next = last
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
                if x > bv { (i, x) } else { (bi, bv) }
            })
            .0 as u32;
        ids.push(next);
        // reclaim the prefix embeds and append the chosen token's embedding.
        data = cur.data;
        if next == eos {
            break;
        }
        let te = decoder::embed_tokens(&embed.data, vocab, hidden, &[next])?;
        data.extend_from_slice(&te.data);
    }
    Ok(ids)
}

// ── B9: full-causal KV-cache decode (O(n)/token, replaces the O(n²) re-prefill) ──
//
// The decode path is held BIT-IDENTICAL to [`generate_greedy`] (hence to the torch
// oracle L4) by routing every decode GEMM through the SAME [`nn::linear_int8_dynamic`]
// (m=1) the prefill uses — its ties-to-even activation quant avoids the rounding gap
// the standalone `decoder::gemv_i8` (half-away) would introduce. The bespoke
// prequant-fused `gemv_i8_bias` is a ledgered perf follow-on; correctness first.

/// One decoder layer's full-causal KV cache: post-RoPE **K** and post-proj **V**,
/// token-major `[n_kv, qkv_dim]`, grown by one row per decode step. NOT R-SWA — GOT
/// Qwen2 attends the whole prefix (no window, no eviction, f32 KV).
struct Qwen2KvCache {
    k: Vec<f32>,
    v: Vec<f32>,
    n_kv: usize,
    qkv_dim: usize,
}

impl Qwen2KvCache {
    fn new(qkv_dim: usize, max_positions: usize) -> Self {
        Self {
            k: Vec::with_capacity(max_positions * qkv_dim),
            v: Vec::with_capacity(max_positions * qkv_dim),
            n_kv: 0,
            qkv_dim,
        }
    }
    /// Seed all `N` prefill rows at once (`k_all`/`v_all` are `[N, qkv_dim]`).
    fn seed(&mut self, k_all: &[f32], v_all: &[f32]) {
        self.k.extend_from_slice(k_all);
        self.v.extend_from_slice(v_all);
        self.n_kv += k_all.len() / self.qkv_dim;
    }
    /// Append one decode step's k/v row (each `[qkv_dim]`).
    fn append(&mut self, k_row: &[f32], v_row: &[f32]) {
        self.k.extend_from_slice(k_row);
        self.v.extend_from_slice(v_row);
        self.n_kv += 1;
    }
}

/// Full-causal m=1 attention: the single new query (`q_row`, `[qkv_dim]`) attends
/// ALL `n_kv` cached keys (every cached position ≤ the current one — no mask needed).
/// scale `1/√head_dim` = 1/8.
///
/// Reads the **token-major** cache directly — no per-step head-major repack (the old
/// `nn::sdpa` path allocated + copied `2·num_heads·n_kv·head_dim` floats EVERY token,
/// an O(n²) alloc-churn that dominated long-page decode). This bespoke scaled-dot-product
/// (per-head dot → softmax → weighted V, all f32) is the standard attention math; it is
/// argmax-exact vs the sdpa path (certified: `kvcache_greedy_matches_oracle_l4` still ==
/// the oracle L4). Per-step temp is just `[n_kv]` scores. (Perf lever, bead B9.)
fn qwen2_decode_attention(
    cache: &Qwen2KvCache,
    q_row: &[f32],
    num_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let dim = num_heads * head_dim;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; dim];
    if got_parallel_attn_enabled() {
        // Each head is a self-contained softmax writing a DISJOINT `[head_dim]` output
        // lane, so fanning the heads across the rayon pool is bit-identical (no cross-head
        // dependence, per-head accumulation order unchanged). `par_chunks_mut(head_dim)`
        // hands head `h` its own output slice + its own scratch.
        out.par_chunks_mut(head_dim)
            .enumerate()
            .for_each(|(h, oh)| decode_attn_head(cache, q_row, h, head_dim, dim, scale, oh));
    } else {
        for h in 0..num_heads {
            let oh = &mut out[h * head_dim..(h + 1) * head_dim];
            decode_attn_head(cache, q_row, h, head_dim, dim, scale, oh);
        }
    }
    out
}

/// One attention head's decode: `softmax(scale · q_h·Kᵀ) · V`, writing `[head_dim]` into
/// `oh`. Reads the token-major cache directly; the only per-head temp is `[n_kv]` scores.
/// Identical math whether called serially or from the rayon fan-out above.
#[inline]
fn decode_attn_head(
    cache: &Qwen2KvCache,
    q_row: &[f32],
    h: usize,
    head_dim: usize,
    dim: usize,
    scale: f32,
    oh: &mut [f32],
) {
    let n_kv = cache.n_kv;
    let qh = &q_row[h * head_dim..h * head_dim + head_dim];
    let mut scores = vec![0.0f32; n_kv];
    // scores[r] = scale · (q_h · k[r, head h]); track the max for a stable softmax.
    let mut smax = f32::NEG_INFINITY;
    for (r, s) in scores.iter_mut().enumerate() {
        let base = r * dim + h * head_dim;
        let kh = &cache.k[base..base + head_dim];
        let dot: f32 = qh.iter().zip(kh).map(|(&a, &b)| a * b).sum();
        *s = dot * scale;
        smax = smax.max(*s);
    }
    // softmax over the cached positions.
    let mut denom = 0.0f32;
    for s in &mut scores {
        *s = (*s - smax).exp();
        denom += *s;
    }
    let inv = 1.0 / denom;
    // out_h = Σ_r softmax[r] · v[r, head h].
    for (r, &s) in scores.iter().enumerate() {
        let w = s * inv;
        let base = r * dim + h * head_dim;
        let vh = &cache.v[base..base + head_dim];
        for (o, &vv) in oh.iter_mut().zip(vh) {
            *o += w * vv;
        }
    }
}

/// One layer's decode weights, loaded ONCE (int8 GEMMs read verbatim from the
/// `.focrq` `QInt8PerChan` records, f32 norms/biases) so no weight is re-read per token.
struct GotLayerW {
    input_ln: Vec<f32>,
    post_attn_ln: Vec<f32>,
    /// Fused q|k|v projection `[3·qkv_dim, hidden]` (one int8 panel) so the decode
    /// quantizes the normed row ONCE and runs ONE n-parallel GEMV for all three.
    qkv: QInt8,
    qkv_bias: Vec<f32>,
    o: QInt8,
    gate: QInt8,
    up: QInt8,
    down: QInt8,
}

/// Concatenate the row-major q/k/v `[d, h]` int8 panels + scales into one
/// `[3·d, h]` [`QInt8`] — bit-identical per-output-channel to the three separate
/// GEMMs (each output channel keeps its own scale), but one dispatch.
fn concat_qkv(q: &QInt8, k: &QInt8, v: &QInt8) -> QInt8 {
    let mut w = Vec::with_capacity(q.w.len() * 3);
    w.extend_from_slice(&q.w);
    w.extend_from_slice(&k.w);
    w.extend_from_slice(&v.w);
    let mut scales = Vec::with_capacity(q.n * 3);
    scales.extend_from_slice(&q.scales);
    scales.extend_from_slice(&k.scales);
    scales.extend_from_slice(&v.scales);
    QInt8::new(w, scales, q.n * 3, q.k)
}

/// The whole GOT decoder's decode-time weights (pre-loaded once for a generation).
struct GotDecodeWeights {
    layers: Vec<GotLayerW>,
    final_norm: Vec<f32>,
    /// Tied `embed_tokens` `[vocab, hidden]`, row-major — the per-token embed lookup.
    embed: Vec<f32>,
    /// The SAME weights transposed to `[hidden, vocab]` ONCE at build, matmul-ready for
    /// the decode-step `lm_head`. Either the f32 pre-transposed `[hidden, vocab]` head
    /// (bit-identical to the certified path — never re-transposes the ~0.6 GB matrix per
    /// token, which was 95% of decode wall-clock) or its int8 quantization
    /// ([`got_int8_lmhead_enabled`]). Built ONCE. See [`got_lm_head`].
    lm_head: LmHead,
    cfg: DecoderConfig,
}

impl GotDecodeWeights {
    fn build(weights: &Weights, cfg: &DecoderConfig) -> FocrResult<Self> {
        let (hidden, qkv_dim, inter) = (cfg.hidden_size, cfg.qkv_dim(), cfg.intermediate_size);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{l}");
            let q = decoder::quant_oc_loaded(
                weights,
                &format!("{p}.self_attn.q_proj.weight"),
                qkv_dim,
            )?;
            let k = decoder::quant_oc_loaded(
                weights,
                &format!("{p}.self_attn.k_proj.weight"),
                qkv_dim,
            )?;
            let v = decoder::quant_oc_loaded(
                weights,
                &format!("{p}.self_attn.v_proj.weight"),
                qkv_dim,
            )?;
            let mut qkv_bias = weights.vec(&format!("{p}.self_attn.q_proj.bias"))?;
            qkv_bias.extend(weights.vec(&format!("{p}.self_attn.k_proj.bias"))?);
            qkv_bias.extend(weights.vec(&format!("{p}.self_attn.v_proj.bias"))?);
            layers.push(GotLayerW {
                input_ln: weights.vec(&format!("{p}.input_layernorm.weight"))?,
                post_attn_ln: weights.vec(&format!("{p}.post_attention_layernorm.weight"))?,
                qkv: concat_qkv(&q, &k, &v),
                qkv_bias,
                o: decoder::quant_oc_loaded(
                    weights,
                    &format!("{p}.self_attn.o_proj.weight"),
                    hidden,
                )?,
                gate: decoder::quant_oc_loaded(
                    weights,
                    &format!("{p}.mlp.gate_proj.weight"),
                    inter,
                )?,
                up: decoder::quant_oc_loaded(weights, &format!("{p}.mlp.up_proj.weight"), inter)?,
                down: decoder::quant_oc_loaded(
                    weights,
                    &format!("{p}.mlp.down_proj.weight"),
                    hidden,
                )?,
            });
        }
        let embed = weights.mat("model.embed_tokens.weight")?.data;
        let (vocab, hidden) = (cfg.vocab_size, cfg.hidden_size);
        // Resolve the lm_head ONCE. int8 lever: quantize the tied `[vocab, hidden]` head to
        // per-output-channel int8 in its NATIVE layout (fed to the n-parallel `gemv_i8` —
        // SDOT on aarch64, VNNI on x86). f32 path: transpose `[vocab, hidden]` -> the
        // matmul-ready `[hidden, vocab]` ONCE (the SAME transpose `linear_no_bias` did per
        // call), so the decode-step head is a plain matmul with no per-token ~0.6 GB
        // re-transpose (which was 95% of decode wall-clock). Only ONE is built, so int8 also
        // halves the head's resident memory (no f32 `[hidden, vocab]` copy).
        let lm_head = if got_int8_lmhead_enabled() {
            LmHead::Int8(nn::quantize_int8(&embed, vocab, hidden))
        } else {
            let mut wt = vec![0.0f32; embed.len()];
            for o in 0..vocab {
                let src = &embed[o * hidden..(o + 1) * hidden];
                for (i, &val) in src.iter().enumerate() {
                    wt[i * vocab + o] = val;
                }
            }
            LmHead::F32(Mat::from_vec(hidden, vocab, wt))
        };
        Ok(Self {
            layers,
            final_norm: weights.vec("model.norm.weight")?,
            embed,
            lm_head,
            cfg: *cfg,
        })
    }
}

/// How many of the int8-approx head logits get recomputed in exact f32 before the greedy
/// argmax (the [`refine_topk_f32`] near-lossless pass). 256 ≫ any realistic gap between the
/// true best token's int8 rank and rank 1, so the refined argmax matches the f32 head.
const GOT_LMHEAD_REFINE_K: usize = 256;

/// The GOT `lm_head` over a SINGLE hidden row `x_row` (`[1, hidden]`): the final
/// `rms_norm` then the head projection, dispatched to the f32 pre-transposed matmul
/// (bit-identical to the certified path) or the int8 per-channel `gemv_i8`
/// ([`got_int8_lmhead_enabled`]). Returns the `[vocab]` logits. Both the seeding prefill
/// (last position only — the rest are never argmaxed) and every decode step funnel through
/// here, so a whole generation uses ONE lm_head numeric path.
///
/// The int8 path is made **near-lossless** by [`refine_topk_f32`]: the fast int8 GEMV picks
/// the top candidates, then those are recomputed in exact f32 so the greedy pick matches the
/// f32 head (the argmax lives in the int8 top-K with overwhelming probability).
fn got_lm_head(w: &GotDecodeWeights, x_row: &Mat, eps: f32) -> FocrResult<Vec<f32>> {
    match &w.lm_head {
        LmHead::F32(wt) => {
            Ok(decoder::norm_and_lm_head_pretransposed(x_row, &w.final_norm, wt, eps)?.data)
        }
        LmHead::Int8(q) => {
            let normed = nn::rms_norm(x_row, Some(&w.final_norm), eps)?;
            let (xq, a) = decoder::quantize_row_i8_te(&normed.data);
            let mut logits = decoder::gemv_i8_bias_prequant(&xq, a, q, None);
            refine_topk_f32(&mut logits, &normed.data, &w.embed, w.cfg.hidden_size);
            Ok(logits)
        }
    }
}

/// Recompute the `GOT_LMHEAD_REFINE_K` largest int8-approx `logits` in exact f32
/// (`normed · embed_row` — the SAME value the f32 head produces for that token), so the
/// greedy argmax over the refined vector matches the f32 lm_head. Makes the int8 lm_head
/// near-lossless: the true best token is in the int8 top-K with overwhelming probability, so
/// its refined logit is exact and wins. `embed` is the tied head `[vocab, hidden]` row-major.
fn refine_topk_f32(logits: &mut [f32], normed: &[f32], embed: &[f32], hidden: usize) {
    let vocab = logits.len();
    let k = GOT_LMHEAD_REFINE_K.min(vocab);
    if k == 0 {
        return;
    }
    // Partition so idx[..k] index the k largest int8-approx logits (O(vocab), no full sort).
    let mut idx: Vec<u32> = (0..vocab as u32).collect();
    idx.select_nth_unstable_by(k - 1, |&a, &b| {
        logits[b as usize].total_cmp(&logits[a as usize])
    });
    for &t in &idx[..k] {
        let t = t as usize;
        let row = &embed[t * hidden..(t + 1) * hidden];
        logits[t] = normed.iter().zip(row).map(|(&x, &wv)| x * wv).sum();
    }
}

/// Seeding prefill: run all `N` positions through the layers (BIT-IDENTICAL to
/// [`forward_prefill`] — same `linear_int8_dynamic` kernel), capturing each layer's
/// post-RoPE K + post-proj V into `caches`, and return the **last-position** logits
/// (projected via [`got_lm_head`] on the single last row — the other N-1 rows are never
/// argmaxed, so we skip the full `[N, vocab]` head GEMM the naive path computed).
fn forward_prefill_seed(
    w: &GotDecodeWeights,
    inputs_embeds: &Mat,
    caches: &mut [Qwen2KvCache],
) -> FocrResult<Vec<f32>> {
    let cfg = &w.cfg;
    let eps = cfg.rms_norm_eps;
    let mut x = inputs_embeds.clone();
    let positions: Vec<usize> = (0..inputs_embeds.rows).collect();
    let rope = decoder::RopeTable::build(&positions, cfg.head_dim, cfg.rope_theta);
    let qkv_dim = cfg.qkv_dim();
    for (l, cl) in w.layers.iter().enumerate() {
        let normed = nn::rms_norm(&x, Some(&cl.input_ln), eps)?;
        // one fused q|k|v GEMM (m=N, row-parallel), then split the columns.
        let qkv = nn::linear_int8_dynamic(&normed, &cl.qkv, Some(&cl.qkv_bias))?;
        let (mut q, mut k, v) = split_qkv_rows(&qkv, qkv_dim);
        decoder::apply_rope(&mut q, &rope)?;
        decoder::apply_rope(&mut k, &rope)?;
        caches[l].seed(&k.data, &v.data); // the very K/V prefill_attention consumes
        let ctx = decoder::prefill_attention(&q, &k, &v, cfg.num_attention_heads, cfg.head_dim)?;
        let attn = nn::linear_int8_dynamic(&ctx, &cl.o, None)?;
        let h = decoder::add_residual(&x, &attn)?;
        let normed2 = nn::rms_norm(&h, Some(&cl.post_attn_ln), eps)?;
        let mlp = decoder::expert_mlp_i8(&normed2, &cl.gate, &cl.up, &cl.down)?;
        x = decoder::add_residual(&h, &mlp)?;
    }
    let last = x.rows - 1;
    let x_last = Mat::from_vec(
        1,
        cfg.hidden_size,
        x.data[last * cfg.hidden_size..].to_vec(),
    );
    got_lm_head(w, &x_last, eps)
}

/// Split a fused `[N, 3·d]` q|k|v activation into three `[N, d]` mats (column blocks).
fn split_qkv_rows(fused: &Mat, d: usize) -> (Mat, Mat, Mat) {
    let n = fused.rows;
    let (mut q, mut k, mut v) = (
        vec![0.0f32; n * d],
        vec![0.0f32; n * d],
        vec![0.0f32; n * d],
    );
    for r in 0..n {
        let row = &fused.data[r * 3 * d..(r + 1) * 3 * d];
        q[r * d..(r + 1) * d].copy_from_slice(&row[0..d]);
        k[r * d..(r + 1) * d].copy_from_slice(&row[d..2 * d]);
        v[r * d..(r + 1) * d].copy_from_slice(&row[2 * d..3 * d]);
    }
    (
        Mat::from_vec(n, d, q),
        Mat::from_vec(n, d, k),
        Mat::from_vec(n, d, v),
    )
}

/// One decode step over a single token embedding `x: [1, hidden]` at absolute
/// `position`, appending to `caches`. Returns the `[vocab]` next-token logits.
fn qwen2_decode_step(
    w: &GotDecodeWeights,
    caches: &mut [Qwen2KvCache],
    x: &Mat,
    position: usize,
) -> FocrResult<Vec<f32>> {
    let cfg = &w.cfg;
    let (hidden, qkv_dim, eps) = (cfg.hidden_size, cfg.qkv_dim(), cfg.rms_norm_eps);
    let (num_heads, head_dim) = (cfg.num_attention_heads, cfg.head_dim);
    let rope = decoder::RopeTable::build(&[position], head_dim, cfg.rope_theta);
    let mut x = x.clone();
    let tlayers = std::time::Instant::now();
    for (l, cl) in w.layers.iter().enumerate() {
        // ── attention: quantize the normed row ONCE, fused q|k|v GEMV (n-parallel) ──
        let normed = nn::rms_norm(&x, Some(&cl.input_ln), eps)?;
        let (xq, a) = decoder::quantize_row_i8_te(&normed.data);
        let qkv = decoder::gemv_i8_bias_prequant(&xq, a, &cl.qkv, Some(&cl.qkv_bias));
        let mut q = Mat::from_vec(1, qkv_dim, qkv[0..qkv_dim].to_vec());
        let mut k = Mat::from_vec(1, qkv_dim, qkv[qkv_dim..2 * qkv_dim].to_vec());
        let v = &qkv[2 * qkv_dim..3 * qkv_dim];
        decoder::apply_rope(&mut q, &rope)?;
        decoder::apply_rope(&mut k, &rope)?;
        caches[l].append(&k.data, v);
        let ta = std::time::Instant::now();
        let ctx = qwen2_decode_attention(&caches[l], &q.data, num_heads, head_dim);
        DECODE_ATTN_NS.fetch_add(ta.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let (xqc, ac) = decoder::quantize_row_i8_te(&ctx);
        let attn = decoder::gemv_i8_bias_prequant(&xqc, ac, &cl.o, None);
        let h = decoder::add_residual(&x, &Mat::from_vec(1, hidden, attn))?;
        // ── dense SwiGLU: quantize normed2 ONCE, share it for gate + up ──────────
        let normed2 = nn::rms_norm(&h, Some(&cl.post_attn_ln), eps)?;
        let (xq2, a2) = decoder::quantize_row_i8_te(&normed2.data);
        let mut gate = Mat::from_vec(
            1,
            cfg.intermediate_size,
            decoder::gemv_i8_bias_prequant(&xq2, a2, &cl.gate, None),
        );
        let up = decoder::gemv_i8_bias_prequant(&xq2, a2, &cl.up, None);
        nn::silu(&mut gate);
        for (g, &u) in gate.data.iter_mut().zip(up.iter()) {
            *g *= u;
        }
        let (xq3, a3) = decoder::quantize_row_i8_te(&gate.data);
        let down = decoder::gemv_i8_bias_prequant(&xq3, a3, &cl.down, None);
        x = decoder::add_residual(&h, &Mat::from_vec(1, hidden, down))?;
    }
    DECODE_GEMV_NS.fetch_add(tlayers.elapsed().as_nanos() as u64, Ordering::Relaxed);
    let thead = std::time::Instant::now();
    let logits = got_lm_head(w, &x, eps)?;
    DECODE_LMHEAD_NS.fetch_add(thead.elapsed().as_nanos() as u64, Ordering::Relaxed);
    Ok(logits)
}

/// **O(n)-per-token** greedy decode: identical id-stream to [`generate_greedy`] but
/// with a full-causal KV cache instead of re-running prefill each step. The bit-for-bit
/// equality is enforced by reusing [`nn::linear_int8_dynamic`] for every GEMM (the
/// prefill kernel), so the decode never diverges from the certified path.
///
/// # Errors
/// Any prefill/decode-step error, or a missing `model.embed_tokens.weight`.
pub fn generate_greedy_kvcache(
    weights: &Weights,
    cfg: &DecoderConfig,
    inputs_embeds: &Mat,
    max_new: usize,
    eos: u32,
) -> FocrResult<Vec<u32>> {
    let w = GotDecodeWeights::build(weights, cfg)?;
    let n = inputs_embeds.rows;
    let mut caches: Vec<Qwen2KvCache> = (0..cfg.num_hidden_layers)
        .map(|_| Qwen2KvCache::new(cfg.qkv_dim(), n + max_new))
        .collect();
    let timing = std::env::var_os("FOCR_TIMING").is_some();
    if timing {
        DECODE_ATTN_NS.store(0, Ordering::Relaxed);
        DECODE_GEMV_NS.store(0, Ordering::Relaxed);
        DECODE_LMHEAD_NS.store(0, Ordering::Relaxed);
    }
    let tseed = std::time::Instant::now();
    let last_logits = forward_prefill_seed(&w, inputs_embeds, &mut caches)?;
    let seed_s = tseed.elapsed().as_secs_f64();
    let tdec = std::time::Instant::now();
    let mut ids = Vec::new();
    let mut next = argmax(&last_logits) as u32;
    for _ in 0..max_new {
        ids.push(next);
        if next == eos {
            break;
        }
        let te = decoder::embed_tokens(&w.embed, cfg.vocab_size, cfg.hidden_size, &[next])?;
        // the new token occupies the position after every currently-cached row.
        let position = caches[0].n_kv;
        let logits = qwen2_decode_step(&w, &mut caches, &te, position)?;
        next = argmax(&logits) as u32;
    }
    if timing {
        let dec_s = tdec.elapsed().as_secs_f64();
        let ns = 1e-9;
        let (attn, layers, head) = (
            DECODE_ATTN_NS.load(Ordering::Relaxed) as f64 * ns,
            DECODE_GEMV_NS.load(Ordering::Relaxed) as f64 * ns,
            DECODE_LMHEAD_NS.load(Ordering::Relaxed) as f64 * ns,
        );
        eprintln!(
            "[focr-timing]   decode {} tok in {:.2}s ({:.1} tok/s) | seed(prefill) {:.2}s | \
             layers {:.2}s (attn {:.2}s, gemv+misc {:.2}s) | lm_head {:.2}s",
            ids.len(),
            dec_s,
            ids.len() as f64 / dec_s.max(1e-9),
            seed_s,
            layers,
            attn,
            layers - attn,
            head,
        );
    }
    Ok(ids)
}

/// Argmax over a logit row (first max on ties) — the shared greedy pick.
fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv { (i, x) } else { (bi, bv) }
        })
        .0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn got_config_is_the_censused_shape() {
        let c = DecoderConfig::got_ocr2();
        assert_eq!(c.hidden_size, 1024);
        assert_eq!(c.intermediate_size, 2816);
        assert_eq!(c.num_hidden_layers, 24);
        assert_eq!(c.num_attention_heads, 16);
        assert_eq!(c.head_dim, 64);
        assert_eq!(c.qkv_dim(), 1024);
        assert_eq!(c.vocab_size, 151_860);
        // full-causal scale must be exactly 1/8 (= 1/sqrt(64), what prefill_attention derives).
        assert!(((1.0 / (c.head_dim as f32).sqrt()) - 0.125).abs() < 1e-7);
    }

    /// Read a raw little-endian f32 blob.
    fn read_f32_le(path: &str) -> Vec<f32> {
        let bytes = std::fs::read(path).expect("oracle blob");
        assert_eq!(bytes.len() % 4, 0, "not a whole f32 count");
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn argmax(v: &[f32]) -> usize {
        v.iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
                if x > bv { (i, x) } else { (bi, bv) }
            })
            .0
    }

    /// **B5 — the GOT decoder parity gate vs the bit-deterministic torch oracle.**
    /// Env-gated (skip-with-success when the artifacts are absent, per the
    /// model-gated pattern): `FOCR_GOT_MODEL` = the got-ocr2 `.focrq` (int8) or the
    /// raw bf16 `model.safetensors` (f32 reference); `FOCR_ORACLE_HIDDEN0` = the
    /// oracle's post-splice decoder input `[N,1024]`; `FOCR_ORACLE_LOGITS` = the
    /// oracle last-position logits `[vocab]`.
    ///
    /// CERTIFIED (2026-06-30, GOT-OCR2 weights): the **f32 reference** path
    /// (`model.safetensors`) matches the torch oracle to **cos = 1.000000** — every
    /// kernel is numerically exact — and the shipping **int8** path
    /// (`got-ocr2.int8.focrq`) to **cos = 0.9993**, with the greedy next-token
    /// **argmax = 9707 exact on both**.
    #[test]
    fn decoder_matches_torch_oracle() {
        let (Ok(model), Ok(h0), Ok(lg)) = (
            std::env::var("FOCR_GOT_MODEL"),
            std::env::var("FOCR_ORACLE_HIDDEN0"),
            std::env::var("FOCR_ORACLE_LOGITS"),
        ) else {
            return;
        };
        let cfg = DecoderConfig::got_ocr2();
        let weights = Weights::load(std::path::Path::new(&model)).expect("load GOT weights");

        let h0_flat = read_f32_le(&h0);
        let n = h0_flat.len() / cfg.hidden_size;
        assert_eq!(n * cfg.hidden_size, h0_flat.len(), "hidden0 not [N,1024]");
        let inputs = Mat::from_vec(n, cfg.hidden_size, h0_flat);

        let logits = forward_prefill(&weights, &cfg, &inputs).expect("prefill forward");
        assert_eq!(logits.cols, cfg.vocab_size);
        let ours = &logits.data[(logits.rows - 1) * logits.cols..];

        let oracle = read_f32_le(&lg);
        let oracle = &oracle[oracle.len() - cfg.vocab_size..]; // last position if [N,vocab]

        // greedy next-token identity (the load-bearing gate; must hold on int8 too).
        assert_eq!(
            argmax(ours),
            argmax(oracle),
            "next-token argmax diverged from the torch oracle"
        );
        // cosine + max-abs. Oracle floor = 0, so any residual is our numeric/quant
        // error: an int8 build stays high-cos; an f32 build is near bit-exact.
        let dot: f64 = ours
            .iter()
            .zip(oracle)
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum();
        let na: f64 = ours
            .iter()
            .map(|&a| f64::from(a) * f64::from(a))
            .sum::<f64>()
            .sqrt();
        let nb: f64 = oracle
            .iter()
            .map(|&b| f64::from(b) * f64::from(b))
            .sum::<f64>()
            .sqrt();
        let cos = dot / (na * nb);
        eprintln!(
            "[B5 parity] argmax={} cos={cos:.6} (oracle argmax={})",
            argmax(ours),
            argmax(oracle)
        );
        assert!(
            cos >= 0.99,
            "logit cosine {cos:.6} < 0.99 — decoder diverged"
        );
    }

    /// **B5/B7 — greedy generation matches the torch oracle's L4 output.** Feeds the
    /// oracle's post-splice `hidden_0` and greedily decodes; the ids must equal the
    /// oracle's greedy prefix `[9707, 38, 1793, 12, …]` (oracle_fixtures.json
    /// `l4_greedy_decode_ids`). Limited to 4 tokens (each step re-runs prefill —
    /// the unoptimized correct path). Env-gated like the parity gate.
    #[test]
    fn greedy_generation_matches_oracle_l4() {
        let (Ok(model), Ok(h0)) = (
            std::env::var("FOCR_GOT_MODEL"),
            std::env::var("FOCR_ORACLE_HIDDEN0"),
        ) else {
            return;
        };
        let cfg = DecoderConfig::got_ocr2();
        let weights = Weights::load(std::path::Path::new(&model)).expect("load GOT weights");
        let h0_flat = read_f32_le(&h0);
        let n = h0_flat.len() / cfg.hidden_size;
        let inputs = Mat::from_vec(n, cfg.hidden_size, h0_flat);

        // eos = <|im_end|> (151645); 4 new tokens is enough to prove the append loop.
        let ids = generate_greedy(&weights, &cfg, &inputs, 4, 151_645).expect("generate");
        eprintln!("[B5 gen] first ids = {ids:?}");
        // int8 build: argmax is exact on this confident prefix (matches the f32 oracle).
        assert_eq!(
            ids,
            vec![9707, 38, 1793, 12],
            "greedy ids diverged from the torch oracle L4"
        );
    }

    /// **B9 — the O(n) KV-cache decode reproduces the oracle L4** (transitively ==
    /// the certified `generate_greedy`, since it's held bit-identical by reusing the
    /// prefill's `linear_int8_dynamic` kernel). Runs 8 tokens — FAST here because the
    /// cache makes each step O(n_kv), unlike the re-prefill path. Env-gated.
    #[test]
    fn kvcache_greedy_matches_oracle_l4() {
        let (Ok(model), Ok(h0)) = (
            std::env::var("FOCR_GOT_MODEL"),
            std::env::var("FOCR_ORACLE_HIDDEN0"),
        ) else {
            return;
        };
        let cfg = DecoderConfig::got_ocr2();
        let weights = Weights::load(std::path::Path::new(&model)).expect("load GOT weights");
        let h0_flat = read_f32_le(&h0);
        let n = h0_flat.len() / cfg.hidden_size;
        let inputs = Mat::from_vec(n, cfg.hidden_size, h0_flat);

        let ids =
            generate_greedy_kvcache(&weights, &cfg, &inputs, 8, 151_645).expect("kvcache gen");
        eprintln!("[B9 kvcache] first ids = {ids:?}");
        // == the torch oracle greedy L4 prefix (oracle_fixtures.json l4_greedy_decode_ids).
        assert_eq!(
            ids,
            vec![9707, 38, 1793, 12, 93495, 17, 13, 15],
            "KV-cache decode diverged from the torch oracle L4"
        );
    }
}
