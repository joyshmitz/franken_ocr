//! TrOMR encoder (E3, bd-3jo6.5.3) — the fifth model lane's vision half
//! (census `docs/zoo/tromr-spec.md` §2a/§2b): a hybrid ResNetV2+ViT over one
//! grayscale staff crop `(1, 128, W)`, W ≤ 1280 a multiple of 16.
//!
//! Graph: TF-'SAME' stem `conv 1→64 k7 s2` → GN32+ReLU → −∞-pad max-pool k3
//! s2 → post-act Bottleneck stages `[2, 3, 7]` (widths 256/512/1024, strides
//! 1/2/2) → `(1024, 8, W/16)` → 1×1 proj to 256 + cls token + CROP-INDEXED
//! learned positions (row-major over an 80-wide table) → 4 pre-LN ViT blocks
//! (8 heads × 32, fused qkv, exact-erf GELU MLP 1024) → final LayerNorm →
//! `[1 + 8·W/16, 256]` — the cls token IS part of the decoder's
//! cross-attention context (§3: the connector is Identity).
//!
//! The stored backbone convs are PRE-WS-FOLDED (E2's export invokes timm's
//! own standardization arithmetic), so every conv here is a plain
//! [`nn::conv2d`] over a [`nn::tf_same_pad`]-prepared input — no runtime
//! weight standardization exists (spec §10.3). No conv biases anywhere in the
//! backbone; the backbone final norm is Identity (both census-confirmed
//! absent from the checkpoint).
//!
//! The 4-head AR decoder (E4, spec §4/§5) lives here too — a self-contained
//! x-transformers graph that does NOT ride `decoder_qwen2` (§10 non-fit):
//! 4 layers of ('a' causal self-attn, 'c' cross-attn over the encoder
//! context, 'f' GEGLU ff), all pre-LN (eps 1e-5) + residual, inner 512 ≠ dim
//! 256, GLU-gated bias-free `on_attn` out-projections, a summed 3-embedding
//! input (+ scaled learned positions), and FOUR parallel heads off one final
//! norm. [`generate`] is the port's DETERMINISTIC per-head-argmax default;
//! upstream's top-k/T=0.2 sampling is the `FOCR_TROMR_SAMPLE` kill-switch
//! (measured divergence — a DISCREPANCIES entry when it lands, spec §5).

use crate::error::{FocrError, FocrResult};

use super::nn;
use super::tensor::Mat;
use super::vision_sam::Linear;
use super::weights::Weights;

/// Staff-crop input height (config `max_height`; spec §6 resizes to this).
pub const IMG_H: usize = 128;
/// ViT patch stride — the backbone's total 16× downsample (spec §2b).
pub const PATCH: usize = 16;
/// Encoder/decoder shared width (`emb_dim == dim == 256`, §3).
pub const DIM: usize = 256;
/// The learned position table is laid out for this many patch COLUMNS
/// (1280/16); a narrower crop crop-indexes its top-left block (§2b).
pub const POS_COLS: usize = 80;
/// Patch rows for the fixed 128-high input (128/16).
pub const POS_ROWS: usize = 8;
const VIT_HEADS: usize = 8;
const VIT_HEAD_DIM: usize = 32;
const GN_GROUPS: usize = 32;
const GN_EPS: f32 = 1e-5;
const LN_EPS: f32 = 1e-6;

/// One flat batch-1 NCHW feature map — the backbone currency.
struct Feature {
    data: Vec<f32>,
    ch: usize,
    h: usize,
    w: usize,
}

/// A backbone conv (no bias — census) + its following GroupNorm params.
/// `norm` is `None` only where the graph has a bare conv (never happens in
/// this backbone: every conv is followed by a GN, with or without ReLU).
struct ConvGn {
    w: Vec<f32>,
    out_ch: usize,
    in_ch: usize,
    k: usize,
    stride: usize,
    gn_w: Vec<f32>,
    gn_b: Vec<f32>,
}

impl ConvGn {
    /// TF-'SAME' pad (zero fill) → conv → GroupNorm(32, 1e-5) with optional
    /// fused ReLU.
    fn apply(&self, x: &Feature, relu: bool) -> FocrResult<Feature> {
        let (padded, ph, pw) = nn::tf_same_pad(
            &x.data,
            1,
            x.ch,
            x.h,
            x.w,
            self.k,
            self.k,
            self.stride,
            self.stride,
            0.0,
        );
        let (oh, ow) = (x.h.div_ceil(self.stride), x.w.div_ceil(self.stride));
        let mut data = nn::conv2d(
            &padded,
            &self.w,
            None,
            1,
            self.in_ch,
            ph,
            pw,
            self.k,
            self.k,
            oh,
            ow,
            self.stride,
            self.stride,
            self.out_ch,
        );
        nn::group_norm(
            &mut data,
            1,
            self.out_ch,
            oh * ow,
            GN_GROUPS,
            GN_EPS,
            &self.gn_w,
            &self.gn_b,
            relu,
        )?;
        Ok(Feature {
            data,
            ch: self.out_ch,
            h: oh,
            w: ow,
        })
    }
}

/// One post-act Bottleneck block (timm ResNetV2, preact=False — spec §2a):
/// `conv1 1×1 → GN+ReLU → conv2 3×3 (stride) → GN+ReLU → conv3 1×1 → GN(no
/// act) → + shortcut → ReLU`; block 0 of a stage downsamples the shortcut
/// with `1×1 (stride) → GN(no act)`.
struct Bottleneck {
    conv1: ConvGn,
    conv2: ConvGn,
    conv3: ConvGn,
    downsample: Option<ConvGn>,
}

impl Bottleneck {
    fn apply(&self, x: &Feature) -> FocrResult<Feature> {
        let shortcut = match &self.downsample {
            Some(d) => d.apply(x, false)?,
            None => Feature {
                data: x.data.clone(),
                ch: x.ch,
                h: x.h,
                w: x.w,
            },
        };
        let h = self.conv1.apply(x, true)?;
        let h = self.conv2.apply(&h, true)?;
        let mut h = self.conv3.apply(&h, false)?;
        if h.data.len() != shortcut.data.len() {
            return Err(FocrError::Other(anyhow::anyhow!(
                "tromr bottleneck: residual len {} != shortcut len {}",
                h.data.len(),
                shortcut.data.len()
            )));
        }
        for (a, b) in h.data.iter_mut().zip(shortcut.data.iter()) {
            *a = (*a + b).max(0.0);
        }
        Ok(h)
    }
}

/// One pre-LN ViT block (spec §2b): LN(1e-6) → fused-qkv MHA (8×32, scale
/// 32^-0.5) → +res; LN → fc1 1024 → exact-erf GELU → fc2 → +res.
struct VitBlock {
    ln1_w: Vec<f32>,
    ln1_b: Vec<f32>,
    qkv: Linear,
    proj: Linear,
    ln2_w: Vec<f32>,
    ln2_b: Vec<f32>,
    fc1: Linear,
    fc2: Linear,
}

/// The hydrated encoder weights.
pub struct TromrEncoderW {
    stem: ConvGn,
    stages: Vec<Vec<Bottleneck>>,
    patch_proj: Linear,
    cls_token: Vec<f32>,
    pos_embed: Vec<f32>,
    blocks: Vec<VitBlock>,
    final_ln_w: Vec<f32>,
    final_ln_b: Vec<f32>,
}

impl TromrEncoderW {
    /// Hydrate from the (WS-pre-folded) artifact — spec §12 names verbatim.
    ///
    /// # Errors
    /// A missing tensor or a shape violation.
    pub fn build(weights: &Weights) -> FocrResult<Self> {
        let b = "encoder.patch_embed.backbone.";
        let conv_gn = |conv: String,
                       norm: String,
                       out_ch: usize,
                       in_ch: usize,
                       k: usize,
                       stride: usize|
         -> FocrResult<ConvGn> {
            Ok(ConvGn {
                w: weights.vec(&conv)?,
                out_ch,
                in_ch,
                k,
                stride,
                gn_w: weights.vec(&format!("{norm}.weight"))?,
                gn_b: weights.vec(&format!("{norm}.bias"))?,
            })
        };

        let stem = conv_gn(
            format!("{b}stem.conv.weight"),
            format!("{b}stem.norm"),
            64,
            1,
            7,
            2,
        )?;

        // Stages [2, 3, 7]; (in, mid, out, stride) per census §2a/§12.
        let plan: [(usize, usize, usize, usize, usize); 3] = [
            (2, 64, 64, 256, 1),
            (3, 256, 128, 512, 2),
            (7, 512, 256, 1024, 2),
        ];
        let mut stages = Vec::with_capacity(3);
        for (s, &(blocks_n, stage_in, mid, out, stage_stride)) in plan.iter().enumerate() {
            let mut blocks = Vec::with_capacity(blocks_n);
            for blk in 0..blocks_n {
                let p = format!("{b}stages.{s}.blocks.{blk}.");
                let (in_ch, stride) = if blk == 0 {
                    (stage_in, stage_stride)
                } else {
                    (out, 1)
                };
                let downsample = if blk == 0 {
                    Some(conv_gn(
                        format!("{p}downsample.conv.weight"),
                        format!("{p}downsample.norm"),
                        out,
                        in_ch,
                        1,
                        stride,
                    )?)
                } else {
                    None
                };
                blocks.push(Bottleneck {
                    conv1: conv_gn(
                        format!("{p}conv1.weight"),
                        format!("{p}norm1"),
                        mid,
                        in_ch,
                        1,
                        1,
                    )?,
                    conv2: conv_gn(
                        format!("{p}conv2.weight"),
                        format!("{p}norm2"),
                        mid,
                        mid,
                        3,
                        stride,
                    )?,
                    conv3: conv_gn(
                        format!("{p}conv3.weight"),
                        format!("{p}norm3"),
                        out,
                        mid,
                        1,
                        1,
                    )?,
                    downsample,
                });
            }
            stages.push(blocks);
        }

        let lin = |wname: String, bname: String, out: usize, in_: usize| -> FocrResult<Linear> {
            Ok(Linear {
                w: weights.vec(&wname)?,
                b: weights.vec(&bname)?,
                out,
                in_,
            })
        };
        let mut blocks = Vec::with_capacity(4);
        for i in 0..4 {
            let p = format!("encoder.blocks.{i}.");
            blocks.push(VitBlock {
                ln1_w: weights.vec(&format!("{p}norm1.weight"))?,
                ln1_b: weights.vec(&format!("{p}norm1.bias"))?,
                qkv: lin(
                    format!("{p}attn.qkv.weight"),
                    format!("{p}attn.qkv.bias"),
                    3 * DIM,
                    DIM,
                )?,
                proj: lin(
                    format!("{p}attn.proj.weight"),
                    format!("{p}attn.proj.bias"),
                    DIM,
                    DIM,
                )?,
                ln2_w: weights.vec(&format!("{p}norm2.weight"))?,
                ln2_b: weights.vec(&format!("{p}norm2.bias"))?,
                fc1: lin(
                    format!("{p}mlp.fc1.weight"),
                    format!("{p}mlp.fc1.bias"),
                    4 * DIM,
                    DIM,
                )?,
                fc2: lin(
                    format!("{p}mlp.fc2.weight"),
                    format!("{p}mlp.fc2.bias"),
                    DIM,
                    4 * DIM,
                )?,
            });
        }

        Ok(Self {
            stem,
            stages,
            patch_proj: lin(
                "encoder.patch_embed.proj.weight".into(),
                "encoder.patch_embed.proj.bias".into(),
                DIM,
                1024,
            )?,
            cls_token: weights.vec("encoder.cls_token")?,
            pos_embed: weights.vec("encoder.pos_embed")?,
            blocks,
            final_ln_w: weights.vec("encoder.norm.weight")?,
            final_ln_b: weights.vec("encoder.norm.bias")?,
        })
    }
}

/// The ResNetV2 backbone: staff tensor `(1, 128, W)` → `(1024, 8, W/16)`.
fn backbone(w: &TromrEncoderW, pixels: &[f32], width: usize) -> FocrResult<Feature> {
    if width == 0 || !width.is_multiple_of(PATCH) || width > POS_COLS * PATCH {
        return Err(FocrError::Other(anyhow::anyhow!(
            "tromr: width {width} must be a non-zero multiple of {PATCH} <= {} (spec §2b \
             crop-indexed positions go undefined past 1280)",
            POS_COLS * PATCH
        )));
    }
    if pixels.len() != IMG_H * width {
        return Err(FocrError::Other(anyhow::anyhow!(
            "tromr: pixel buffer {} != 1*{IMG_H}*{width}",
            pixels.len()
        )));
    }
    let x = Feature {
        data: pixels.to_vec(),
        ch: 1,
        h: IMG_H,
        w: width,
    };
    // Stem: conv7 s2 (GN+ReLU) then the −∞-padded s2 max-pool.
    let x = w.stem.apply(&x, true)?;
    let (padded, ph, pw) =
        nn::tf_same_pad(&x.data, 1, x.ch, x.h, x.w, 3, 3, 2, 2, f32::NEG_INFINITY);
    let (oh, ow) = (x.h.div_ceil(2), x.w.div_ceil(2));
    let mut x = Feature {
        data: nn::max_pool2d(&padded, 1, x.ch, ph, pw, 3, 2, oh, ow),
        ch: x.ch,
        h: oh,
        w: ow,
    };
    for stage in &w.stages {
        for block in stage {
            x = block.apply(&x)?;
        }
    }
    Ok(x)
}

/// Channel-major `(C, H·W)` → token-major `[H·W, C]`.
fn tokens_from_feature(f: &Feature) -> Mat {
    let spatial = f.h * f.w;
    let mut out = vec![0.0f32; spatial * f.ch];
    for c in 0..f.ch {
        for s in 0..spatial {
            out[s * f.ch + c] = f.data[c * spatial + s];
        }
    }
    Mat::from_vec(spatial, f.ch, out)
}

/// Fused-qkv bidirectional MHA (8 heads × 32, scale 32^-0.5).
fn self_attention(blk: &VitBlock, x: &Mat) -> FocrResult<Mat> {
    let seq = x.rows;
    let qkv = blk.qkv.apply(x)?; // [seq, 768] = q|k|v
    let head_span = seq * VIT_HEAD_DIM;
    let mut qf = vec![0.0f32; VIT_HEADS * head_span];
    let mut kf = vec![0.0f32; VIT_HEADS * head_span];
    let mut vf = vec![0.0f32; VIT_HEADS * head_span];
    for s in 0..seq {
        let row = qkv.row(s);
        for h in 0..VIT_HEADS {
            let dst = h * head_span + s * VIT_HEAD_DIM;
            let src = h * VIT_HEAD_DIM;
            qf[dst..dst + VIT_HEAD_DIM].copy_from_slice(&row[src..src + VIT_HEAD_DIM]);
            kf[dst..dst + VIT_HEAD_DIM].copy_from_slice(&row[DIM + src..DIM + src + VIT_HEAD_DIM]);
            vf[dst..dst + VIT_HEAD_DIM]
                .copy_from_slice(&row[2 * DIM + src..2 * DIM + src + VIT_HEAD_DIM]);
        }
    }
    let scale = 1.0 / (VIT_HEAD_DIM as f32).sqrt();
    let ctx = nn::sdpa(
        &qf,
        &kf,
        &vf,
        VIT_HEADS,
        seq,
        seq,
        VIT_HEAD_DIM,
        VIT_HEAD_DIM,
        scale,
        false,
    );
    // Head-major back to [seq, 256].
    let mut merged = vec![0.0f32; seq * DIM];
    for h in 0..VIT_HEADS {
        for s in 0..seq {
            let src = h * head_span + s * VIT_HEAD_DIM;
            let dst = s * DIM + h * VIT_HEAD_DIM;
            merged[dst..dst + VIT_HEAD_DIM].copy_from_slice(&ctx[src..src + VIT_HEAD_DIM]);
        }
    }
    blk.proj.apply(&Mat::from_vec(seq, DIM, merged))
}

fn add_assign(x: &mut Mat, y: &Mat) -> FocrResult<()> {
    if x.rows != y.rows || x.cols != y.cols {
        return Err(FocrError::Other(anyhow::anyhow!(
            "tromr add_assign: [{}, {}] += [{}, {}]",
            x.rows,
            x.cols,
            y.rows,
            y.cols
        )));
    }
    for (a, b) in x.data.iter_mut().zip(y.data.iter()) {
        *a += b;
    }
    Ok(())
}

/// The full E3 encoder: staff tensor `(1, 128, W)` flat → the decoder's
/// cross-attention context `[1 + 8·(W/16), 256]` (cls first — §2b).
///
/// # Errors
/// Shape violations, a missing tensor, or a kernel failure.
pub fn encode(w: &TromrEncoderW, pixels: &[f32], width: usize) -> FocrResult<Mat> {
    let feat = backbone(w, pixels, width)?;
    let x = tokens_from_feature(&feat); // [8·wp, 1024] row-major (r, c)
    let x = w.patch_proj.apply(&x)?; // [8·wp, 256]

    let (rows, wp) = (feat.h, feat.w);
    let seq = 1 + rows * wp;
    let mut tok = Mat::from_vec(seq, DIM, vec![0.0f32; seq * DIM]);
    // cls token + pos[0].
    for d in 0..DIM {
        tok.data[d] = w.cls_token[d] + w.pos_embed[d];
    }
    // Patch tokens + CROP-INDEXED positions: (r, c) → pos_embed[1 + r·80 + c].
    for r in 0..rows {
        for c in 0..wp {
            let t = 1 + r * wp + c;
            let pos = (1 + r * POS_COLS + c) * DIM;
            let src = (r * wp + c) * DIM;
            for d in 0..DIM {
                tok.data[t * DIM + d] = x.data[src + d] + w.pos_embed[pos + d];
            }
        }
    }

    for blk in &w.blocks {
        let h = nn::layer_norm(&tok, Some(&blk.ln1_w), Some(&blk.ln1_b), LN_EPS)?;
        let attn = self_attention(blk, &h)?;
        add_assign(&mut tok, &attn)?;
        let h2 = nn::layer_norm(&tok, Some(&blk.ln2_w), Some(&blk.ln2_b), LN_EPS)?;
        let mut m = blk.fc1.apply(&h2)?;
        nn::gelu(&mut m);
        let m = blk.fc2.apply(&m)?;
        add_assign(&mut tok, &m)?;
    }
    nn::layer_norm(&tok, Some(&w.final_ln_w), Some(&w.final_ln_b), LN_EPS)
}

// ───────────────────────── E4: the 4-head AR decoder ─────────────────────────

/// Decoder pre-branch LayerNorm eps (torch default — x-transformers passes
/// none; spec §4. NOTE: 1e-5, unlike the encoder's 1e-6).
const DEC_LN_EPS: f32 = 1e-5;
/// Attention inner width (8 heads × 64 — inner 512 ≠ dim 256, spec §4).
const DEC_INNER: usize = 512;
const DEC_HEADS: usize = 8;
const DEC_HEAD_DIM: usize = 64;
/// `max_seq_len` (config): the position table height AND the generate cap.
pub const MAX_SEQ: usize = 256;
/// Learned positions are scaled by `dim^-0.5 = 1/16` (x_transformers §4).
const POS_SCALE: f32 = 1.0 / 16.0;
/// Rhythm-stream generate seeds (config `bos_token`/`nonote_token`).
const SEED_RHYTHM: u32 = 1;
const SEED_NONOTE: u32 = 0;

/// One attention sublayer's weights: `to_{q,k,v} [512, 256]` and the
/// `on_attn` out projection `[512, 512]` — ALL bias-free (census §12/§16).
struct AttnW {
    to_q: Vec<f32>,
    to_k: Vec<f32>,
    to_v: Vec<f32>,
    to_out: Vec<f32>,
}

/// A pre-branch LayerNorm's affine params.
struct Ln {
    w: Vec<f32>,
    b: Vec<f32>,
}

/// One of the 4 decoder layers: ('a' self-attn, 'c' cross-attn, 'f' GEGLU
/// feed-forward), each pre-norm + residual (spec §4).
struct DecLayer {
    ln_a: Ln,
    self_attn: AttnW,
    ln_c: Ln,
    cross_attn: AttnW,
    ln_f: Ln,
    ff_proj: Linear,
    ff_out: Linear,
}

/// The hydrated TrOMR decoder (spec §12 names verbatim).
pub struct TromrDecoderW {
    rhythm_emb: Vec<f32>,
    pitch_emb: Vec<f32>,
    lift_emb: Vec<f32>,
    pos_emb: Vec<f32>,
    layers: Vec<DecLayer>,
    final_ln: Ln,
    /// The four parallel per-stream heads (spec §4) — public: E7's assembly
    /// applies rhythm/pitch/lift per step, and the note head (inference-dead
    /// upstream, spec §5) stays exposed for the cert + future consistency
    /// diagnostics.
    pub head_rhythm: Linear,
    /// Pitch head `[71, 256]`.
    pub head_pitch: Linear,
    /// Lift head `[7, 256]`.
    pub head_lift: Linear,
    /// Note head `[2, 256]` (output-only; discarded at inference upstream).
    pub head_note: Linear,
}

impl TromrDecoderW {
    /// Hydrate from the artifact. The flat x-transformers layout indexes
    /// sublayers `layers.{i}` with `i%3` ⇒ 0='a', 1='c', 2='f' (spec §4);
    /// `layers.{i}.0.0` is the pre-branch norm, `layers.{i}.1` the branch.
    ///
    /// # Errors
    /// A missing tensor or a shape violation.
    pub fn build(weights: &Weights) -> FocrResult<Self> {
        let ln = |name: String| -> FocrResult<Ln> {
            Ok(Ln {
                w: weights.vec(&format!("{name}.weight"))?,
                b: weights.vec(&format!("{name}.bias"))?,
            })
        };
        let attn = |i: usize| -> FocrResult<AttnW> {
            let p = format!("decoder.net.attn_layers.layers.{i}.1.");
            Ok(AttnW {
                to_q: weights.vec(&format!("{p}to_q.weight"))?,
                to_k: weights.vec(&format!("{p}to_k.weight"))?,
                to_v: weights.vec(&format!("{p}to_v.weight"))?,
                to_out: weights.vec(&format!("{p}to_out.0.weight"))?,
            })
        };
        let head = |stream: &str, vocab: usize| -> FocrResult<Linear> {
            Ok(Linear {
                w: weights.vec(&format!("decoder.net.to_logits_{stream}.weight"))?,
                b: weights.vec(&format!("decoder.net.to_logits_{stream}.bias"))?,
                out: vocab,
                in_: DIM,
            })
        };
        let mut layers = Vec::with_capacity(4);
        for l in 0..4 {
            let base = 3 * l;
            layers.push(DecLayer {
                ln_a: ln(format!("decoder.net.attn_layers.layers.{base}.0.0"))?,
                self_attn: attn(base)?,
                ln_c: ln(format!("decoder.net.attn_layers.layers.{}.0.0", base + 1))?,
                cross_attn: attn(base + 1)?,
                ln_f: ln(format!("decoder.net.attn_layers.layers.{}.0.0", base + 2))?,
                ff_proj: Linear {
                    w: weights.vec(&format!(
                        "decoder.net.attn_layers.layers.{}.1.net.0.proj.weight",
                        base + 2
                    ))?,
                    b: weights.vec(&format!(
                        "decoder.net.attn_layers.layers.{}.1.net.0.proj.bias",
                        base + 2
                    ))?,
                    out: 2048,
                    in_: DIM,
                },
                ff_out: Linear {
                    w: weights.vec(&format!(
                        "decoder.net.attn_layers.layers.{}.1.net.3.weight",
                        base + 2
                    ))?,
                    b: weights.vec(&format!(
                        "decoder.net.attn_layers.layers.{}.1.net.3.bias",
                        base + 2
                    ))?,
                    out: DIM,
                    in_: 1024,
                },
            });
        }
        Ok(Self {
            rhythm_emb: weights.vec("decoder.net.rhythm_emb.emb.weight")?,
            pitch_emb: weights.vec("decoder.net.pitch_emb.emb.weight")?,
            lift_emb: weights.vec("decoder.net.lift_emb.emb.weight")?,
            pos_emb: weights.vec("decoder.net.pos_emb.emb.weight")?,
            layers,
            final_ln: ln("decoder.net.norm".into())?,
            head_rhythm: head("rhythm", 260)?,
            head_pitch: head("pitch", 71)?,
            head_lift: head("lift", 7)?,
            head_note: head("note", 2)?,
        })
    }
}

/// Bias-free `[out, in]` projection: `y = x @ w^T`.
fn proj_no_bias(x: &Mat, w: &[f32], out: usize) -> FocrResult<Mat> {
    let lin = Linear {
        w: w.to_vec(),
        b: Vec::new(),
        out,
        in_: x.cols,
    };
    lin.apply(x)
}

/// One `on_attn` attention branch (self or cross — spec §4): q from `x_q`,
/// k/v from `kv`, 8 heads × 64 at scale 1/8 (stable softmax inside the sdpa
/// kernel — OQ-T4), then `Linear(512→512, no bias)` + GLU (`a · σ(b)`).
fn glu_attention(a: &AttnW, x_q: &Mat, kv: &Mat, causal: bool) -> FocrResult<Mat> {
    let (seq_q, seq_k) = (x_q.rows, kv.rows);
    let q = proj_no_bias(x_q, &a.to_q, DEC_INNER)?;
    let k = proj_no_bias(kv, &a.to_k, DEC_INNER)?;
    let v = proj_no_bias(kv, &a.to_v, DEC_INNER)?;

    // Repack [seq, 512] → head-major [8, seq, 64].
    let pack = |m: &Mat, seq: usize| -> Vec<f32> {
        let span = seq * DEC_HEAD_DIM;
        let mut out = vec![0.0f32; DEC_HEADS * span];
        for s in 0..seq {
            let row = m.row(s);
            for h in 0..DEC_HEADS {
                let dst = h * span + s * DEC_HEAD_DIM;
                out[dst..dst + DEC_HEAD_DIM]
                    .copy_from_slice(&row[h * DEC_HEAD_DIM..(h + 1) * DEC_HEAD_DIM]);
            }
        }
        out
    };
    let (qf, kf, vf) = (pack(&q, seq_q), pack(&k, seq_k), pack(&v, seq_k));
    let scale = 1.0 / (DEC_HEAD_DIM as f32).sqrt();
    let ctx = nn::sdpa(
        &qf,
        &kf,
        &vf,
        DEC_HEADS,
        seq_q,
        seq_k,
        DEC_HEAD_DIM,
        DEC_HEAD_DIM,
        scale,
        causal,
    );
    // Merge back to [seq_q, 512].
    let span = seq_q * DEC_HEAD_DIM;
    let mut merged = vec![0.0f32; seq_q * DEC_INNER];
    for h in 0..DEC_HEADS {
        for s in 0..seq_q {
            let src = h * span + s * DEC_HEAD_DIM;
            let dst = s * DEC_INNER + h * DEC_HEAD_DIM;
            merged[dst..dst + DEC_HEAD_DIM].copy_from_slice(&ctx[src..src + DEC_HEAD_DIM]);
        }
    }
    // on_attn: Linear(512→512, no bias) then GLU split 2×256: `a · σ(b)`.
    let o = proj_no_bias(
        &Mat::from_vec(seq_q, DEC_INNER, merged),
        &a.to_out,
        DEC_INNER,
    )?;
    let mut out = vec![0.0f32; seq_q * DIM];
    for s in 0..seq_q {
        let row = o.row(s);
        for d in 0..DIM {
            out[s * DIM + d] = row[d] * (1.0 / (1.0 + (-row[DIM + d]).exp()));
        }
    }
    Ok(Mat::from_vec(seq_q, DIM, out))
}

/// The full-prefix decoder forward (upstream-faithful: NO KV cache — spec §4
/// notes upstream re-forwards the whole prefix; at 256×256 this is trivially
/// cheap, and a cache is a later bit-proven lever). Returns the final-normed
/// hidden `[t, 256]` for the (rhythm, pitch, lift) prefix over the encoder
/// `ctx` (`[1+8·wp, 256]`).
///
/// # Errors
/// Length mismatches between the three streams, an empty prefix, or a
/// prefix past [`MAX_SEQ`].
pub fn decoder_forward(
    w: &TromrDecoderW,
    ctx: &Mat,
    rhythm: &[u32],
    pitch: &[u32],
    lift: &[u32],
) -> FocrResult<Mat> {
    let t = rhythm.len();
    if t == 0 || t > MAX_SEQ || pitch.len() != t || lift.len() != t {
        return Err(FocrError::Other(anyhow::anyhow!(
            "tromr decoder: stream lens (r {}, p {}, l {}) must be equal, 1..={MAX_SEQ}",
            rhythm.len(),
            pitch.len(),
            lift.len()
        )));
    }
    // x_t = rhythm_emb[r] + pitch_emb[p] + lift_emb[l] + pos[t]/16 (spec §4).
    let mut x = Mat::from_vec(t, DIM, vec![0.0f32; t * DIM]);
    for (i, ((&r, &p), &l)) in rhythm.iter().zip(pitch).zip(lift).enumerate() {
        let (r, p, l) = (r as usize, p as usize, l as usize);
        if r >= 260 || p >= 71 || l >= 7 {
            return Err(FocrError::Other(anyhow::anyhow!(
                "tromr decoder: id out of table at step {i} (r {r}, p {p}, l {l})"
            )));
        }
        for d in 0..DIM {
            x.data[i * DIM + d] = w.rhythm_emb[r * DIM + d]
                + w.pitch_emb[p * DIM + d]
                + w.lift_emb[l * DIM + d]
                + w.pos_emb[i * DIM + d] * POS_SCALE;
        }
    }
    for layer in &w.layers {
        let h = nn::layer_norm(&x, Some(&layer.ln_a.w), Some(&layer.ln_a.b), DEC_LN_EPS)?;
        let a = glu_attention(&layer.self_attn, &h, &h, true)?;
        add_assign(&mut x, &a)?;
        let h = nn::layer_norm(&x, Some(&layer.ln_c.w), Some(&layer.ln_c.b), DEC_LN_EPS)?;
        let c = glu_attention(&layer.cross_attn, &h, ctx, false)?;
        add_assign(&mut x, &c)?;
        let h = nn::layer_norm(&x, Some(&layer.ln_f.w), Some(&layer.ln_f.b), DEC_LN_EPS)?;
        // GEGLU: proj → chunk (x, gate) 2×1024 → x · GELU(gate) → out. The
        // gate halves are gathered into one Mat so the exact-erf GELU runs
        // vectorized, then multiplied back against the value halves.
        let pr = layer.ff_proj.apply(&h)?;
        let mut gate = Mat::from_vec(t, 1024, vec![0.0f32; t * 1024]);
        for s in 0..t {
            gate.data[s * 1024..(s + 1) * 1024].copy_from_slice(&pr.row(s)[1024..2048]);
        }
        nn::gelu(&mut gate);
        let mut gated = Mat::from_vec(t, 1024, vec![0.0f32; t * 1024]);
        for s in 0..t {
            let row = pr.row(s);
            for (g, (&x_val, &g_val)) in gated.data[s * 1024..(s + 1) * 1024].iter_mut().zip(
                row[..1024]
                    .iter()
                    .zip(gate.data[s * 1024..(s + 1) * 1024].iter()),
            ) {
                *g = x_val * g_val;
            }
        }
        let f = layer.ff_out.apply(&gated)?;
        add_assign(&mut x, &f)?;
    }
    nn::layer_norm(&x, Some(&w.final_ln.w), Some(&w.final_ln.b), DEC_LN_EPS)
}

/// The three generated id streams (seeds excluded), positionally rhythm /
/// pitch / lift end-to-end (the §4 naming-swap trap cancels; never "fix" it).
pub struct MusicStreams {
    /// Rhythm ids (the stream that carries `[EOS]`; includes it when emitted).
    pub rhythm: Vec<u32>,
    /// Pitch ids.
    pub pitch: Vec<u32>,
    /// Lift (accidental) ids.
    pub lift: Vec<u32>,
}

/// Greedy (per-head argmax) generation — the port's DETERMINISTIC default
/// (spec §5 port decision; upstream samples at T=0.2 behind top-k, which is
/// the `FOCR_TROMR_SAMPLE` kill-switch, NOT implemented here). Seeds
/// rhythm=[BOS]=1, pitch=lift=nonote=0; stops on rhythm `[EOS]`=2 or after
/// [`MAX_SEQ`] steps. The note head is inference-dead (spec §5) and skipped.
///
/// # Errors
/// A decoder-forward failure.
pub fn generate(w: &TromrDecoderW, ctx: &Mat) -> FocrResult<MusicStreams> {
    let mut rhythm = vec![SEED_RHYTHM];
    let mut pitch = vec![SEED_NONOTE];
    let mut lift = vec![SEED_NONOTE];
    for _ in 0..MAX_SEQ {
        // Upstream windows the prefix to the LAST max_seq_len positions.
        let start = rhythm.len().saturating_sub(MAX_SEQ);
        let hidden = decoder_forward(w, ctx, &rhythm[start..], &pitch[start..], &lift[start..])?;
        let last = Mat::from_vec(1, DIM, hidden.row(hidden.rows - 1).to_vec());
        let argmax = |head: &Linear| -> FocrResult<u32> {
            let logits = head.apply(&last)?;
            Ok(logits
                .data
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                    if v > bv { (i, v) } else { (bi, bv) }
                })
                .0 as u32)
        };
        let r = argmax(&w.head_rhythm)?;
        rhythm.push(r);
        pitch.push(argmax(&w.head_pitch)?);
        lift.push(argmax(&w.head_lift)?);
        if r == crate::tokenizer::music::EOS_ID {
            break;
        }
    }
    Ok(MusicStreams {
        rhythm: rhythm[1..].to_vec(),
        pitch: pitch[1..].to_vec(),
        lift: lift[1..].to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zoo_dir() -> Option<std::path::PathBuf> {
        let dir = std::env::var_os("FOCR_TROMR_DIR").map(std::path::PathBuf::from)?;
        dir.join("tromr.focrq").is_file().then_some(dir)
    }

    fn read_f32(path: &std::path::Path) -> Vec<f32> {
        let bytes = std::fs::read(path).expect("fixture bin reads");
        bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    }

    fn cos(a: &[f32], b: &[f32]) -> f64 {
        let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b.iter()) {
            dot += f64::from(x) * f64::from(y);
            na += f64::from(x) * f64::from(x);
            nb += f64::from(y) * f64::from(y);
        }
        dot / (na.sqrt() * nb.sqrt()).max(1e-30)
    }

    fn maxabs(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn width_and_buffer_guards_reject() {
        // Guards fire BEFORE weight access, so a dummy hydration works:
        // synthesize via the error path (no zoo needed).
        let Some(dir) = zoo_dir() else {
            eprintln!("[tromr-test] skip_no_model: FOCR_TROMR_DIR unset (guard leg included)");
            return;
        };
        let weights = Weights::load(&dir.join("tromr.focrq")).expect("artifact loads");
        let w = TromrEncoderW::build(&weights).expect("hydrates");
        // width not ×16, width 0, width > 1280, short buffer — all clean errors.
        assert!(encode(&w, &vec![0.0; IMG_H * 100], 100).is_err());
        assert!(encode(&w, &[], 0).is_err());
        assert!(encode(&w, &vec![0.0; IMG_H * 1296], 1296).is_err());
        assert!(encode(&w, &[0.0; 7], 800).is_err());
    }

    /// The E4 L3 cert (step-0 head logits) + L4 cert (argmax generate
    /// token-exact): the decoder runs over the ORACLE's encoder context
    /// (isolation — the encoder has its own cert), so any divergence is the
    /// decoder's. The oracle's argmax generate is proven deterministic in
    /// the fixture (`argmax_generate_deterministic: true`), so L4 expects
    /// EXACT streams. Model-gated skip-with-SUCCESS.
    #[test]
    fn tromr_decoder_matches_argmax_oracle() {
        let Some(dir) = zoo_dir() else {
            eprintln!("[tromr-test] skip_no_model: FOCR_TROMR_DIR unset");
            return;
        };
        let fx_path = dir.join("tromr_oracle_fixtures.json");
        if !fx_path.is_file() || !dir.join("tromr_seam_head0_rhythm.bin").is_file() {
            eprintln!("[tromr-test] skip_no_model: decoder fixtures absent");
            return;
        }
        let fx: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(fx_path).unwrap()).unwrap();
        assert_eq!(
            fx["nondeterminism_floor"]["argmax_generate_deterministic"],
            serde_json::Value::Bool(true),
            "the oracle argmax run must be deterministic for an exact L4 gate"
        );

        let weights = Weights::load(&dir.join("tromr.focrq")).expect("artifact loads");
        let dec = TromrDecoderW::build(&weights).expect("decoder hydrates");
        let ctx_flat = read_f32(&dir.join("tromr_seam_encoder_out.bin"));
        let seq = ctx_flat.len() / DIM;
        let ctx = Mat::from_vec(seq, DIM, ctx_flat);

        // L3: step-0 hidden over the seeds → all four heads vs the oracle.
        let hidden = decoder_forward(&dec, &ctx, &[1], &[0], &[0]).expect("prefill runs");
        let last = Mat::from_vec(1, DIM, hidden.row(hidden.rows - 1).to_vec());
        for (stream, head) in [
            ("rhythm", &dec.head_rhythm),
            ("pitch", &dec.head_pitch),
            ("lift", &dec.head_lift),
            ("note", &dec.head_note),
        ] {
            let ours = head.apply(&last).expect("head applies");
            let oracle = read_f32(&dir.join(format!("tromr_seam_head0_{stream}.bin")));
            assert_eq!(ours.data.len(), oracle.len(), "{stream} head width");
            let (c, m) = (cos(&ours.data, &oracle), maxabs(&ours.data, &oracle));
            eprintln!("[tromr-cert] head0_{stream} cos {c:.8} maxabs {m:.3e}");
            assert!(c >= 0.9999, "head0_{stream} cos {c}");
        }

        // L4: full argmax generate over the oracle context — token-EXACT.
        let streams = generate(&dec, &ctx).expect("generate runs");
        let want = |k: &str| -> Vec<u32> {
            fx["argmax_generate"][k]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
                .collect()
        };
        assert_eq!(streams.rhythm, want("rhythm"), "rhythm stream");
        assert_eq!(streams.pitch, want("pitch"), "pitch stream");
        assert_eq!(streams.lift, want("lift"), "lift stream");
        eprintln!(
            "[tromr-cert] L4 argmax generate EXACT: {} steps, rhythm ends [barline, EOS]",
            streams.rhythm.len()
        );
    }

    /// The E3 L1/L2 cert: every oracle seam (stem, stages, patch proj, each
    /// ViT block, the final norm) at cosine ≥ 0.9999 with maxabs ledgered;
    /// the oracle's own floor on this stack is 0.0 (same- AND cross-thread),
    /// so every divergence below is OUR summation-order envelope, reported
    /// per seam. Model-gated skip-with-SUCCESS.
    #[test]
    fn tromr_encoder_matches_torch_oracle() {
        let Some(dir) = zoo_dir() else {
            eprintln!("[tromr-test] skip_no_model: FOCR_TROMR_DIR unset");
            return;
        };
        let fx_path = dir.join("tromr_oracle_fixtures.json");
        if !fx_path.is_file() {
            eprintln!(
                "[tromr-test] skip_no_model: oracle fixtures absent (gen_reference_fixtures_tromr.py)"
            );
            return;
        }
        let fx: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(fx_path).unwrap()).unwrap();
        let width = fx["preproc"]["shape"][2].as_u64().unwrap() as usize;
        let pixels = read_f32(&dir.join("tromr_preproc.bin"));
        assert_eq!(pixels.len(), IMG_H * width, "preproc fixture shape");

        let weights = Weights::load(&dir.join("tromr.focrq")).expect("artifact loads");
        let w = TromrEncoderW::build(&weights).expect("hydrates");

        // Backbone seams (channel-major in the fixture, ours identical layout).
        let feat = backbone(&w, &pixels, width).expect("backbone runs");
        let stage2 = read_f32(&dir.join("tromr_seam_stage2.bin"));
        assert_eq!(feat.data.len(), stage2.len(), "stage2 shape");
        let (c, m) = (cos(&feat.data, &stage2), maxabs(&feat.data, &stage2));
        eprintln!("[tromr-cert] stage2 cos {c:.8} maxabs {m:.3e}");
        assert!(c >= 0.9999, "stage2 cos {c}");

        // Full encoder vs the final oracle output [1, seq, 256].
        let out = encode(&w, &pixels, width).expect("encode runs");
        let oracle = read_f32(&dir.join("tromr_seam_encoder_out.bin"));
        assert_eq!(out.data.len(), oracle.len(), "encoder_out shape");
        let (c, m) = (cos(&out.data, &oracle), maxabs(&out.data, &oracle));
        eprintln!(
            "[tromr-cert] encoder_out cos {c:.8} maxabs {m:.3e} (oracle floor 0.0 both legs)"
        );
        assert!(c >= 0.9999, "encoder_out cos {c}");
    }
}
