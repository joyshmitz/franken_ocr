//! SmolVLM2 describe/VQA assembly (C7, bd-3jo6.3.7) — the third end-to-end
//! model path, mirroring [`super::got`]'s shape: preprocess → vision tower →
//! connector → prompt splice → dense decoder greedy → detokenize.
//!
//! * **Preprocess**: [`preprocess::preprocess_smolvlm2`] — Pillow-bit-exact
//!   LANCZOS resize/split (L0b proven maxabs 0.0 vs the torch oracle).
//! * **Vision**: [`vision_siglip`] (C3, certified cos 1.00000000) →
//!   [`token_compress::pixel_shuffle`] ×4 → the high-precision
//!   `modality_projection` GEMM (C4, certified) → `64` rows per frame.
//! * **Prompt**: the SmolVLM2 chat template (spec §5) — one literal
//!   `<|im_start|>`, `User:` (no space before an image-first content list),
//!   the `<fake_token_around_image>`/`<row_r_col_c>`/`<global-img>` image
//!   expansion with 64 `<image>` slots per frame (tiles row-major, global
//!   LAST), the question text, `<end_of_utterance>\n`, and the `Assistant:`
//!   generation suffix. Encoded by the C6 SmolLM2 tokenizer (id-exact).
//! * **Decoder**: [`DecoderConfig::smolvlm2`] (C5, certified) through the
//!   O(n)-per-token KV-cache greedy decode, stop at `<end_of_utterance>`
//!   (49279). Upstream has NO repetition guard (`no_repeat_ngram = 0`).
//!
//! Task = the natural-language question (there are no GOT-style instruction
//! modes): describe/caption and VQA are the same machine with a different
//! question string.

use image::DynamicImage;

use crate::error::{FocrError, FocrResult};
use crate::preprocess;
use crate::tokenizer::{Tokenizer, special_smollm2};

use super::decoder_qwen2::{self, DecoderConfig};
use super::tensor::Mat;
use super::vision_sam::Linear;
use super::weights::Weights;
use super::{connector, decoder, token_compress, vision_siglip};

/// The generation stop id — `<end_of_utterance>` (spec §8).
pub const EOS_ID: u32 = special_smollm2::END_OF_UTTERANCE;
/// `<image>` splice-slot id.
const IMAGE_ID: u32 = special_smollm2::IMAGE;
/// `<image>` slots per 512² frame (`processor_config.json image_seq_len`).
const IMG_SLOTS_PER_FRAME: usize = 64;
/// Pixel-shuffle scale factor (`config.scale_factor`).
const PS_SCALE: usize = 4;
/// The model-card describe question (the oracle's L0c/L4 anchor prompt).
pub const DESCRIBE_QUESTION: &str = "Can you describe this image?";
/// The model-card decode cap (`max_new_tokens=64` in every README example).
pub const DEFAULT_MAX_NEW: usize = 64;
/// `max_position_embeddings` (spec §4) — the architectural sequence budget;
/// generation is clamped to `MAX_POSITION - prompt_len`.
pub const MAX_POSITION: usize = 8192;

/// The §5 image-expansion string for an `R×C` split image: per tile
/// `<fake_token_around_image><row_r_col_c>` + 64 `<image>`s (row-major, a
/// `"\n"` after each row), then `"\n"` + the global frame bracketed by
/// `<fake_token_around_image>`. The trailing `"\n\n"` abutment BPE-merges —
/// pinned by the L0c fixture, never hand-computed (OQ-4).
fn image_prompt_string(rows: usize, cols: usize) -> String {
    const FAKE: &str = "<fake_token_around_image>";
    const GLOBAL: &str = "<global-img>";
    let slots = "<image>".repeat(IMG_SLOTS_PER_FRAME);
    let mut s = String::new();
    for r in 1..=rows {
        for c in 1..=cols {
            s.push_str(FAKE);
            s.push_str(&format!("<row_{r}_col_{c}>"));
            s.push_str(&slots);
        }
        s.push('\n');
    }
    s.push('\n');
    s.push_str(FAKE);
    s.push_str(GLOBAL);
    s.push_str(&slots);
    s.push_str(FAKE);
    s
}

/// Render the full describe/VQA chat prompt (image-first content, spec §5):
/// `<|im_start|>User:{expansion}{question}<end_of_utterance>\nAssistant:`.
fn describe_prompt(rows: usize, cols: usize, question: &str) -> String {
    format!(
        "<|im_start|>User:{}{}<end_of_utterance>\nAssistant:",
        image_prompt_string(rows, cols),
        question
    )
}

/// Encode the describe/VQA prompt to ids (the C6 tokenizer owns the specials
/// splitting; nothing is auto-prepended — the template supplies
/// `<|im_start|>` literally).
///
/// # Errors
/// A tokenizer encode error (impossible for a valid vocab).
pub fn describe_prompt_ids(
    tk: &Tokenizer,
    rows: usize,
    cols: usize,
    question: &str,
) -> FocrResult<Vec<u32>> {
    tk.encode(&describe_prompt(rows, cols, question))
}

/// Run the certified vision stack over preprocessed frames: SigLIP per frame
/// → pixel-shuffle ×4 → one stacked `modality_projection` GEMM. Returns the
/// `[n_frames * 64, 960]` vision rows in frame order (tiles row-major, global
/// LAST — the same order the prompt expansion emits `<image>` slots).
///
/// # Errors
/// A hydration/forward error, or a shape violation.
pub fn vision_rows(weights: &Weights, frames: &[f32], n_frames: usize) -> FocrResult<Mat> {
    let sw = vision_siglip::siglip_weights_from(weights, "model.vision_model")?;
    let post = vision_siglip::forward_frames(&sw, frames, n_frames)?;

    let ps_cols = vision_siglip::EMBED_DIM * PS_SCALE * PS_SCALE; // 12288
    let mut ps = Mat::zeros(n_frames * IMG_SLOTS_PER_FRAME, ps_cols);
    for (f, frame) in post.iter().enumerate() {
        let shuffled = token_compress::pixel_shuffle(frame, PS_SCALE)?;
        if shuffled.cols != ps_cols || shuffled.rows != IMG_SLOTS_PER_FRAME {
            return Err(FocrError::Other(anyhow::anyhow!(
                "smolvlm2 vision: pixel_shuffle produced [{}, {}], want [{IMG_SLOTS_PER_FRAME}, {ps_cols}]",
                shuffled.rows,
                shuffled.cols
            )));
        }
        let dst_start = f * IMG_SLOTS_PER_FRAME * ps_cols;
        ps.data[dst_start..dst_start + shuffled.data.len()].copy_from_slice(&shuffled.data);
    }

    let proj = weights.mat("model.connector.modality_projection.proj.weight")?;
    if (proj.rows, proj.cols) != (960, ps_cols) {
        return Err(FocrError::Other(anyhow::anyhow!(
            "smolvlm2 connector: proj shape [{}, {}], want [960, {ps_cols}]",
            proj.rows,
            proj.cols
        )));
    }
    let lin = Linear {
        w: proj.data,
        b: Vec::new(),
        out: 960,
        in_: ps_cols,
    };
    lin.apply(&ps)
}

/// Build the decoder `inputs_embeds`: embed the prompt ids against the
/// (untied) `model.text_model.embed_tokens.weight`, then scatter the vision
/// rows into the `<image>` (49190) slots in prompt order — the
/// `inputs_merger` splice.
///
/// # Errors
/// An embed error, or a [`connector::masked_scatter`] mismatch (the number of
/// `<image>` slots must equal `vision.rows`).
pub fn build_inputs_embeds(weights: &Weights, vision: &Mat, prompt_ids: &[u32]) -> FocrResult<Mat> {
    let embed = weights.mat("model.text_model.embed_tokens.weight")?;
    let (vocab, hidden) = (embed.rows, embed.cols);
    let mut inputs_embeds = decoder::embed_tokens(&embed.data, vocab, hidden, prompt_ids)?;
    let mask: Vec<bool> = prompt_ids.iter().map(|&id| id == IMAGE_ID).collect();
    connector::masked_scatter(&mut inputs_embeds, vision, &mask)?;
    Ok(inputs_embeds)
}

/// End-to-end SmolVLM2 describe/VQA: LANCZOS split preprocess → SigLIP +
/// pixel-shuffle + connector → `<image>` splice → SmolLM2 KV-cache greedy →
/// detokenize (specials stripped, trimmed). `question` is the task; `max_new`
/// caps generation (model-card default [`DEFAULT_MAX_NEW`]). Stops at
/// `<end_of_utterance>`.
///
/// # Errors
/// A preprocess, vision, decode, or tokenizer error.
pub fn recognize(
    weights: &Weights,
    tk: &Tokenizer,
    img: &DynamicImage,
    question: &str,
    max_new: usize,
) -> FocrResult<String> {
    let tv = std::time::Instant::now();
    let pre = preprocess::preprocess_smolvlm2(img)?;
    let vision = vision_rows(weights, &pre.frames, pre.n_frames)?;
    let prompt_ids = describe_prompt_ids(tk, pre.rows, pre.cols, question)?;
    let inputs_embeds = build_inputs_embeds(weights, &vision, &prompt_ids)?;
    super::timing_log(&format!(
        "  smolvlm2.vision+splice {:.2}s ({} frames, {} prompt ids)",
        tv.elapsed().as_secs_f64(),
        pre.n_frames,
        prompt_ids.len()
    ));
    let tg = std::time::Instant::now();
    let cfg = DecoderConfig::smolvlm2();
    // Clamp to the architectural position budget net of the prompt (upstream
    // has no config max_new; the RoPE table stops at max_position 8192).
    let max_new = max_new.min(MAX_POSITION.saturating_sub(inputs_embeds.rows));
    let ids =
        decoder_qwen2::generate_greedy_kvcache(weights, &cfg, &inputs_embeds, max_new, EOS_ID)?;
    super::timing_log(&format!(
        "  smolvlm2.generate {} tokens {:.2}s",
        ids.len(),
        tg.elapsed().as_secs_f64()
    ));
    Ok(tk.decode_skip_special(&ids)?.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_prompt_string_shape() {
        let s = image_prompt_string(2, 2);
        // 4 tiles + 1 global = 5 slot runs of 64.
        assert_eq!(s.matches("<image>").count(), 5 * IMG_SLOTS_PER_FRAME);
        assert_eq!(s.matches("<fake_token_around_image>").count(), 4 + 2);
        assert_eq!(s.matches("<global-img>").count(), 1);
        for marker in [
            "<row_1_col_1>",
            "<row_1_col_2>",
            "<row_2_col_1>",
            "<row_2_col_2>",
        ] {
            assert_eq!(s.matches(marker).count(), 1, "{marker}");
        }
        // Rows end with \n; the global section starts after the \n\n abutment.
        assert!(s.contains("\n\n<fake_token_around_image><global-img>"));
        assert!(s.ends_with("<fake_token_around_image>"));
    }

    #[test]
    fn describe_prompt_template_shape() {
        let p = describe_prompt(1, 2, "What color is the car?");
        assert!(p.starts_with("<|im_start|>User:<fake_token_around_image>"));
        assert!(p.contains("What color is the car?<end_of_utterance>\nAssistant:"));
        assert!(p.ends_with("Assistant:"));
        // No auto-space after "User:" for image-first content.
        assert!(!p.contains("User: <"));
    }

    /// The splice mask targets exactly the `<image>` ids — a synthetic-vocab
    /// end-to-end of prompt→embed→scatter without real weights is covered by
    /// `build_inputs_embeds`'s callers (`connector::masked_scatter` has its
    /// own unit suite); here we pin the slot-count arithmetic.
    #[test]
    fn slot_count_matches_vision_rows() {
        // 3×4 grid + global = 13 frames → 832 slots — the l0c fixture's count.
        let s = image_prompt_string(3, 4);
        assert_eq!(s.matches("<image>").count(), 13 * IMG_SLOTS_PER_FRAME);
    }

    // ── armed certs (env-gated, real weights + oracle fixtures) ─────────────

    fn load_vision_fixture() -> Option<serde_json::Value> {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/smolvlm2/vision_oracle_fixtures.json"
        );
        let text = std::fs::read_to_string(path).ok()?;
        Some(serde_json::from_str(&text).expect("vision_oracle_fixtures.json parses"))
    }

    fn load_real_tokenizer() -> Option<Tokenizer> {
        let dir = std::env::var("FOCR_SMOLVLM2_DIR").ok()?;
        let path = format!("{dir}/tokenizer.json");
        if !std::path::Path::new(&path).is_file() {
            eprintln!("skip-with-SUCCESS: {path} absent");
            return None;
        }
        Some(Tokenizer::from_file(std::path::Path::new(&path)).expect("tokenizer loads"))
    }

    /// **C7 L0c — the rendered describe prompt is id-EXACT vs the processor
    /// oracle** (876 ids incl. the OQ-4 `\n\n` merge and all 832 slots).
    #[test]
    fn describe_prompt_ids_match_oracle_l0c() {
        let Some(tk) = load_real_tokenizer() else {
            return;
        };
        let Some(fx) = load_vision_fixture() else {
            return;
        };
        let want: Vec<u32> = fx["l0c_describe_prompt"]["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let got = describe_prompt_ids(&tk, 3, 4, DESCRIBE_QUESTION).expect("encode");
        if got != want {
            let pos = got
                .iter()
                .zip(&want)
                .position(|(a, b)| a != b)
                .unwrap_or_else(|| got.len().min(want.len()));
            panic!(
                "L0c prompt ids diverged at {pos}: got len {} want len {} \
                 (got[{pos}..+4]={:?} want[{pos}..+4]={:?})",
                got.len(),
                want.len(),
                &got[pos.min(got.len().saturating_sub(1))..got.len().min(pos + 4)],
                &want[pos.min(want.len().saturating_sub(1))..want.len().min(pos + 4)]
            );
        }
        eprintln!("[C7 L0c] {} prompt ids exact", got.len());
    }

    /// Assert a decode's divergence from the oracle greedy stream is a
    /// MEASURED near-tie flip, not a defect (DISC-003): the exact prefix must
    /// stay long, and the first divergent token must be the oracle's rank-2
    /// candidate at a step whose top-2 logit gap is small (the per-step
    /// `step_top2` ledger the oracle script replays). A real bug (wrong math,
    /// not reordered math) picks far-from-tie tokens and fails both gates.
    fn assert_near_tie_divergence(label: &str, ids: &[u32], want: &[u32], fx: &serde_json::Value) {
        let prefix = ids.iter().zip(want).take_while(|(a, b)| a == b).count();
        eprintln!("[C8 L4v] {label}: exact prefix {prefix}/{} ids", want.len());
        assert!(
            prefix >= 16,
            "{label}: exact prefix {prefix} < 16 — more than a near-tie flip"
        );
        if prefix == want.len() {
            eprintln!("[C8 L4v] {label}: id-EXACT");
            return;
        }
        let steps = fx["l4_describe_greedy"]["step_top2"]
            .as_array()
            .expect("regenerate the vision oracle fixture for the step_top2 ledger");
        let step = &steps[prefix];
        let top2_id = step["top2"][0].as_u64().unwrap() as u32;
        let gap = step["gap"].as_f64().unwrap();
        eprintln!(
            "[C8 L4v] {label}: diverged at step {prefix} (oracle top-2 gap {gap:.4}); \
             ours={} oracle-rank2={top2_id}",
            ids[prefix]
        );
        assert_eq!(
            ids[prefix], top2_id,
            "{label}: divergent token is not the oracle's rank-2 candidate — a defect, \
             not a near-tie flip"
        );
        // Budget from measurement (DISC-003, 2026-07-03): the kvcache fast
        // path's bespoke decode-attention rounding compounds along the
        // autoregressive chain; the observed flip sits at gap 0.353 by step
        // 20 (while the re-prefill greedy path — same rounding as the sdpa
        // prefill — is 64/64 id-EXACT, so the decoder MATH is certified).
        // A wrong-math defect picks far-from-tie tokens (median ledger gap is
        // ~1.0, spikes ≫ 3), which still fails this gate.
        assert!(
            gap <= 0.5,
            "{label}: oracle top-2 gap {gap:.4} at step {prefix} is not a near-tie \
             (measured compounded-drift flips sit ≤ ~0.35; a wide-gap flip is a defect)"
        );
    }

    /// **C8 L4v — describe greedy vs the torch oracle**, two legs (spec §13
    /// L4 is "id-exact to first divergence"; every divergence must be a
    /// ledger-verified near-tie — DISC-003):
    ///
    /// 1. **Decoder-from-oracle-vision:** splice the ORACLE's own
    ///    `connector_out.bin` rows into our id-exact prompt and decode — this
    ///    isolates prompt + splice + decoder from the vision drift.
    /// 2. **Full pipeline:** our L0b-exact preprocess + certified
    ///    SigLIP/connector feed the same decode.
    ///
    /// Both legs must hold a ≥16-token exact prefix (measured 20/22 on
    /// 2026-07-02) and any first divergence must land on the oracle's rank-2
    /// token at a measured near-tie step.
    #[test]
    fn describe_e2e_matches_oracle_l4() {
        let Ok(dir) = std::env::var("FOCR_SMOLVLM2_DIR") else {
            return;
        };
        let Some(tk) = load_real_tokenizer() else {
            return;
        };
        let Some(fx) = load_vision_fixture() else {
            return;
        };
        let model_path = format!("{dir}/model.safetensors");
        let conn_path = format!("{dir}/smolvlm2_connector_out.bin");
        if !std::path::Path::new(&model_path).is_file()
            || !std::path::Path::new(&conn_path).is_file()
        {
            eprintln!("skip-with-SUCCESS: {model_path} / {conn_path} absent");
            return;
        }
        let want: Vec<u32> = fx["l4_describe_greedy"]["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let want_text = fx["l4_describe_greedy"]["text"].as_str().unwrap();

        let weights = Weights::load(std::path::Path::new(&model_path)).expect("weights");
        let photo = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/smolvlm2/sample_photo.png"
        );
        let img = image::open(photo).expect("sample photo decodes");
        let pre = preprocess::preprocess_smolvlm2(&img).expect("preprocess");
        let prompt_ids =
            describe_prompt_ids(&tk, pre.rows, pre.cols, DESCRIBE_QUESTION).expect("prompt");
        let cfg = DecoderConfig::smolvlm2();

        // Leg 1: decoder-from-oracle-vision.
        let conn: Vec<f32> = std::fs::read(&conn_path)
            .expect("connector blob reads")
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let oracle_vision = Mat::from_vec(pre.n_frames * IMG_SLOTS_PER_FRAME, 960, conn);
        let embeds_ov = build_inputs_embeds(&weights, &oracle_vision, &prompt_ids).expect("splice");

        // Leg 0: MEASURE our decoder's logit drift at the prefill seam using
        // the ledger's step-0 anchors (oracle top-2 ids + exact logit values):
        // the drift magnitude is what justifies (or refutes) a near-tie flip.
        {
            let logits = decoder_qwen2::forward_prefill(&weights, &cfg, &embeds_ov)
                .expect("prefill (oracle vision)");
            let last = &logits.data[(logits.rows - 1) * logits.cols..];
            let s0 = &fx["l4_describe_greedy"]["step_top2"][0];
            let (t1_id, t1_val) = (
                s0["top1"][0].as_u64().unwrap() as usize,
                s0["top1"][1].as_f64().unwrap(),
            );
            let (t2_id, t2_val) = (
                s0["top2"][0].as_u64().unwrap() as usize,
                s0["top2"][1].as_f64().unwrap(),
            );
            let drift1 = (f64::from(last[t1_id]) - t1_val).abs();
            let drift2 = (f64::from(last[t2_id]) - t2_val).abs();
            let our_gap = f64::from(last[t1_id]) - f64::from(last[t2_id]);
            let argmax = last
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .unwrap();
            eprintln!(
                "[C8 L4v] leg 0: prefill logit drift top1={drift1:.4} top2={drift2:.4} \
                 our_gap={our_gap:.4} oracle_gap={:.4} argmax={} (oracle {})",
                s0["gap"].as_f64().unwrap(),
                argmax,
                t1_id
            );
            assert_eq!(argmax, t1_id, "prefill argmax diverged at step 0");
        }

        let ids_ov = decoder_qwen2::generate_greedy_kvcache(
            &weights,
            &cfg,
            &embeds_ov,
            DEFAULT_MAX_NEW,
            EOS_ID,
        )
        .expect("generate (oracle vision)");
        // FULL leg (opt-in, ~64 re-prefills — hours in a dev build): the
        // O(n²) `generate_greedy` path must be id-EXACT vs the oracle — it
        // shares the sdpa prefill math end to end, so it certifies prompt +
        // splice + decoder absolutely. PROVEN 64/64 on 2026-07-03 (DISC-003).
        // The kvcache fast path below uses the bespoke decode-attention whose
        // different f32 rounding compounds autoregressively — near-tie flips
        // are expected there and ledger-gated instead.
        if std::env::var_os("FOCR_SMOLVLM2_CERT_FULL").is_some() {
            let ids_greedy =
                decoder_qwen2::generate_greedy(&weights, &cfg, &embeds_ov, DEFAULT_MAX_NEW, EOS_ID)
                    .expect("generate_greedy (oracle vision)");
            assert_eq!(
                ids_greedy, want,
                "re-prefill greedy from oracle vision must be id-EXACT \
                 (prompt/splice/decoder math diverged — a real defect)"
            );
            eprintln!(
                "[C8 L4v] FULL leg: re-prefill greedy id-EXACT ({} ids)",
                want.len()
            );
        }
        assert_near_tie_divergence("leg 1 (oracle vision)", &ids_ov, &want, &fx);

        // Leg 2: full native pipeline + faithfulness eyeball.
        let vision = vision_rows(&weights, &pre.frames, pre.n_frames).expect("vision");
        let inputs_embeds = build_inputs_embeds(&weights, &vision, &prompt_ids).expect("splice");
        let ids = decoder_qwen2::generate_greedy_kvcache(
            &weights,
            &cfg,
            &inputs_embeds,
            DEFAULT_MAX_NEW,
            EOS_ID,
        )
        .expect("generate");
        let text = tk.decode_skip_special(&ids).expect("decode");
        eprintln!("[C8 L4v] ours:   {:?}", text.trim());
        eprintln!("[C8 L4v] oracle: {want_text:?}");
        assert_near_tie_divergence("leg 2 (full pipeline)", &ids, &want, &fx);
    }
}
