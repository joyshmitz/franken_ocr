//! OneChart assembly (sub-epic D) — the fourth model path, a direct GOT
//! sibling (Vary-tiny splice; census `docs/zoo/onechart-spec.md`):
//! squash-bicubic 1024² RAW-[0,1] preprocess ([`preprocess::onechart_view_tensor`])
//! → the SAME certified SAM-ViT-B tower as GOT (prefix `model.vision_tower`,
//! share-by-import per the B3 precedent) → a `Linear(1024→768, bias)`
//! `mm_projector` → 256 `<imgpad>` (50265) slots → the OPT-125M decoder (D4,
//! pending on the shared dense engine) + the `num_decoder` number head (D5).
//!
//! This module currently ships the D3 vision seam ([`vision_features`]); the
//! decoder/recognize assembly lands with D4/D5.

use crate::error::FocrResult;

use super::connector;
use super::decoder;
use super::tensor::Mat;
use super::vision_sam::{self, Linear};
use super::weights::Weights;

/// The vision-token count (SAM 1024² → 16× compressor → 256 tokens, as GOT).
pub const VISION_TOKENS: usize = 256;
/// The decoder hidden width the projector emits (OPT hidden 768 — census §3:
/// the connector currency is 768, NOT GOT's 1024 or Baidu's 1280).
pub const HIDDEN: usize = 768;

/// D3: the OneChart vision features — the certified SAM tower at the
/// `model.vision_tower` prefix, then the `model.mm_projector`
/// `Linear(1024→768, bias=True)` (census §3). Returns `[256, 768]`
/// token-major rows, ready for the `<imgpad>` splice.
///
/// # Errors
/// A tower/hydration error, or a projector shape violation.
pub fn vision_features(weights: &Weights, image: &Mat, prefix: &str) -> FocrResult<Mat> {
    let sam = vision_sam::forward_prefix(weights, image, prefix)?; // [1024, 256] channel-major
    let sam_t = transpose(&sam); // [256, 1024] token-major
    let w = weights.vec("model.mm_projector.weight")?; // [768*1024] row-major [out,in]
    let b = weights.vec("model.mm_projector.bias")?; // [768]
    let proj = Linear {
        w,
        b,
        out: HIDDEN,
        in_: 1024,
    };
    proj.apply(&sam_t) // [256, 768]
}

/// Build the OPT decoder `inputs_embeds`: embed the prompt ids against the
/// tied `model.decoder.embed_tokens.weight`, then scatter the vision rows
/// into the 256 `<imgpad>` (50265) slots in prompt order. (The learned
/// position table is added INSIDE the decoder prefill — census §4/OQ-D6.)
///
/// # Errors
/// An embed error, or a [`connector::masked_scatter`] slot-count mismatch.
pub fn build_inputs_embeds(weights: &Weights, vision: &Mat, prompt_ids: &[u32]) -> FocrResult<Mat> {
    let embed = weights.mat("model.decoder.embed_tokens.weight")?;
    let (vocab, hidden) = (embed.rows, embed.cols);
    let mut inputs_embeds = decoder::embed_tokens(&embed.data, vocab, hidden, prompt_ids)?;
    let mask: Vec<bool> = prompt_ids
        .iter()
        .map(|&id| id == crate::tokenizer::special_opt::IMG_PAD)
        .collect();
    connector::masked_scatter(&mut inputs_embeds, vision, &mask)?;
    Ok(inputs_embeds)
}

/// `[r, c]` row-major → `[c, r]` row-major (channel-major SAM output →
/// token-major rows; the same reshape GOT's assembly performs).
fn transpose(m: &Mat) -> Mat {
    let (r, c) = (m.rows, m.cols);
    let mut out = vec![0.0f32; r * c];
    for i in 0..r {
        for j in 0..c {
            out[j * r + i] = m.data[i * c + j];
        }
    }
    Mat::from_vec(c, r, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_round_trips() {
        let m = Mat::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = transpose(&m);
        assert_eq!((t.rows, t.cols), (3, 2));
        assert_eq!(t.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(transpose(&t).data, m.data);
    }

    #[test]
    fn vision_features_error_handling() {
        // An empty weights bundle must fail loud (missing tower tensors), not
        // panic — the ModelNotFound/FormatMismatch rail.
        let w = Weights::default();
        let img = Mat::from_vec(3, 4, vec![0.0; 12]);
        assert!(vision_features(&w, &img, "model.vision_tower").is_err());
    }

    /// **D4-prefill — the OPT decoder vs the torch oracle** (skip-with-SUCCESS
    /// without `FOCR_ONECHART_DIR`): embed the committed 309-id prompt, splice
    /// the ORACLE's own projector rows into the 256 `<imgpad>` slots
    /// (seam-isolated from the D3 vision drift), run the new
    /// `DecoderConfig::onechart()` prefill (LayerNorm+bias, learned offset-2
    /// positions, ReLU fc1/fc2, tied head), and hold the last-pos logits to
    /// argmax-exact + cosine ≥ 0.9999 vs `onechart_final_logits.bin`.
    #[test]
    fn opt_prefill_matches_torch_oracle() {
        let Ok(dir) = std::env::var("FOCR_ONECHART_DIR") else {
            return;
        };
        let proj_path = format!("{dir}/onechart_proj_out.bin");
        let logits_path = format!("{dir}/onechart_final_logits.bin");
        let model_path = format!("{dir}/model.safetensors");
        if !std::path::Path::new(&proj_path).is_file() {
            eprintln!("skip-with-SUCCESS: {proj_path} absent (run the oracle script)");
            return;
        }
        let read_f32 = |p: &str| -> Vec<f32> {
            std::fs::read(p)
                .expect("oracle blob reads")
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let fx: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/onechart/oracle_fixtures.json"
            ))
            .expect("oracle fixtures read"),
        )
        .expect("oracle fixtures parse");
        let prompt_ids: Vec<u32> = fx["l0c_prompt"]["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        // 308 measured ids (the census §5 estimated 309; the fixture's own
        // `n` is the truth — 256 <imgpad> + 52 text/bracket ids, no bos).
        assert_eq!(
            prompt_ids.len(),
            fx["l0c_prompt"]["n"].as_u64().unwrap() as usize,
            "prompt drifted from its own fixture"
        );
        assert_eq!(prompt_ids.len(), 308, "measured census prompt length");

        let weights = Weights::load(std::path::Path::new(&model_path)).expect("weights");
        let vision = Mat::from_vec(VISION_TOKENS, HIDDEN, read_f32(&proj_path));
        let embeds = build_inputs_embeds(&weights, &vision, &prompt_ids).expect("splice");
        let cfg = super::super::decoder_qwen2::DecoderConfig::onechart();
        let logits =
            super::super::decoder_qwen2::forward_prefill(&weights, &cfg, &embeds).expect("prefill");
        let ours = &logits.data[(logits.rows - 1) * logits.cols..];
        let want = read_f32(&logits_path);
        assert_eq!(ours.len(), want.len(), "vocab width");

        let argmax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .unwrap()
        };
        let mut dot = 0.0f64;
        let (mut na, mut nb) = (0.0f64, 0.0f64);
        let mut max_abs = 0.0f64;
        for (a, b) in ours.iter().zip(&want) {
            let (a, b) = (f64::from(*a), f64::from(*b));
            dot += a * b;
            na += a * a;
            nb += b * b;
            max_abs = max_abs.max((a - b).abs());
        }
        let cos = dot / (na.sqrt() * nb.sqrt());
        eprintln!(
            "[D4 prefill] argmax={} (oracle {}) cos={cos:.8} maxabs={max_abs:.3e}",
            argmax(ours),
            argmax(&want)
        );
        assert_eq!(argmax(ours), argmax(&want), "next-token argmax diverged");
        assert!(cos >= 0.9999, "prefill logit cosine {cos:.8} < 0.9999");
    }

    /// **D4-decode — the Opt KV-cache path** (skip-with-SUCCESS without
    /// `FOCR_ONECHART_DIR`): from the same oracle-vision embeds as the prefill
    /// cert, (a) the O(n) KV-cache greedy and the O(n²) re-prefill greedy must
    /// agree on a 24-token window (the B9 identity at OPT geometry), (b) the
    /// first generated id must be 50268 `<Number>` (the certified prefill
    /// argmax / census §8 protocol), and (c) the decoded text must
    /// prefix-match the oracle `chat()` answer.
    #[test]
    fn opt_kvcache_matches_greedy_and_oracle() {
        let Ok(dir) = std::env::var("FOCR_ONECHART_DIR") else {
            return;
        };
        let proj_path = format!("{dir}/onechart_proj_out.bin");
        let model_path = format!("{dir}/model.safetensors");
        if !std::path::Path::new(&proj_path).is_file() {
            eprintln!("skip-with-SUCCESS: {proj_path} absent (run the oracle script)");
            return;
        }
        let read_f32 = |p: &str| -> Vec<f32> {
            std::fs::read(p)
                .expect("oracle blob reads")
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let fx: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/onechart/oracle_fixtures.json"
            ))
            .expect("oracle fixtures read"),
        )
        .expect("oracle fixtures parse");
        let prompt_ids: Vec<u32> = fx["l0c_prompt"]["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let weights = Weights::load(std::path::Path::new(&model_path)).expect("weights");
        let vision = Mat::from_vec(VISION_TOKENS, HIDDEN, read_f32(&proj_path));
        let embeds = build_inputs_embeds(&weights, &vision, &prompt_ids).expect("splice");
        let cfg = super::super::decoder_qwen2::DecoderConfig::onechart();

        let ids_kv = super::super::decoder_qwen2::generate_greedy_kvcache(
            &weights,
            &cfg,
            &embeds,
            24,
            crate::tokenizer::special_opt::BOS_EOS,
        )
        .expect("kvcache greedy");
        let ids_greedy = super::super::decoder_qwen2::generate_greedy(
            &weights,
            &cfg,
            &embeds,
            24,
            crate::tokenizer::special_opt::BOS_EOS,
        )
        .expect("re-prefill greedy");
        eprintln!("[D4 decode] kvcache: {ids_kv:?}");
        assert_eq!(
            ids_kv, ids_greedy,
            "B9 identity: kvcache vs re-prefill greedy diverged at OPT geometry"
        );
        assert_eq!(
            ids_kv[0],
            crate::tokenizer::special_opt::NUMBER,
            "first generated id must be the <Number> trigger (census §8)"
        );

        // Text prefix vs the oracle chat() answer (same greedy trajectory).
        let tk = crate::tokenizer::Tokenizer::from_opt_dir(std::path::Path::new(&dir))
            .expect("onechart tokenizer");
        let ours = tk.decode_skip_special(&ids_kv).expect("decode");
        let oracle = fx["l4_chat"]["answer"].as_str().unwrap();
        eprintln!("[D4 decode] ours:   {:?}", ours.trim());
        eprintln!("[D4 decode] oracle: {:?}", &oracle[..oracle.len().min(80)]);
        let ours_t = ours.trim();
        let prefix = ours_t
            .chars()
            .zip(oracle.trim().chars())
            .take_while(|(a, b)| a == b)
            .count();
        eprintln!("[D4 decode] text prefix match: {prefix} chars");
        assert!(
            prefix >= 12,
            "decoded text diverged from the oracle chat() answer too early ({prefix} chars)"
        );
    }

    /// **D3 — OneChart vision + projector vs the torch oracle**
    /// (skip-with-SUCCESS without `FOCR_ONECHART_DIR`): feed the oracle's own
    /// preprocessed tensor (seam-isolated from resize parity, OQ-D3) through
    /// the certified SAM tower at the OneChart prefix + the `mm_projector`,
    /// and hold the `[256, 768]` output to cosine ≥ 0.9999 + a bounded
    /// max-abs vs `onechart_proj_out.bin`.
    #[test]
    fn vision_features_match_torch_oracle() {
        let Ok(dir) = std::env::var("FOCR_ONECHART_DIR") else {
            return;
        };
        let pre_path = format!("{dir}/onechart_preproc.bin");
        let want_path = format!("{dir}/onechart_proj_out.bin");
        let model_path = format!("{dir}/model.safetensors");
        if !std::path::Path::new(&pre_path).is_file() {
            eprintln!("skip-with-SUCCESS: {pre_path} absent (run the oracle script)");
            return;
        }
        let read_f32 = |p: &str| -> Vec<f32> {
            std::fs::read(p)
                .expect("oracle blob reads")
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let pre = read_f32(&pre_path);
        assert_eq!(pre.len(), 3 * 1024 * 1024, "preproc not [3,1024,1024]");
        let want = read_f32(&want_path);
        assert_eq!(want.len(), VISION_TOKENS * HIDDEN, "proj_out not [256,768]");

        let weights = Weights::load(std::path::Path::new(&model_path)).expect("weights");
        let image = Mat::from_vec(3, 1024 * 1024, pre);
        let ours = vision_features(&weights, &image, "model.vision_tower").expect("vision");
        assert_eq!((ours.rows, ours.cols), (VISION_TOKENS, HIDDEN));

        let mut dot = 0.0f64;
        let (mut na, mut nb) = (0.0f64, 0.0f64);
        let mut max_abs = 0.0f64;
        for (a, b) in ours.data.iter().zip(&want) {
            let (a, b) = (f64::from(*a), f64::from(*b));
            dot += a * b;
            na += a * a;
            nb += b * b;
            max_abs = max_abs.max((a - b).abs());
        }
        let cos = dot / (na.sqrt() * nb.sqrt());
        eprintln!("[D3 parity] proj_out cos={cos:.8} maxabs={max_abs:.3e}");
        assert!(cos >= 0.9999, "OneChart vision cosine {cos:.8} < 0.9999");
        assert!(
            max_abs <= 1e-2,
            "OneChart proj_out maxabs {max_abs:.3e} > 1e-2 — investigate before tightening"
        );
    }
}
