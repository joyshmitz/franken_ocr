//! Model-architecture descriptors — the foundation of the multi-model "model zoo"
//! (epic `bd-3jo6`, task A1).
//!
//! This is the FIRST, deliberately **additive** step of the multi-model
//! generalization: it describes each model's identity + graph shape + decode
//! contract + tokenizer + tasks as plain data behind a [`ModelArch`] trait, and a
//! [`registry`] that maps a model id to its descriptor — WITHOUT yet rewiring the
//! live forward. So the Baidu Unlimited-OCR engine stays byte-identical (its
//! goldens, CER, and `robot selftest` are unchanged); nothing here is on the hot
//! path. Later foundation tasks (A2 `.focrq` v2 + registry, A3 `convert`, A4
//! `pull`, A5 the multi-task CLI, A6 tokenizers, A7 the shared decoder) dispatch
//! real behavior through this descriptor.
//!
//! The Unlimited-OCR descriptor's values are asserted against the live engine
//! constants (`FOCR_MODEL_LICENSE_NOTICE`, `DEFAULT_MODEL_PATH`,
//! `sampler::DecodeParams::default()`) in the unit tests, so the description can
//! never silently drift from the real model.

/// The vision-encoder family a model uses. Only [`VisionEncoder::SamClip`] (the
/// Baidu Unlimited-OCR tower) is implemented today; the rest are the zoo targets,
/// declared so the registry + `focr models` can describe them before they land.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VisionEncoder {
    /// SAM-ViT-B → 16× conv token-compressor → CLIP-L/14 (Baidu Unlimited-OCR).
    SamClip,
    /// SAM / VitDet-style ViT (GOT-OCR2.0, OneChart). [planned: B3/D3]
    SamVit,
    /// SigLIP ViT (SmolVLM2). [planned: C3]
    Siglip,
    /// ResNet stem + ViT (Polyphonic-TrOMR, pix2tex). [planned: E3/F2]
    ResNetVit,
    /// BEiT / DeiT ViT (TrOCR). [planned: F1]
    BeitVit,
}

/// The autoregressive-decoder family a model uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Decoder {
    /// DeepSeek-V2 MoE (64 routed + 2 shared experts, top-6) with R-SWA
    /// (Reference Sliding Window Attention, window 128) — the Baidu decoder.
    DeepSeekV2MoeRswa,
    /// Qwen2 dense (GOT-OCR2.0, OneChart). [planned: shared engine A7]
    Qwen2Dense,
    /// SmolLM2 / Llama-style dense (SmolVLM2). [planned: A7]
    LlamaDense,
    /// mBART / RoBERTa-style seq2seq decoder (TrOCR, pix2tex). [planned: A7]
    Seq2SeqDense,
}

/// The tokenizer family a model uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TokenizerKind {
    /// DeepSeek BPE over the committed `tokenizer.json` (Baidu Unlimited-OCR).
    DeepSeekBpe,
    /// Qwen2 BPE (GOT-OCR2.0, OneChart). [planned: A6/B6/D9]
    Qwen2Bpe,
    /// SmolLM2 BPE (SmolVLM2). [planned: A6/C6]
    SmolLm2Bpe,
    /// A music-symbol vocabulary (Polyphonic-TrOMR). [planned: A6/E6]
    MusicVocab,
    /// SentencePiece / BART (TrOCR, pix2tex). [planned: A6/F]
    SentencePiece,
}

/// A task a model can serve — the routing key for the multi-task CLI (A5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Task {
    /// Document text → markdown (the default `focr ocr`).
    Ocr,
    /// Math / formula → LaTeX.
    Formula,
    /// Tables → structured markdown / TEDS.
    Tables,
    /// Chart / graph → structured data.
    Chart,
    /// Molecular structure → SMILES.
    Molecular,
    /// Geometry → tikz.
    Geometry,
    /// Sheet music → MusicXML / **kern.
    Music,
    /// Photo description / captioning.
    Describe,
    /// Visual question answering.
    Vqa,
    /// Handwriting recognition.
    Handwriting,
}

/// The greedy-decode contract a model ships with — a plain-data mirror of the
/// load-bearing knobs in [`sampler::DecodeParams`] (so a descriptor needs no
/// runtime). Asserted equal to the live `DecodeParams::default()` in tests.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecodeContract {
    /// Sampling temperature; `0.0` ⇒ greedy argmax.
    pub temperature: f32,
    /// EOS token id.
    pub eos_token_id: u32,
    /// No-repeat n-gram size; `0` disables.
    pub no_repeat_ngram_size: usize,
    /// Sliding window for the no-repeat n-gram processor.
    pub ngram_window: usize,
}

/// The quantization policy (doctrine #2): which tensor groups are int8 vs kept
/// high-precision. Every franken_ocr arch shares the SAME validated policy —
/// decoder GEMMs int8, the entire vision tower + projector + router + embed + ALL
/// norms high-precision — but each declares it explicitly so `focr convert` (A3)
/// is data-driven rather than hardcoded per model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuantPolicy {
    /// Decoder FFN / expert / projection GEMMs are int8.
    pub decoder_gemms_int8: bool,
    /// Vision tower, projector, router gate, `embed_tokens`, and ALL norms stay
    /// BF16/F32 (quantizing the vision path wrecks accuracy — CER 0.37).
    pub vision_high_precision: bool,
    /// `lm_head` int8 is BEYOND the validated set — it ships only behind a
    /// measured-CER kill-switch (doctrine #2 / OQ-14), never assumed lossless.
    pub lm_head_int8_killswitched: bool,
}

impl QuantPolicy {
    /// The fixed, validated franken_ocr quant policy (doctrine #2). Every arch
    /// uses this shape.
    pub const DOCTRINE: Self = Self {
        decoder_gemms_int8: true,
        vision_high_precision: true,
        lm_head_int8_killswitched: true,
    };
}

/// A model-architecture descriptor — the foundation `ModelArch` of the zoo. A
/// descriptor is pure metadata (no weights, no runtime); the forward dispatch is
/// wired through it in later A-tasks. `Send + Sync` so a `&'static dyn ModelArch`
/// can live in the process-global registry.
pub trait ModelArch: Send + Sync {
    /// Stable model id — the cache subdir, the `--model` value, and the manifest
    /// key, e.g. `"unlimited-ocr"`.
    fn id(&self) -> &'static str;
    /// Human display name, e.g. `"Baidu Unlimited-OCR"`.
    fn display_name(&self) -> &'static str;
    /// The redistribution / license notice that must travel with the model's
    /// `.focrq` and appear on agent-facing provenance surfaces.
    fn license_notice(&self) -> &'static str;
    /// The default `.focrq` basename in the cache, e.g. `"unlimited-ocr.focrq"`.
    fn default_artifact_basename(&self) -> &'static str;
    /// The vision-encoder family.
    fn vision_encoder(&self) -> VisionEncoder;
    /// The decoder family.
    fn decoder(&self) -> Decoder;
    /// The tokenizer family.
    fn tokenizer(&self) -> TokenizerKind;
    /// The quant policy (defaults to the doctrine policy; every arch uses it).
    fn quant_policy(&self) -> QuantPolicy {
        QuantPolicy::DOCTRINE
    }
    /// The greedy-decode contract this model ships with.
    fn decode_contract(&self) -> DecodeContract;
    /// The tasks this model serves (the CLI routes these to this arch).
    fn tasks(&self) -> &'static [Task];
    /// Whether this arch's forward is implemented TODAY (vs a planned zoo target
    /// the registry merely describes). Only Unlimited-OCR is `true` so far.
    fn implemented(&self) -> bool;
}

/// The Baidu Unlimited-OCR architecture — the FIRST [`ModelArch`] implementation
/// and the project default (`focr ocr`). Its descriptor values are asserted
/// against the live engine constants in tests, so the description cannot drift.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnlimitedOcr;

impl ModelArch for UnlimitedOcr {
    fn id(&self) -> &'static str {
        "unlimited-ocr"
    }
    fn display_name(&self) -> &'static str {
        "Baidu Unlimited-OCR"
    }
    fn license_notice(&self) -> &'static str {
        crate::FOCR_MODEL_LICENSE_NOTICE
    }
    fn default_artifact_basename(&self) -> &'static str {
        "unlimited-ocr.focrq"
    }
    fn vision_encoder(&self) -> VisionEncoder {
        VisionEncoder::SamClip
    }
    fn decoder(&self) -> Decoder {
        Decoder::DeepSeekV2MoeRswa
    }
    fn tokenizer(&self) -> TokenizerKind {
        TokenizerKind::DeepSeekBpe
    }
    fn decode_contract(&self) -> DecodeContract {
        // The frozen single-image greedy contract (plan §6.10 / SPEC-100..103);
        // asserted equal to `sampler::DecodeParams::default()` in the tests.
        DecodeContract {
            temperature: 0.0,
            eos_token_id: 1,
            no_repeat_ngram_size: 35,
            ngram_window: 128,
        }
    }
    fn tasks(&self) -> &'static [Task] {
        &[Task::Ocr]
    }
    fn implemented(&self) -> bool {
        true
    }
}

/// The one process-global Unlimited-OCR descriptor instance.
static UNLIMITED_OCR: UnlimitedOcr = UnlimitedOcr;

/// The model registry, in priority order (the default first) — every architecture
/// the runtime knows about. This is the single place new zoo models register as
/// their sub-epics land (A2+). Today it holds the one IMPLEMENTED arch
/// (Unlimited-OCR); planned archs are appended here when they ship.
static REGISTRY: &[&dyn ModelArch] = &[&UNLIMITED_OCR];

/// The model registry slice (see [`static@REGISTRY`]).
#[must_use]
pub fn registry() -> &'static [&'static dyn ModelArch] {
    REGISTRY
}

/// Look up an architecture by its stable id; `None` if unknown.
#[must_use]
pub fn arch_by_id(id: &str) -> Option<&'static dyn ModelArch> {
    registry().iter().copied().find(|a| a.id() == id)
}

/// The default architecture (`focr ocr`): the Baidu Unlimited-OCR model.
#[must_use]
pub fn default_arch() -> &'static dyn ModelArch {
    &UNLIMITED_OCR
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_engine::sampler;
    use crate::{DEFAULT_MODEL_PATH, FOCR_MODEL_LICENSE_NOTICE};

    #[test]
    fn registry_holds_the_default_and_only_implemented_arch() {
        let archs = registry();
        assert_eq!(archs.len(), 1, "only Unlimited-OCR is implemented today");
        assert_eq!(archs[0].id(), "unlimited-ocr");
        assert!(archs.iter().all(|a| a.implemented()));
        // Every registered id is unique (a registry invariant the zoo relies on).
        let mut ids: Vec<&str> = archs.iter().map(|a| a.id()).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), archs.len(), "model ids must be unique");
    }

    #[test]
    fn lookup_and_default_resolve_unlimited_ocr() {
        assert_eq!(default_arch().id(), "unlimited-ocr");
        assert_eq!(
            arch_by_id("unlimited-ocr").map(ModelArch::id),
            Some("unlimited-ocr")
        );
        assert!(
            arch_by_id("got-ocr2").is_none(),
            "planned archs are not registered yet"
        );
    }

    /// The descriptor must match the LIVE engine constants, so it can never drift.
    #[test]
    fn unlimited_ocr_descriptor_matches_the_live_engine() {
        let a = UnlimitedOcr;
        // License notice is the single source of truth in lib.rs.
        assert_eq!(a.license_notice(), FOCR_MODEL_LICENSE_NOTICE);
        // The default artifact basename matches the resolved DEFAULT_MODEL_PATH.
        let want_basename = std::path::Path::new(DEFAULT_MODEL_PATH)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap();
        assert_eq!(a.default_artifact_basename(), want_basename);
        // Graph shape + tasks.
        assert_eq!(a.vision_encoder(), VisionEncoder::SamClip);
        assert_eq!(a.decoder(), Decoder::DeepSeekV2MoeRswa);
        assert_eq!(a.tokenizer(), TokenizerKind::DeepSeekBpe);
        assert_eq!(a.tasks(), &[Task::Ocr]);
        // Quant policy is the fixed doctrine policy.
        assert_eq!(a.quant_policy(), QuantPolicy::DOCTRINE);
        assert!(a.quant_policy().vision_high_precision);
        assert!(a.quant_policy().decoder_gemms_int8);
    }

    /// The descriptor's decode contract must equal the live frozen
    /// `DecodeParams::default()` — the contract the AR loop actually runs.
    #[test]
    fn unlimited_ocr_decode_contract_matches_sampler_default() {
        let c = UnlimitedOcr.decode_contract();
        let d = sampler::DecodeParams::default();
        assert_eq!(c.temperature, d.temperature);
        assert_eq!(c.eos_token_id, d.eos_token_id);
        assert_eq!(c.no_repeat_ngram_size, d.no_repeat_ngram_size);
        assert_eq!(c.ngram_window, d.ngram_window);
        // And the documented frozen values.
        assert_eq!(c.temperature, 0.0);
        assert_eq!(c.eos_token_id, 1);
        assert_eq!(c.no_repeat_ngram_size, 35);
        assert_eq!(c.ngram_window, 128);
    }
}
