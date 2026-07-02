# franken_ocr — Known Conformance Discrepancies

This document is the honest-divergence ledger: every place where `franken_ocr`'s
output or behavior **intentionally or measurably differs** from the reference
Baidu Unlimited-OCR model (the PyTorch `transformers` oracle pinned by
`scripts/gen_reference_fixtures.py`).

A discrepancy is only recorded once its impact has been **measured** against the
reference. Speculation does not belong here; the cost of a divergence must be a
real number tied to a real test before it is accepted. Every accepted divergence
carries a **kill-switch** (an environment variable that restores reference
behavior) so it can be toggled off for bit-exact comparison.

This is an **artifact-graph ledger** (plan §8.4): every entry carries the same
FrankenSuite provenance fields as `NEGATIVE_EVIDENCE.md` / `PERF_LEDGER.md`, so a
divergence is reproducible and traceable to the exact model version and command
that measured it.

## Canonical provenance source (the truth pack)

Every entry's `claim_id`/`evidence_id` and provenance fields resolve against the
**Phase −1 truth pack**:

- **Model source commit:** Hugging Face `baidu/Unlimited-OCR`
  **`3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5`** (GitHub
  `7e98affeacba24e95562fbaa234ddb89b856874a`), verified 2026-06-25 — see
  `docs/truth-pack/PINNED_SOURCES.md`.
- **Source / fixture hashes:** SHA-256 of every load-bearing source in
  `docs/truth-pack/SOURCE_HASHES.md`. The `Reference behavior` of every entry
  cites the oracle code by `(file_sha256, line range)` against that table (e.g.
  `modeling_deepseekv2.py` 74e36e6b… for R-SWA semantics), and the measured impact
  cites the **fixture hash** of the parity corpus it ran against.
- **Runtime pin:** the oracle stack is `torch==2.10.0`, `transformers==4.57.1`,
  `Pillow==12.1.1` (`PINNED_SOURCES.md`). A "measured impact" produced against any
  other stack is **not comparable** and may not be recorded as ACCEPTED.

If `SOURCE_HASHES.md` fails to verify, the model moved: STOP, re-pin, and
re-confirm every entry. A DISC entry whose `Reference behavior` cannot be
resolved to a truth-pack source line is **incomplete**.

## Per-entry schema

```
## DISC-NNN: <short title>
- claim_id / evidence_id: <CLAIM-… → artifacts/perf/<bead>/ or artifacts/parity/<bead>/>
- Provenance (model commit + fixture hash): HF 3a7f4db… + <oracle file_sha256:lines from
    SOURCE_HASHES.md> + parity corpus fixture sha256 + <.focrq sha256 for the precision under test>
- CPU feature string: <dispatched SIMD tier the divergence was observed on, e.g.
    aarch64+neon+dotprod or aarch64+neon+i8mm — a divergence can be arch-specific
    (rounding/order)>
- Exact command + env: <gauntlet/parity invocation + FOCR_*/OMP_NUM_THREADS set>
- Reference behavior: <what the torch/transformers oracle does — quote the source line>
- Our impl: <what franken_ocr does, and where (file:fn)>
- Fallback / kill-switch state: <FOCR_* var, default value, and what the ON value restores>
- Measured impact: <real numbers vs reference — CER / token diff / TEDS / ULP / timing,
    plus the AF-2 tail figure (CVaR_0.1 / EVT_p999) for accuracy divergences>
- Resolution: ACCEPTED / INVESTIGATING / REVERT
- Tests affected: <test names / fixture corpus> (XFAIL, never SKIP — §8.6)
- Review date: <YYYY-MM-DD>
```

`Kill switch` is folded into **Fallback / kill-switch state** (the same field the
other two ledgers carry) so the three ledgers share one provenance vocabulary:
the env var name, its default, and exactly what restoring it gives back
(reference-bit-exact behavior).

Quantization-induced divergences (int8, then int4) are the expected source of
most future entries: each will record the per-bit-width measured accuracy delta
against the bf16 reference, the kill switch (e.g. forcing a layer back to higher
precision via `FOCR_INT8_ATTN=0` / `FOCR_INT8_LMHEAD=0` / dropping a tensor one
tier under AF-1), and the corpus slice (dense text / tables / formulas / numbers)
where the impact was measured — with the AF-2 tail bound, not just the mean,
since exact-token OCR fails in the tail.

---

## DISC-001: L0 resampling kernel — `image` crate CatmullRom in place of PIL BICUBIC

- claim_id / evidence_id: CLAIM-l0-resample-catmullrom → artifacts/parity/bd-30me/
    (directory to be populated when the armed L0 EXACT gate runs; today the evidence is
    the in-tree Pillow-12.1.1 goldens + the differential log below)
- Provenance (model commit + fixture hash): HF 3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5
    + `modeling_unlimitedocr.py` sha256 268bdcbe…: `ImageOps.pad(image, (base_size,
    base_size), …)` at :872-873 and `dynamic_preprocess`'s `resized_img =
    image.resize((target_width, target_height))` at :197 (BICUBIC is `Image.resize`'s
    default) — SOURCE_HASHES.md; GOT-OCR2's `GOTImageEvalProcessor` squash resize is the
    same kernel (spec §13b). Oracle resampler pin: **Pillow 12.1.1** (PINNED_SOURCES.md
    runtime pin). **No parity-corpus fixture hash yet** — the oracle preprocessed-tensor
    fixture (bd-1gv.3.1) is still blocked, which is exactly why the impact below is TBD.
- CPU feature string: n/a — preprocess is scalar integer/f64 image code with no SIMD
    dispatch; the divergence is kernel-semantics, not arch-rounding (same output on
    aarch64 and x86_64)
- Exact command + env: `cargo test -p franken_ocr preprocess::` (Pillow-12.1.1 goldens +
    default-path regression, no env); reference path armed via `FOCR_RESAMPLE=pil-bicubic`
    (e.g. `FOCR_RESAMPLE=pil-bicubic focr ocr <img>`); golden generator + 370-case
    differential: `scripts/gen_pil_bicubic_goldens.py` (header has the uv-venv recipe)
- Reference behavior: PIL `Image.resize(…, Resampling.BICUBIC)` / `ImageOps.pad(…,
    method=BICUBIC)` — Pillow's `src/libImaging/Resample.c` 8-bit path: two passes
    (horizontal then vertical) with u8 clip between them; per-window f64 coefficients with
    the sample window clamped to the image **before** weighting and renormalized by their
    own sum; coefficients fixed-pointed at 2^22 (`PRECISION_BITS = 32-8-2`) rounded half
    away from zero; accumulation from `1 << 21` with `clip8` (>>22, saturate 0..255).
- Our impl: default `image::imageops::FilterType::CatmullRom` at the single resize funnel
    `src/preprocess/mod.rs::resample_exact_with` (feeding `pad_to_square`,
    `build_gundam_tiles`, `got_view_tensor`). Same `a = -0.5` continuous cubic, but
    clamp-at-edge sampling + float accumulation ⇒ NOT bit-identical to PIL. The reference
    path `src/preprocess/pil_resample.rs::resize_bicubic` is a step-for-step `Resample.c`
    port proven bit-exact against Pillow 12.1.1: 370/370 randomized differential cases
    (sources 1×1..640×480 → targets 1×1..1024×1024, random + solid-extreme pixels,
    2026-07-01, `scripts/gen_pil_bicubic_goldens.py`) + 6 Pillow-generated goldens
    embedded in its unit tests.
- Fallback / kill-switch state: `FOCR_RESAMPLE` (default **unset** ⇒ CatmullRom,
    byte-identical to the pre-DISC-001 pipeline — doctrine #2); `=pil-bicubic` restores
    reference-bit-exact PIL BICUBIC at ALL preprocess resize sites for L0 EXACT
    comparison. Note the polarity: unlike most entries, here the DEFAULT is the divergence
    and the switch arms the reference.
- Measured impact: **TBD — honestly not yet measured.** The CER/token-diff cost of
    CatmullRom-vs-BICUBIC needs (a) the armed L0 EXACT gate over an oracle
    preprocessed-tensor fixture (bd-1gv.3.1, blocked) and (b) an e2e A/B
    (`FOCR_RESAMPLE` unset vs `=pil-bicubic`) over the parity corpus, on the pinned
    oracle stack. Known today: aggregate tensor stats match the torch oracle to
    ~5e-3 (`preprocess::tests::got_preprocess_matches_oracle_l0b` tolerances exist
    BECAUSE of this divergence) — that bounds it as small but does NOT license calling
    it zero, and per §above this entry may not be promoted to ACCEPTED until the real
    numbers land here.
- Resolution: INVESTIGATING (divergence ledgered + bit-exact reference path and
    kill-switch shipped; measurement pending the armed L0 gate)
- Tests affected: `preprocess::pil_resample::tests::*` (Pillow-12.1.1 goldens, copy /
    solid / 1×1 / zero-dim semantics), `preprocess::tests::
    resample_kind_default_and_kill_switch_parse`, `preprocess::tests::
    default_resample_is_catmullrom_byte_identical` (doctrine-#2 byte-identity regression),
    `preprocess::tests::pil_kill_switch_dispatch_routes_to_pil_resampler`,
    `preprocess::tests::got_preprocess_matches_oracle_l0b` (stats-tolerance test; goes
    EXACT under the kill-switch once the L0 fixture exists)
- Review date: 2026-07-01

---

Entry shape reference (a **template**, not a measurement):

```
## DISC-001: <e.g. int8 attention q/k/v/o drifts a sub-script token on dense formulas>
- claim_id / evidence_id: CLAIM-int8-attn-qkvo → artifacts/parity/<bead>/
- Provenance (model commit + fixture hash): HF 3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5
    + modeling_deepseekv2.py sha256 74e36e6b…: <attn lines>  (SOURCE_HASHES.md)
    + parity corpus fixture sha256 <…>  +  <model>.focrq sha256 <int8-attn build>
- CPU feature string: aarch64+neon+i8mm   (and re-checked on x86_64+avx512vnni)
- Exact command + env: cargo test -p focr --test parity -- disc_int8_attn  /
    OMP_NUM_THREADS=8  (reference torch set_num_threads(8), §9.3)
- Reference behavior: f32 Q·Kᵀ / scores·V bmm (modeling_deepseekv2.py:<lines>)
- Our impl: int8 SMMLA attention in src/decode/attention.rs::<fn>
- Fallback / kill-switch state: FOCR_INT8_ATTN (default 0 = reference f32 attention);
    =1 enables the int8 path under test
- Measured impact: CER Δ <x.xx>%, token diff <n> on the dense-formula slice;
    CVaR_0.1 <x.xx>%, EVT_p999 <x.xx>% (AF-2)
- Resolution: <ACCEPTED|INVESTIGATING|REVERT>
- Tests affected: parity::disc_int8_attn (XFAIL while kill-switch ON)
- Review date: 2026-MM-DD
```
