# PARITY_LADDER.md — the L0–L5 conformance ladder + differential suite (DESIGN)

> **Beads:** `VERIFY-ladder-l0` (`bd-re8.4`), `VERIFY-ladder-l1-l2` (`bd-re8.5`),
> `VERIFY-ladder-l3-l4` (`bd-re8.6`), `VERIFY-ladder-l5` (`bd-re8.7`),
> `VERIFY-differential-suite` (`bd-re8.9`). Parent epic **E-VERIFY** (`bd-re8`).
> Integration runner: `VERIFY-ladder-runner` (`bd-re8.19`).
>
> **This is the DESIGN document for plan §8.2 (the layered ladder) + §8.3
> (differential).** It is the single specification each ladder-gate bead and the
> differential-suite bead implement against, and the contract the three-pillar
> conformance pillar (`bd-re8.13`) and the per-commit parity receipt (§9.2) read.
> It does not itself run anything — it fixes *what each gate compares, at what
> granularity, against which oracle, and within what tolerance*, and names the bead
> that builds it.

---

## 0. The one rule the whole ladder enforces

> **The discrete output is BIT-EXACT where the reference is deterministic; the
> continuous output is held only within the MEASURED tolerance** (plan §8.2,
> AGENTS.md doctrine #1, the frankensearch `parity_logits_and_ranking_match_reference`
> shape).

Concretely:

- **Discrete (decoded token ids, decoded text, tile geometry, image-token id
  stream, argmax index):** **bit-exact** — wherever the reference is deterministic
  (L0 always; L4 over the reproducible prefix defined in §2).
- **Continuous (logits, hidden states, per-op activations):** **within a documented,
  *measured* tolerance** — cosine ≥ 0.9999 in f32, and an int8/int4 budget **derived
  from the oracle's own nondeterminism floor (§2)**, never a guessed epsilon and
  never the *imported* frankensearch `max_diff ~0.055` (that is a BERT-reranker figure
  on a different shape/depth — plan §3.x, §8.2, line-backed below).
- **Determinism (self-consistency):** a *separate* gate (`VERIFY-determinism-gate`,
  `bd-re8.18`) asserts same-input-twice → byte-identical output. This is "match
  yourself", distinct from L4's "match the reference".

A perf commit may only land if it **re-states its parity receipt** (e.g.
`text exact, max logit diff 0.05, deterministic`) and the ladder is green
(§9.2 5-pass loop; the receipt is one row of the `bd-re8.19` scorecard).

---

## 1. Inputs the ladder consumes (the oracle, and why it is split)

Every gate is measured against the **bf16 Python/HF reference oracle**
(`scripts/gen_reference_fixtures.py`, bead `VERIFY-oracle-harness` = `bd-re8.1`),
pinned to `torch==2.10.0`, `transformers==4.57.1`, `Pillow==12.1.1`
(`docs/truth-pack/PINNED_SOURCES.md`), against HF model commit
**`3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5`**.

**Oracle split (OQ-17, RESOLVED — `oq/preprocess-infer.md`):** the official
`infer()` path is CUDA-oriented (`.cuda()` + autocast). So:

- **Correctness** fixtures come from the **unmodified official model on a CUDA host**,
  frozen once and committed. **Parity NEVER depends on CPU HF.** L0–L5 compare
  against these frozen `.npy`/`.json` artifacts.
- **Performance** uses a *separate* CPU baseline (the gauntlet, `bd-re8.17`), never
  the parity oracle.

The oracle dumps two artifact families (`ActivationCapture`, `_run_one` in
`gen_reference_fixtures.py`):

| Artifact | Path | Consumed by |
|---|---|---|
| Per-stage prefill activations (`.npy`) | `<out>/activations/<doc>/<stage>.npy` | L1, L2, L3 |
| End-to-end golden (`.json`: decoded text, tokens, bbox tags) | `<out>/<doc>.json` | L4, L5 |

The **frozen** `.npy`/`.json` are the committed bar. A **live** test-only
PyO3/subprocess bridge (`VERIFY-oracle-bridge` = `bd-re8.3`,
`EngineIdentity::{Subject, Oracle}` asserted-distinct, **never linked into the
shipping `focr` binary**) supplies *ad-hoc* inputs for the differential suite (§8)
and carries the **per-op ULP tolerance table** (4 ULP f32 matmul / 2 ULP
elementwise) used as the L1/L2 comparator.

### Stage seam names (verbatim from `ActivationCapture`)

The oracle hooks emit exactly these stage keys (`gen_reference_fixtures.py`
`register()`):

```
sam_output            # SAM-ViT-B output (post-neck/compress)   SPEC-040..046
clip_output           # CLIP-L/14 output                        SPEC-047..050
projector_output      # 2048→1280 linear projector              SPEC-051..052
decoder_layer_00_hidden … decoder_layer_11_hidden   # per-decoder-layer hidden  SPEC-070..072
lm_head_logits        # full prefill logits, all positions      SPEC-081
```

**Seam-coverage honesty (a real gap, ledgered here so L2 does not over-claim):**
the hooks expose **module-output** seams (`sam_output`, `clip_output`,
`projector_output`, per-layer, `lm_head`). The §8.1 wish-list also names
`post-patch-embed`, `post-bridge/compress`, and `post-connector` as L2 seams.
Of those:

- **post-patch-embed / post-bridge-compress** are *internal* to `sam_model`'s
  forward — not separately hooked. They are reached either by adding a sub-module
  hook in a follow-up to `bd-re8.1`, or (preferred for bring-up) by the **live
  bridge** (`bd-re8.3`) capturing the intermediate tensor on an ad-hoc input. Until
  then, L2 asserts the SAM seam at `sam_output` only and **records the missing
  sub-seams as enumerated coverage debt** (FEATURE_PARITY / the `bd-re8.12`
  coverage matrix), never silently skips them.
- **post-connector** (the masked-scatter result, `inputs_embeds` feeding the
  decoder) equals the **input** to `decoder_layer_00`; L2 captures it by hooking the
  decoder embedding seam (a `bd-re8.1` follow-up) or via the bridge. The connector
  ordering invariant `[SPEC-066]` is *additionally* checked structurally in L0
  (image-token id-stream layout) so a connector mis-order surfaces at L0/L4, not only
  as a fuzzy L2 cosine.

This gap is the kind §8.2 warns about ("a wrong SAM window/pos-embed surfaces ONLY
as a fuzzy L2 cosine drop" — OQ-15); L2 therefore **logs per-stage cosine** so a
silent vision corruption is visible at the coarsest seam we *do* have.

---

## 2. The keystone: establish the oracle's OWN nondeterminism floor FIRST

> **Bead `VERIFY-nondeterminism-floor` = `bd-re8.2`. No L3/L4 tolerance is set
> before this runs.** (plan §8.2 ⚠️ block; AGENTS.md Testing Policy.)

The HF reference is frequently non-deterministic across torch thread counts / BLAS
reduction order at the logit-tie level (bf16 arithmetic, 129280-vocab argmax). If we
set tolerances *before* measuring the oracle's own noise we either (a) chase phantom
"bugs" that are the oracle disagreeing with itself, or (b) set tolerances so loose
that real drift hides inside them.

**Procedure (committed once as a fixture):**

1. Run the oracle **twice** over the full golden corpus, at **two thread counts**
   (e.g. `torch.set_num_threads(8)` and `(32)`), with
   `torch.use_deterministic_algorithms(True)` where it does not error.
2. Diff the runs; record the **nondeterminism envelope** as
   `tests/fixtures/oracle_nondeterminism_envelope.json`:
   - per-token divergence rate,
   - first-divergence token position per document,
   - per-logit max-abs spread at matching positions.
3. **Derive the two tolerances every downstream gate uses:**
   - **L4 "exact" prefix** = the longest decoded-token prefix the oracle reproduces
     *identically* across all runs/threads, **per document**. L4 asserts bit-exact
     only over that prefix.
   - **L3 logit tolerance** = derived from the measured oracle logit variance, **NOT**
     the imported `0.055`.

> **A franken_ocr int8 divergence *inside* the oracle's own bf16 noise is not a bug.**
> This sentence is the entire reason this gate runs first.

---

## 3. The ladder (plan §8.2) — every gate, granularity, oracle, tolerance, bead

The "Tolerance type" column states the §0 rule per gate: **EXACT** = bit-exact
discrete; **COSINE** = continuous within measured cosine/ULP; **MEASURED** =
continuous within the §2-derived quant budget; **EXACT-PREFIX** = bit-exact over the
§2 reproducible prefix; **BUDGET** = aggregate metric within a documented, ledgered
budget.

| Gate | Granularity (what is compared) | Oracle artifact | Tolerance | Tol. type | Bead |
|------|-------------------------------|-----------------|-----------|-----------|------|
| **L0** preprocessing | resized/normalized/padded tensor + tile geometry + image-token id stream | the oracle's preprocessed input tensor (regenerated by re-running the deterministic front end) | **EXACT** — gray pad `127`, `find_closest_aspect_ratio` selection, `[-1,1]` normalize, tile/token census all byte/value-exact | EXACT | `bd-re8.4` |
| **L1** per-op | each kernel's output vs the matching oracle activation | per-stage `.npy` + live bridge | **cosine ≥ 0.9999 (f32)**; documented int8 tolerance; bridge path uses the **4 ULP f32-matmul / 2 ULP elementwise** table | COSINE | `bd-re8.5` |
| **L2** per-layer | all 12 `decoder_layer_NN_hidden`; each vision-stage seam (`sam_output`, `clip_output`, `projector_output`, + connector/sub-seams per §1) | per-stage `.npy` | **cosine ≈ 1.0**; **max-abs-diff LEDGERED per layer** (so slow cross-layer drift is visible) | COSINE | `bd-re8.5` |
| **L3** logits | pre-sampling logits (`lm_head_logits`, all prefill positions) | `lm_head_logits.npy` | within the **MEASURED int8/int4 quant tolerance derived from §2** (the oracle nondeterminism floor — NOT `0.055`); **argmax MUST match** at every position where the reference is deterministic | MEASURED + EXACT-argmax | `bd-re8.6` |
| **L4** token | decoded token id sequence under greedy | golden `.json` tokens | **EXACT under greedy**, defined ONLY over the §2 reproducible prefix | EXACT-PREFIX | `bd-re8.6` |
| **L5** end-to-end OCR | decoded text + bbox tags on the golden corpus | golden `.json` (text + bbox) | **exact-match where deterministic**; aggregate **CER / TEDS / Formula-CDM within a documented budget** (int8 within noise of bf16; int4 within a small ledgered budget; **no catastrophic regression on dense numeric/table content** — held to the AF-2 tail bound) | EXACT-where-det + BUDGET | `bd-re8.7` |

A failed lower gate makes higher gates meaningless, so the **integration runner**
(`bd-re8.19`) executes L0→L5 **in order** and short-circuits sensibly, emitting one
structured scorecard `{gate, granularity, tolerance, measured, pass}` (§7).

### 3.1 L0 — preprocessing parity (EXACT) — `VERIFY-ladder-l0` / `bd-re8.4`

Preprocessing is deterministic integer/float arithmetic with **no quantization**, so
any drift is a bug → **EXACT** tolerance, not cosine. If preprocessing diverges,
every downstream stage is fed a different tensor and L1+ become meaningless. L0 is a
hard prerequisite for Phase 1 (fp32 parity).

What is asserted exact (line-backed to THE SPEC and `oq/preprocess-infer.md`):

- **Normalize** — `image_mean = image_std = [0.5,0.5,0.5]` → scale to `[0,1]` then
  `(x-0.5)/0.5` → **`[-1,1]`**, CHW layout `[SPEC-021]`.
- **Pad-to-square, gray** `(127,127,127)` = `tuple(int(m*255))`, aspect preserved
  (`ImageOps.pad` equivalent) `[SPEC-022]`.
- **Tile geometry** — `patch_size=16`, `downsample_ratio=4`,
  `num_queries = ceil((image_size//patch_size)/downsample_ratio)`. Two modes:
  **Base** (`1024×1024`, `crop_mode=false`) and **Gundam** (`base_size=1024`,
  `image_size=640`, `crop_mode=true`, global padded-1024 view `global_view_pos='head'`
  + `640×640` local crops via `dynamic_preprocess()` / `find_closest_aspect_ratio()`
  over `i*j ∈ [min_num, max_num]`, `min_num=2/max_num=32`, OQ-7) `[SPEC-023..029]`.
- **Ratio selection** (`find_closest_aspect_ratio`, tie-break larger area),
  **gray pad value**, and the **`[-1,1]` normalize** all match exactly.
- **Image-token id stream** — the 2D layout, `273` slots per 1024-view
  (`(16+1)·16+1`, CENSUS (c), OQ-18) `[SPEC-028]`; BOS prepend + `images_seq_mask`
  `[SPEC-030]`; image-tensor packing `images=[(crop, ori)]` `[SPEC-031]`.

The tile/token census is additionally asserted against the **machine-readable census**
(`docs/truth-pack/CENSUS.md`, E-PM1) — the source of truth — so a token-count drift is
caught against a pinned number, not a magic constant.

> **Why exact and not cosine:** the only floating arithmetic here is the resize +
> normalize; the *reference* itself is deterministic (PIL/torch on CPU for the front
> end), so there is no nondeterminism floor to absorb. A `[-1,1]` value mismatch or a
> wrong pad pixel is, by construction, a port bug.

### 3.2 L1 per-op + L2 per-layer (cosine) — `VERIFY-ladder-l1-l2` / `bd-re8.5`

L1/L2 catch kernel drift at the finest granularity the oracle exposes. A correct
end-to-end answer can hide a *compensating pair* of per-op bugs; per-op/per-layer
cosine pins each kernel to the reference independently. Every kernel bead in
P1/P2/P3 re-proves L1/L2 on each change (the operational form of "every fused op at
peak AND correct").

- **L1 (per-op):** each kernel's output vs the oracle activation, **cosine ≥ 0.9999
  in f32** (documented int8 tolerance for the quant path). The **bridge path**
  (`bd-re8.3`) compares through the **per-op ULP table — 4 ULP f32 matmul, 2 ULP
  elementwise** — the comparator, *not* a hand-guessed epsilon.
- **L2 (per-layer):** all 12 decoder-layer hidden states + each vision-stage seam
  (§1). **cosine ≈ 1.0; max-abs-diff LEDGERED per layer.** The per-layer max-abs
  ledger makes a slow drift *across* layers visible even when the final cosine still
  looks fine — and is the only signal that catches a wrong SAM window size / windowed
  pos-embed (OQ-15), which surfaces *solely* as a fuzzy L2 cosine drop. L2 therefore
  **logs per-stage cosine**, never a single aggregate.
- **Cosine, not bit-exact, because** f32-vs-bf16 (and later int8-vs-bf16) are
  *continuous-value* divergences within a documented tolerance — exactly the
  continuous half of the §0 rule.

### 3.3 L3 logits + L4 token — `VERIFY-ladder-l3-l4` / `bd-re8.6`

This is where quantization risk is *adjudicated*. L3 bounds the continuous logit
divergence; L4 demands the discrete decoded sequence be exact where the reference is
deterministic. It is the gate that says "int8/int4 did not change what the model
says", and the gate P2 (int8) / P4 (int4) must clear.

- **L3 (logits):** pre-sampling logits within the int8/int4 quant tolerance
  **DERIVED from §2** (the nondeterminism floor — NOT the precedent `0.055`).
  **argmax MUST match** at every position where the reference is deterministic. This
  is the continuous-within-tolerance + discrete-argmax-exact split applied to the
  logit row.
- **L4 (token):** the decoded token sequence is **EXACT under greedy where the
  reference is deterministic** — and "where deterministic" is the §2 reproducible
  prefix, **per document**, nothing looser.
- **Sampler semantics replicated exactly** (`[SPEC-100..103]`, OQ-18): greedy /
  low-temp argmax (`do_sample = temperature>0`, default `0.0` ⇒ argmax); strip
  trailing EOS `<｜end▁of▁sentence｜>`; **`no_repeat_ngram_size=35`** with a sliding
  `ngram_window` that **DIFFERS by mode — `128` single-image vs `1024` multi-image**
  (the custom `SlidingWindowNoRepeatNgramProcessor` bans repeated 35-grams within the
  window). These are first-class generation semantics, not CLI flags — a wrong window
  silently changes the token stream and would fail L4 inside the reproducible prefix.

### 3.4 L5 end-to-end OCR — `VERIFY-ladder-l5` / `bd-re8.7`

L5 is the top of the ladder and the only gate a non-expert can read: does
franken_ocr produce the *same document parse* as the reference? It is the operational
definition of **G1**. The lower gates can all be green and L5 still catch a
**postprocess** bug (tag parsing, bbox rescale, markdown assembly).

- Compare decoded text + bbox tags on the **frozen golden corpus**, **exact-match
  where the reference is deterministic** (the reproducible-prefix discipline of §2,
  applied to text).
- Aggregate **CER / TEDS / Formula-CDM within a documented quantization budget**:
  - **int8 within noise** of bf16 (NVFP4's reported "OCR identical to BF16" is the
    bar);
  - **int4 within a small ledgered budget**, **with no catastrophic regression on
    dense numeric / table content** — held to the AF-2 **tail** bound
    (`VERIFY-tail-risk` = `bd-1xfa.2`: `CVaR_0.1` / `EVT_p999`, not the mean), because
    exact-token OCR fails in the tail and mean-CER under-predicts it.
  - Metrics: **CER** (character error rate), **TEDS / TEDS-S** (table structure),
    **Formula-CDM** (LaTeX) — the reference's own reported strengths (printed-text
    Text-Edit `0.038`, tables TEDS `90.93`, math Formula-CDM `92.61`).
- **Postprocess parity:** EOS strip; `<|ref|>…<|/ref|><|det|>…<|/det|>` regex parse;
  bbox coords `/999 → ×W/H` rescale `[SPEC-110..115]`; markdown assembly (HTML
  tables, LaTeX, `<page>` separators for multi-page).

> **Multi-page is CROSS-PAGE DEPENDENT under R-SWA (OQ-13, RESOLVED).** `infer_multi`
> concatenates ALL pages into ONE prefill/`generate()` call, so page *N* attends to
> pages `1..N-1`. **L5 multi-page comparison is against the multi-page reference
> output, NEVER a sum of single-page parses.** The full prompt-mode taxonomy and
> whether bboxes emit in all modes is OQ-8 (RESOLVED, `oq/preprocess-infer.md`); L5
> bbox assertions follow that taxonomy.

---

## 4. Phase-gating: which gate must be green when

The ladder is not a one-time check — it is re-proven on every perf commit (§9.2) and
is the per-phase exit gate.

| Phase | Numerics | Ladder requirement (exit gate) |
|-------|----------|--------------------------------|
| **Phase 1** — fp32 reference-parity | pure f32 | **L0–L5 all green in f32**: per-layer cosine ≥ 0.9999, decoded text exact where reference deterministic, CER ≈ reference; determinism gate green; census passes; no blocking `[OPEN]`. |
| **Phase 2** — int8 | weight-only int8 decoder | L0–L4 hold; **L3 within the documented int8 tolerance, argmax-match where deterministic**; L5 **int8 within noise** of bf16; every accepted int8 divergence is a `DISC-NNN`. |
| **Phase 3** — SIMD kernels | int8 + hand-SIMD islands | **SIMD path == scalar path BIT-IDENTICAL** (a *separate* invariant, monitored as an e-process); the ladder result is unchanged vs Phase 2 (SIMD is a bit-identical layer, never a numerics change). |
| **Phase 4** — int4 | int4-group experts | L5 **int4 within a small ledgered budget**, **AF-2 tail bound respected** on the dense-numeric/table slice; each int4 divergence a `DISC-NNN` with kill-switch (drop the tensor one tier under AF-1). |

The §9.2 5-pass optimization loop **re-proves L0–L5 on every lever**; a lever that
drifts any gate is **reverted, no source landed**, and memorialized in
`docs/NEGATIVE_EVIDENCE.md` (the W/L/N ledger). The keep-gate and the ladder are the
two halves of "correctness before speed, falsifiably".

---

## 5. Where bit-exact applies and where it cannot (the explicit map)

The §0 rule, made concrete per artifact, so no gate over- or under-claims:

| Artifact | Reference deterministic? | Comparison |
|----------|--------------------------|------------|
| Preprocessed tensor (`[-1,1]`, pad 127), tile geometry, image-token id stream | **Yes** (deterministic CPU front end) | **bit/value-exact** (L0) |
| Per-op / per-layer activations | No (bf16 reduction order) | **cosine ≥ 0.9999 / ULP table** (L1/L2) |
| Logit row values | No (bf16, BLAS order) | **measured tolerance from §2** (L3) |
| Logit **argmax** index | Yes, *where deterministic* (§2) | **exact** (L3) |
| Decoded token id (greedy) | Yes, over the §2 reproducible prefix | **exact-prefix** (L4) |
| Decoded text / bbox tags | Yes, where deterministic | **exact-where-det** (L5) |
| Aggregate CER / TEDS / Formula-CDM | n/a (aggregate metric) | **within documented budget** (L5) |
| SIMD vs scalar kernel | **Yes** (same integer/float ops) | **bit-identical** (Phase-3 invariant, e-process) |
| franken_ocr run vs itself | **Yes** | **byte-identical** (determinism gate, `bd-re8.18`) |

The two columns are the entire safety argument: every "Yes" is held to bit/value
exactness; every "No" is held only to a number we **measured** (the §2 floor or the
ULP table), never one we guessed.

---

## 6. The differential suite (plan §8.3) — `VERIFY-differential-suite` / `bd-re8.9`

Differential testing (`/testing-conformance-harnesses` Pattern 1) is the gold-standard
conformance method: run **our path** and the **oracle** on the *same* inputs and diff.
It catches what golden snapshots miss — snapshots only cover the *frozen* corpus, while
the differential harness runs on **any** input via the live bridge. It is how we prove
"same answer as the bf16 reference" for arbitrary inputs.

### 6.1 Two axes: granularity × oracle

The suite diffs along **two granularities** against **a primary and two secondary
oracles**:

**Granularity (each wired into the `ConformanceTest` trait as
`category = differential`, `bd-re8.12`):**

- **Per-op** — each kernel's output vs the oracle activation on the *same* input,
  through the **per-op ULP table** (4 ULP f32 matmul / 2 ULP elementwise). This is the
  L1 comparator applied to ad-hoc (non-corpus) inputs via the live bridge.
- **End-to-end** — the full `focr` forward vs the oracle's decoded output, through the
  **L3/L4/L5 tolerances** (measured logit tol + reproducible-prefix exactness +
  CER/TEDS/Formula-CDM budget).

**Oracles:**

| # | Oracle | Role | Trust basis | Compare scope |
|---|--------|------|-------------|---------------|
| **Primary** | **bf16 HF reference** via frozen `.npy`/`.json` (corpus) + the live PyO3/subprocess bridge (ad-hoc) | the gold standard for both per-op and e2e | the unmodified official model; the source of L0–L5 truth | **all** modules / full e2e |
| **Secondary A** | **community NVFP4** per-layer activation dumps | independent confirmation of the **quant** path during bring-up | the NVFP4 author reports "OCR identical to BF16" | **only the modules NVFP4 actually quantizes** (§6.3) |
| **Secondary B** | **community GGUF** per-layer dumps | independent confirmation of an alternative quant scheme | a second, independently-produced quantization | the modules GGUF quantizes |

The secondaries are **frozen fixtures only — no framework runtime dependency**
(`gen_reference_fixtures.py` dumps them alongside the bf16 oracle, §8.1). They confirm
our quant path *independently of the bf16 oracle*, which is exactly the cross-check that
catches a quant-loader bug the bf16 oracle (running full precision) cannot see.

### 6.2 The contract each differential test emits

The harness emits one structured row per test (consumed by the `bd-re8.12` coverage
matrix and the `bd-re8.19` scorecard):

```
{ scope: "op" | "e2e",
  oracle: "bf16" | "nvfp4" | "gguf",
  module: "<op or stage name>" | "e2e",
  max_diff: <f32>,          # or cosine / ULP for op scope
  within_tol: bool,
  xfail: bool,              # an intentional divergence → a DISC-NNN
  disc: "DISC-NNN" | null }
```

- **Intentional divergences are `XFAIL`, never `SKIP`** — each one a `DISC-NNN` in
  `docs/DISCREPANCIES.md` (reference behavior, our impl, **measured** impact,
  kill-switch env var, resolution, review date). A `SKIP` silently drops coverage; an
  `XFAIL` keeps the clause in the matrix as a known, ledgered divergence.
- **Model-gated:** e2e differential runs need the reference env and the 6.67 GB
  weights → **skip-with-SUCCESS** when absent, but **prove the native path ran** by
  pointing any fallback at `/nonexistent` (so a silent skip is never mistaken for a
  pass).

### 6.3 ⚠️ OQ-14 — compare ONLY the modules a secondary oracle quantizes

> **OQ-14 status: PARTIAL** (`docs/truth-pack/oq/secondary.md`,
> `docs/truth-pack/OQ_INDEX.md`). **OUR** quantizable-Linear candidate set is
> **definitive** from the pinned bf16 index — **2244 `*_proj.weight` (+ `lm_head` =
> 2245)**:

| Bucket | Count |
|---|---:|
| Routed experts (`*.mlp.experts.*.{gate,up,down}_proj.weight`) | 2112 |
| Language attention (`*.self_attn.{q,k,v,o}_proj.weight`) | 48 |
| Shared experts (`*.mlp.shared_experts.{gate,up,down}_proj.weight`) | 33 |
| Dense MLP layer 0 (`layers.0.mlp.{gate,up,down}_proj.weight`) | 3 |
| Vision tower (`*.vision_model.*.self_attn.{qkv,out}_proj.weight`) | 48 |
| `lm_head.weight` | 1 |
| **TOTAL** | **2245** |

> The bead's "~2229" / "~2196" estimates are superseded by the line-backed **2244 /
> 2245** (the estimate dropped the 48 vision projections and over-counted shared as 66
> instead of 33).

The **exact NVFP4 subset** of those 2245 (and the per-block scales) is **BLOCKED** —
it is defined only by the external `sahilchachra/Unlimited-OCR-NVFP4` checkpoint, which
is **not present locally** (the pinned bf16 snapshot has **no** `quantization_config`
and **no** `*.weight_scale`/`fp4`/`amax` keys — verified absent).

**Therefore the differential rule for the NVFP4 secondary oracle:**

> **Compare ONLY the modules NVFP4 actually quantizes — its `*.weight_scale` /
> `quantization_config.ignore` set — never the full 2245.** Diffing a module NVFP4
> left in bf16 (commonly the **whole vision tower**, the router `mlp.gate`,
> embeddings, and often `lm_head`) against our quantized version is
> **apples-to-oranges** and would manufacture a fake divergence.

To resolve OQ-14 (and unblock the full NVFP4 secondary-oracle comparison), dump from
the external repo (per `oq/secondary.md`):

1. its `model.safetensors.index.json` `weight_map` → diff suffixes vs our 2244; record
   the scale-key naming (`*.weight_scale`, `*.weight_scale_2`, `*.input_scale`,
   `*.weight_global_scale`);
2. its `config.json` `quantization_config` → `quant_method`
   (`nvfp4`/`modelopt`/`compressed-tensors`), `group_size` (NVFP4 block = 16),
   and the **`ignore`/`exclude` list** (the bf16-kept Linears — **this list is exactly
   the differential compare-scope filter**);
3. per-tensor shapes + dtypes from the safetensors headers (confirm the fp4-packing
   layout, scale shapes `[out, in/group_size]`).

Until that dump lands, the NVFP4/GGUF secondaries are exercised **only on the modules
whose scale keys are present in the dump we have**; the rest are enumerated coverage
debt, not a green pass.

### 6.4 Relationship to metamorphic + golden (the rest of §8.3)

The differential suite is one of three §8.3 pillars; the design is consistent with the
sibling beads but they are *not* owned here:

- **Metamorphic** (`VERIFY-metamorphic-suite` = `bd-re8.10`): oracle-free invariants —
  identity-resize, 90°-rotation/transpose bbox relations, whitespace-pad invariance,
  Base-mode determinism. **⚠️ Multi-page is cross-page DEPENDENT (OQ-13):** the
  defensible property is "changing page order / earlier-page content **MAY** change
  later-page output" — **never** "multi-page = sum of single-page parses".
- **Golden artifacts** (`VERIFY-golden-artifacts` = `bd-re8.11`): exact (insta) for
  `focr ocr --json` + `--help`; **fuzzy (ULP)** for logits/activations (consumes the
  L1/L2 comparator); scrubbed (timing/run-id) for robot NDJSON; canonicalized for
  cross-platform. `UPDATE_GOLDENS=1` + mandatory `git diff`; CI never auto-updates.

Differential = "same as reference (any input)"; metamorphic = "self-consistent under
transforms (no oracle)"; golden = "no regression vs frozen good output". The three are
complementary, not redundant.

---

## 7. The integration runner + per-commit parity receipt — `bd-re8.19`

A single ordered runner (`tests/conformance_harness.rs`, `VERIFY-ladder-runner` =
`bd-re8.19`) executes **L0→L5 in order** over the golden corpus, short-circuits
sensibly (a failed L0 makes L1+ meaningless and is reported as such, not as six
confusing failures), and emits **one** structured scorecard fixture:

```
{ gate: "L0".."L5",
  granularity: "<what was compared>",
  tolerance: "<the rule from §3>",
  measured: <exact | cosine | logit_diff | CER/TEDS/Formula_CDM>,
  pass: bool }
```

From that scorecard the **per-commit parity receipt** is derived (e.g.
`text exact, max logit diff 0.05, deterministic`) — the single line every perf commit
re-states under §9.2. The runner is:

- **Model-gated:** skip-with-SUCCESS without the 6.67 GB weights; **prove the native
  path ran** (`/nonexistent` fallback) so a silently-skipped suite is never a fake pass.
- **The single entry** the phase exit gates and the three-pillar **conformance pillar**
  (`bd-re8.13`) call; the per-category scorecard rows feed the **conformal lower-bound
  ratchet** (`bd-re8.14`, release decision on the LOWER bound, no per-category
  regression) and the **CVaR/EVT tail monitor** (`bd-1xfa.2`).

---

## 8. Provenance + traceability (the artifact-graph contract)

Every gate result and differential row is traceable to the exact model version it was
measured against — the same artifact-graph discipline as
`DISCREPANCIES.md` / `NEGATIVE_EVIDENCE.md` / `PERF_LEDGER.md` (`bd-re8.16`):

- **Model source commit:** HF `3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5` (GitHub
  `7e98affeacba24e95562fbaa234ddb89b856874a`), verified 2026-06-25
  (`docs/truth-pack/PINNED_SOURCES.md`).
- **Source / fixture hashes:** SHA-256 of every load-bearing source in
  `docs/truth-pack/SOURCE_HASHES.md`; each gate's "reference behavior" cites the oracle
  code by `(file_sha256, line range)`, and each measured result cites the **fixture
  hash** of the parity corpus it ran against.
- **Runtime pin:** the oracle stack is `torch==2.10.0`, `transformers==4.57.1`,
  `Pillow==12.1.1`. A result produced against any other stack is **not comparable** and
  may not be recorded as a pass.
- **Fixture provenance** (`PROVENANCE.md`, `bd-re8.12`): `transformers==4.57.1`,
  `torch==2.10.0`, git ref, exact command.

If `SOURCE_HASHES.md` fails to verify, the model moved: **STOP**, re-pin, re-confirm.

---

## 9. Bead → gate index (the implementation map)

| Bead key | bd-id | Delivers |
|----------|-------|----------|
| `VERIFY-oracle-harness` | `bd-re8.1` | `gen_reference_fixtures.py` — frozen per-stage `.npy` + e2e `.json` + secondary NVFP4/GGUF dumps |
| `VERIFY-nondeterminism-floor` | `bd-re8.2` | the §2 envelope; derives the L3 tolerance + L4 reproducible prefix |
| `VERIFY-oracle-bridge` | `bd-re8.3` | live test-only PyO3/subprocess bridge + per-op ULP table + `EngineIdentity` |
| **`VERIFY-ladder-l0`** | **`bd-re8.4`** | **L0 preprocessing (EXACT)** — §3.1 |
| **`VERIFY-ladder-l1-l2`** | **`bd-re8.5`** | **L1 per-op + L2 per-layer (cosine + max-abs ledger)** — §3.2 |
| **`VERIFY-ladder-l3-l4`** | **`bd-re8.6`** | **L3 logits (measured tol + argmax) + L4 token (exact-prefix)** — §3.3 |
| **`VERIFY-ladder-l5`** | **`bd-re8.7`** | **L5 end-to-end (exact-where-det + CER/TEDS/Formula-CDM budget)** — §3.4 |
| **`VERIFY-differential-suite`** | **`bd-re8.9`** | **differential: our path vs bf16 oracle (per-op + e2e) + NVFP4/GGUF secondaries (OQ-14 scope)** — §6 |
| `VERIFY-metamorphic-suite` | `bd-re8.10` | oracle-free invariants (§6.4) |
| `VERIFY-golden-artifacts` | `bd-re8.11` | insta/fuzzy/scrubbed/canonicalized goldens (§6.4) |
| `VERIFY-conformance-trait` | `bd-re8.12` | `ConformanceTest` trait + coverage matrix + XFAIL/DISC discipline |
| `VERIFY-three-pillar-cert` | `bd-re8.13` | reads the L0–L5 scorecard as the conformance pillar |
| `VERIFY-conformal-ratchet` | `bd-re8.14` | Beta-posterior × conformal lower-bound release ratchet over the scorecard |
| `VERIFY-ledger-discipline` | `bd-re8.16` | artifact-graph ledgers (DISCREPANCIES / NEGATIVE_EVIDENCE / PERF_LEDGER) |
| `VERIFY-determinism-gate` | `bd-re8.18` | same-input-twice byte-identical + `many_pages_without_deadlock` watchdog |
| `VERIFY-ladder-runner` | `bd-re8.19` | the ordered L0→L5 integration runner + structured scorecard (§7) |
| `VERIFY-tail-risk` (AF-2) | `bd-1xfa.2` | `tail_risk_monitor` — `CVaR_0.1` / `EVT_p999` the L5 budget gates on |

---

## 10. Open items this design carries forward (honest gaps)

- **OQ-14 (PARTIAL):** the exact NVFP4 quantized-module set + per-block scales need the
  external `sahilchachra/Unlimited-OCR-NVFP4` dump (§6.3). Until then the NVFP4/GGUF
  secondaries compare only the modules whose scale keys are present; the rest are
  enumerated coverage debt.
- **L2 vision sub-seams:** `post-patch-embed` / `post-bridge-compress` /
  `post-connector` are not separately hooked by the current `gen_reference_fixtures.py`
  (§1); they are reached via a `bd-re8.1` follow-up hook or the live bridge, and tracked
  as coverage debt in the meantime — never silently skipped.
- **OQ-8 (RESOLVED) bbox-per-mode:** L5 bbox assertions follow the resolved prompt-mode
  taxonomy (`oq/preprocess-infer.md`); the mode matrix is enumerated in FEATURE_PARITY.
- All three are line-backed retry conditions, not blockers: **zero blocking OQs remain**
  for the f32 reference-parity forward (`OQ_INDEX.md`).
</content>
</invoke>
