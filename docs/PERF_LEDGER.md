# franken_ocr — Performance Ledger

> Head-to-head, **MEASURED** performance log for the `focr` engine. Every row is
> a real wall-clock measurement against a real reference on the same machine.
> No row is added without a number; projections and targets do not go here (they
> live in `COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md`). Levers that show ~0 gain or a
> regression are reverted and recorded in `docs/NEGATIVE_EVIDENCE.md`, not here.

This is an **artifact-graph ledger** (plan §8.4): every row carries the same
FrankenSuite provenance fields as `NEGATIVE_EVIDENCE.md` / `DISCREPANCIES.md`,
plus the §9.1 **roofline** columns and the §9.3 **fairness** columns that make
each ratio honest.

## Canonical provenance source (the truth pack)

Every row's `claim_id`/`evidence_id`/`model_commit`/`fixture_hash` resolve against
the **Phase −1 truth pack**:

- **Model source commit:** Hugging Face `baidu/Unlimited-OCR`
  **`3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5`** (GitHub
  `7e98affeacba24e95562fbaa234ddb89b856874a`), verified 2026-06-25 — see
  `docs/truth-pack/PINNED_SOURCES.md`. This is the `model_commit` column value for
  every franken_ocr row.
- **Source / fixture hashes:** `docs/truth-pack/SOURCE_HASHES.md`. The
  `fixture_hash` column is the SHA-256 of the parity/perf fixture the row was
  measured on, plus the `.focrq` conversion hash for the precision the row reports.
- **Runtime pin:** the reference stack is `torch==2.10.0`, `transformers==4.57.1`
  (`PINNED_SOURCES.md`); the **reference** column names which proven CPU baseline
  ran (CPU-patched HF, else llama.cpp GGUF / ONNX Runtime / MLAS, labeled as such —
  §9.3). A ratio against an unpinned stack is **not comparable** and is not added.

If `SOURCE_HASHES.md` fails to verify, the model moved: STOP, re-pin, and re-run.
A row whose `model_commit`/`fixture_hash` cannot be resolved to the truth pack is
**incomplete and may not be cited**.

## Measurement protocol

- **Correctness reference** is the pinned CUDA PyTorch / `transformers`
  Unlimited-OCR model (bf16) from `scripts/gen_reference_fixtures.py`.
- **CPU performance reference** is whichever CPU baseline Phase -1 proves is
  valid and runnable on the same host: CPU-patched HF if it reproduces the CUDA
  oracle's tokens, otherwise llama.cpp GGUF / ONNX Runtime / MLAS labeled as such.
- **`focr`** is measured in the `release-perf` profile (`debug=line-tables-only,
  lto=thin, codegen-units=1`), warm, with a fixed thread budget recorded per row.
- **Precision column** states what is being compared: `focr-int8` (or `-int4`)
  vs `torch-bf16`. A speed ratio is only meaningful alongside the accuracy delta
  for that precision (see `docs/DISCREPANCIES.md`).
- **ratio** = reference_time / focr_time (>1.0 means focr is faster). Stages are
  measured per the pipeline boundary they name.

### Fairness controls (all mandatory — §9.3; a row without them is invalid)

- **Thread parity:** pin `OMP_NUM_THREADS` / torch `set_num_threads(N)` **equal
  to** focr's thread budget. **NEVER benchmark torch at @64** — oversubscription
  inflates fake "wins" (a hardened frankentorch lesson); measure at @8 / @32. The
  `thread budget` column records the N used on **both** sides.
- **Allocator fairness:** build focr with the same allocator posture as the claim
  (system by default; `mimalloc` only behind its feature, §9.6), wired into the
  measured binary — the `command/env` column records which. When a row uses
  mimalloc, both sides use the same allocator.
- **Precision fairness:** the `precision` column annotates focr-int8/-int4 vs
  torch-bf16 (and torch-int8 if available) — a raw ratio across different numerics
  is meaningless without it.
- **Best-of-N with warmup discard:** report the min and the per-side precision
  (cv%); `cv_pct > 5%` is noise and ineligible to land a claim.

### Roofline / compute-floor (§9.1 — recorded next to every ratio)

For each stage, compute the **compute floor** (int8/int4 GEMM FLOPs ÷ the arch's
peak int8 throughput) and the **memory floor** (bytes streamed ÷ DRAM bandwidth),
and take the **max** — that is what a perfect kernel would hit. The ledger records
that floor and focr's **distance above it** (`focr_time / floor_time`). A stage
sitting **>~1.3× above its floor** is a named, attackable lever (which kernel,
which arch), never an excuse — that is the falsifiable perf target. (Sibling
lesson: ONNX-int8 ran *near* the compute floor while the Rust path sat ~2× above
it, and the gap was kernels below peak, not framework overhead — see
`NEGATIVE_EVIDENCE.md` NE-INH-5.)

## Ratio table

The table carries, per row: the **artifact-graph provenance** fields
(`claim_id`, `evidence_id`, `model_commit`, `fixture_hash`,
`arch/cpu_features`, `fallback/kill-switch`), the **measured ratio**, the
reference backend identity, the **roofline** fields (`floor_kind`, `floor_ms`,
`dist_above_floor`), and the **fairness** fields (`precision`,
`threads (focr=ref N)`, `allocator`, `command/env`), plus the mandatory
`correctness_proof` reference that proves the row did not buy speed by breaking
parity. Every field is mandatory; a row missing any is invalid.

| date | claim_id | evidence_id | model_commit | fixture_hash | arch/cpu_features | stage | reference_backend | focr_ms | ref_ms | ratio | floor_kind | floor_ms | dist_above_floor | precision (focr vs ref) | threads (focr=ref N) | allocator | command/env | fallback/kill-switch state | correctness_proof | notes |
|------|----------|-------------|--------------|--------------|-------------------|-------|-------------------|--------:|-------:|------:|------------|---------:|-----------------:|-------------------------|----------------------|-----------|-------------|----------------------------|-------------------|-------|
| 2026-07-02 | G2-unlimited-int8-0014-20260702 | artifacts/perf/bd-re8.17/G2-unlimited-int8-0014-20260702 | 3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5 | page=page_0014.png sha256=f1e35a58f673036b16302c86676f1ec6a89218fb4e666c477dbb4ae21b904df2; weights=model-00001-of-000001.safetensors sha256=2bc48a7a110061ea58fff65d3169367eebe3aee371ca6968dc2219c1b2855fc6 (bf16 shard, runtime int8 quant — no .focrq) | aarch64+neon+dotprod | preprocess | hf | 20.000 | 24.399 | 1.220 | memory | 0.105 | 190.73 | focr-int8 vs hf-bf16 | focr=ref=8 | system | focr: /private/tmp/cc_tgt_dev/release/focr ocr /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/pages/page_0014.png --model /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/model [threads=8 FOCR_TIMING=1]; ref: scripts/gauntlet_reference.py --stage all --page /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/pages/page_0014.png --model-dir /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/model --backend hf --precision bf16 --entry gauntlet_ref_unlimited:run_stage --setup gauntlet_ref_unlimited:setup --runs 5 --warmup 1 --threads 8 --out /Users/jemanuel/projects/franken_ocr/artifacts/perf/bd-re8.17/ref_page_0014/ref_stages.json [torch==2.10.0 transformers==4.57.1] | FOCR_DECODE_INT8=1 FOCR_MODEL_PATH=/Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/model FOCR_THREADS=8 FOCR_TIMING=1 | CER_norm=0.03529 focr-int8 text vs pinned HF bf16 reference text on page_0014.png, same prompt/runs as the timings, ref <;det;> grounding spans stripped (scripts/baseline/compare_ocr.py; artifacts/perf/bd-re8.17/cer_page_0014/cer.json) | quiet-host runbook (scripts/gauntlet_runbook.sh); best-of-5 warm, N=8 both sides; FORCE=1 loadavg-gate bypassed |
| 2026-07-02 | G2-unlimited-int8-0014-20260702 | artifacts/perf/bd-re8.17/G2-unlimited-int8-0014-20260702 | 3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5 | page=page_0014.png sha256=f1e35a58f673036b16302c86676f1ec6a89218fb4e666c477dbb4ae21b904df2; weights=model-00001-of-000001.safetensors sha256=2bc48a7a110061ea58fff65d3169367eebe3aee371ca6968dc2219c1b2855fc6 (bf16 shard, runtime int8 quant — no .focrq) | aarch64+neon+dotprod | vision-encode | hf | 6930.000 | 25785.159 | 3.721 | compute | 1934.504 | 3.58 | focr-int8 vs hf-bf16 | focr=ref=8 | system | focr: /private/tmp/cc_tgt_dev/release/focr ocr /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/pages/page_0014.png --model /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/model [threads=8 FOCR_TIMING=1]; ref: scripts/gauntlet_reference.py --stage all --page /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/pages/page_0014.png --model-dir /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/model --backend hf --precision bf16 --entry gauntlet_ref_unlimited:run_stage --setup gauntlet_ref_unlimited:setup --runs 5 --warmup 1 --threads 8 --out /Users/jemanuel/projects/franken_ocr/artifacts/perf/bd-re8.17/ref_page_0014/ref_stages.json [torch==2.10.0 transformers==4.57.1] | FOCR_DECODE_INT8=1 FOCR_MODEL_PATH=/Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/model FOCR_THREADS=8 FOCR_TIMING=1 | CER_norm=0.03529 focr-int8 text vs pinned HF bf16 reference text on page_0014.png, same prompt/runs as the timings, ref <;det;> grounding spans stripped (scripts/baseline/compare_ocr.py; artifacts/perf/bd-re8.17/cer_page_0014/cer.json) | quiet-host runbook (scripts/gauntlet_runbook.sh); best-of-5 warm, N=8 both sides; FORCE=1 loadavg-gate bypassed |
| 2026-07-02 | G2-unlimited-int8-0014-20260702 | artifacts/perf/bd-re8.17/G2-unlimited-int8-0014-20260702 | 3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5 | page=page_0014.png sha256=f1e35a58f673036b16302c86676f1ec6a89218fb4e666c477dbb4ae21b904df2; weights=model-00001-of-000001.safetensors sha256=2bc48a7a110061ea58fff65d3169367eebe3aee371ca6968dc2219c1b2855fc6 (bf16 shard, runtime int8 quant — no .focrq) | aarch64+neon+dotprod | prefill | hf | 460.000 | 2295.186 | 4.990 | compute | 108.634 | 4.23 | focr-int8 vs hf-bf16 | focr=ref=8 | system | focr: /private/tmp/cc_tgt_dev/release/focr ocr /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/pages/page_0014.png --model /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/model [threads=8 FOCR_TIMING=1]; ref: scripts/gauntlet_reference.py --stage all --page /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/pages/page_0014.png --model-dir /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/model --backend hf --precision bf16 --entry gauntlet_ref_unlimited:run_stage --setup gauntlet_ref_unlimited:setup --runs 5 --warmup 1 --threads 8 --out /Users/jemanuel/projects/franken_ocr/artifacts/perf/bd-re8.17/ref_page_0014/ref_stages.json [torch==2.10.0 transformers==4.57.1] | FOCR_DECODE_INT8=1 FOCR_MODEL_PATH=/Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/model FOCR_THREADS=8 FOCR_TIMING=1 | CER_norm=0.03529 focr-int8 text vs pinned HF bf16 reference text on page_0014.png, same prompt/runs as the timings, ref <;det;> grounding spans stripped (scripts/baseline/compare_ocr.py; artifacts/perf/bd-re8.17/cer_page_0014/cer.json) | quiet-host runbook (scripts/gauntlet_runbook.sh); best-of-5 warm, N=8 both sides; FORCE=1 loadavg-gate bypassed |
| 2026-07-02 | G2-unlimited-int8-0014-20260702 | artifacts/perf/bd-re8.17/G2-unlimited-int8-0014-20260702 | 3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5 | page=page_0014.png sha256=f1e35a58f673036b16302c86676f1ec6a89218fb4e666c477dbb4ae21b904df2; weights=model-00001-of-000001.safetensors sha256=2bc48a7a110061ea58fff65d3169367eebe3aee371ca6968dc2219c1b2855fc6 (bf16 shard, runtime int8 quant — no .focrq) | aarch64+neon+dotprod | decode-per-token | hf | 18.121 | 29.343 | 1.619 | memory | 4.953 | 3.66 | focr-int8 vs hf-bf16 | focr=ref=8 | system | focr: /private/tmp/cc_tgt_dev/release/focr ocr /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/pages/page_0014.png --model /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/model [threads=8 FOCR_TIMING=1]; ref: scripts/gauntlet_reference.py --stage all --page /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/pages/page_0014.png --model-dir /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/model --backend hf --precision bf16 --entry gauntlet_ref_unlimited:run_stage --setup gauntlet_ref_unlimited:setup --runs 5 --warmup 1 --threads 8 --out /Users/jemanuel/projects/franken_ocr/artifacts/perf/bd-re8.17/ref_page_0014/ref_stages.json [torch==2.10.0 transformers==4.57.1] | FOCR_DECODE_INT8=1 FOCR_MODEL_PATH=/Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/model FOCR_THREADS=8 FOCR_TIMING=1 | CER_norm=0.03529 focr-int8 text vs pinned HF bf16 reference text on page_0014.png, same prompt/runs as the timings, ref <;det;> grounding spans stripped (scripts/baseline/compare_ocr.py; artifacts/perf/bd-re8.17/cer_page_0014/cer.json) | quiet-host runbook (scripts/gauntlet_runbook.sh); best-of-5 warm, N=8 both sides; FORCE=1 loadavg-gate bypassed |
| 2026-07-02 | G2-unlimited-int8-0014-20260702 | artifacts/perf/bd-re8.17/G2-unlimited-int8-0014-20260702 | 3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5 | page=page_0014.png sha256=f1e35a58f673036b16302c86676f1ec6a89218fb4e666c477dbb4ae21b904df2; weights=model-00001-of-000001.safetensors sha256=2bc48a7a110061ea58fff65d3169367eebe3aee371ca6968dc2219c1b2855fc6 (bf16 shard, runtime int8 quant — no .focrq) | aarch64+neon+dotprod | end-to-end | hf | 18307.842 | 42841.792 | 2.340 | memory | 4495.040 | 4.07 | focr-int8 vs hf-bf16 | focr=ref=8 | system | focr: /private/tmp/cc_tgt_dev/release/focr ocr /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/pages/page_0014.png --model /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/model [threads=8 FOCR_TIMING=1]; ref: scripts/gauntlet_reference.py --stage all --page /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/pages/page_0014.png --model-dir /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/model --backend hf --precision bf16 --entry gauntlet_ref_unlimited:run_stage --setup gauntlet_ref_unlimited:setup --runs 5 --warmup 1 --threads 8 --out /Users/jemanuel/projects/franken_ocr/artifacts/perf/bd-re8.17/ref_page_0014/ref_stages.json [torch==2.10.0 transformers==4.57.1] | FOCR_DECODE_INT8=1 FOCR_MODEL_PATH=/Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/model FOCR_THREADS=8 FOCR_TIMING=1 | CER_norm=0.03529 focr-int8 text vs pinned HF bf16 reference text on page_0014.png, same prompt/runs as the timings, ref <;det;> grounding spans stripped (scripts/baseline/compare_ocr.py; artifacts/perf/bd-re8.17/cer_page_0014/cer.json) | quiet-host runbook (scripts/gauntlet_runbook.sh); best-of-5 warm, N=8 both sides; FORCE=1 loadavg-gate bypassed |
| 2026-07-02 | G2-unlimited-int8-0009-20260702 | artifacts/perf/bd-re8.17/G2-unlimited-int8-0009-20260702 | 3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5 | page=page_0009.png sha256=62526eb02efd588005ba70eba171d0e4bf64f4a0f0d258f6e72e8a14830b74da; weights=model-00001-of-000001.safetensors sha256=2bc48a7a110061ea58fff65d3169367eebe3aee371ca6968dc2219c1b2855fc6 (bf16 shard, runtime int8 quant — no .focrq) | aarch64+neon+dotprod | preprocess | hf | 30.000 | 27.493 | 0.916 | memory | 0.105 | 286.10 | focr-int8 vs hf-bf16 | focr=ref=8 | system | focr: /private/tmp/cc_tgt_dev/release/focr ocr /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/pages/page_0009.png --model /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/model [threads=8 FOCR_TIMING=1]; ref: scripts/gauntlet_reference.py --stage all --page /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/pages/page_0009.png --model-dir /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/model --backend hf --precision bf16 --entry gauntlet_ref_unlimited:run_stage --setup gauntlet_ref_unlimited:setup --runs 5 --warmup 1 --threads 8 --out /Users/jemanuel/projects/franken_ocr/artifacts/perf/bd-re8.17/ref_page_0009/ref_stages.json [torch==2.10.0 transformers==4.57.1] | FOCR_DECODE_INT8=1 FOCR_MODEL_PATH=/Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/model FOCR_THREADS=8 FOCR_TIMING=1 | CER_norm=0.00943 focr-int8 text vs pinned HF bf16 reference text on page_0009.png, same prompt/runs as the timings, ref <;det;> grounding spans stripped (scripts/baseline/compare_ocr.py; artifacts/perf/bd-re8.17/cer_page_0009/cer.json) | quiet-host runbook (scripts/gauntlet_runbook.sh); best-of-5 warm, N=8 both sides; FORCE=1 loadavg-gate bypassed |
| 2026-07-02 | G2-unlimited-int8-0009-20260702 | artifacts/perf/bd-re8.17/G2-unlimited-int8-0009-20260702 | 3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5 | page=page_0009.png sha256=62526eb02efd588005ba70eba171d0e4bf64f4a0f0d258f6e72e8a14830b74da; weights=model-00001-of-000001.safetensors sha256=2bc48a7a110061ea58fff65d3169367eebe3aee371ca6968dc2219c1b2855fc6 (bf16 shard, runtime int8 quant — no .focrq) | aarch64+neon+dotprod | vision-encode | hf | 7190.000 | 25721.901 | 3.577 | compute | 1934.504 | 3.72 | focr-int8 vs hf-bf16 | focr=ref=8 | system | focr: /private/tmp/cc_tgt_dev/release/focr ocr /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/pages/page_0009.png --model /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/model [threads=8 FOCR_TIMING=1]; ref: scripts/gauntlet_reference.py --stage all --page /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/pages/page_0009.png --model-dir /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/model --backend hf --precision bf16 --entry gauntlet_ref_unlimited:run_stage --setup gauntlet_ref_unlimited:setup --runs 5 --warmup 1 --threads 8 --out /Users/jemanuel/projects/franken_ocr/artifacts/perf/bd-re8.17/ref_page_0009/ref_stages.json [torch==2.10.0 transformers==4.57.1] | FOCR_DECODE_INT8=1 FOCR_MODEL_PATH=/Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/model FOCR_THREADS=8 FOCR_TIMING=1 | CER_norm=0.00943 focr-int8 text vs pinned HF bf16 reference text on page_0009.png, same prompt/runs as the timings, ref <;det;> grounding spans stripped (scripts/baseline/compare_ocr.py; artifacts/perf/bd-re8.17/cer_page_0009/cer.json) | quiet-host runbook (scripts/gauntlet_runbook.sh); best-of-5 warm, N=8 both sides; FORCE=1 loadavg-gate bypassed |
| 2026-07-02 | G2-unlimited-int8-0009-20260702 | artifacts/perf/bd-re8.17/G2-unlimited-int8-0009-20260702 | 3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5 | page=page_0009.png sha256=62526eb02efd588005ba70eba171d0e4bf64f4a0f0d258f6e72e8a14830b74da; weights=model-00001-of-000001.safetensors sha256=2bc48a7a110061ea58fff65d3169367eebe3aee371ca6968dc2219c1b2855fc6 (bf16 shard, runtime int8 quant — no .focrq) | aarch64+neon+dotprod | prefill | hf | 480.000 | 2297.617 | 4.787 | compute | 108.634 | 4.42 | focr-int8 vs hf-bf16 | focr=ref=8 | system | focr: /private/tmp/cc_tgt_dev/release/focr ocr /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/pages/page_0009.png --model /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/model [threads=8 FOCR_TIMING=1]; ref: scripts/gauntlet_reference.py --stage all --page /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/pages/page_0009.png --model-dir /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/model --backend hf --precision bf16 --entry gauntlet_ref_unlimited:run_stage --setup gauntlet_ref_unlimited:setup --runs 5 --warmup 1 --threads 8 --out /Users/jemanuel/projects/franken_ocr/artifacts/perf/bd-re8.17/ref_page_0009/ref_stages.json [torch==2.10.0 transformers==4.57.1] | FOCR_DECODE_INT8=1 FOCR_MODEL_PATH=/Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/model FOCR_THREADS=8 FOCR_TIMING=1 | CER_norm=0.00943 focr-int8 text vs pinned HF bf16 reference text on page_0009.png, same prompt/runs as the timings, ref <;det;> grounding spans stripped (scripts/baseline/compare_ocr.py; artifacts/perf/bd-re8.17/cer_page_0009/cer.json) | quiet-host runbook (scripts/gauntlet_runbook.sh); best-of-5 warm, N=8 both sides; FORCE=1 loadavg-gate bypassed |
| 2026-07-02 | G2-unlimited-int8-0009-20260702 | artifacts/perf/bd-re8.17/G2-unlimited-int8-0009-20260702 | 3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5 | page=page_0009.png sha256=62526eb02efd588005ba70eba171d0e4bf64f4a0f0d258f6e72e8a14830b74da; weights=model-00001-of-000001.safetensors sha256=2bc48a7a110061ea58fff65d3169367eebe3aee371ca6968dc2219c1b2855fc6 (bf16 shard, runtime int8 quant — no .focrq) | aarch64+neon+dotprod | decode-per-token | hf | 18.367 | 29.643 | 1.614 | memory | 4.953 | 3.71 | focr-int8 vs hf-bf16 | focr=ref=8 | system | focr: /private/tmp/cc_tgt_dev/release/focr ocr /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/pages/page_0009.png --model /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/model [threads=8 FOCR_TIMING=1]; ref: scripts/gauntlet_reference.py --stage all --page /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/pages/page_0009.png --model-dir /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/model --backend hf --precision bf16 --entry gauntlet_ref_unlimited:run_stage --setup gauntlet_ref_unlimited:setup --runs 5 --warmup 1 --threads 8 --out /Users/jemanuel/projects/franken_ocr/artifacts/perf/bd-re8.17/ref_page_0009/ref_stages.json [torch==2.10.0 transformers==4.57.1] | FOCR_DECODE_INT8=1 FOCR_MODEL_PATH=/Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/model FOCR_THREADS=8 FOCR_TIMING=1 | CER_norm=0.00943 focr-int8 text vs pinned HF bf16 reference text on page_0009.png, same prompt/runs as the timings, ref <;det;> grounding spans stripped (scripts/baseline/compare_ocr.py; artifacts/perf/bd-re8.17/cer_page_0009/cer.json) | quiet-host runbook (scripts/gauntlet_runbook.sh); best-of-5 warm, N=8 both sides; FORCE=1 loadavg-gate bypassed |
| 2026-07-02 | G2-unlimited-int8-0009-20260702 | artifacts/perf/bd-re8.17/G2-unlimited-int8-0009-20260702 | 3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5 | page=page_0009.png sha256=62526eb02efd588005ba70eba171d0e4bf64f4a0f0d258f6e72e8a14830b74da; weights=model-00001-of-000001.safetensors sha256=2bc48a7a110061ea58fff65d3169367eebe3aee371ca6968dc2219c1b2855fc6 (bf16 shard, runtime int8 quant — no .focrq) | aarch64+neon+dotprod | end-to-end | hf | 11449.067 | 31013.439 | 2.709 | compute | 2528.649 | 4.53 | focr-int8 vs hf-bf16 | focr=ref=8 | system | focr: /private/tmp/cc_tgt_dev/release/focr ocr /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/pages/page_0009.png --model /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/model [threads=8 FOCR_TIMING=1]; ref: scripts/gauntlet_reference.py --stage all --page /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/pages/page_0009.png --model-dir /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/model --backend hf --precision bf16 --entry gauntlet_ref_unlimited:run_stage --setup gauntlet_ref_unlimited:setup --runs 5 --warmup 1 --threads 8 --out /Users/jemanuel/projects/franken_ocr/artifacts/perf/bd-re8.17/ref_page_0009/ref_stages.json [torch==2.10.0 transformers==4.57.1] | FOCR_DECODE_INT8=1 FOCR_MODEL_PATH=/Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/model FOCR_THREADS=8 FOCR_TIMING=1 | CER_norm=0.00943 focr-int8 text vs pinned HF bf16 reference text on page_0009.png, same prompt/runs as the timings, ref <;det;> grounding spans stripped (scripts/baseline/compare_ocr.py; artifacts/perf/bd-re8.17/cer_page_0009/cer.json) | quiet-host runbook (scripts/gauntlet_runbook.sh); best-of-5 warm, N=8 both sides; FORCE=1 loadavg-gate bypassed |
| _—_  | _—_      | _—_         | _—_          | _—_          | _—_               | _—_   | _—_              | _—_     | _—_    | _—_   | _—_        | _—_      | _—_              | _—_                     | _—_                  | _—_       | _—_         | _—_                        | _—_               | _no measurements yet_ |

**Column legend.**
- `model_commit` — always `3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5` (HF; truth pack).
- `fixture_hash` — parity/perf fixture sha256 + `.focrq` hash for the precision in `precision` (`SOURCE_HASHES.md`).
- `arch/cpu_features` — the **dispatched** SIMD tier (e.g.
  `aarch64+neon+dotprod` for SDOT, `aarch64+neon+i8mm` for SMMLA,
  `x86_64+avx512vnni`).
- `reference_backend` — the proven CPU perf baseline actually run (`hf`, `onnx`,
  `gguf`, `mlas`, etc.); backend identity is separate from numeric precision.
- `floor_kind` — `compute` or `memory`, whichever floor binds for the stage (§9.1); `floor_ms` is that floor's time.
- `dist_above_floor` = `focr_ms / floor_ms`; **> ~1.3 ⇒ named attackable lever**, not an excuse.
- `precision` — e.g. `focr-int8 vs torch-bf16`; the accuracy delta lives in `DISCREPANCIES.md`.
- `threads (focr=ref N)` — the single N pinned on **both** sides; **torch is NEVER @64** (§9.3). A row whose reference ran oversubscribed is rejected.
- `allocator` — `system` (default, no FFI) or `mimalloc-feature`; both sides match.
- `fallback/kill-switch state` — e.g. `FOCR_INT8_ATTN=0 FOCR_INT8_LMHEAD=0 int4=off`.
- `correctness_proof` — the exact parity receipt for the row: test name and
  result, or a pointer into the committed evidence manifest that includes text
  exactness, max logit/ULP delta, CER/TEDS/Formula-CDM budget, and determinism.

**Stage vocabulary:** `preprocess` (image decode/resize/normalize) · `vision-encode`
(DeepEncoder + projector, per page) · `prefill` (build reference KV: visual + prompt) ·
`decode-per-token` (R-SWA + MoE, amortized per output token) · `end-to-end`
(`focr ocr`, image in → text out). Per G2: **decode-per-token faster** than the
Phase −1 proven CPU reference is the gate; **vision-prefill parity-or-slower in
f32 v1 is acceptable and recorded honestly**; end-to-end-faster is a tracked
stretch — record every stage, never just the favorable one.

---

_No performance numbers recorded yet. The inference path is not implemented, so
there is nothing to measure. This table stays empty until a real head-to-head
ratio exists — no fabricated or projected numbers. The first row MUST carry full
truth-pack provenance (`model_commit 3a7f4db…` + `fixture_hash` from
`SOURCE_HASHES.md`), all roofline columns, and all fairness columns; its raw
paired baseline/after gauntlet logs + SHA-256 manifest live in
`artifacts/perf/<bead>/` (the `evidence_id`). The manifest file must be one of
`SHA256SUMS`, `SHA256SUMS.txt`, `sha256sums.txt`, `sha256.txt`,
`manifest.sha256`, or `manifest.json` so `scripts/check_ledgers.py` can verify
the raw evidence is hash-anchored._
