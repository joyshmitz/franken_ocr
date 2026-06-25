# bd-3n16 SAM rel-pos bias precompute evidence

This directory records local before/after evidence for the first safe sub-lever
inside `src/native_engine/vision_sam.rs::attention`: precomputing the decomposed
relative-position H/W bias once per query instead of recomputing those dot
products inside every key loop.

This is **not** a `docs/PERF_LEDGER.md` row: it does not compare against the
pinned Phase -1 CPU reference and it uses deterministic synthetic inputs instead
of model fixtures. It is local negative-evidence discipline for keeping or
reverting a single structural kernel lever before the riskier QK^T/probs@V GEMM
rewrite.

## Harness

- Probe: ignored unit test
  `native_engine::vision_sam::tests::sam_attention_relpos_bias_local_probe`
- Real path under test: `vision_sam::attention`
- Shape: SAM attention window, `14 x 14 = 196` tokens, width `768`,
  `12` heads, `head_dim=64`
- Rel-pos tables: deterministic non-zero synthetic H/W tables
- QKV/proj: identity weights, so the probe exercises the real attention loop
  while keeping output checksums stable
- Runs per benchmark command: `5`
- Profile: `release-perf`
- Thread env: default local Rust/frankentorch settings
- CPU feature string: `arm64 Apple M4, dotprod=1, i8mm=1`
- Fallback / kill-switch state: no `FOCR_*` performance kill-switches set

## Command

```text
FOCR_SAM_ATTN_PROBE_RUNS=5 CARGO_TARGET_DIR=target-codex-verify cargo +nightly --config patch.crates-io.asupersync.path='"/dp/asupersync"' test --profile release-perf sam_attention_relpos_bias_local_probe --lib -- --ignored --nocapture
```

## Results

Baseline, old nested rel-pos dot inside every key loop:

- `avg_ms`: `26.8490916`
- `total_ms`: `134.245458`
- `warm_checksum`: `-0.0019174511544406414`
- `checksum`: `-0.009587256237864494`

After, per-query decomposed rel-pos bias precompute:

- `avg_ms`: `17.787375`
- `total_ms`: `88.936875`
- `warm_checksum`: `-0.0019174511544406414`
- `checksum`: `-0.009587256237864494`

Local speedup for this probe: `26.8490916 / 17.787375 = 1.509x`
(`33.75%` less wall time for the same 5 attention calls).

## Rejected Follow-Up: QKV Copy + Grid-Coord Hoist

A narrower layout-only follow-up was tested after `afcd9b7`:

- replace the nested scalar Q/K/V split loop with per-head contiguous slice
  copies;
- precompute key `ky/kx` grid coordinates instead of computing `j / gw` and
  `j % gw` inside the logits loop.

It preserved the checksum but regressed the 7-run local probe:

- baseline: `17.53473214285714 ms`
- attempted: `18.066375 ms`
- delta: `+3.03%` wall time, so the code was reverted and not committed.

Retry condition: only revisit this layout lever as part of a larger batched
QK^T/probs@V rewrite where profiling shows the QKV split or coordinate math has
become a measurable hotspot.

## Kept Follow-Up: Per-Head GEMM QK^T + Probs@V

The larger structural rewrite requested by `bd-3n16` was then tested:

- replace the per-query scalar QK dot loop with `nn::matmul(Q, K^T)` per head;
- transpose each head's K with contiguous stores before the GEMM;
- replace the scalar `probs @ V` weighted-sum loop with a second `nn::matmul`.

It kept the focused SAM test suite green and improved the 7-run local probe:

- fresh baseline: `18.075154714285716 ms`
- attempted/kept: `12.591648857142857 ms`
- delta: `1.435488x` speedup, `30.3373%` less wall time

Correctness note: the GEMM path is not bit-exact with the scalar loop because the
kernel changes f32 accumulation order. The probe checksum drift was tiny
(`-0.013422159478068352` -> `-0.013422181829810143` over 7 runs), and the
committed focused test `attention_gemm_matches_scalar_reference_with_relpos`
compares GEMM attention against the old scalar loop with non-zero decomposed
rel-pos tables and requires `max_abs <= 2e-6`.

## Files

- `baseline_sam_attention_probe.txt`: old nested rel-pos loop.
- `after_sam_attention_probe.txt`: rel-pos bias precompute.
- `qkv_grid_hoist_baseline_7run.txt`: baseline for the rejected layout-only
  follow-up.
- `qkv_grid_hoist_attempt_7run.txt`: attempted layout-only follow-up result.
- `gemm_baseline_7run.txt`: fresh baseline before the kept per-head GEMM rewrite.
- `gemm_attempt_7run.txt`: kept per-head GEMM QK^T/probs@V rewrite result.
- `SHA256SUMS`: hash manifest for this evidence bundle.
