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

## Files

- `baseline_sam_attention_probe.txt`: old nested rel-pos loop.
- `after_sam_attention_probe.txt`: rel-pos bias precompute.
- `SHA256SUMS`: hash manifest for this evidence bundle.
