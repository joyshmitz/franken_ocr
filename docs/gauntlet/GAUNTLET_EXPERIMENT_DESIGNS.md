# Gauntlet Experiment Designs

## Experiment `EXP-1401` - Stream source BF16 through decode GEMVs

| field | value |
|---|---|
| `experiment_id` | `EXP-1401` |
| `pillar` | `perf` |
| `created_at_utc` | `2026-07-10T02:08:00Z` |
| `created_by_agent` | `BrownFox` |
| `bead_id` | `bd-2mo.30` |
| `parent_hypothesis_id` | `PASS-01-BF16-STREAM` |
| `status` | `CLOSED` |

### Hypothesis

Reading BF16 decoder weights directly inside GEMV would reduce widened-cache
traffic without changing accumulation order or decoded output.

### Motivation

Decoder expert and head weight traffic dominated the first measured profile, so
removing an expanded f32 cache appeared to offer an accretive bandwidth win.

### Minimal Reproducer

Run the interleaved six-sample page workload with the streaming mode disabled and
enabled, holding model, page, worker count, precision gates, and decode cap fixed.

### Expected Signal

Byte-identical Markdown and at least a two-percent median decode improvement with
no material regression in the measured `lm_head` phase.

### Falsifiability Criteria

Any output mismatch, or a stable decode or `lm_head` regression beyond noise,
falsifies the proposed source-BF16 consumption path.

### One-Line Invocation

`FOCR_BF16_STREAM=1 FOCR_THREADS=8 FOCR_TIMING=1 FOCR_PROFILE_DECODE=1 focr ocr page_0014.png --model unlimited-ocr.recipe-v1.focrq`

### Results Inline

```yaml
result_status: NO_EVIDENCE
result_summary: outputs remained byte-identical but median decode regressed 4.59 percent and lm_head regressed 7.44 percent, so the lever was reverted
result_evidence_paths:
  - artifacts/perf/bd-2mo.30/profile-recipe-5733407/ab-bf16-stream
closed_at_utc: 2026-07-10T03:20:00Z
retry_condition_predicate: retry only if a profiler attributes a clearly-above-noise share to BF16 widening after a native packed-BF16 kernel exists
```

### Closure Predicate

Close after output identity, interleaved measurements, a hash manifest, and the
negative-evidence retry predicate are all recorded.
