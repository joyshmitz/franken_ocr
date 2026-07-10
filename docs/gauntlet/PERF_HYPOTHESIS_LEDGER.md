# Performance Hypothesis Ledger

## Experiment `EXP-1402` - Whole-program PGO over the representative corpus

| field | value |
|---|---|
| `experiment_id` | `EXP-1402` |
| `pillar` | `perf` |
| `created_at_utc` | `2026-07-10T07:37:36Z` |
| `created_by_agent` | `BrownFox` |
| `bead_id` | `bd-2mo.23` |
| `parent_hypothesis_id` | `PASS-12-PGO` |
| `status` | `CLOSED` |

### Hypothesis

Whole-program profile guidance would improve both dense decode and the capped
ten-page wall time without moving any numerical or dispatch contract.

### Motivation

The decode loop contains branch-heavy routing and ring-buffer control whose
layout can improve without runtime complexity or model-specific numerics.

### Minimal Reproducer

Compare ordinary `release-perf` and PGO-use binaries in a reversed/interleaved
page workload and a separately measured capped ten-page workload.

### Expected Signal

At least a two-percent win in both decode and capped ten-page wall time, stable
CV, byte-identical page output, and semantic identity after stripping timings.

### Falsifiability Criteria

Failure of either precommitted workload gate, any output drift, or a greater than
two-percent regression in a major phase rejects the whole-program profile.

### One-Line Invocation

`RUSTFLAGS='-Cprofile-use=focr.profdata -Cllvm-args=-pgo-warn-missing-function' cargo build --profile release-perf --bin focr`

### Results Inline

```yaml
result_status: NO_EVIDENCE
result_summary: decode improved 14.52 percent but capped ten-page wall improved only 1.14 percent while prefill regressed about 56 percent, so the PGO build was rejected
result_evidence_paths:
  - artifacts/perf/bd-2mo.30/profile-recipe-5733407/pass12-pgo-20260710T073736Z
closed_at_utc: 2026-07-10T08:10:00Z
retry_condition_predicate: reconsider only inside the broader phase-selective or prefill-weighted PGO experiment with the same two-workload gate
```

### Closure Predicate

Close only when both workloads, output identity, variance, provenance, and the
explicit retry condition are present in a hash-verified evidence directory.
