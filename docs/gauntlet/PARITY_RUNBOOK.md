# PARITY RUNBOOK — reproducing the franken_ocr parity verification (bd-wp8.9)

How a future agent/maintainer re-verifies every parity claim in the
certification bundle, from a fresh checkout. Every step is a committed script
with its own receipt; nothing here depends on session state.

## 0. Prerequisites

* The real model artifacts (see `docs/FEATURE_PARITY.md` §weights + the
  baseline workspace notes): `unlimited-ocr.int8.focrq` + `tokenizer.json`
  in the model cache (`focr pull` or `FOCR_MODEL_PATH=<dir>`); the raw
  safetensors dir for f32 legs.
* The torch oracle env for reference regeneration (pinned
  `torch==2.10.0 transformers==4.57.1`, CPU) — only needed to REGENERATE
  frozen fixtures; verification against committed fixtures needs no Python.

## 1. The L0–L5 ladder receipt (the core parity proof)

```
bash scripts/ladder_scorecard.sh          # writes tests/fixtures/ladder_scorecard/scorecard_armed.json
```

All-green armed receipt = L0 preprocess / L1 vision / L2 hidden / L3 logits /
L4 token-exact / L5 CER within the documented budgets, against the frozen
torch-oracle fixtures. `scorecard_armed.json` carries the per-rung numbers the
readiness gate reads.

## 2. Multi-page cross-page parity (infer_multi)

```
FOCR_MODEL_PATH=<model dir> FOCR_CORPUS_DIR=<pages dir> \
  cargo test --release --test parity_ladder l5_multi -- --nocapture
```

Two armed rungs: the 2-page leg (CER ≤ 0.25, plate byte-exact) and the
10-page long-horizon leg (CER ≤ 0.50, subject capped at 7600 tokens — a true
prefix). Oracle fixtures: `tests/fixtures/multi_page/` (regenerate via
`scripts/baseline/run_baidu_reference_multi.py`). Accepted divergence class:
DISC-004.

## 3. Kernel parity on the host silicon

```
focr robot selftest                       # 44/44 per-model int8-kernel parity vs the scalar oracle
```

Machine-readable per-model verdicts (`models[]`), including each registered
decoder's worst-case-K overflow row. `FOCR_FORCE_ARCH=<tier>` sweeps tiers.

## 4. Property + fuzz hardening

```
cargo test --test property_suite          # PROPTEST_CASES=2048 for the deep lane
PATH=$CARGO_HOME/bin:$PATH cargo fuzz run <target> -- -max_total_time=300
```

Targets: `focrq_parse`, `safetensors_parse`, `image_decode`, `pretok_split`
(committed seed corpus under `fuzz/corpus/`).

## 5. The perf pillar (quiet host REQUIRED)

```
bash scripts/gauntlet_runbook.sh all      # preflight refuses loadavg >= 2.0
```

Serial: preflight → focr timing → torch reference (pinned env) → roofline →
CER → the PERF_LEDGER row draft. Fairness discipline per
`docs/gauntlet/METHODOLOGY.md` §9.3 (thread pins, warmup discard, cv% bound,
within-regime pairs only).

## 6. The full gate + the bundle

```
bash scripts/check.sh                     # every validator + fmt/check/clippy/test/ubs
python3 scripts/gauntlet_cert.py --release-readiness   # the ship-gate cells
python3 scripts/gauntlet_cert.py --bundle              # assemble + certify (exit 1 until certified)
```

The bundle refuses certification unless readiness is all-green, convergence
(≥10 rounds, last 2 clean) is met, and every core evidence artifact is
fresher than 24 hours — regenerate steps 1–5 first.
