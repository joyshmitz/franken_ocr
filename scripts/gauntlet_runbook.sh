#!/usr/bin/env bash
# gauntlet_runbook.sh — the EXACT serial command sequence for the bd-re8.17
# quiet-host head-to-head measurement (focr vs the pinned CPU HF baseline).
#
# EVERYTHING is embedded: truth-pack provenance (model_commit, fixture hashes),
# fairness pins (N=8 threads on BOTH sides), page/model/venv paths, claim ids.
# On the quiet host you run ONE command and read the printed ledger row:
#
#   bash scripts/gauntlet_runbook.sh all            # preflight -> ... -> row
#
# or step-by-step (each step is idempotent and strictly serial):
#
#   bash scripts/gauntlet_runbook.sh preflight      # untimed gates + self-tests
#   bash scripts/gauntlet_runbook.sh focr           # TIMED: focr best-of-5 warm @8
#   bash scripts/gauntlet_runbook.sh reference      # TIMED: HF baseline best-of-5 warm @8
#   bash scripts/gauntlet_runbook.sh roofline       # untimed: §9.1 floors
#   bash scripts/gauntlet_runbook.sh cer            # untimed: correctness proof (CER)
#   bash scripts/gauntlet_runbook.sh row            # untimed: PERF_LEDGER row draft + validation
#
# QUIET-HOST DOCTRINE: the two TIMED steps poison their cv% on a contended
# host. preflight refuses when 1-min loadavg >= LOAD_MAX (default 2.0);
# FORCE=1 overrides (recorded). Close editors/indexers/other agents first.
#
# Knobs (all optional):
#   APPLY=1          insert the validated row(s) into docs/PERF_LEDGER.md
#   VERIFY_SHARD=1   re-hash the 6.7GB weights shard in preflight (~40s)
#   FORCE=1          bypass the loadavg gate (the row notes must say why)
#   PAGES="page_0009.png page_0014.png"   override the measured page set
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── EMBEDDED PROVENANCE (docs/truth-pack/PINNED_SOURCES.md + SOURCE_HASHES.md
#    + the verified baseline workspace) — resolve nothing at measurement time ──
MODEL_COMMIT="3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5"    # HF baidu/Unlimited-OCR pin
PINNED_TORCH="2.10.0"
PINNED_TRANSFORMERS="4.57.1"
# model-00001-of-000001.safetensors (6672547120 bytes) — HF LFS etag, VERIFIED 2026-06-26
SHARD_SHA256="2bc48a7a110061ea58fff65d3169367eebe3aee371ca6968dc2219c1b2855fc6"
# Perf fixture pages (200 DPI renders, seed 20260626; hashed 2026-07-02).
# (A function, not `declare -A`: /bin/bash on macOS is 3.2.)
page_sha256() {
  case "$1" in
    page_0009.png) echo "62526eb02efd588005ba70eba171d0e4bf64f4a0f0d258f6e72e8a14830b74da" ;;
    page_0014.png) echo "f1e35a58f673036b16302c86676f1ec6a89218fb4e666c477dbb4ae21b904df2" ;;
    *) echo "" ;;
  esac
}

# ── paths (the proven baseline workspace; override via env if it moved) ─────
WORK="${FOCR_GAUNTLET_WORK:-/Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work}"
MODEL_DIR="${FOCR_GAUNTLET_MODEL_DIR:-$WORK/model}"
PAGES_DIR="${FOCR_GAUNTLET_PAGES_DIR:-$WORK/pages}"
VENV_PY="${FOCR_GAUNTLET_VENV_PY:-/Volumes/focrvenv/venv/bin/python}"
FOCR_BIN="${FOCR_BIN:-}"
if [[ -z "$FOCR_BIN" ]]; then
  for c in "$REPO_ROOT/target/release-perf/focr" "$REPO_ROOT/target/release/focr" \
           /private/tmp/cc_tgt_dev/release/focr; do
    [[ -x "$c" ]] && FOCR_BIN="$c" && break
  done
fi

# ── measurement contract (docs/PERF_LEDGER.md §9.3) ─────────────────────────
THREADS=8                      # ONE budget, BOTH sides; NEVER 64
RUNS=5                         # best-of-5 ...
WARMUP=1                       # ... after 1 discarded warm run
LOAD_MAX="${LOAD_MAX:-2.0}"
PAGES="${PAGES:-page_0009.png page_0014.png}"
STAMP="$(date -u +%Y%m%d)"
OUT="$REPO_ROOT/artifacts/perf/bd-re8.17"
ARCH_JSON="$OUT/arch.json"

# Reference-side pool pins — MUST be in the environment BEFORE python starts
# (gauntlet_reference.py refuses otherwise; torch/OMP read them at import).
ref_env() {
  env FOCR_THREADS="$THREADS" \
      OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
      OPENBLAS_NUM_THREADS="$THREADS" VECLIB_MAXIMUM_THREADS="$THREADS" \
      NUMEXPR_NUM_THREADS="$THREADS" \
      HF_HOME=/Volumes/focrvenv/hf_home TMPDIR=/Volumes/focrvenv/tmp \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      "$@"
}

die() { echo "RUNBOOK-FATAL: $*" >&2; exit 1; }
note() { echo "runbook: $*" >&2; }

stem() { local p="$1"; p="${p%.png}"; echo "$p"; }

# ── preflight — untimed gates; nothing here touches a stopwatch ─────────────
step_preflight() {
  note "preflight: quiet-host + fixtures + pins + harness self-tests"
  local load
  load="$(sysctl -n vm.loadavg | awk '{print $2}')"
  if awk -v l="$load" -v m="$LOAD_MAX" 'BEGIN{exit !(l>=m)}'; then
    [[ "${FORCE:-0}" == "1" ]] \
      || die "1-min loadavg $load >= $LOAD_MAX — host is NOT quiet (FORCE=1 to override; a contended host poisons cv%)"
    note "WARNING: loadavg $load >= $LOAD_MAX but FORCE=1 — timings from this host are suspect"
  else
    note "loadavg $load OK (< $LOAD_MAX)"
  fi

  [[ -x "$FOCR_BIN" ]] || die "no focr binary (FOCR_BIN=$FOCR_BIN)"
  [[ -d "$MODEL_DIR" ]] || die "model dir missing: $MODEL_DIR"
  [[ -x "$VENV_PY" ]] || die "reference venv python missing: $VENV_PY"

  # Fixture hashes must match the embedded truth-pack values (moved page = STOP).
  local page have want
  for page in $PAGES; do
    [[ -f "$PAGES_DIR/$page" ]] || die "page fixture missing: $PAGES_DIR/$page"
    want="$(page_sha256 "$page")"
    [[ -n "$want" ]] || die "no embedded sha256 for $page — add it to page_sha256() first"
    have="$(shasum -a 256 "$PAGES_DIR/$page" | awk '{print $1}')"
    [[ "$have" == "$want" ]] || die "$page sha256 drifted: $have != $want"
    note "$page sha256 verified"
  done
  if [[ "${VERIFY_SHARD:-0}" == "1" ]]; then
    have="$(shasum -a 256 "$MODEL_DIR/model-00001-of-000001.safetensors" | awk '{print $1}')"
    [[ "$have" == "$SHARD_SHA256" ]] || die "weights shard sha256 drifted: $have"
    note "weights shard sha256 verified"
  fi

  # Truth-pack runtime pins (the reference harness re-verifies fail-closed).
  "$VENV_PY" - <<PY
import sys, torch, transformers
ok = (torch.__version__.split("+")[0] == "$PINNED_TORCH"
      and transformers.__version__ == "$PINNED_TRANSFORMERS")
print(f"runbook: venv torch={torch.__version__} transformers={transformers.__version__}",
      file=sys.stderr)
sys.exit(0 if ok else 1)
PY

  # Harness self-tests (all untimed, no model needed).
  python3 scripts/gauntlet_timing.py --self-test >/dev/null
  python3 scripts/gauntlet_reference.py --self-test >/dev/null
  python3 scripts/gauntlet_ref_unlimited.py --self-test >/dev/null
  python3 scripts/gauntlet_roofline.py --self-test >/dev/null
  python3 scripts/gauntlet_row.py --self-test >/dev/null
  bash scripts/gauntlet_focr.sh --self-test >/dev/null
  note "harness self-tests: all pass"

  # Dispatched SIMD tier -> the ledger arch/cpu_features cell (recorded once).
  mkdir -p "$OUT"
  "$FOCR_BIN" robot selftest >"$ARCH_JSON"
  python3 - "$ARCH_JSON" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1]))
assert doc["all_ok"] is True, "focr robot selftest FAILED — kernels unproven"
print(f"runbook: arch/cpu_features = {doc['selected_feature']} (selftest 24/24)",
      file=sys.stderr)
PY
}

arch_features() {
  [[ -f "$ARCH_JSON" ]] || die "run preflight first ($ARCH_JSON missing)"
  python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['selected_feature'])" "$ARCH_JSON"
}

# ── focr side — TIMED (best-of-5 warm, N=8, int8 decode) ────────────────────
step_focr() {
  local page s
  for page in $PAGES; do
    s="$(stem "$page")"
    note "focr side: $page (runs=$RUNS warmup=$WARMUP threads=$THREADS int8)"
    # FOCR_DECODE_INT8=1 = the shipped int8 decode lever (timing lines prove it
    # via the _i8 suffix -> precision focr-int8). FOCR_MODEL_PATH: the raw
    # safetensors dir loads directly (no .focrq needed on this workspace).
    FOCR_DECODE_INT8=1 FOCR_MODEL_PATH="$MODEL_DIR" \
      bash scripts/gauntlet_focr.sh \
        --binary "$FOCR_BIN" \
        --page "$PAGES_DIR/$page" \
        --model "$MODEL_DIR" \
        --runs "$RUNS" --warmup "$WARMUP" --threads "$THREADS" \
        --out-dir "$OUT/focr_$s"
  done
}

# ── reference side — TIMED (same N, pinned stack, instrumented stages) ──────
step_reference() {
  local page s
  for page in $PAGES; do
    s="$(stem "$page")"
    note "reference side: $page (runs=$RUNS warmup=$WARMUP threads=$THREADS bf16)"
    mkdir -p "$OUT/ref_$s"
    ref_env FOCR_REF_TEXT_DIR="$OUT/ref_$s/text" \
      "$VENV_PY" scripts/gauntlet_reference.py \
        --stage all \
        --page "$PAGES_DIR/$page" \
        --model-dir "$MODEL_DIR" \
        --backend hf --precision bf16 \
        --entry gauntlet_ref_unlimited:run_stage \
        --setup gauntlet_ref_unlimited:setup \
        --runs "$RUNS" --warmup "$WARMUP" --threads "$THREADS" \
        --out "$OUT/ref_$s/ref_stages.json"
  done
}

# ── roofline — untimed derivation from the focr measurement ─────────────────
step_roofline() {
  local page s
  for page in $PAGES; do
    s="$(stem "$page")"
    python3 scripts/gauntlet_roofline.py \
      --arch unlimited-ocr --precision int8 --profile m4 \
      --stages-json "$OUT/focr_$s/focr_stages.json" \
      --out "$OUT/roofline_$s.json"
  done
}

# ── correctness proof — CER of focr text vs the reference text from the SAME
#    timed runs. The HF reference emits <|det|>…<|/det|> grounding spans that
#    focr's plain-text mode never produces (smoke-proven 2026-07-02; the raw
#    diff is ~0.61 CER purely from the markers) — strip them from the ref side
#    before scoring, and say so in the proof string. ──────────────────────────
step_cer() {
  local page s
  for page in $PAGES; do
    s="$(stem "$page")"
    mkdir -p "$OUT/cer_$s/hyp" "$OUT/cer_$s/ref"
    [[ -f "$OUT/focr_$s/raw/run_001.stdout" ]] || die "focr raw stdout missing for $page — run 'focr' first"
    [[ -f "$OUT/ref_$s/text/$s.md" ]] || die "reference text missing for $page — run 'reference' first"
    cp "$OUT/focr_$s/raw/run_001.stdout" "$OUT/cer_$s/hyp/$s.md"
    python3 - "$OUT/ref_$s/text/$s.md" "$OUT/cer_$s/ref/$s.md" <<'PY'
import re, sys
text = open(sys.argv[1], encoding="utf-8").read()
open(sys.argv[2], "w", encoding="utf-8").write(re.sub(r"<\|det\|>.*?<\|/det\|>", "", text))
PY
    python3 scripts/baseline/compare_ocr.py \
      --ref "$OUT/cer_$s/ref" --hyp "$OUT/cer_$s/hyp" \
      --json "$OUT/cer_$s/cer.json"
  done
}

# ── row — merge, bundle, validate (shadow check_ledgers), optionally apply ──
step_row() {
  local page s claim fixture proof cer arch apply=()
  arch="$(arch_features)"
  [[ "${APPLY:-0}" == "1" ]] && apply=(--apply)
  for page in $PAGES; do
    s="$(stem "$page")"
    claim="G2-unlimited-int8-${s#page_}-$STAMP"
    fixture="page=$page sha256=${PAGE_SHA256[$page]}; weights=model-00001-of-000001.safetensors sha256=$SHARD_SHA256 (bf16 shard, runtime int8 quant — no .focrq)"
    cer="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['aggregate']['cer_norm'])" "$OUT/cer_$s/cer.json")"
    proof="CER_norm=$cer focr-int8 text vs pinned HF bf16 reference text on $page, same prompt/runs as the timings, ref <;det;> grounding spans stripped (scripts/baseline/compare_ocr.py; artifacts/perf/bd-re8.17/cer_$s/cer.json)"
    note "row: $page claim_id=$claim"
    python3 scripts/gauntlet_row.py \
      --focr-stages "$OUT/focr_$s/focr_stages.json" \
      --ref-stages "$OUT/ref_$s/ref_stages.json" \
      --roofline "$OUT/roofline_$s.json" \
      --claim-id "$claim" \
      --model-commit "$MODEL_COMMIT" \
      --fixture-hash "$fixture" \
      --arch-features "$arch" \
      --correctness-proof "$proof" \
      --notes "quiet-host runbook (scripts/gauntlet_runbook.sh); best-of-$RUNS warm, N=$THREADS both sides${FORCE:+; FORCE=1 loadavg-gate bypassed}" \
      ${apply[@]+"${apply[@]}"}
  done
  python3 scripts/check_ledgers.py >/dev/null && note "check_ledgers: pass"
}

case "${1:-}" in
  preflight) step_preflight ;;
  focr) step_focr ;;
  reference) step_reference ;;
  roofline) step_roofline ;;
  cer) step_cer ;;
  row) step_row ;;
  all)
    step_preflight
    step_focr
    step_reference
    step_roofline
    step_cer
    step_row
    ;;
  *) sed -n '2,28p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
