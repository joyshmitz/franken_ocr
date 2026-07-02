#!/usr/bin/env bash
# gauntlet_focr.sh — the focr side of the head-to-head gauntlet (bd-re8.17).
#
# Runs the release `focr` binary N times WARM on one page with FOCR_TIMING=1,
# captures each run's stderr/stdout + a high-resolution wall clock, then folds
# the `[focr-timing]` lines into per-stage best-of-N + cv% JSON records via
# scripts/gauntlet_timing.py. The output (`focr_stages.json`) is one of the
# three inputs `scripts/gauntlet_row.py` merges into a PERF_LEDGER row.
#
# Honesty contract (docs/PERF_LEDGER.md §fairness):
#   - every number is a real wall-clock sample of a real run; a missing model /
#     page / binary SKIPS with exit 0 (never fabricates, never reds CI);
#   - the thread budget is pinned into the environment and recorded;
#   - warmup runs are discarded; all measured samples are kept in the record;
#   - all raw stderr/stdout/meta land beside the JSON for the evidence bundle.
#
# Usage:
#   scripts/gauntlet_focr.sh --page PAGE [--model MODEL] [--binary BIN]
#                            [--runs N] [--warmup W] [--threads T]
#                            [--allocator LABEL] [--precision P] [--out-dir DIR]
#   scripts/gauntlet_focr.sh --self-test   # stub-binary dry run, no model needed
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMING_PY="$REPO_ROOT/scripts/gauntlet_timing.py"

PAGE=""
MODEL=""
BINARY=""
RUNS=5
WARMUP=1
THREADS="${FOCR_THREADS:-8}"
ALLOCATOR="system"
PRECISION=""
OUT_DIR=""
SELF_TEST=0
SYNTHETIC=0

usage() { sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --page) PAGE="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --binary) BINARY="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --allocator) ALLOCATOR="$2"; shift 2 ;;
    --precision) PRECISION="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --self-test) SELF_TEST=1; shift ;;
    --synthetic) SYNTHETIC=1; shift ;;   # stamp records synthetic (stub runs)
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

skip() { # reason [detail] — graceful fixture-absent skip: exit 0, one JSON line
  printf '{"event":"skip","harness":"gauntlet_focr","reason":"%s","detail":"%s"}\n' "$1" "${2:-}"
  exit 0
}

if ! [[ "$RUNS" =~ ^[1-9][0-9]*$ ]] || ! [[ "$WARMUP" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --runs must be a positive integer, --warmup a non-negative integer" >&2
  exit 2
fi
if ! [[ "$THREADS" =~ ^[1-9][0-9]*$ ]] || (( THREADS > 32 )); then
  # NEVER 64 — oversubscription fakes wins (docs/PERF_LEDGER.md §fairness).
  echo "ERROR: --threads must be a positive integer <= 32 (got '$THREADS')" >&2
  exit 2
fi

# ── --self-test: full dry run against a stub binary that emits synthetic
#    FOCR_TIMING lines; asserts the pipeline produces the expected stages. ────
if (( SELF_TEST )); then
  TMP="$(mktemp -d "${TMPDIR:-/tmp}/focr-gauntlet-selftest.XXXXXX")"
  trap 'rm -rf "$TMP"' EXIT
  STUB="$TMP/focr-stub"
  cat >"$STUB" <<'STUB_EOF'
#!/usr/bin/env bash
# Stub focr: proves the harness plumbing without weights. Refuses to pretend
# to be real: it only runs when FOCR_TIMING is set by the harness.
[[ -n "${FOCR_TIMING:-}" ]] || { echo "stub requires FOCR_TIMING" >&2; exit 9; }
cat >&2 <<'ERR'
[focr-timing] preprocess 0.10s
[focr-timing]   vision.sam 0.80s
[focr-timing]   vision.clip 0.30s
[focr-timing]   vision.bridge 0.04s
[focr-timing] vision_tower 1.14s
[focr-timing] weight_cache_build_i8 0.90s
[focr-timing] prefill_i8 1.80s (289 tokens)
[focr-timing] decode_i8 6.00s (600 tokens, 0.010s/tok)
ERR
echo "stub page text"
STUB_EOF
  chmod +x "$STUB"
  printf 'not-really-a-png' >"$TMP/page.png"
  "$0" --binary "$STUB" --page "$TMP/page.png" --runs 3 --warmup 1 \
       --threads "$THREADS" --out-dir "$TMP/out" --synthetic
  python3 - "$TMP/out/focr_stages.json" <<'PY_EOF'
import json, sys
doc = json.load(open(sys.argv[1]))
stages = {r["stage"]: r for r in doc["stages"]}
assert doc["synthetic"] is True, "self-test output must be stamped synthetic"
assert doc["stdout_identical_across_runs"] is True
for want in ("preprocess", "vision_encode", "prefill", "decode_per_token", "end_to_end"):
    assert want in stages, f"missing stage {want}"
    assert stages[want]["n"] == 3, f"{want}: expected 3 samples"
assert stages["decode_per_token"]["tokens"] == 600
assert abs(stages["decode_per_token"]["best_ms"] - 10.0) < 1e-6
assert doc["precision"] == "focr-int8"
print(json.dumps({"check": "gauntlet-focr-self-test", "result": "pass"}))
PY_EOF
  exit 0
fi

# ── resolve inputs; absence of any fixture is a graceful skip ────────────────
[[ -n "$PAGE" ]] || { echo "ERROR: --page is required (or --self-test)" >&2; exit 2; }
if [[ -z "$BINARY" ]]; then
  for candidate in "$REPO_ROOT/target/release-perf/focr" "$REPO_ROOT/target/release/focr"; do
    [[ -x "$candidate" ]] && BINARY="$candidate" && break
  done
  [[ -n "$BINARY" ]] || skip "no_release_binary" "build focr with the release-perf profile first"
fi
[[ -x "$BINARY" ]] || skip "binary_not_executable" "$BINARY"
[[ -f "$PAGE" ]] || skip "page_fixture_absent" "$PAGE"
if [[ -n "$MODEL" && ! -e "$MODEL" ]]; then
  skip "model_fixture_absent" "$MODEL"
fi

OUT_DIR="${OUT_DIR:-$REPO_ROOT/artifacts/perf/bd-re8.17/focr}"
RAW_DIR="$OUT_DIR/raw"
mkdir -p "$RAW_DIR"
# Refuse a dirty raw dir (fresh-eyes fix): the aggregator globs EVERY
# run_*.meta.json under it, so re-running into the same out dir would fold a
# previous session's samples into this one's best-of-N — silently mixed
# measurements. Fail closed; the operator picks a fresh dir (nothing deleted).
if compgen -G "$RAW_DIR/run_*.meta.json" > /dev/null 2>&1; then
  echo "FATAL: $RAW_DIR already holds run_*.meta.json from a previous session;" >&2
  echo "       re-aggregating would mix runs. Use a fresh out dir." >&2
  exit 2
fi

CMD=("$BINARY" "ocr" "$PAGE")
[[ -n "$MODEL" ]] && CMD+=("--model" "$MODEL")

# Thread parity pins: focr's own budget plus the pool knobs a helper could
# read. Recorded verbatim into every run's meta (the ledger `command/env`).
export FOCR_TIMING=1
export FOCR_THREADS="$THREADS"
export OMP_NUM_THREADS="$THREADS"
export RAYON_NUM_THREADS="$THREADS"

TOTAL=$(( WARMUP + RUNS ))
echo "gauntlet_focr: $TOTAL runs ($WARMUP warmup + $RUNS measured) of: ${CMD[*]}" >&2

for i in $(seq 1 "$TOTAL"); do
  if (( i <= WARMUP )); then
    TAG="warmup_$i"
  else
    TAG="run_$(printf '%03d' $(( i - WARMUP )))"
  fi
  # The python wrapper provides the high-resolution wall clock (BSD `date` has
  # no %N) and writes the run meta the aggregator consumes.
  FOCR_GAUNTLET_TAG="$TAG" \
  FOCR_GAUNTLET_RAW_DIR="$RAW_DIR" \
  FOCR_GAUNTLET_WARMUP="$WARMUP" \
  python3 - "${CMD[@]}" <<'PY_EOF'
import hashlib, json, os, subprocess, sys, time

cmd = sys.argv[1:]
tag = os.environ["FOCR_GAUNTLET_TAG"]
raw = os.environ["FOCR_GAUNTLET_RAW_DIR"]
stdout_path = os.path.join(raw, f"{tag}.stdout")
stderr_path = os.path.join(raw, f"{tag}.stderr")

t0 = time.perf_counter()
with open(stdout_path, "wb") as out, open(stderr_path, "wb") as err:
    rc = subprocess.run(cmd, stdout=out, stderr=err).returncode
wall_ms = (time.perf_counter() - t0) * 1000.0

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

pins = ("FOCR_TIMING", "FOCR_THREADS", "OMP_NUM_THREADS", "RAYON_NUM_THREADS")
meta = {
    "tag": tag,
    "command": cmd,
    "exit_code": rc,
    "wall_ms": round(wall_ms, 3),
    "stdout": os.path.basename(stdout_path),
    "stderr": os.path.basename(stderr_path),
    "binary": cmd[0],
    "binary_sha256": sha256(cmd[0]),
    "page": cmd[2],
    "page_sha256": sha256(cmd[2]),
    "model": cmd[4] if len(cmd) > 4 else None,
    "threads": int(os.environ["FOCR_THREADS"]),
    "warmup": int(os.environ["FOCR_GAUNTLET_WARMUP"]),
    "env_pins": {k: os.environ[k] for k in pins},
    # The full FOCR_* surface = the row's fallback/kill-switch state evidence.
    "focr_env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("FOCR_")
                 and not k.startswith("FOCR_GAUNTLET_")},
}
with open(os.path.join(raw, f"{tag}.meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
    f.write("\n")
if rc != 0:
    sys.stderr.write(open(stderr_path, errors="replace").read())
    sys.exit(rc)
print(f"  {tag}: {wall_ms:.0f} ms", file=sys.stderr)
PY_EOF
done

AGG_ARGS=(aggregate --run-dir "$RAW_DIR" --out "$OUT_DIR/focr_stages.json"
          --threads "$THREADS" --allocator "$ALLOCATOR")
[[ -n "$PRECISION" ]] && AGG_ARGS+=(--precision "$PRECISION")
(( SYNTHETIC )) && AGG_ARGS+=(--synthetic)
python3 "$TIMING_PY" "${AGG_ARGS[@]}"
