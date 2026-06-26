#!/usr/bin/env bash
# Run franken_ocr's native forward on the 20 test pages -> franken_out/page_*.md,
# for parity comparison against the baidu oracle (scripts/baseline/compare_ocr.py).
#
# Builds `focr` release with isolated CARGO_HOME + CARGO_TARGET_DIR on the APFS
# scratch image (root disk is ~99% full) and debuginfo off, then runs
# `focr ocr <page.png> --model <model_dir>` per page, capturing markdown.
#
# Usage: run_franken.sh [model_dir] [pages_dir] [out_dir]
set -euo pipefail

WORK=/Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work
SCRATCH=/Volumes/focrvenv
MODEL_DIR="${1:-$WORK/model}"
PAGES_DIR="${2:-$WORK/pages}"
OUT_DIR="${3:-$WORK/franken_out}"
mkdir -p "$OUT_DIR"

export CARGO_TARGET_DIR="$SCRATCH/focr_target"
export CARGO_HOME="$SCRATCH/focr_cargo_home"
export CARGO_PROFILE_RELEASE_DEBUG=0
mkdir -p "$CARGO_TARGET_DIR" "$CARGO_HOME"

echo "[franken] building focr (release, isolated dirs on $SCRATCH) ..."
cargo build --release --bin focr

BIN="$CARGO_TARGET_DIR/release/focr"
[ -x "$BIN" ] || { echo "focr binary not found at $BIN" >&2; exit 1; }
echo "[franken] binary: $BIN"
echo "[franken] model:  $MODEL_DIR"

shopt -s nullglob
n=0
for png in "$PAGES_DIR"/page_*.png; do
  base="$(basename "${png%.png}")"
  out="$OUT_DIR/${base}.md"
  printf '[franken] %s -> %s ... ' "$base" "$out"
  t0=$SECONDS
  if "$BIN" ocr "$png" --model "$MODEL_DIR" > "$out" 2>"$OUT_DIR/${base}.err"; then
    echo "ok ($((SECONDS-t0))s, $(wc -c <"$out") bytes)"
  else
    echo "FAILED (see ${base}.err)"; tail -3 "$OUT_DIR/${base}.err" >&2 || true
  fi
  n=$((n+1))
done
echo "[franken] ran $n page(s); output in $OUT_DIR"
