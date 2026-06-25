#!/usr/bin/env sh
# Re-fetch the load-bearing Baidu Unlimited-OCR SOURCE files (small text files only;
# NOT the 6.67 GB weights — see scripts/fetch_model.sh) into docs/truth-pack/snapshots/,
# pinned to the HF commit recorded in docs/truth-pack/PINNED_SOURCES.md.
#
#   scripts/fetch_sources.sh            # fetch
#   scripts/fetch_sources.sh --verify   # fetch AND verify SHA-256 vs SOURCE_HASHES.md
#
# Snapshots are git-ignored; only their hashes + the derived truth-pack artifacts
# are committed. Re-fetch + verify gives byte-identical reproducibility.
set -eu

PIN="3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5"   # HF commit (see PINNED_SOURCES.md)
BASE="https://huggingface.co/baidu/Unlimited-OCR/resolve/${PIN}"
DIR="$(cd "$(dirname "$0")/.." && pwd)/docs/truth-pack/snapshots"
mkdir -p "$DIR"

FILES="config.json model.safetensors.index.json tokenizer_config.json \
special_tokens_map.json processor_config.json modeling_unlimitedocr.py \
modeling_deepseekv2.py deepencoder.py configuration_deepseek_v2.py \
conversation.py README.md LICENSE"

echo "Fetching sources at HF commit ${PIN} ..."
for f in $FILES; do
  curl -fsSL --max-time 120 "${BASE}/${f}" -o "${DIR}/${f}"
  printf "  %-32s %8s bytes\n" "$f" "$(wc -c < "${DIR}/${f}")"
done

if [ "${1:-}" = "--verify" ]; then
  echo "Verifying against SOURCE_HASHES.md ..."
  HASHES="$(cd "$(dirname "$0")/.." && pwd)/docs/truth-pack/SOURCE_HASHES.md"
  awk -F'`' '/\| `[0-9a-f]{64}` \|/{print $2"  "$4}' "$HASHES" | (cd "$DIR" && shasum -a 256 -c -)
  echo "OK: all snapshots match the pinned hashes."
fi
