#!/bin/sh
# tromr_music_e2e.sh — the E9 music-task gate (bd-3jo6.5.9).
#
# Drives the REAL `focr` binary over the REAL tromr.focrq:
#   1/4 GATE      zoo dir present (artifact + tokenizer tables + the upstream
#                 example staff) else SKIP-with-SUCCESS
#   2/4 BIN       build (or take FOCR_BIN) — release profile
#   3/4 NEGATIVE  /nonexistent model => exit 3; music at an unlimited-named
#                 model => exit 2 (Usage — knowably neither tromr nor got;
#                 ambiguous names pass through to the engine arch tag by design)
#   4/4 MUSIC     --task music on the example staff => partwise MusicXML
#                 carrying the CERTIFIED structure (clef G2, key CM fifths 0,
#                 3 measures — the token-exact argmax result for this staff)
#
# Env: FOCR_TROMR_DIR (default the USB zoo); FOCR_BIN (skips the build).
#
# Logging contract: stdout DATA-ONLY NDJSON, schema "tromr_music_e2e/v1"
# (events gate|bin|negative|music|result); human telemetry `TRMU `-prefixed
# on stderr. Exit 0 = PASS or gated SKIP; non-zero = a real divergence.
#
# POSIX sh; passes `sh -n`. python3 required (NDJSON emission).
set -eu

log()   { printf 'TRMU %s\n' "$*" >&2; }
step()  { printf 'TRMU ==== STEP %s ====\n' "$*" >&2; }
ok()    { printf 'TRMU   PASS  %s\n' "$*" >&2; }
skip()  { printf 'TRMU   SKIP  %s\n' "$*" >&2; }
fail()  { printf 'TRMU   FAIL  %s\n' "$*" >&2; }

ndj() {
  python3 - "$@" <<'PY'
import json, sys, time
rec = {"schema": "tromr_music_e2e/v1", "ts_ms": int(time.time() * 1000)}
for kv in sys.argv[1:]:
    k, v = kv.split("=", 1)
    try:
        rec[k] = int(v)
    except ValueError:
        rec[k] = v
print(json.dumps(rec, sort_keys=True))
PY
}

if ! command -v python3 >/dev/null 2>&1; then
  fail "python3 not found — required for NDJSON emission"
  exit 2
fi

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
ZOO="${FOCR_TROMR_DIR:-/Volumes/USBNVME16TB/temp_agent_space/zoo/tromr}"
MODEL="$ZOO/tromr.focrq"
STAFF="$ZOO/../tromr-upstream/examples/1.png"

step "1/4 GATE"
if [ ! -f "$MODEL" ] || [ ! -f "$ZOO/tokenizer_rhythm.json" ] || [ ! -f "$STAFF" ]; then
  skip "tromr artifact/tokenizers/example staff absent under $ZOO — skipped with SUCCESS"
  ndj event=gate result=skip reason=no_weights zoo="$ZOO"
  ndj event=result result=skip
  exit 0
fi
ok "artifact + tokenizer tables + example staff present"
ndj event=gate result=pass zoo="$ZOO"

step "2/4 BIN"
if [ -n "${FOCR_BIN:-}" ]; then
  BIN="$FOCR_BIN"
else
  ( cd "$REPO_ROOT" && cargo build --release --bin focr ) >&2
  BIN="${CARGO_TARGET_DIR:-$REPO_ROOT/target}/release/focr"
fi
if [ ! -x "$BIN" ]; then
  fail "focr binary not found/executable at $BIN"
  ndj event=bin result=fail path="$BIN"
  exit 1
fi
ok "binary ready: $BIN"
ndj event=bin result=pass path="$BIN"

step "3/4 NEGATIVE"
rc=0
"$BIN" ocr "$STAFF" --task music --model /nonexistent/tromr.focrq >/dev/null 2>&1 || rc=$?
if [ "$rc" -ne 3 ]; then
  fail "/nonexistent model must exit 3 (ModelNotFound); got $rc"
  ndj event=negative check=nonexistent_model result=fail exit_code="$rc" expect=3
  exit 1
fi
ok "/nonexistent model => exit 3"
ndj event=negative check=nonexistent_model result=pass exit_code="$rc"

rc=0
"$BIN" ocr "$STAFF" --task music --model unlimited-ocr.int8.focrq >/dev/null 2>&1 || rc=$?
if [ "$rc" -ne 2 ]; then
  fail "music at an unlimited-named model must exit 2 (Usage); got $rc"
  ndj event=negative check=wrong_family_guard result=fail exit_code="$rc" expect=2
  exit 1
fi
ok "music at an unlimited-named model => exit 2 (knowably neither tromr nor got)"
ndj event=negative check=wrong_family_guard result=pass exit_code="$rc"

step "4/4 MUSIC"
t0=$(python3 -c 'import time; print(int(time.time()*1000))')
OUT=$("$BIN" ocr "$STAFF" --task music --model "$MODEL" 2>/dev/null) || {
  rc=$?
  fail "music run failed (exit $rc)"
  ndj event=music result=fail exit_code="$rc"
  exit 1
}
t1=$(python3 -c 'import time; print(int(time.time()*1000))')
case "$OUT" in
  "<?xml"*) ok "output opens MusicXML" ;;
  *)
    fail "output is not MusicXML: $(printf '%s' "$OUT" | head -c 80)"
    ndj event=music result=fail reason=not_xml
    exit 1
    ;;
esac
for want in "<clef><sign>G</sign><line>2</line></clef>" "<key><fifths>0</fifths></key>" "<measure number=\"3\">" "score-partwise"; do
  case "$OUT" in
    *"$want"*) : ;;
    *)
      fail "MusicXML missing the certified structure: $want"
      ndj event=music result=fail reason=structure
      exit 1
      ;;
  esac
done
ok "MusicXML carries the certified structure (clef G2, CM, 3 measures)"
ndj event=music result=pass elapsed_ms=$((t1 - t0))

log "ALL STEPS PASSED"
ndj event=result result=pass
exit 0
