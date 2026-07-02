#!/usr/bin/env python3
"""Parse `FOCR_TIMING=1` stderr into per-stage gauntlet records (bd-re8.17).

`focr` emits `[focr-timing] <stage> <secs>s` lines to stderr when `FOCR_TIMING`
is set (`src/native_engine/mod.rs::timing_log`, plus the GOT decode line in
`src/native_engine/decoder_qwen2.rs`). This module turns N warm runs of that
stderr (captured by `scripts/gauntlet_focr.sh`) into the shared
`focr-gauntlet-stage/v1` JSON records that `scripts/gauntlet_row.py` merges
with the reference side (`scripts/gauntlet_reference.py`) and the roofline
(`scripts/gauntlet_roofline.py`) into a PERF_LEDGER row.

Honesty contract: every number here is read from a real run's stderr/meta
files. There is no default, estimate, or fill-in anywhere — a run whose stderr
carries no `[focr-timing]` lines is a hard error, never a zero.

Stage vocabulary (docs/PERF_LEDGER.md): `preprocess`, `vision_encode`,
`prefill`, `decode_per_token`, `end_to_end` (underscored here; the ledger row
uses hyphens). Extra informational stages (`weight_cache_build`,
`decode_total`, `vision_sam`, ...) are carried alongside but are not ledger
stages.

Usage:
  gauntlet_timing.py aggregate --run-dir DIR --out FILE [--threads N]
                               [--precision P] [--allocator A] [--synthetic]
  gauntlet_timing.py --self-test
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time

SCHEMA_STAGE = "focr-gauntlet-stage/v1"
SCHEMA_DOC = "focr-gauntlet-stages/v1"

LEDGER_STAGES = ("preprocess", "vision_encode", "prefill", "decode_per_token", "end_to_end")

PREFIX = "[focr-timing]"

# One regex per timing_log format string in src (kept in emission order of the
# pipeline; the GOT decode line must be tried before the Unlimited decode line
# because both start with "decode").
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("preprocess", re.compile(r"^preprocess (?P<s>\d+(?:\.\d+)?)s$")),
    ("vision_sam", re.compile(r"^vision\.sam (?P<s>\d+(?:\.\d+)?)s$")),
    ("vision_clip", re.compile(r"^vision\.clip (?P<s>\d+(?:\.\d+)?)s$")),
    ("vision_bridge", re.compile(r"^vision\.bridge (?P<s>\d+(?:\.\d+)?)s$")),
    # Batched-vision lines (bd-t6a / bd-1azu.10, spine multi-page path,
    # src/native_engine/mod.rs::vision_tower_batched_pages). The sam/clip/bridge
    # batch lines fold into the same stage names as their per-view twins (both
    # are the run's total time in that tower); hydrate is its own info stage.
    (
        "vision_hydrate",
        re.compile(r"^vision\.hydrate\(batch\) (?P<s>\d+(?:\.\d+)?)s$"),
    ),
    (
        "vision_sam_batch",
        re.compile(
            r"^vision\.sam\(batch of (?P<views>\d+), side (?P<side>\d+)\)"
            r" (?P<s>\d+(?:\.\d+)?)s$"
        ),
    ),
    (
        "vision_clip_batch",
        re.compile(r"^vision\.clip\(batch of (?P<views>\d+)\) (?P<s>\d+(?:\.\d+)?)s$"),
    ),
    (
        "vision_bridge_batch",
        re.compile(r"^vision\.bridge\(batch of (?P<views>\d+)\) (?P<s>\d+(?:\.\d+)?)s$"),
    ),
    ("vision_encode", re.compile(r"^vision_tower (?P<s>\d+(?:\.\d+)?)s$")),
    (
        "weight_cache_build",
        re.compile(r"^weight_cache_build(?P<i8>_i8)? (?P<s>\d+(?:\.\d+)?)s$"),
    ),
    (
        "prefill",
        re.compile(r"^prefill(?P<i8>_i8)? (?P<s>\d+(?:\.\d+)?)s \((?P<tok>\d+) tokens\)$"),
    ),
    (
        "got_decode",
        re.compile(
            r"^decode (?P<tok>\d+) tok in (?P<s>\d+(?:\.\d+)?)s \(\d+(?:\.\d+)? tok/s\)"
            r" \| seed\(prefill\) (?P<seed>\d+(?:\.\d+)?)s"
            r" \| layers (?P<layers>\d+(?:\.\d+)?)s"
            r" \(attn (?P<attn>\d+(?:\.\d+)?)s, gemv\+misc (?P<gemv>\d+(?:\.\d+)?)s\)"
            r" \| lm_head (?P<head>\d+(?:\.\d+)?)s$"
        ),
    ),
    (
        "decode_total",
        re.compile(
            r"^decode(?P<i8>_i8)? (?P<s>\d+(?:\.\d+)?)s"
            r" \((?P<tok>\d+) tokens, \d+(?:\.\d+)?s/tok\)$"
        ),
    ),
    (
        "decode_phases",
        re.compile(
            r"^decode_i8 phases \(ms\): lm_head (?P<head>\d+)\s+attn (?P<attn>\d+)"
            r"\s+experts (?P<experts>\d+)\s+route (?P<route>\d+)$"
        ),
    ),
    ("got_vision_splice", re.compile(r"^got\.vision\+splice (?P<s>\d+(?:\.\d+)?)s$")),
    ("got_generate", re.compile(r"^got\.generate (?P<tok>\d+) tokens (?P<s>\d+(?:\.\d+)?)s$")),
    ("got_forward", re.compile(r"^got forward (?P<s>\d+(?:\.\d+)?)s$")),
]


class TimingParseError(ValueError):
    """A run's stderr could not be parsed into timing evidence."""


def parse_run(stderr_text: str) -> dict:
    """Parse one run's stderr into `{stage: {"s": secs, "tokens": n?}, ...}`.

    Repeated occurrences of a stage (multi-view vision, multi-page PDFs) are
    summed and counted in `occurrences`; token counts sum likewise. Raises
    [`TimingParseError`] when no `[focr-timing]` line is present (the honest
    refusal: no evidence, no record) or when a `[focr-timing]` line does not
    match any known format (the binary's format moved — fix the parser, do not
    guess).
    """
    stages: dict[str, dict] = {}
    saw_prefix = False
    for raw in stderr_text.splitlines():
        line = raw.strip()
        if not line.startswith(PREFIX):
            continue
        saw_prefix = True
        body = line[len(PREFIX) :].strip()
        matched = False
        for name, pattern in _PATTERNS:
            m = pattern.match(body)
            if m is None:
                continue
            matched = True
            _fold(stages, name, m)
            break
        if not matched:
            raise TimingParseError(f"unrecognized [focr-timing] line: {body!r}")
    if not saw_prefix:
        raise TimingParseError(
            "no [focr-timing] lines in stderr — was FOCR_TIMING=1 exported and is "
            "this a real focr run?"
        )
    if not stages:
        raise TimingParseError("[focr-timing] prefix seen but no stage line parsed")
    return stages


def _fold(stages: dict[str, dict], name: str, m: re.Match[str]) -> None:
    groups = m.groupdict()
    if name == "decode_phases":
        entry = stages.setdefault(name, {"occurrences": 0})
        for key in ("head", "attn", "experts", "route"):
            entry[f"{key}_ms"] = entry.get(f"{key}_ms", 0.0) + float(groups[key])
        entry["occurrences"] += 1
        return
    if name == "got_decode":
        # The GOT decoder's one-line breakdown carries BOTH the decode total and
        # the seeding prefill; fan it out to the shared stage names.
        _add(stages, "decode_total", float(groups["s"]), int(groups["tok"]))
        _add(stages, "prefill", float(groups["seed"]), None)
        _add(stages, "decode_layers", float(groups["layers"]), None)
        _add(stages, "decode_attn", float(groups["attn"]), None)
        _add(stages, "decode_lm_head", float(groups["head"]), None)
        return
    if name.endswith("_batch"):
        # Batched vision (bd-t6a): same tower work as the per-view lines, one
        # line per side-group. Fold into the per-view stage name and keep the
        # batching evidence (`batched`, summed `views`) alongside.
        base = name[: -len("_batch")]
        _add(stages, base, float(groups["s"]), None)
        entry = stages[base]
        entry["batched"] = True
        entry["views"] = entry.get("views", 0) + int(groups["views"])
        return
    tokens = int(groups["tok"]) if groups.get("tok") is not None else None
    _add(stages, name, float(groups["s"]), tokens)
    if groups.get("i8") is not None:
        stages[name]["int8"] = True


def _add(stages: dict[str, dict], name: str, secs: float, tokens: int | None) -> None:
    entry = stages.setdefault(name, {"s": 0.0, "occurrences": 0})
    entry["s"] += secs
    entry["occurrences"] += 1
    if tokens is not None:
        entry["tokens"] = entry.get("tokens", 0) + tokens


def infer_precision(runs: list[dict]) -> str | None:
    """`focr-int8` / `focr-f32` when the timing lines themselves prove it.

    The Unlimited-OCR path suffixes its prefill/decode lines with `_i8`; the
    GOT path's lines carry no precision marker, so `None` is returned and the
    caller must supply `--precision` (refusing beats guessing).
    """
    saw_marked = False
    saw_int8 = False
    for run in runs:
        if "got_forward" in run:
            continue  # GOT timing lines carry no `_i8` marker; they prove nothing
        for name in ("prefill", "decode_total", "weight_cache_build"):
            entry = run.get(name)
            if entry is None:
                continue
            saw_marked = True
            saw_int8 = saw_int8 or bool(entry.get("int8"))
    if not saw_marked:
        return None
    return "focr-int8" if saw_int8 else "focr-f32"


def _stats(samples_ms: list[float]) -> dict:
    best = min(samples_ms)
    mean = statistics.fmean(samples_ms)
    cv_pct = None
    if len(samples_ms) > 1 and mean > 0:
        cv_pct = statistics.stdev(samples_ms) / mean * 100.0
    return {
        "samples_ms": [round(v, 6) for v in samples_ms],
        "best_ms": round(best, 6),
        "p50_ms": round(statistics.median(samples_ms), 6),
        "mean_ms": round(mean, 6),
        "cv_pct": None if cv_pct is None else round(cv_pct, 3),
        "n": len(samples_ms),
    }


def aggregate_runs(
    runs: list[dict],
    wall_ms: list[float],
    *,
    threads: int,
    precision: str,
    allocator: str,
    backend: str = "focr",
    warmup_discarded: int = 0,
    synthetic: bool = False,
) -> list[dict]:
    """Fold per-run parses into best-of-N stage records.

    `runs` and `wall_ms` are index-aligned (warmups already discarded by the
    caller). Token counts must agree across runs for `prefill`/`decode_total`
    — greedy decode is deterministic, so a variance is flagged
    (`tokens_consistent: false`) rather than averaged away; `gauntlet_row.py`
    refuses such records.
    """
    if len(runs) != len(wall_ms):
        raise TimingParseError(f"runs ({len(runs)}) and wall_ms ({len(wall_ms)}) misaligned")
    if not runs:
        raise TimingParseError("no measured runs to aggregate")

    names: list[str] = []
    for run in runs:
        for name in run:
            if name not in names:
                names.append(name)

    records: list[dict] = []
    for name in names:
        if name == "decode_phases":
            continue
        present = [run[name] for run in runs if name in run]
        if len(present) != len(runs):
            raise TimingParseError(
                f"stage {name!r} present in only {len(present)}/{len(runs)} runs — "
                "runs are not comparable"
            )
        samples_ms = [entry["s"] * 1000.0 for entry in present]
        tokens = [entry.get("tokens") for entry in present]
        record = {
            "schema": SCHEMA_STAGE,
            "source": "focr",
            "stage": name,
            "ledger_stage": name in LEDGER_STAGES,
            "unit": "ms",
            **_stats(samples_ms),
            "warmup_discarded": warmup_discarded,
            "threads": threads,
            "precision": precision,
            "backend": backend,
            "allocator": allocator,
            "occurrences": present[0].get("occurrences", 1),
            "synthetic": synthetic,
        }
        if any(t is not None for t in tokens):
            record["tokens"] = tokens[0]
            record["tokens_consistent"] = len(set(tokens)) == 1
        records.append(record)

    # decode_per_token derives per run from the decode TOTAL and its token
    # count (never from the printed rounded s/tok).
    decode = [run.get("decode_total") for run in runs]
    if all(d is not None and d.get("tokens") for d in decode):
        per_tok_ms = [d["s"] * 1000.0 / d["tokens"] for d in decode]
        tokens = [d["tokens"] for d in decode]
        records.append(
            {
                "schema": SCHEMA_STAGE,
                "source": "focr",
                "stage": "decode_per_token",
                "ledger_stage": True,
                "unit": "ms",
                **_stats(per_tok_ms),
                "warmup_discarded": warmup_discarded,
                "threads": threads,
                "precision": precision,
                "backend": backend,
                "allocator": allocator,
                "occurrences": 1,
                "tokens": tokens[0],
                "tokens_consistent": len(set(tokens)) == 1,
                "synthetic": synthetic,
            }
        )

    records.append(
        {
            "schema": SCHEMA_STAGE,
            "source": "focr",
            "stage": "end_to_end",
            "ledger_stage": True,
            "unit": "ms",
            **_stats(list(wall_ms)),
            "warmup_discarded": warmup_discarded,
            "threads": threads,
            "precision": precision,
            "backend": backend,
            "allocator": allocator,
            "occurrences": 1,
            "note": "process wall clock: includes binary startup + model load + weight-cache build",
            "synthetic": synthetic,
        }
    )
    return records


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cmd_aggregate(args: argparse.Namespace) -> int:
    metas = sorted(glob.glob(os.path.join(args.run_dir, "run_*.meta.json")))
    if not metas:
        print(f"ERROR: no run_*.meta.json under {args.run_dir}", file=sys.stderr)
        return 2

    runs: list[dict] = []
    wall_ms: list[float] = []
    stdout_hashes: set[str] = set()
    raw_files: list[str] = []
    meta0: dict = {}
    for meta_path in metas:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if not meta0:
            meta0 = meta
        stderr_path = os.path.join(args.run_dir, meta["stderr"])
        with open(stderr_path, encoding="utf-8", errors="replace") as f:
            stderr_text = f.read()
        if meta.get("exit_code", 0) != 0:
            print(
                f"ERROR: {meta_path}: run exited {meta['exit_code']} — a failed run "
                "is not perf evidence",
                file=sys.stderr,
            )
            return 1
        try:
            runs.append(parse_run(stderr_text))
        except TimingParseError as err:
            print(f"ERROR: {stderr_path}: {err}", file=sys.stderr)
            return 1
        wall_ms.append(float(meta["wall_ms"]))
        stdout_path = os.path.join(args.run_dir, meta["stdout"])
        stdout_hashes.add(sha256_file(stdout_path))
        raw_files += [meta_path, stderr_path, stdout_path]

    precision = infer_precision(runs)
    if args.precision:
        if precision is not None and precision != args.precision:
            print(
                f"ERROR: --precision {args.precision} contradicts the timing lines "
                f"({precision})",
                file=sys.stderr,
            )
            return 1
        precision = args.precision
    if precision is None:
        print(
            "ERROR: precision not inferable from the timing lines (GOT path) — "
            "pass --precision focr-int8|focr-f32",
            file=sys.stderr,
        )
        return 1

    threads = args.threads if args.threads is not None else meta0.get("threads")
    if not isinstance(threads, int) or threads <= 0:
        print("ERROR: positive --threads (or meta threads) required", file=sys.stderr)
        return 1

    records = aggregate_runs(
        runs,
        wall_ms,
        threads=threads,
        precision=precision,
        allocator=args.allocator,
        warmup_discarded=int(meta0.get("warmup", 0)),
        synthetic=args.synthetic,
    )
    doc = {
        "schema": SCHEMA_DOC,
        "source": "focr",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_dir": os.path.abspath(args.run_dir),
        "command": meta0.get("command"),
        "env_pins": meta0.get("env_pins", {}),
        "focr_env": meta0.get("focr_env", {}),
        "binary": meta0.get("binary"),
        "binary_sha256": meta0.get("binary_sha256"),
        "page": meta0.get("page"),
        "page_sha256": meta0.get("page_sha256"),
        "model": meta0.get("model"),
        "threads": threads,
        "precision": precision,
        "allocator": args.allocator,
        "runs": len(runs),
        "warmup": int(meta0.get("warmup", 0)),
        "stdout_identical_across_runs": len(stdout_hashes) == 1,
        "stages": records,
        "synthetic": args.synthetic,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, sort_keys=False)
        f.write("\n")
    print(
        json.dumps(
            {
                "event": "focr_stages_written",
                "out": args.out,
                "runs": len(runs),
                "stages": [r["stage"] for r in records],
                "stdout_identical_across_runs": doc["stdout_identical_across_runs"],
                "synthetic": args.synthetic,
            }
        )
    )
    return 0


# ── self-test ────────────────────────────────────────────────────────────────

_SYNTHETIC_UNLIMITED = """\
[focr-timing] preprocess 0.12s
[focr-timing]   vision.sam 1.20s
[focr-timing]   vision.clip 0.40s
[focr-timing]   vision.bridge 0.05s
[focr-timing] vision_tower 1.65s
[focr-timing] weight_cache_build_i8 1.10s
[focr-timing] prefill_i8 2.40s (289 tokens)
[focr-timing] decode_i8 9.00s (750 tokens, 0.012s/tok)
[focr-timing] decode_i8 phases (ms): lm_head 1200  attn 2400  experts 4300  route 100
noise line the parser must ignore
"""

_SYNTHETIC_GOT = """\
[focr-timing]   got.vision+splice 1.30s
[focr-timing]   decode 512 tok in 12.00s (42.7 tok/s) | seed(prefill) 0.55s | \
layers 10.00s (attn 3.00s, gemv+misc 7.00s) | lm_head 1.50s
[focr-timing]   got.generate 512 tokens 12.60s
[focr-timing] got forward 14.10s
"""

# The bd-t6a spine batched-vision line set (multi-page: two side groups for
# sam, single lines for hydrate/clip/bridge; the second sam group proves the
# same-stage fold sums seconds and view counts).
_SYNTHETIC_BATCH = """\
[focr-timing]   vision.hydrate(batch) 0.62s
[focr-timing]   vision.sam(batch of 3, side 1024) 2.40s
[focr-timing]   vision.sam(batch of 2, side 640) 0.80s
[focr-timing]   vision.clip(batch of 5) 0.90s
[focr-timing]   vision.bridge(batch of 5) 0.12s
"""


def _self_test() -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, **fields: object) -> None:
        print(json.dumps({"check": name, "result": "pass" if ok else "fail", **fields}))
        if not ok:
            failures.append(name)

    # Unlimited-OCR int8 line set parses stage-for-stage.
    run = parse_run(_SYNTHETIC_UNLIMITED)
    check("unlimited-preprocess", math.isclose(run["preprocess"]["s"], 0.12))
    check("unlimited-vision", math.isclose(run["vision_encode"]["s"], 1.65))
    check("unlimited-prefill-tokens", run["prefill"]["tokens"] == 289)
    check("unlimited-decode-tokens", run["decode_total"]["tokens"] == 750)
    check("unlimited-int8-marker", run["decode_total"].get("int8") is True)
    check("unlimited-phases", math.isclose(run["decode_phases"]["experts_ms"], 4300.0))
    check("unlimited-precision", infer_precision([run]) == "focr-int8")

    # GOT line set: the one-line decode breakdown fans out to shared stages.
    got = parse_run(_SYNTHETIC_GOT)
    check("got-decode-total", math.isclose(got["decode_total"]["s"], 12.0))
    check("got-decode-tokens", got["decode_total"]["tokens"] == 512)
    check("got-prefill-seed", math.isclose(got["prefill"]["s"], 0.55))
    check("got-vision", math.isclose(got["got_vision_splice"]["s"], 1.30))
    check("got-precision-unknown", infer_precision([got]) is None)

    # bd-t6a batched-vision lines: hydrate is its own stage; sam/clip/bridge
    # batch lines fold into the per-view stage names, summing seconds + views.
    batch = parse_run(_SYNTHETIC_BATCH)
    check("batch-hydrate", math.isclose(batch["vision_hydrate"]["s"], 0.62))
    check("batch-sam-sum", math.isclose(batch["vision_sam"]["s"], 3.20))
    check("batch-sam-views", batch["vision_sam"]["views"] == 5)
    check("batch-sam-occurrences", batch["vision_sam"]["occurrences"] == 2)
    check("batch-sam-marker", batch["vision_sam"].get("batched") is True)
    check("batch-clip", math.isclose(batch["vision_clip"]["s"], 0.90))
    check("batch-bridge", math.isclose(batch["vision_bridge"]["s"], 0.12))
    check("batch-precision-unknown", infer_precision([batch]) is None)

    # Per-view and batched lines for the same tower coexist in one run (a
    # mixed spine run): the fold sums into a single stage entry.
    mixed = parse_run(_SYNTHETIC_BATCH + "[focr-timing]   vision.sam 1.00s\n")
    check("batch-mixed-sam-sum", math.isclose(mixed["vision_sam"]["s"], 4.20))
    check("batch-mixed-occurrences", mixed["vision_sam"]["occurrences"] == 3)

    # No timing lines / unknown timing line both refuse (never fabricate).
    for name, text in (
        ("refuses-empty", "plain stderr, no timing\n"),
        ("refuses-unknown-line", "[focr-timing] warpdrive 9.99s\n"),
    ):
        try:
            parse_run(text)
            check(name, False)
        except TimingParseError:
            check(name, True)

    # Aggregation: best-of-N is the min; cv% is sample stdev over mean;
    # per-token decode derives from totals; token drift is flagged.
    run_a = parse_run(_SYNTHETIC_UNLIMITED)
    run_b = parse_run(_SYNTHETIC_UNLIMITED.replace("9.00s (750", "9.60s (750"))
    records = aggregate_runs(
        [run_a, run_b],
        [15000.0, 15400.0],
        threads=8,
        precision="focr-int8",
        allocator="system",
        warmup_discarded=1,
        synthetic=True,
    )
    by_stage = {r["stage"]: r for r in records}
    check("agg-decode-best", math.isclose(by_stage["decode_total"]["best_ms"], 9000.0))
    per_tok = by_stage["decode_per_token"]
    check("agg-per-token-best", math.isclose(per_tok["best_ms"], 9000.0 / 750))
    check("agg-per-token-consistent", per_tok["tokens_consistent"] is True)
    expected_cv = statistics.stdev([12.0, 12.8]) / statistics.fmean([12.0, 12.8]) * 100
    check("agg-per-token-cv", math.isclose(per_tok["cv_pct"], round(expected_cv, 3)))
    check("agg-e2e-best", math.isclose(by_stage["end_to_end"]["best_ms"], 15000.0))
    check("agg-synthetic-stamp", all(r["synthetic"] for r in records))

    run_drift = parse_run(_SYNTHETIC_UNLIMITED.replace("(750 tokens", "(751 tokens"))
    drift = aggregate_runs(
        [run_a, run_drift],
        [1.0, 1.0],
        threads=8,
        precision="focr-int8",
        allocator="system",
        synthetic=True,
    )
    drift_tok = next(r for r in drift if r["stage"] == "decode_per_token")
    check("agg-token-drift-flagged", drift_tok["tokens_consistent"] is False)

    # A stage missing from one run is a comparability error, not a silent gap.
    try:
        aggregate_runs(
            [run_a, {"preprocess": {"s": 0.1, "occurrences": 1}}],
            [1.0, 1.0],
            threads=8,
            precision="focr-int8",
            allocator="system",
            synthetic=True,
        )
        check("agg-missing-stage-refused", False)
    except TimingParseError:
        check("agg-missing-stage-refused", True)

    if failures:
        print(f"SELF-TEST FAILED: {failures}", file=sys.stderr)
        return 1
    print(json.dumps({"check": "gauntlet-timing-self-test", "result": "pass"}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run the parser self-test")
    sub = parser.add_subparsers(dest="cmd")
    agg = sub.add_parser("aggregate", help="aggregate a gauntlet_focr.sh run dir")
    agg.add_argument("--run-dir", required=True)
    agg.add_argument("--out", required=True)
    agg.add_argument("--threads", type=int, default=None)
    agg.add_argument("--precision", default=None, help="focr-int8|focr-f32 (GOT path only)")
    agg.add_argument("--allocator", default="system")
    agg.add_argument(
        "--synthetic",
        action="store_true",
        help="stamp records synthetic (self-test stubs; gauntlet_row.py refuses them)",
    )
    args = parser.parse_args()
    if args.self_test:
        return _self_test()
    if args.cmd == "aggregate":
        return _cmd_aggregate(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
