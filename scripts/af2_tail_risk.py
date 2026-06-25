#!/usr/bin/env python3
"""AF-2 tail-risk monitor: mean / CVaR_alpha / EVT-p999 over per-document CER.

This is the OFFLINE reference implementation of the franken_ocr alien-artifact
family **AF-2** (plan section 9.7, design doc `docs/alien/AF-2-tail-risk-cvar-evt.md`).
It is a COMPLETE, runnable tool (NOT a skeleton): given a list of per-document
Character Error Rates (CER, one float in [0, 1] per golden-corpus document) it
computes the three tail-risk statistics the release scorecard gates on:

  * ``mean``       -- the naive average CER (reported, but NEVER the gate);
  * ``cvar_<a>``   -- CVaR_alpha, the mean of the worst ``alpha`` fraction of
                      per-document CER (default alpha = 0.10), i.e. the average
                      over the heaviest documents, not just the boundary quantile;
  * ``evt_p999``   -- the 99.9th-percentile document CER from an Extreme-Value
                      (Peaks-Over-Threshold) Generalized-Pareto tail fit.

WHY THIS EXISTS (the load-bearing intuition)
--------------------------------------------
OCR fails in the TAIL. Most pages decode cleanly, but a quantization choice that
wrecks dense tables, sub/superscripts, long digit runs, or code is unacceptable
even when the *mean* CER looks great -- perplexity and mean-CER systematically
under-predict exact-token failure on those few hard documents. So the AF-2 gate
optimizes/bounds the worst-case fraction (CVaR) and extrapolates past the corpus
size to the p99.9 document (EVT), instead of trusting the mean.

WHAT THIS TOOL IS / IS NOT
--------------------------
OFFLINE TOOLING ONLY. franken_ocr's Rust engine NEVER invokes this script at
inference time and imports no Python. The shipping ``tail_risk_monitor`` artifact
is the Rust re-implementation of exactly these formulas (held numerically
equivalent to this reference). This script is the human-/CI-run oracle: feed it a
JSON array / newline list / CSV column of per-document CER and it emits one NDJSON
record with ``{mean, cvar, evt_p999, ...}`` plus, with ``--compare`` /
``--budget``, the release verdict.

DEPENDENCIES
------------
Python standard library only (no numpy/scipy) -- matches the repo convention that
offline tooling stays dependency-light and reproducible. The Generalized-Pareto
fit uses the method-of-moments / probability-weighted-moments estimator (closed
form, no optimizer), which is robust on the small samples a golden corpus yields.

USAGE
-----
    # from a JSON array of per-doc CER
    python3 scripts/af2_tail_risk.py --input cer.json

    # from stdin (one float per line, '#' comments allowed)
    focr ... | python3 scripts/af2_tail_risk.py --stdin

    # inline values, custom alpha, gate against an f32 baseline + budget
    python3 scripts/af2_tail_risk.py --values 0.01 0.02 0.5 0.9 \\
        --alpha 0.1 --baseline-cvar 0.05 --budget 0.01

Exit codes:
    0  computed (and, if a gate was requested, the gate PASSED)
    1  bad input / usage error
    3  a release gate was requested and the tail bound FAILED it
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

# Exit codes (mirrors the project's stable-exit-code discipline, AGENTS.md / plan 7.4).
EXIT_OK = 0
EXIT_USAGE = 1
EXIT_GATE_FAILED = 3

# Default worst-case fraction for CVaR. 0.10 == "the worst 10% of documents".
DEFAULT_ALPHA = 0.10
# EVT target quantile: the 99.9th-percentile document.
DEFAULT_EVT_Q = 0.999
# Default Peaks-Over-Threshold exceedance fraction: fit the GPD to the worst 15%.
# (Standard POT practice keeps enough exceedances for a stable fit while staying
# in the genuine tail; configurable via --pot-frac.)
DEFAULT_POT_FRAC = 0.15
# Minimum exceedances below which the GPD fit is not trusted and we fall back to
# the empirical quantile (the conservative deterministic fallback).
MIN_EXCEEDANCES = 8


# --------------------------------------------------------------------------- #
# Core statistics                                                             #
# --------------------------------------------------------------------------- #


def empirical_quantile(sorted_vals: Sequence[float], q: float) -> float:
    """Type-7 (linear-interpolation) empirical quantile of an ascending list.

    Matches numpy's default and R's type 7; used as the EVT fallback and to seed
    the POT threshold.
    """
    n = len(sorted_vals)
    if n == 0:
        raise ValueError("empirical_quantile of empty sample")
    if n == 1:
        return float(sorted_vals[0])
    q = min(max(q, 0.0), 1.0)
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def mean(vals: Sequence[float]) -> float:
    return math.fsum(vals) / len(vals)


def cvar(vals: Sequence[float], alpha: float) -> float:
    """CVaR_alpha = mean of the worst ``alpha`` fraction of ``vals`` (upper tail).

    For risk we treat LARGER CER as worse, so the "worst alpha fraction" is the
    top ``alpha`` of the distribution. We use the coherent (Rockafellar-Uryasev)
    definition that stays exact for fractional cutoffs:

        CVaR_alpha = (1/(alpha*n)) * [ sum of the ceil(alpha*n) largest values
                       MINUS the over-counted partial document at the boundary ].

    Concretely: let k = ceil(alpha * n) be the number of documents in the worst
    fraction; average the k largest, then correct for the fact that alpha*n may
    be non-integer so the k-th (boundary) document is only partially included.
    This makes CVaR continuous in alpha and >= VaR_alpha always.
    """
    n = len(vals)
    if n == 0:
        raise ValueError("cvar of empty sample")
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    ordered = sorted(vals, reverse=True)  # worst (largest CER) first
    target = alpha * n
    k = math.ceil(target - 1e-12)  # documents fully or partially in the tail
    k = max(1, min(k, n))
    full = math.fsum(ordered[: k - 1]) if k >= 1 else 0.0
    # Weight of the boundary (k-th) document: the residual fraction of alpha*n.
    boundary_weight = target - (k - 1)
    boundary_weight = min(max(boundary_weight, 0.0), 1.0)
    weighted_sum = full + boundary_weight * ordered[k - 1]
    return weighted_sum / target


def value_at_risk(sorted_vals: Sequence[float], alpha: float) -> float:
    """VaR_alpha for the upper tail: the (1 - alpha) quantile of CER.

    Reported alongside CVaR for context (CVaR >= VaR always).
    """
    return empirical_quantile(sorted_vals, 1.0 - alpha)


# --------------------------------------------------------------------------- #
# EVT: Peaks-Over-Threshold Generalized-Pareto tail fit                       #
# --------------------------------------------------------------------------- #


@dataclass
class GpdFit:
    """A Generalized-Pareto fit to the peaks-over-threshold exceedances.

    The GPD CDF for exceedance y = x - u (y >= 0) is
        G(y) = 1 - (1 + xi*y/beta)^(-1/xi),  xi != 0
        G(y) = 1 - exp(-y/beta),             xi == 0
    with shape ``xi`` and scale ``beta`` > 0.
    """

    threshold: float  # u
    scale: float  # beta
    shape: float  # xi
    n_exceed: int  # number of exceedances used
    n_total: int  # total sample size
    method: str  # "pwm" | "empirical-fallback"

    @property
    def exceed_rate(self) -> float:
        """P(X > u): the fraction of documents above the POT threshold."""
        return self.n_exceed / self.n_total if self.n_total else 0.0

    def quantile(self, q: float) -> float:
        """Return the GPD-extrapolated quantile x_q for q in (exceed-anchor, 1).

        Inverting the conditional GPD and composing with the exceedance rate:
            x_q = u + (beta/xi) * ( ( (1-q) / zeta_u )^(-xi) - 1 ),  xi != 0
            x_q = u - beta * ln( (1-q) / zeta_u ),                   xi == 0
        where zeta_u = P(X > u) is the exceedance rate.
        """
        zeta = self.exceed_rate
        if zeta <= 0.0:
            return self.threshold
        ratio = (1.0 - q) / zeta
        ratio = max(ratio, 1e-300)
        if abs(self.shape) < 1e-8:
            return self.threshold - self.scale * math.log(ratio)
        return self.threshold + (self.scale / self.shape) * (ratio ** (-self.shape) - 1.0)


def fit_gpd_pwm(sorted_vals: Sequence[float], pot_frac: float) -> GpdFit:
    """Fit a GPD to the upper-tail exceedances via probability-weighted moments.

    PWM (Hosking & Wallis 1987) is a closed-form, optimizer-free estimator that
    is stable on the small exceedance counts a golden corpus produces, where MLE
    can fail to converge. For ascending exceedances y_(1)..y_(m) it uses the
    plotting-position moments ``a0`` (the mean) and ``a1``:

        a0 = mean(y)
        a1 = (1/m) * sum_j (1 - (j - 0.35)/m) * y_(j)
        xi   = 2 - a0 / (a0 - 2*a1)
        beta = 2 * a0 * a1 / (a0 - 2*a1)

    When the exceedance count is too small (< MIN_EXCEEDANCES) or the moment
    relation is degenerate, we fall back to the empirical quantile (the
    conservative deterministic fallback -- no extrapolated bound is invented).
    """
    n = len(sorted_vals)
    if n == 0:
        raise ValueError("fit_gpd of empty sample")

    # Choose threshold u as the (1 - pot_frac) quantile; exceedances are the
    # documents strictly above it.
    u = empirical_quantile(sorted_vals, 1.0 - pot_frac)
    exceed = [v - u for v in sorted_vals if v > u]
    m = len(exceed)

    if m < MIN_EXCEEDANCES:
        # Not enough tail data to trust an EVT extrapolation. Fall back to the
        # empirical p999 (clamped to the data max) -- never invent a bound.
        return GpdFit(
            threshold=u,
            scale=0.0,
            shape=0.0,
            n_exceed=m,
            n_total=n,
            method="empirical-fallback",
        )

    exceed.sort()  # ascending exceedances y_(1) <= ... <= y_(m)
    # Hosking & Wallis (1987) GPD probability-weighted moments:
    #   a0 = mean(y)
    #   a1 = (1/m) * sum_j (1 - p_j) * y_(j),  p_j = (j - 0.35)/m   (j = 1..m)
    # In the xi = -k convention (CDF G(y) = 1 - (1 + xi*y/beta)^(-1/xi)):
    #   xi   = 2 - a0/(a0 - 2*a1)
    #   beta = 2 * a0 * a1 / (a0 - 2*a1)
    # Note a1 uses the *plotting-position* weight (1 - p_j); using the unbiased
    # rank weight (j-1)/(m-1) instead collapses a1 -> a0/2 and makes the GPD
    # moment relation degenerate, so the plotting-position form is required.
    a0 = math.fsum(exceed) / m
    a1 = math.fsum((1.0 - ((j + 1) - 0.35) / m) * y for j, y in enumerate(exceed)) / m

    denom = a0 - 2.0 * a1
    if abs(denom) < 1e-12 or a0 <= 0.0:
        return GpdFit(
            threshold=u,
            scale=0.0,
            shape=0.0,
            n_exceed=m,
            n_total=n,
            method="empirical-fallback",
        )

    shape = 2.0 - a0 / denom
    scale = 2.0 * a0 * a1 / denom

    if not (math.isfinite(shape) and math.isfinite(scale)) or scale <= 0.0:
        return GpdFit(
            threshold=u,
            scale=0.0,
            shape=0.0,
            n_exceed=m,
            n_total=n,
            method="empirical-fallback",
        )

    return GpdFit(
        threshold=u,
        scale=scale,
        shape=shape,
        n_exceed=m,
        n_total=n,
        method="pwm",
    )


def evt_quantile(sorted_vals: Sequence[float], q: float, pot_frac: float) -> tuple[float, GpdFit]:
    """EVT (POT/GPD) estimate of the q-quantile, with empirical clamping.

    Returns ``(x_q, fit)``. CER is bounded in [0, 1], so we clamp the EVT
    estimate into that range; we also never report a tail quantile *below* the
    empirical quantile (the fit must not under-state observed risk).
    """
    n = len(sorted_vals)
    if n == 0:
        raise ValueError("evt_quantile of empty sample")
    emp = empirical_quantile(sorted_vals, q)
    fit = fit_gpd_pwm(sorted_vals, pot_frac)
    if fit.method == "empirical-fallback":
        x_q = emp
    else:
        x_q = fit.quantile(q)
        if not math.isfinite(x_q):
            x_q = emp
    # The tail bound is a *worst-case* estimate: never let it under-state the
    # empirical quantile, and clamp to the CER domain [0, 1].
    x_q = max(x_q, emp)
    x_q = min(max(x_q, 0.0), 1.0)
    return x_q, fit


# --------------------------------------------------------------------------- #
# Top-level report                                                            #
# --------------------------------------------------------------------------- #


@dataclass
class TailReport:
    n: int
    alpha: float
    mean: float
    var_alpha: float
    cvar_alpha: float
    evt_q: float
    evt_pXXX: float
    pot_frac: float
    gpd_shape: float
    gpd_scale: float
    gpd_threshold: float
    gpd_n_exceed: int
    gpd_method: str
    max_cer: float
    min_cer: float
    gate: dict[str, object] = field(default_factory=dict)

    def to_record(self) -> dict[str, object]:
        # Key the CVaR/EVT fields by their parameter so the NDJSON is
        # self-describing (e.g. cvar_0.1, evt_p999).
        cvar_key = f"cvar_{_fmt_frac(self.alpha)}"
        evt_key = f"evt_p{_fmt_pctile(self.evt_q)}"
        rec: dict[str, object] = {
            "artifact": "af2_tail_risk",
            "n": self.n,
            "alpha": self.alpha,
            "mean": self.mean,
            f"var_{_fmt_frac(self.alpha)}": self.var_alpha,
            cvar_key: self.cvar_alpha,
            evt_key: self.evt_pXXX,
            "evt_q": self.evt_q,
            "pot_frac": self.pot_frac,
            "gpd_shape": self.gpd_shape,
            "gpd_scale": self.gpd_scale,
            "gpd_threshold": self.gpd_threshold,
            "gpd_n_exceed": self.gpd_n_exceed,
            "gpd_method": self.gpd_method,
            "max_cer": self.max_cer,
            "min_cer": self.min_cer,
        }
        if self.gate:
            rec["gate"] = self.gate
        return rec


def _fmt_frac(alpha: float) -> str:
    """Format alpha=0.1 -> '0.1' for stable field names."""
    s = f"{alpha:.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _fmt_pctile(q: float) -> str:
    """Format q=0.999 -> '999', q=0.99 -> '99' for stable field names."""
    s = f"{q * 100:.4f}".rstrip("0").rstrip(".")
    return s.replace(".", "")


def compute_report(
    vals: Sequence[float],
    alpha: float = DEFAULT_ALPHA,
    evt_q: float = DEFAULT_EVT_Q,
    pot_frac: float = DEFAULT_POT_FRAC,
) -> TailReport:
    """Compute the full AF-2 tail-risk report for a list of per-doc CER."""
    if not vals:
        raise ValueError("no CER values provided")
    for v in vals:
        if not math.isfinite(v):
            raise ValueError(f"non-finite CER value: {v!r}")
        if v < 0.0 or v > 1.0:
            raise ValueError(f"CER out of [0,1]: {v!r} (CER is a rate)")
    sorted_vals = sorted(vals)
    evt_pXXX, fit = evt_quantile(sorted_vals, evt_q, pot_frac)
    return TailReport(
        n=len(vals),
        alpha=alpha,
        mean=mean(vals),
        var_alpha=value_at_risk(sorted_vals, alpha),
        cvar_alpha=cvar(vals, alpha),
        evt_q=evt_q,
        evt_pXXX=evt_pXXX,
        pot_frac=pot_frac,
        gpd_shape=fit.shape,
        gpd_scale=fit.scale,
        gpd_threshold=fit.threshold,
        gpd_n_exceed=fit.n_exceed,
        gpd_method=fit.method,
        max_cer=sorted_vals[-1],
        min_cer=sorted_vals[0],
    )


def apply_gate(
    report: TailReport,
    baseline_cvar: float | None,
    baseline_evt: float | None,
    budget: float,
) -> dict[str, object]:
    """Evaluate the AF-2 release gate.

    The release scorecard gates on the **CVaR / EVT bound vs the f32 baseline**,
    NOT the mean. A candidate (e.g. an int4 config) PASSES iff its tail
    statistics stay within ``budget`` of the f32 baseline's:

        cvar_candidate  <= baseline_cvar + budget
        evt_candidate   <= baseline_evt  + budget   (when a baseline EVT given)

    Returns a structured verdict. If no baseline is supplied, the gate reports
    the raw bounds with ``verdict = "no-baseline"`` (informational, not a fail).
    """
    gate: dict[str, object] = {"budget": budget}
    checks: list[dict[str, object]] = []
    passed = True
    have_baseline = False

    if baseline_cvar is not None:
        have_baseline = True
        limit = baseline_cvar + budget
        ok = report.cvar_alpha <= limit + 1e-12
        passed = passed and ok
        checks.append(
            {
                "name": f"cvar_{_fmt_frac(report.alpha)}",
                "candidate": report.cvar_alpha,
                "baseline": baseline_cvar,
                "limit": limit,
                "pass": ok,
            }
        )

    if baseline_evt is not None:
        have_baseline = True
        limit = baseline_evt + budget
        ok = report.evt_pXXX <= limit + 1e-12
        passed = passed and ok
        checks.append(
            {
                "name": f"evt_p{_fmt_pctile(report.evt_q)}",
                "candidate": report.evt_pXXX,
                "baseline": baseline_evt,
                "limit": limit,
                "pass": ok,
            }
        )

    gate["checks"] = checks
    if not have_baseline:
        gate["verdict"] = "no-baseline"
    else:
        gate["verdict"] = "pass" if passed else "fail"
        # The plan's documented fallback when the gate fails: keep the
        # tail-offending tensor one precision tier higher (llama.cpp _M
        # discipline). We surface that as the actionable remediation.
        if not passed:
            gate["fallback"] = (
                "Tail bound exceeds the ledgered budget: keep the tail-offending "
                "tensor one precision tier higher (int4->int8 or int8->bf16) and "
                "re-measure (plan section 9.7 AF-2 fallback)."
            )
    return gate


# --------------------------------------------------------------------------- #
# Input parsing                                                               #
# --------------------------------------------------------------------------- #


def parse_values_text(text: str) -> list[float]:
    """Parse per-doc CER from free text.

    Accepts (auto-detected):
      * a JSON array of numbers, optionally a JSON object of {doc_id: cer}, or a
        JSON array of {"cer": x} / {"doc": ..., "cer": x} records;
      * otherwise, one number per line (commas/whitespace separated), '#'
        comments and blank lines ignored.
    """
    stripped = text.strip()
    if not stripped:
        return []
    # Try JSON first.
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        obj = None
    if obj is not None:
        return _coerce_json_values(obj)
    # Fallback: line/token oriented.
    vals: list[float] = []
    for raw in stripped.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        for tok in line.replace(",", " ").split():
            vals.append(float(tok))
    return vals


def _coerce_json_values(obj: object) -> list[float]:
    if isinstance(obj, dict):
        # {doc_id: cer} or a single {"cer": x}
        if "cer" in obj and isinstance(obj["cer"], (int, float)):
            return [float(obj["cer"])]
        return [float(v) for v in obj.values()]
    if isinstance(obj, list):
        out: list[float] = []
        for item in obj:
            if isinstance(item, (int, float)):
                out.append(float(item))
            elif isinstance(item, dict) and "cer" in item:
                out.append(float(item["cer"]))
            else:
                raise ValueError(f"cannot read CER from list item {item!r}")
        return out
    if isinstance(obj, (int, float)):
        return [float(obj)]
    raise ValueError(f"unsupported JSON shape for CER input: {type(obj).__name__}")


def load_values(args: argparse.Namespace) -> list[float]:
    if args.values:
        return [float(v) for v in args.values]
    if args.input:
        text = Path(args.input).read_text(encoding="utf-8")
        return parse_values_text(text)
    if args.stdin or not sys.stdin.isatty():
        return parse_values_text(sys.stdin.read())
    raise ValueError(
        "no input: pass --values, --input FILE, or pipe CER on stdin (--stdin)"
    )


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="af2_tail_risk.py",
        description=(
            "AF-2 tail-risk monitor: mean / CVaR_alpha / EVT-p999 over per-document "
            "CER. The release scorecard gates on the CVaR/EVT bound, NOT the mean "
            "(plan section 9.7)."
        ),
    )
    src = p.add_argument_group("input (choose one)")
    src.add_argument("--input", metavar="FILE", help="read CER from FILE (JSON array/object or one-per-line)")
    src.add_argument("--stdin", action="store_true", help="read CER from stdin")
    src.add_argument(
        "--values",
        nargs="+",
        type=float,
        metavar="CER",
        help="inline per-document CER values",
    )

    cfg = p.add_argument_group("parameters")
    cfg.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help=f"CVaR worst-fraction (default {DEFAULT_ALPHA} = worst 10%% of docs)",
    )
    cfg.add_argument(
        "--evt-q",
        type=float,
        default=DEFAULT_EVT_Q,
        help=f"EVT target quantile (default {DEFAULT_EVT_Q} = p99.9 document)",
    )
    cfg.add_argument(
        "--pot-frac",
        type=float,
        default=DEFAULT_POT_FRAC,
        help=f"peaks-over-threshold exceedance fraction for the GPD fit (default {DEFAULT_POT_FRAC})",
    )

    gate = p.add_argument_group("release gate (optional; gates on tail, not mean)")
    gate.add_argument("--baseline-cvar", type=float, help="f32 baseline CVaR_alpha to gate against")
    gate.add_argument("--baseline-evt", type=float, help="f32 baseline EVT-p999 to gate against")
    gate.add_argument(
        "--budget",
        type=float,
        default=0.0,
        help="ledgered tolerance the candidate tail bound may exceed the baseline by",
    )

    out = p.add_argument_group("output")
    out.add_argument(
        "--pretty",
        action="store_true",
        help="pretty-print the JSON record (default: single-line NDJSON)",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not (0.0 < args.alpha <= 1.0):
        print(f"ERROR: --alpha must be in (0, 1], got {args.alpha}", file=sys.stderr)
        return EXIT_USAGE
    if not (0.0 < args.evt_q < 1.0):
        print(f"ERROR: --evt-q must be in (0, 1), got {args.evt_q}", file=sys.stderr)
        return EXIT_USAGE
    if not (0.0 < args.pot_frac < 1.0):
        print(f"ERROR: --pot-frac must be in (0, 1), got {args.pot_frac}", file=sys.stderr)
        return EXIT_USAGE

    try:
        vals = load_values(args)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_USAGE

    if not vals:
        print("ERROR: empty CER input", file=sys.stderr)
        return EXIT_USAGE

    try:
        report = compute_report(vals, alpha=args.alpha, evt_q=args.evt_q, pot_frac=args.pot_frac)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_USAGE

    if args.baseline_cvar is not None or args.baseline_evt is not None:
        report.gate = apply_gate(
            report,
            baseline_cvar=args.baseline_cvar,
            baseline_evt=args.baseline_evt,
            budget=args.budget,
        )

    record = report.to_record()
    if args.pretty:
        print(json.dumps(record, indent=2, sort_keys=True))
    else:
        print(json.dumps(record, sort_keys=True))

    if report.gate and report.gate.get("verdict") == "fail":
        return EXIT_GATE_FAILED
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
