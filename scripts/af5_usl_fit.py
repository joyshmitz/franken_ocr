#!/usr/bin/env python3
"""AF-5 — Universal Scalability Law (USL) fitter for many-core pool sizing.

OFFLINE TOOLING ONLY. franken_ocr's Rust engine NEVER invokes this script. This
is the human-run, out-of-band step that turns a thread sweep — measured by the
Rust bench harness (bd-2mo.21, ``focr`` decode-GEMV / prefill-GEMM sweep) — into
the ``pool_sizing`` table the runtime bakes in. The runtime's deterministic
fallback is the *physical-core count*; it never depends on this fit having run
(plan §6.9, §9.7 AF-5; AGENTS.md doctrine #5, #8).

THE MODEL (transcribed verbatim from plan §6.9 / §9.7)
------------------------------------------------------
The Universal Scalability Law gives the speedup ``C(N)`` of a workload run on
``N`` parallel workers relative to ``N = 1``::

                              N
    C(N) = ----------------------------------------
            1 + alpha * (N - 1) + beta * N * (N - 1)

  * ``alpha`` (contention / serialization) is the Amdahl term: the fraction of
    work that cannot run in parallel. It makes ``C(N)`` *saturate*.
  * ``beta``  (cross-core coherency / crosstalk) is the *retrograde* term unique
    to USL: coordination cost that grows as ``N*(N-1)`` pairwise interactions.
    A non-zero ``beta`` makes ``C(N)`` turn over and DROP past a peak.

Amdahl (``beta = 0``) only saturates; it never regresses, so it cannot model the
measured anti-win of over-threading memory-bandwidth-bound decode. USL can. This
is why §6.9 mandates USL, not Amdahl, for the decode pool.

THE PEAK (closed form)
----------------------
``C(N)`` is maximized (over the reals, ``beta > 0``) at::

    N* = sqrt((1 - alpha) / beta)

We then take the integer pool size as the best of ``floor(N*)`` / ``ceil(N*)``
under the *sampled-and-extrapolated* curve, clamped to ``[1, num_cpus]``. On a
64-core Threadripper, decode (bandwidth-bound, ``beta``-dominated) peaks far
below 64 (~8-16 effective cores); prefill (compute-bound, ``beta`` ~ 0) peaks
much higher. **Cap each pool at its USL peak, not at ``num_cpus``** — that cap
is the whole AF-5 win.

WHY A POLYNOMIAL FIT (no SciPy / no NumPy)
-----------------------------------------
Substituting ``C_i = T_i / T_1`` (speedup at ``N_i`` threads relative to one)
and rearranging the USL into a *deficiency* form linearizes it exactly::

    N_i / C_i - 1 = alpha * (N_i - 1) + beta * N_i * (N_i - 1)

The left side is observed; the right side is linear in (``alpha``, ``beta``)
with the two known regressors ``x1 = (N_i - 1)`` and ``x2 = N_i*(N_i - 1)`` and
NO intercept (USL pins ``C(1) = 1`` by construction). So the fit is an ordinary
least-squares solve of a 2x2 normal-equation system — closed form, deterministic,
stdlib-only, no third-party dependency. (This is the canonical Gunther USL
linearization; we keep the intercept pinned at 0 to honor ``C(1)=1`` exactly.)

This is the *seed* fit. We then optionally refine (``--refine``) with a few
Gauss-Newton steps on the NONLINEAR residual ``C_i - C_hat(N_i)`` directly in
speedup space, which is what we actually report R^2 against — the linearized fit
can be biased by the ``1/C_i`` transform at noisy high-``N`` points. If refine
does not improve the nonlinear SSE it is discarded (monotone, never worse).

INPUT
-----
A JSON file (``--samples``) OR stdin, of the shape::

    {
      "arch": "threadripper-7980x",
      "op_class": "decode_gemv",
      "num_cpus": 64,
      "physical_cores": 64,
      "samples": [
        {"n": 1,  "throughput": 1.000, "cv_pct": 0.8},
        {"n": 2,  "throughput": 1.93,  "cv_pct": 1.1},
        ...
        {"n": 64, "throughput": 6.10,  "cv_pct": 4.9}
      ]
    }

``throughput`` is any consistent rate (tokens/s, GEMV/s, GFLOP/s) — only ratios
matter, so absolute units cancel. ``cv_pct`` (coefficient of variation across the
best-of-N timing repeats, §9.3) is optional; if any sample exceeds ``--cv-max``
(default 5%) the run is flagged ``noisy`` and the decision is marked advisory.

If no ``--samples`` is given and stdin is a TTY, a synthetic self-check instance
is fit and the recovered (alpha, beta) are asserted against the ground truth —
this is what makes the script ``py_compile`` *and* smoke-runnable with no inputs.

OUTPUT
------
One JSON object on stdout (the ``pool_sizing`` row the Rust converter bakes; the
schema ``focr robot backends`` reports, bd-2mo.2), e.g.::

    {
      "schema_version": 1,
      "arch": "threadripper-7980x",
      "op_class": "decode_gemv",
      "alpha": 0.041, "beta": 0.0019,
      "peak_n_real": 12.34, "peak_n": 12,
      "r2": 0.998, "rmse": 0.07,
      "speedup_at_peak": 7.21, "speedup_at_num_cpus": 6.10,
      "num_cpus": 64, "physical_cores": 64,
      "cap_is_win": true, "predicted_gain_pct": 18.2,
      "noisy": false, "degenerate": false,
      "fallback_used": false, "chosen_pool_n": 12,
      "decision": "cap-at-usl-peak"
    }

DECISION / FALLBACK (deterministic, AGENTS.md #5)
-------------------------------------------------
  * ``degenerate``      -> beta <= 0 or fit unusable    -> fallback = physical cores.
  * ``poor fit``        -> non-finite or low R^2         -> fallback = physical cores.
  * ``noisy``           -> any cv_pct > cv-max          -> peak kept but advisory.
  * ``cap_is_win``      -> predicted speedup(peak) >= speedup(num_cpus)
                           (the AF-5 PROOF OBLIGATION, predicted here, MEASURED
                           by the Rust E2E in bd-1xfa.5.1).
  * On any error / unusable fit the ``chosen_pool_n`` is the physical-core count
    and ``fallback_used`` is true. The runtime NEVER requires this script.

The proof obligation (measured throughput at chosen N >= throughput at num_cpus)
is *predicted* here and PROVEN by the Rust E2E harness; this script's job is to
produce the candidate ``chosen_pool_n`` and the transparency numbers.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SCHEMA_VERSION = 1
DEFAULT_CV_MAX = 5.0
MIN_R2 = 0.90


@dataclass
class Sample:
    """One thread-sweep data point: N workers -> throughput (any rate)."""

    n: int
    throughput: float
    cv_pct: float = 0.0


@dataclass
class Sweep:
    """A full per-(arch, op-class) thread sweep plus host facts."""

    arch: str
    op_class: str
    num_cpus: int
    physical_cores: int
    samples: list[Sample] = field(default_factory=list)


@dataclass
class UslFit:
    """The fitted USL plus the derived pool-sizing decision."""

    alpha: float
    beta: float
    peak_n_real: float
    peak_n: int
    r2: float
    rmse: float
    speedup_at_peak: float
    speedup_at_num_cpus: float
    noisy: bool
    degenerate: bool


# --------------------------------------------------------------------------- #
# Core USL math                                                               #
# --------------------------------------------------------------------------- #


def usl_speedup(n: float, alpha: float, beta: float) -> float:
    """C(N) = N / (1 + alpha*(N-1) + beta*N*(N-1)) — the USL itself (§6.9)."""
    denom = 1.0 + alpha * (n - 1.0) + beta * n * (n - 1.0)
    if denom <= 0.0:
        # Past the model's validity floor; treat as zero useful speedup.
        return 0.0
    return n / denom


def usl_peak_real(alpha: float, beta: float) -> float:
    """Closed-form argmax over the reals: N* = sqrt((1-alpha)/beta).

    Requires ``beta > 0`` (a real retrograde term) and ``alpha < 1`` for a peak
    above N=1. Returns ``+inf`` when ``beta == 0`` (pure-Amdahl, never regresses)
    so the caller clamps to ``num_cpus``.
    """
    if beta <= 0.0:
        return math.inf
    num = 1.0 - alpha
    if num <= 0.0:
        return 1.0
    return math.sqrt(num / beta)


def _normalize_to_speedup(samples: list[Sample]) -> list[tuple[float, float]]:
    """Return [(N, C=throughput/throughput@N=1)], using the N=1 sample as base.

    If no N=1 sample exists, the smallest-N sample is used as the base and its
    speedup is scaled to that N (so C(base_n) == base_n under the convention).
    """
    if not samples:
        return []
    base = min(samples, key=lambda s: s.n)
    if base.throughput <= 0.0:
        raise ValueError("base-N throughput must be positive")
    # USL is defined relative to a single worker. If the base is N=1, this is the
    # raw ratio. If the base is N=k>1, we anchor C(k)=k (best-available proxy).
    scale = base.n / base.throughput
    return [(float(s.n), s.throughput * scale) for s in samples]


def fit_usl_linear(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Seed fit: OLS of the linearized USL deficiency with the intercept pinned.

    Linearization (exact): N/C - 1 = alpha*(N-1) + beta*N*(N-1).
    Regressors x1=(N-1), x2=N*(N-1); response y=N/C-1; NO intercept (C(1)=1).
    Solve the 2x2 normal equations [[Sx1x1, Sx1x2],[Sx1x2, Sx2x2]] [a;b] = [Sx1y; Sx2y].
    """
    s11 = s12 = s22 = s1y = s2y = 0.0
    used = 0
    for n, c in points:
        if n <= 1.0:
            # n=1 carries no information (both regressors are 0); skip it.
            continue
        if c <= 0.0:
            continue
        x1 = n - 1.0
        x2 = n * (n - 1.0)
        y = n / c - 1.0
        s11 += x1 * x1
        s12 += x1 * x2
        s22 += x2 * x2
        s1y += x1 * y
        s2y += x2 * y
        used += 1
    if used < 2:
        raise ValueError("need at least two N>1 samples to fit USL")
    det = s11 * s22 - s12 * s12
    if abs(det) < 1e-18:
        raise ValueError("degenerate normal-equation system (collinear regressors)")
    alpha = (s1y * s22 - s2y * s12) / det
    beta = (s11 * s2y - s12 * s1y) / det
    return alpha, beta


def _nonlinear_sse(points: list[tuple[float, float]], alpha: float, beta: float) -> float:
    """Sum of squared residuals in SPEEDUP space (what we report R^2 against)."""
    sse = 0.0
    for n, c in points:
        sse += (c - usl_speedup(n, alpha, beta)) ** 2
    return sse


def refine_usl_gauss_newton(
    points: list[tuple[float, float]],
    alpha0: float,
    beta0: float,
    iters: int = 12,
) -> tuple[float, float]:
    """Refine (alpha, beta) by Gauss-Newton on the nonlinear speedup residual.

    Damped, monotone: each step is accepted only if it lowers the nonlinear SSE,
    otherwise the step is halved (a simple Levenberg-style backtrack). The seed
    is never made worse, so this is safe to always run.
    """
    alpha, beta = alpha0, beta0
    best_sse = _nonlinear_sse(points, alpha, beta)
    for _ in range(iters):
        # Build the 2x2 Gauss-Newton normal equations from analytic Jacobians.
        jtj00 = jtj01 = jtj11 = 0.0
        jtr0 = jtr1 = 0.0
        for n, c in points:
            denom = 1.0 + alpha * (n - 1.0) + beta * n * (n - 1.0)
            if denom <= 0.0:
                continue
            chat = n / denom
            r = c - chat
            # d(chat)/d(alpha) = -n*(n-1)/denom^2 ; d(chat)/d(beta) = -n*n*(n-1)/denom^2
            d_alpha = -n * (n - 1.0) / (denom * denom)
            d_beta = -n * n * (n - 1.0) / (denom * denom)
            jtj00 += d_alpha * d_alpha
            jtj01 += d_alpha * d_beta
            jtj11 += d_beta * d_beta
            jtr0 += d_alpha * r
            jtr1 += d_beta * r
        det = jtj00 * jtj11 - jtj01 * jtj01
        if abs(det) < 1e-20:
            break
        # Solve J^T J delta = J^T r  (note residual r = observed - model, so the
        # step direction uses +(J^T r) because d(model)/dparam was folded above).
        d_a = (jtr0 * jtj11 - jtr1 * jtj01) / det
        d_b = (jtj00 * jtr1 - jtj01 * jtr0) / det
        step = 1.0
        improved = False
        for _ in range(8):  # backtracking line search
            na = alpha + step * d_a
            nb = beta + step * d_b
            sse = _nonlinear_sse(points, na, nb)
            if sse < best_sse and math.isfinite(sse):
                alpha, beta, best_sse = na, nb, sse
                improved = True
                break
            step *= 0.5
        if not improved:
            break
    return alpha, beta


def _r2_rmse(points: list[tuple[float, float]], alpha: float, beta: float) -> tuple[float, float]:
    """Coefficient of determination + RMSE in speedup space."""
    ys = [c for _, c in points]
    mean = sum(ys) / len(ys)
    ss_tot = sum((c - mean) ** 2 for c in ys)
    ss_res = _nonlinear_sse(points, alpha, beta)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    rmse = math.sqrt(ss_res / len(points))
    return r2, rmse


def _choose_integer_peak(alpha: float, beta: float, n_real: float, num_cpus: int) -> int:
    """Pick the integer pool size: best of floor/ceil(N*) under C(N), clamped."""
    if not math.isfinite(n_real):
        return max(1, num_cpus)
    lo = max(1, int(math.floor(n_real)))
    hi = max(1, int(math.ceil(n_real)))
    candidates = sorted({lo, hi, min(hi, num_cpus), min(lo, num_cpus)})
    candidates = [min(max(1, c), num_cpus) for c in candidates]
    best = max(candidates, key=lambda k: usl_speedup(float(k), alpha, beta))
    return best


# --------------------------------------------------------------------------- #
# Top-level fit + decision                                                     #
# --------------------------------------------------------------------------- #


def fit_sweep(sweep: Sweep, cv_max: float = DEFAULT_CV_MAX, refine: bool = True) -> UslFit:
    """Fit the USL to one sweep and derive the integer peak + win prediction."""
    points = _normalize_to_speedup(sweep.samples)
    noisy = any(s.cv_pct > cv_max for s in sweep.samples)

    try:
        alpha, beta = fit_usl_linear(points)
        if refine:
            alpha, beta = refine_usl_gauss_newton(points, alpha, beta)
    except ValueError:
        # Unfittable -> degenerate; caller falls back to physical cores.
        return UslFit(
            alpha=float("nan"),
            beta=float("nan"),
            peak_n_real=float("nan"),
            peak_n=max(1, sweep.physical_cores),
            r2=0.0,
            rmse=float("inf"),
            speedup_at_peak=float("nan"),
            speedup_at_num_cpus=float("nan"),
            noisy=noisy,
            degenerate=True,
        )

    # A non-positive beta (no retrograde term) or alpha>=1 (no parallel fraction)
    # means USL predicts no interior peak -> degenerate, fall back.
    degenerate = (not math.isfinite(beta)) or beta <= 0.0 or alpha >= 1.0
    n_real = usl_peak_real(alpha, beta)
    peak_n = (
        max(1, sweep.physical_cores)
        if degenerate
        else _choose_integer_peak(alpha, beta, n_real, sweep.num_cpus)
    )
    r2, rmse = _r2_rmse(points, alpha, beta)
    speedup_at_peak = usl_speedup(float(peak_n), alpha, beta)
    speedup_at_num_cpus = usl_speedup(float(sweep.num_cpus), alpha, beta)
    return UslFit(
        alpha=alpha,
        beta=beta,
        peak_n_real=n_real,
        peak_n=peak_n,
        r2=r2,
        rmse=rmse,
        speedup_at_peak=speedup_at_peak,
        speedup_at_num_cpus=speedup_at_num_cpus,
        noisy=noisy,
        degenerate=degenerate,
    )


def decide(sweep: Sweep, fit: UslFit) -> dict:
    """Produce the ``pool_sizing`` row + deterministic fallback decision."""
    poor_fit = (not fit.degenerate) and ((not math.isfinite(fit.r2)) or fit.r2 < MIN_R2)
    fallback_used = fit.degenerate or poor_fit
    if fallback_used:
        chosen = max(1, sweep.physical_cores)
        decision = "fallback-physical-cores"
        cap_is_win = False
        predicted_gain_pct = 0.0
    else:
        chosen = fit.peak_n
        # Predicted AF-5 proof obligation: speedup(peak) >= speedup(num_cpus).
        cap_is_win = fit.speedup_at_peak >= fit.speedup_at_num_cpus
        base = fit.speedup_at_num_cpus
        predicted_gain_pct = (
            100.0 * (fit.speedup_at_peak - base) / base if base > 0.0 else 0.0
        )
        if chosen >= sweep.num_cpus:
            decision = "no-cap-needed"  # peak at/above num_cpus (compute-bound)
        else:
            decision = "cap-at-usl-peak"

    row = {
        "schema_version": SCHEMA_VERSION,
        "arch": sweep.arch,
        "op_class": sweep.op_class,
        "alpha": _round(fit.alpha),
        "beta": _round(fit.beta, 6),
        "peak_n_real": _round(fit.peak_n_real),
        "peak_n": fit.peak_n,
        "r2": _round(fit.r2),
        "rmse": _round(fit.rmse),
        "speedup_at_peak": _round(fit.speedup_at_peak),
        "speedup_at_num_cpus": _round(fit.speedup_at_num_cpus),
        "num_cpus": sweep.num_cpus,
        "physical_cores": sweep.physical_cores,
        "cap_is_win": bool(cap_is_win),
        "predicted_gain_pct": _round(predicted_gain_pct, 1),
        "noisy": bool(fit.noisy),
        "degenerate": bool(fit.degenerate),
        "fallback_used": bool(fallback_used),
        "chosen_pool_n": chosen,
        "decision": decision,
    }
    return row


def _round(x: float, ndigits: int = 4) -> Optional[float]:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    return round(float(x), ndigits)


# --------------------------------------------------------------------------- #
# I/O                                                                         #
# --------------------------------------------------------------------------- #


def parse_sweep(obj: dict) -> Sweep:
    """Validate and parse the input JSON into a Sweep."""
    if "samples" not in obj or not isinstance(obj["samples"], list):
        raise ValueError("input must carry a non-empty 'samples' list")
    samples = []
    for raw in obj["samples"]:
        samples.append(
            Sample(
                n=int(raw["n"]),
                throughput=float(raw["throughput"]),
                cv_pct=float(raw.get("cv_pct", 0.0)),
            )
        )
    if not samples:
        raise ValueError("'samples' is empty")
    num_cpus = int(obj.get("num_cpus", max(s.n for s in samples)))
    physical = int(obj.get("physical_cores", num_cpus))
    return Sweep(
        arch=str(obj.get("arch", "unknown")),
        op_class=str(obj.get("op_class", "unknown")),
        num_cpus=num_cpus,
        physical_cores=physical,
        samples=samples,
    )


def _synthetic_selfcheck() -> int:
    """Fit a known decode-like instance and assert recovery; print the row.

    Used when no input is supplied (smoke / py_compile run). Ground-truth
    (alpha, beta) chosen so the USL peak lands well below num_cpus, mimicking
    bandwidth-bound decode on a 64-core Threadripper.
    """
    true_alpha, true_beta = 0.040, 0.0020
    num_cpus = 64
    ns = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
    samples = [
        {"n": n, "throughput": usl_speedup(float(n), true_alpha, true_beta), "cv_pct": 0.5}
        for n in ns
    ]
    sweep = parse_sweep(
        {
            "arch": "selfcheck-threadripper",
            "op_class": "decode_gemv",
            "num_cpus": num_cpus,
            "physical_cores": num_cpus,
            "samples": samples,
        }
    )
    fit = fit_sweep(sweep)
    row = decide(sweep, fit)
    print(json.dumps(row, sort_keys=True))

    jagged_sweep = parse_sweep(
        {
            "arch": "jagged-selfcheck",
            "op_class": "decode_gemv",
            "num_cpus": 32,
            "physical_cores": 16,
            "samples": [
                {"n": 1, "throughput": 1.0},
                {"n": 2, "throughput": 5.0},
                {"n": 4, "throughput": 1.2},
                {"n": 8, "throughput": 6.0},
                {"n": 16, "throughput": 1.1},
                {"n": 32, "throughput": 4.0},
            ],
        }
    )
    jagged_fit = fit_sweep(jagged_sweep)
    jagged_row = decide(jagged_sweep, jagged_fit)
    poor_fit_ok = (
        (not jagged_fit.degenerate)
        and jagged_fit.r2 < MIN_R2
        and jagged_row["fallback_used"]
        and jagged_row["decision"] == "fallback-physical-cores"
        and jagged_row["chosen_pool_n"] == 16
    )

    # Assertions: the fit must recover the truth and cap below num_cpus.
    expected_peak = usl_peak_real(true_alpha, true_beta)  # ~ sqrt(0.96/0.002) ~ 21.9
    ok = (
        abs(fit.alpha - true_alpha) < 5e-3
        and abs(fit.beta - true_beta) < 5e-4
        and abs(fit.peak_n_real - expected_peak) < 1.5
        and fit.peak_n < num_cpus
        and row["cap_is_win"]
        and fit.r2 > 0.999
        and poor_fit_ok
    )
    diag = {
        "selfcheck": "af5_usl_fit",
        "recovered_alpha": _round(fit.alpha),
        "recovered_beta": _round(fit.beta, 6),
        "expected_peak_real": _round(expected_peak),
        "poor_fit_r2": _round(jagged_fit.r2),
        "poor_fit_fallback_ok": bool(poor_fit_ok),
        "ok": bool(ok),
    }
    print(json.dumps(diag, sort_keys=True), file=sys.stderr)
    return 0 if ok else 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit the Universal Scalability Law to a thread sweep and "
        "report the USL peak (the AF-5 pool cap). Stdlib-only, offline tooling.",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default=None,
        help="Path to the sweep JSON (default: stdin; synthetic self-check if a TTY).",
    )
    parser.add_argument(
        "--cv-max",
        type=float,
        default=DEFAULT_CV_MAX,
        help="Max per-sample coefficient-of-variation %% before flagging 'noisy' (default 5).",
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Skip the Gauss-Newton nonlinear refinement (use the linearized seed only).",
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="Run the synthetic recovery self-check and exit (no input needed).",
    )
    args = parser.parse_args(argv)

    if args.selfcheck:
        return _synthetic_selfcheck()

    try:
        if args.samples:
            obj = json.JSONDecoder().decode(Path(args.samples).read_text(encoding="utf-8"))
        elif not sys.stdin.isatty():
            obj = json.load(sys.stdin)
        else:
            # No input at all -> run the self-check so the script is always runnable.
            return _synthetic_selfcheck()
        sweep = parse_sweep(obj)
    except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": f"bad input: {exc}"}, sort_keys=True), file=sys.stderr)
        return 2

    fit = fit_sweep(sweep, cv_max=args.cv_max, refine=not args.no_refine)
    row = decide(sweep, fit)
    print(json.dumps(row, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
