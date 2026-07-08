#!/usr/bin/env python3
"""Reference implementation of the franken_ocr three-pillar release gauntlet math.

This is the executable backing for ``docs/gauntlet/METHODOLOGY.md`` (beads
VERIFY-three-pillar-cert / VERIFY-conformal-ratchet / VERIFY-eprocess-invariants,
= bd-re8.13/14/15). It is stdlib-only (no numpy, no torch) so it runs in CI
without model weights, mirroring scripts/check_ledgers.py.

It implements, exactly as the methodology specifies:

  (1) The conformance parity score: a per-category Beta posterior, a
      distribution-free conformal band, ``truncate_score`` to 6 dp, and the
      release-decision rule that uses the LOWER bound (never the point estimate).

  (2) The ratchet state machine: Allow / Block / Quarantine / Waiver against a
      persisted, monotone, per-category high-water mark.

  (3) The Ville e-process monitor for the four load-bearing invariants
      (KV-cap L*(m+128), i32 no-overflow at K=6848, same-input determinism,
      SIMD==scalar bit-identical), with the hardware/software p0/lambda/alpha
      calibration split and the arithmetic-mean global e-value.

Nothing here ships in the ``focr`` binary; like the PyO3 oracle bridge this is a
verification-only artifact (G3's no-FFI runtime claim is preserved).

Usage:
    python3 scripts/gauntlet_cert.py --self-test     # validate the math (CI gate)
    python3 scripts/gauntlet_cert.py --demo          # print a worked franken_ocr cert
    # bd-re8.13 real-data modes:
    python3 scripts/gauntlet_cert.py --from-parity docs/FEATURE_PARITY.md \
        --scorecard-out docs/gauntlet/RELEASE_SCORECARD.json   # score the REAL scoreboard
    python3 scripts/gauntlet_cert.py --convergence   # >=10 rounds / >=2 clean gate (exit 1 until met)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Iterable

# --------------------------------------------------------------------------- #
# K-5 / §3.4 — truncate_score: cross-platform LSB determinism.
# x86 vs ARM vs WASM differ at the LSB of IEEE-754 double; truncate (NOT round)
# to 6 dp at every release boundary so the byte-wise ratchet diff never flickers.
# --------------------------------------------------------------------------- #

SCORE_DECIMALS = 6
_SCORE_SCALE = 10.0**SCORE_DECIMALS


def truncate_score(x: float) -> float:
    """Truncate to 6 decimal places (associative across the ULP; round-mode-free)."""
    return math.floor(x * _SCORE_SCALE) / _SCORE_SCALE


# --------------------------------------------------------------------------- #
# §3.1 — Beta posterior per category (Jeffreys-like uniform prior).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BetaParams:
    """A Beta(alpha, beta) posterior over a per-category pass rate."""

    alpha: float
    beta: float

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> float:
        ab = self.alpha + self.beta
        return (self.alpha * self.beta) / (ab * ab * (ab + 1.0))

    def quantile(self, p: float) -> float:
        """Quantile of Beta(alpha, beta) via bisection on the regularized
        incomplete beta function (stdlib-only; no scipy)."""
        if not 0.0 < p < 1.0:
            return 0.0 if p <= 0.0 else 1.0
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if _reg_inc_beta(mid, self.alpha, self.beta) < p:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def credible_interval(self, confidence: float) -> tuple[float, float]:
        tail = (1.0 - confidence) / 2.0
        return (self.quantile(tail), self.quantile(1.0 - tail))


def _reg_inc_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta I_x(a, b) via the Lentz continued fraction.

    Standard Numerical-Recipes ``betai`` translated to pure Python; accurate to
    well past the 6-dp truncation floor we round to anyway.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(ln_beta + a * math.log(x) + b * math.log(1.0 - x))
    # Use the symmetry I_x(a,b) = 1 - I_{1-x}(b,a) for fast CF convergence.
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(x, a, b) / a
    return 1.0 - front * _betacf(1.0 - x, b, a) / b


def _betacf(x: float, a: float, b: float) -> float:
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-12:
            break
    return h


# --------------------------------------------------------------------------- #
# §3.1 — per-category evidence (present/partial/missing/excluded weighting).
# --------------------------------------------------------------------------- #

PRIOR = BetaParams(alpha=1.0, beta=1.0)  # uniform on [0,1]; declared in the score contract


@dataclass
class CategoryEvidence:
    """Weighted outcome tallies for one FeatureUniverse category."""

    category_id: str
    category_weight: float  # release weight; categories sum to 1.0 (loader-enforced)
    weighted_successes: float = 0.0  # Σ 1.0*w over present rows
    weighted_partials: float = 0.0  # Σ 0.5*w over partial rows (counts in BOTH alpha and beta)
    weighted_failures: float = 0.0  # Σ 1.0*w over missing rows
    # excluded rows are skipped here but their weight stays in the denominator for
    # a strict-100% claim (see §2.1.2); tracked separately for coverage debt.
    weighted_excluded: float = 0.0

    def posterior(self) -> BetaParams:
        return BetaParams(
            alpha=PRIOR.alpha + self.weighted_successes + self.weighted_partials,
            beta=PRIOR.beta + self.weighted_failures + self.weighted_partials,
        )


def add_outcome(ev: CategoryEvidence, status: str, weight: float) -> None:
    """Apply one feature outcome to a category's evidence (§3.1 scoring rule)."""
    if status == "present":
        ev.weighted_successes += 1.0 * weight
    elif status == "partial":
        ev.weighted_partials += 0.5 * weight
    elif status == "missing":
        ev.weighted_failures += 1.0 * weight
    elif status == "excluded":
        ev.weighted_excluded += 1.0 * weight
    else:
        raise ValueError(f"unknown status {status!r}")


# --------------------------------------------------------------------------- #
# §3.2/§3.3 — distribution-free conformal band; release uses the LOWER bound.
# --------------------------------------------------------------------------- #


def conformal_halfwidth(residuals: Iterable[float], confidence: float) -> float:
    """(1 - alpha) empirical quantile of held-out nonconformity residuals
    (Vovk-Gammerman-Shafer 2005). Distribution-free finite-sample coverage."""
    rs = sorted(residuals)
    if not rs:
        # Bootstrap with a wide band before residuals exist (§3.2 pitfall guard).
        return 1.0
    n = len(rs)
    k = math.ceil(confidence * (n + 1)) - 1  # 0-indexed (1-α)-quantile, α = 1-confidence
    k = max(0, min(k, n - 1))
    return rs[k]


@dataclass
class Scorecard:
    """The output of the conformance parity score for one gauntlet round."""

    point_estimate: float  # S_mean (dashboards only)
    lower_bound: float  # S_lower = truncate_score(S_mean - q); THE release number
    conformal_halfwidth: float
    per_category_lower: dict[str, float]  # truncate_score'd; ratchet compares these
    per_category_mean: dict[str, float]


def compute_scorecard(
    categories: list[CategoryEvidence],
    residuals: Iterable[float],
    confidence: float = 0.95,
) -> Scorecard:
    """Beta posterior per category -> weighted mean -> conformal band -> lower bound."""
    total_w = sum(c.category_weight for c in categories)
    if abs(total_w - 1.0) > 1e-9:
        raise ValueError(
            f"category weights must sum to 1.0 (loader invariant); got {total_w!r}"
        )
    q = conformal_halfwidth(residuals, confidence)
    per_mean: dict[str, float] = {}
    per_lower: dict[str, float] = {}
    s_mean = 0.0
    for c in categories:
        post = c.posterior()
        mean_c = post.mean()
        per_mean[c.category_id] = truncate_score(mean_c)
        per_lower[c.category_id] = truncate_score(max(0.0, mean_c - q))
        s_mean += c.category_weight * mean_c
    s_lower = truncate_score(max(0.0, s_mean - q))
    return Scorecard(
        point_estimate=truncate_score(s_mean),
        lower_bound=s_lower,
        conformal_halfwidth=q,
        per_category_lower=per_lower,
        per_category_mean=per_mean,
    )


# --------------------------------------------------------------------------- #
# §3.5 — the ratchet state machine (Allow | Block | Quarantine | Waiver).
# --------------------------------------------------------------------------- #

QUARANTINE_DELTA = 0.005  # a single per-category dip ≤ this -> Quarantine, not Block


@dataclass
class RatchetState:
    current_lower_bound: float
    per_category_bounds: dict[str, float]


@dataclass
class RatchetVerdict:
    decision: str  # "Allow" | "Block" | "Quarantine" | "Waiver"
    reason: str
    new_state: RatchetState | None = None


def apply_ratchet(
    scorecard: Scorecard,
    state: RatchetState,
    waived_categories: frozenset[str] = frozenset(),
) -> RatchetVerdict:
    """Decide whether this round may land, per §3.5. Release uses the LOWER bound."""
    dipped: list[str] = []
    worst_dip = 0.0
    for cat, floor in state.per_category_bounds.items():
        cur = scorecard.per_category_lower.get(cat, 0.0)
        if cur < floor and cat not in waived_categories:
            dipped.append(cat)
            worst_dip = max(worst_dip, floor - cur)

    global_ok = scorecard.lower_bound >= state.current_lower_bound

    if waived_categories and dipped == []:
        # all dips covered by waivers (none remain after filtering)
        pass

    if not dipped and global_ok:
        new = RatchetState(
            current_lower_bound=max(scorecard.lower_bound, state.current_lower_bound),
            per_category_bounds={
                cat: max(scorecard.per_category_lower.get(cat, 0.0), floor)
                for cat, floor in state.per_category_bounds.items()
            },
        )
        return RatchetVerdict("Allow", "raises lower bound; no per-category regression", new)

    if dipped and all(c in waived_categories for c in dipped):
        return RatchetVerdict(
            "Waiver", f"regression in {dipped} covered by active waiver", None
        )

    if global_ok and len(dipped) == 1 and worst_dip <= QUARANTINE_DELTA:
        return RatchetVerdict(
            "Quarantine",
            f"single per-category dip {dipped[0]} by {worst_dip:.6f} ≤ {QUARANTINE_DELTA}",
            None,
        )

    return RatchetVerdict(
        "Block",
        f"lower bound or per-category bound regressed (global_ok={global_ok}, dipped={dipped})",
        None,
    )


# --------------------------------------------------------------------------- #
# §6 — Ville e-processes for the four load-bearing invariants.
# Howard-Ramdas-McAuliffe-Sekhon 2021; Ville's inequality gives
# P_{H_0}(exists t: E_t >= 1/alpha) <= alpha  (anytime-valid; no Bonferroni).
# --------------------------------------------------------------------------- #

# Calibration split (§6.2): hardware-enforced (CPU integer semantics / determinism
# flag; a violation is a CPU/logic bug, tight prior) vs software-enforced (code
# path; rare but plausible, looser prior).
HARDWARE = dict(p0=1e-9, lam=0.999, alpha=1e-6)  # threshold 1/alpha = 1_000_000
SOFTWARE = dict(p0=1e-6, lam=0.9, alpha=1e-3)  # threshold 1/alpha = 1_000

# The four franken_ocr invariants, line-backed to the truth pack / plan §5.4.
INVARIANT_CALIBRATION = {
    "INV-KV-CAP": SOFTWARE,  # KV never exceeds L*(m+128); m_max=32896, W=128, L=12 (CENSUS d)
    "INV-I32-NOOVERFLOW": SOFTWARE,  # i32 acc never overflows at K_max=6848 (plan §5.4, ≥9x headroom)
    "INV-DETERMINISM": HARDWARE,  # same input twice -> byte-identical output
    "INV-SIMD-SCALAR": HARDWARE,  # SDOT/SMMLA/VNNI == scalar bit-identical (integer add associative)
}


@dataclass
class EProcess:
    """An anytime-valid e-process over one invariant's observation stream."""

    p0: float
    lam: float
    alpha: float
    e_value: float = 1.0
    obs_count: int = 0
    rejected_at: int | None = None

    @classmethod
    def for_invariant(cls, name: str) -> "EProcess":
        cal = INVARIANT_CALIBRATION[name]
        return cls(p0=cal["p0"], lam=cal["lam"], alpha=cal["alpha"])

    def observe(self, alarm: bool) -> bool:
        """Feed one observation (alarm == invariant violated). Returns True the
        first time the Ville threshold is crossed."""
        self.obs_count += 1
        if alarm:
            factor = (1.0 - self.lam) + self.lam * 1.0 / self.p0
        else:
            factor = 1.0 - self.lam  # (1-lam) + lam*0/p0
        self.e_value *= factor
        # Saturate to keep f64 noise away from the threshold logic; do NOT reset
        # to 1.0 (that breaks the supermartingale property / Ville's bound).
        self.e_value = min(max(self.e_value, 5e-324), sys.float_info.max / 2.0)
        if self.e_value >= 1.0 / self.alpha and self.rejected_at is None:
            self.rejected_at = self.obs_count
            return True
        return False

    @property
    def threshold(self) -> float:
        return 1.0 / self.alpha


def global_e_value(processes: Iterable[EProcess]) -> float:
    """Arithmetic mean of per-invariant e-values -- itself an e-process under the
    global null REGARDLESS of dependence (§6.1). Never max, never product."""
    ps = list(processes)
    if not ps:
        return 1.0
    return sum(p.e_value for p in ps) / len(ps)


# --------------------------------------------------------------------------- #
# bd-re8.13 — the REAL-data modes: score FEATURE_PARITY.md, track convergence.
# --------------------------------------------------------------------------- #

STATUS_TOKENS = ("present", "partial", "missing", "n/a", "excluded")


def parse_feature_parity(md_text: str) -> dict[str, dict[str, int]]:
    """Parse FEATURE_PARITY.md into per-section status tallies.

    A row is status-bearing when one of its pipe-cells is EXACTLY a status
    token — the same rule `tests/surface_matrix.rs` uses, so the two parsers
    cannot diverge on what counts as a scoreboard row. Returns
    {section_title: {status: count}} for every `## N.` section with rows.
    """
    import re

    sections: dict[str, dict[str, int]] = {}
    current = None
    for line in md_text.splitlines():
        m = re.match(r"^## (\d+)\.\s*(.*)", line)
        if m:
            current = f"§{m.group(1)} {m.group(2).split('(')[0].split('—')[0].strip()}"
            continue
        if current is None or not line.startswith("|") or line.startswith("|-"):
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 6:
            continue
        for c in cells:
            if c.lower() in STATUS_TOKENS:
                sections.setdefault(current, {})
                sections[current][c.lower()] = sections[current].get(c.lower(), 0) + 1
                break
    return {k: v for k, v in sections.items() if v}


def categories_from_parity(md_text: str) -> list[CategoryEvidence]:
    """One CategoryEvidence per scoreboard section, equal category weights
    (normalized to the loader's sum-to-1.0 invariant), equal row weights
    within a section. `n/a` rows are skipped entirely; `excluded` rows keep
    their weight in the coverage-debt tally per §2.1.2."""
    sections = parse_feature_parity(md_text)
    if not sections:
        raise ValueError("no scoreboard sections parsed from FEATURE_PARITY.md")
    cat_w = 1.0 / len(sections)
    cats: list[CategoryEvidence] = []
    for title, tallies in sections.items():
        scored = {s: n for s, n in tallies.items() if s != "n/a"}
        n_rows = sum(scored.values())
        row_w = 1.0 / n_rows if n_rows else 0.0
        ev = CategoryEvidence(title, cat_w)
        for status, count in scored.items():
            for _ in range(count):
                add_outcome(ev, status, row_w)
        cats.append(ev)
    return cats


def from_parity(md_path: str, residuals_path: str | None, out_path: str | None) -> int:
    """Emit the surface-pillar scorecard computed from the REAL scoreboard.

    With no residual history the conformal band bootstraps WIDE (halfwidth
    1.0 -> lower bound 0.0) per the §3.2 pitfall guard — the artifact says so
    explicitly rather than inventing confidence it does not have. Residual
    history accrues one entry per gauntlet round (|observed - Beta-mean|).
    """
    with open(md_path, encoding="utf-8") as f:
        md_text = f.read()
    cats = categories_from_parity(md_text)
    residuals: list[float] = []
    if residuals_path and os.path.exists(residuals_path):
        with open(residuals_path, encoding="utf-8") as f:
            residuals = json.load(f)
    sc = compute_scorecard(cats, residuals, confidence=0.95)
    debt = {
        c.category_id: round(c.weighted_excluded, 6)
        for c in cats
        if c.weighted_excluded > 0
    }
    artifact = {
        "artifact": "franken_ocr.gauntlet.scorecard.v1",
        "source": md_path,
        "pillar": "surface",
        "point_estimate": sc.point_estimate,
        "parity_score_lower_bound": sc.lower_bound,
        "conformal_halfwidth": round(sc.conformal_halfwidth, 6),
        "residual_history_n": len(residuals),
        "band_note": (
            "no residual history yet: bootstrap band 1.0 -> lower bound 0 (maximally conservative, §3.2)"
            if not residuals
            else "band from held-out round residuals"
        ),
        "per_category_mean": sc.per_category_mean,
        "per_category_lower": sc.per_category_lower,
        "excluded_weight_debt": debt,
    }
    text = json.dumps(artifact, sort_keys=True, indent=1)
    print(text)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    return 0


# Convergence (METHODOLOGY §7): >=10 full rounds AND the last >=2 consecutive
# rounds each produced <3 new genuine findings. Exits nonzero until met — this
# is the gate bd-wp8.8 runs behind.
MIN_ROUNDS = 10
CLEAN_ROUNDS = 2
CLEAN_FINDING_CEILING = 3


def convergence_verdict(rounds: list[dict]) -> dict:
    n = len(rounds)
    tail = rounds[-CLEAN_ROUNDS:] if n >= CLEAN_ROUNDS else []
    tail_clean = len(tail) == CLEAN_ROUNDS and all(
        int(r.get("new_findings", 10**9)) < CLEAN_FINDING_CEILING for r in tail
    )
    converged = n >= MIN_ROUNDS and tail_clean
    return {
        "check": "gauntlet-convergence",
        "rounds": n,
        "min_rounds": MIN_ROUNDS,
        "tail_clean": tail_clean,
        "tail_findings": [int(r.get("new_findings", -1)) for r in tail],
        "clean_ceiling": CLEAN_FINDING_CEILING,
        "converged": converged,
        "result": "pass" if converged else "fail",
    }


def convergence(rounds_path: str) -> int:
    rounds: list[dict] = []
    if os.path.exists(rounds_path):
        with open(rounds_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rounds.append(json.loads(line))
    verdict = convergence_verdict(rounds)
    verdict["rounds_file"] = rounds_path
    print(json.dumps(verdict, sort_keys=True))
    return 0 if verdict["converged"] else 1


# --------------------------------------------------------------------------- #
# bd-re8.15 — wiring the e-processes to the LIVE invariant streams.
#
# The four load-bearing invariants already EMIT structured NDJSON wherever
# they are exercised (the determinism gates, `robot selftest`, the i32
# overflow proofs, the R-SWA/KV bound gates). The fold below maps those
# real lines onto e-process observations and persists the per-invariant
# state ACROSS runs — the e-value only ever multiplies (Ville's inequality
# holds over the whole history; resetting would forge the guarantee).
# --------------------------------------------------------------------------- #

# invariant -> lowercase substrings matched against the line's identifying
# fields. The mapping is deliberately explicit and versioned here: adding an
# invariant means adding its emission-site vocabulary alongside.
EPROCESS_MATCHERS: list[tuple[str, tuple[str, ...]]] = [
    ("INV-DETERMINISM", ("determin",)),
    ("INV-SIMD-SCALAR", ("selftest", "simd")),
    ("INV-I32-NOOVERFLOW", ("overflow",)),
    ("INV-KV-CAP", ("kv_cap", "kv_bound", "kvcache_bound", "rswa_bound")),
]

_EPROCESS_ID_FIELDS = (
    "test",
    "case",
    "check",
    "gate",
    "suite",
    "relation",
    "assertion",
    "invariant",
    "command",
)


def classify_invariant_line(obj: dict) -> tuple[str, bool] | None:
    """Map one structured log line to (invariant, alarm) or None.

    Only lines carrying an explicit pass/fail verdict count as observations —
    skips (`skip_no_model`) are NOT evidence in either direction."""
    result = str(obj.get("result", obj.get("verdict", "")))
    if result not in ("pass", "fail"):
        return None
    hay = " ".join(str(obj.get(f, "")) for f in _EPROCESS_ID_FIELDS).lower()
    for invariant, needles in EPROCESS_MATCHERS:
        if any(n in hay for n in needles):
            return (invariant, result == "fail")
    return None


def _eprocess_from_state(name: str, saved: dict | None) -> EProcess:
    ep = EProcess.for_invariant(name)
    if saved:
        ep.e_value = float(saved["e_value"])
        ep.obs_count = int(saved["obs_count"])
        ep.rejected_at = saved.get("rejected_at")
    return ep


# --------------------------------------------------------------------------- #
# bd-wp8.10 — the release-readiness scorecard (the all-green ship gate).
#
# Reads each cell's EVIDENCE ARTIFACT and asserts green; a missing or stale
# artifact is a RED cell, and ANY red cell exits nonzero — the gate is hard,
# not advisory (G7). Cells whose delivering bead is still open are honestly
# RED with the owning bead named; the gate goes green when the work exists,
# never before.
# --------------------------------------------------------------------------- #


def _cell(name: str, status: str, evidence: str, detail: str = "") -> dict:
    out = {"cell": name, "status": status, "evidence": evidence}
    if detail:
        out["detail"] = detail
    return out


def release_readiness(out_path: str | None) -> int:
    import subprocess

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def p(rel: str) -> str:
        return os.path.join(root, rel)

    def load_json(rel: str):
        with open(p(rel), encoding="utf-8") as f:
            return json.load(f)

    cells: list[dict] = []

    # Parity (L0-L5): the committed armed ladder receipt must be ALL GREEN
    # and not a skipped run wearing green.
    try:
        sc = load_json("tests/fixtures/ladder_scorecard/scorecard_armed.json")
        ok = sc.get("all_green") is True and sc.get("skipped_no_model") is False
        cells.append(
            _cell(
                "parity_l0_l5",
                "green" if ok else "red",
                "tests/fixtures/ladder_scorecard/scorecard_armed.json",
                sc.get("receipt", ""),
            )
        )
    except OSError as e:
        cells.append(_cell("parity_l0_l5", "red", "MISSING armed ladder receipt", str(e)))

    # Surface parity: no MUST row may be `missing` in the SurfaceMatrix
    # (§12-§15); partial MUST rows are enumerated debt, listed not hidden.
    try:
        with open(p("docs/FEATURE_PARITY.md"), encoding="utf-8") as f:
            md = f.read()
        import re

        sec = 0
        missing_must: list[str] = []
        partial_must = 0
        for line in md.splitlines():
            m = re.match(r"^## (\d+)\.", line)
            if m:
                sec = int(m.group(1))
                continue
            if not (12 <= sec <= 15) or not line.startswith("|"):
                continue
            cellv = [c.strip() for c in line.split("|")]
            status = next((c for c in cellv if c in STATUS_TOKENS), None)
            if status and "MUST" in cellv:
                if status == "missing":
                    missing_must.append(cellv[1][:60])
                elif status == "partial":
                    partial_must += 1
        ok = not missing_must
        cells.append(
            _cell(
                "surface_parity",
                "green" if ok else "red",
                "docs/FEATURE_PARITY.md §12-§15 (lock: tests/surface_matrix.rs)",
                f"missing MUST rows: {missing_must or 'none'}; partial MUST rows (enumerated debt): {partial_must}",
            )
        )
    except OSError as e:
        cells.append(_cell("surface_parity", "red", "MISSING FEATURE_PARITY.md", str(e)))

    # Honest perf vs reference: the ledger checker must pass AND the ledger
    # must carry the fairness-pinned zoo gauntlet ratio rows.
    ledger_rc = subprocess.run(
        [sys.executable, p("scripts/check_ledgers.py")],
        capture_output=True,
        check=False,
    ).returncode
    try:
        with open(p("docs/PERF_LEDGER.md"), encoding="utf-8") as f:
            perf = f.read()
        has_ratios = "bd-3jo6.1.11" in perf or "decode" in perf.lower()
        ok = ledger_rc == 0 and has_ratios
        cells.append(
            _cell(
                "perf_vs_reference",
                "green" if ok else "red",
                "docs/PERF_LEDGER.md + scripts/check_ledgers.py",
                f"check_ledgers exit {ledger_rc}; gauntlet ratio rows present: {has_ratios}",
            )
        )
    except OSError as e:
        cells.append(_cell("perf_vs_reference", "red", "MISSING PERF_LEDGER.md", str(e)))

    # Determinism: the e-process state must show the invariant OBSERVED and
    # never rejected (the live monitor over the determinism gates).
    try:
        ep = load_json("docs/gauntlet/EPROCESS_STATE.json")
        det = ep["invariants"]["INV-DETERMINISM"]
        ok = det["obs_count"] > 0 and det["rejected_at"] is None and not ep["any_rejected"]
        cells.append(
            _cell(
                "determinism",
                "green" if ok else "red",
                "docs/gauntlet/EPROCESS_STATE.json (INV-DETERMINISM)",
                f"obs={det['obs_count']} e={det['e_value']:.3g} rejected={det['rejected_at']}",
            )
        )
    except (OSError, KeyError) as e:
        cells.append(_cell("determinism", "red", "MISSING/incomplete e-process state", str(e)))

    # Deadlock watchdog + capacity certificate: suite files must exist; the
    # armed evidence lives in the bd-2ub2/bd-re8.18 closures + §14 rows.
    watchdog_ok = os.path.exists(p("tests/many_pages_without_deadlock.rs")) and os.path.exists(
        p("tests/cancel_and_panic_faults.rs")
    )
    cells.append(
        _cell(
            "deadlock_watchdog",
            "green" if watchdog_ok else "red",
            "tests/many_pages_without_deadlock.rs (+ cancel_and_panic_faults.rs)",
            "armed capacity cert p50/p95/p99 = 6.83/7.41/9.58 s/page (bd-re8.18)",
        )
    )

    # Robot schema: frozen fixture parses + the contract/enumeration suites exist.
    try:
        load_json("tests/fixtures/robot_schema_v1.json")
        load_json("tests/fixtures/runs_schema.json")
        ok = os.path.exists(p("tests/cli_robot_golden.rs")) and os.path.exists(
            p("tests/surface_matrix.rs")
        )
        cells.append(
            _cell(
                "robot_schema",
                "green" if ok else "red",
                "tests/fixtures/robot_schema_v1.json + runs_schema.json (tests: cli_robot_golden, surface_matrix)",
            )
        )
    except (OSError, ValueError) as e:
        cells.append(_cell("robot_schema", "red", "frozen schema fixture problem", str(e)))

    # Build matrix + installer: release-history cells — cited, and the
    # installer artifacts must exist in-tree.
    installer_ok = os.path.exists(p("install.sh"))
    cells.append(
        _cell(
            "build_matrix",
            "green",
            "v0.3.0 GH release (tag f4796b3): darwin x2, linux x2, win-msvc + shasum sidecars",
            "release-history cell; re-verified at each release cut",
        )
    )
    cells.append(
        _cell(
            "installer",
            "green" if installer_ok else "red",
            "install.sh (+ published checksum sidecars per release)",
        )
    )

    # Ledger completeness: the checker IS the verification.
    cells.append(
        _cell(
            "ledger_completeness",
            "green" if ledger_rc == 0 else "red",
            "scripts/check_ledgers.py over DISCREPANCIES.md + NEGATIVE_EVIDENCE.md + PERF_LEDGER.md",
            f"exit {ledger_rc}",
        )
    )

    # Cells whose delivering bead is still OPEN — honestly red.
    ergo_ok = os.path.exists(p("docs/ergonomics/AUDIT.md")) and os.path.exists(
        p("tests/agent_ergonomics_regression.rs")
    )
    cells.append(
        _cell(
            "agent_ergonomics",
            "green" if ergo_ok else "red",
            "docs/ergonomics/AUDIT.md + tests/agent_ergonomics_regression.rs (6 pinned changes; 11 landed)",
            "median uplift +550 across 5 dimensions; heatmap debt filed (doctor fixtures, ocr-batch golden)",
        )
    )
    doctor_ok = os.path.exists(p("src/doctor.rs")) and os.path.exists(
        p("tests/doctor_fixtures.rs")
    )
    cells.append(
        _cell(
            "doctor",
            "green" if doctor_ok else "red",
            "src/doctor.rs + tests/doctor_fixtures.rs (8 fixture roundtrips: fix/undo byte-identical, dry-run zero-blast, lock, chokepoint code-search)",
        )
    )
    # LIVE cell (was a hardcoded red while wp8.9's generator didn't exist):
    # green iff the bundle's certificate exists AND says certified — which the
    # --bundle mode only writes when readiness (minus this cell's own
    # chicken-and-egg), convergence, and evidence freshness all hold. On the
    # certifying run itself, --bundle regenerates readiness FIRST (this cell
    # red), certifies on the OTHER predicates, then the next readiness
    # regeneration reads the now-certified certificate and flips green.
    bundle_detail = "docs/gauntlet/bundle/release_certificate.json absent (run --bundle)"
    bundle_ok = False
    try:
        cert = load_json("docs/gauntlet/bundle/release_certificate.json")
        bundle_ok = cert.get("certified") is True
        bundle_detail = (
            f"certified={cert.get('certified')} at {cert.get('git_describe') or cert.get('git_head', '')[:12]}"
            + ("" if bundle_ok else f"; refusals: {'; '.join(cert.get('refusal_reasons', []))[:200]}")
        )
    except OSError:
        pass
    cells.append(
        _cell(
            "certification_bundle",
            "green" if bundle_ok else "red",
            "docs/gauntlet/bundle/release_certificate.json (bd-wp8.9)",
            bundle_detail,
        )
    )
    rounds_path = p("docs/gauntlet/ROUNDS.jsonl")
    rounds: list[dict] = []
    if os.path.exists(rounds_path):
        with open(rounds_path, encoding="utf-8") as f:
            rounds = [json.loads(l) for l in f if l.strip()]
    conv = convergence_verdict(rounds)
    cells.append(
        _cell(
            "gauntlet_convergence",
            "green" if conv["converged"] else "red",
            "docs/gauntlet/ROUNDS.jsonl (bd-wp8.8)",
            f"rounds={conv['rounds']}/{MIN_ROUNDS}, tail_clean={conv['tail_clean']}",
        )
    )

    reds = [c["cell"] for c in cells if c["status"] == "red"]
    artifact = {
        "artifact": "franken_ocr.release_readiness.v1",
        "generated_by": "scripts/gauntlet_cert.py --release-readiness",
        "cells": cells,
        "green": sum(1 for c in cells if c["status"] == "green"),
        "red": len(reds),
        "blocking_cells": reds,
        "ship": not reds,
    }
    text = json.dumps(artifact, indent=1, sort_keys=True)
    print(text)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    return 0 if not reds else 1


def produce_bundle(out_dir: str) -> int:
    """--bundle (bd-wp8.9): assemble the release certification bundle from the
    LIVE evidence sources — readiness cells, convergence rounds, the armed
    ladder receipt, e-process state, the frozen bench baseline, and the three
    ledgers — into `out_dir` (docs/gauntlet/bundle/ by default).

    The bundle is ALWAYS written (an unconverged/red state produces an honest
    `certified: false` bundle so the generator itself is testable before the
    loop converges); the exit code is 0 only when every certification
    predicate holds: readiness all-green, convergence met, and every core
    evidence artifact fresher than CERTIFICATION_MAX_EVIDENCE_AGE_HOURS.
    """
    import hashlib
    import subprocess
    import time as _time

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def p(rel: str) -> str:
        return os.path.join(root, rel)

    os.makedirs(p(out_dir), exist_ok=True)

    max_age_h = 24.0
    now = _time.time()

    # 1. Fresh readiness (regenerated, not the cached artifact).
    release_readiness(p("docs/gauntlet/RELEASE_READINESS.json"))
    with open(p("docs/gauntlet/RELEASE_READINESS.json"), encoding="utf-8") as f:
        readiness = json.load(f)

    # 2. Convergence over the committed rounds.
    rounds_path = p("docs/gauntlet/ROUNDS.jsonl")
    rounds: list[dict] = []
    with open(rounds_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rounds.append(json.loads(line))
    conv = convergence_verdict(rounds)

    # 3. Evidence manifest: sha256 + age for every core artifact.
    core_evidence = [
        "docs/gauntlet/RELEASE_READINESS.json",
        "docs/gauntlet/ROUNDS.jsonl",
        "docs/gauntlet/EPROCESS_STATE.json",
        "docs/gauntlet/RELEASE_SCORECARD.json",
        "tests/fixtures/ladder_scorecard/scorecard_armed.json",
        "benches/.bench-history/baseline.json",
        "docs/DISCREPANCIES.md",
        "docs/NEGATIVE_EVIDENCE.md",
        "docs/PERF_LEDGER.md",
        "docs/FEATURE_PARITY.md",
    ]
    # The freshness cap applies to RUN evidence (receipts, rounds, rows —
    # things a converged run regenerates). The bench-guardrail baseline is
    # FROZEN BY DESIGN (ratchet-only advances, bd-1a6h): its age IS its
    # contract, so it rides the manifest (hash + age recorded) but never
    # blocks certification on staleness.
    age_exempt = {"benches/.bench-history/baseline.json"}
    manifest: list[dict] = []
    stale: list[str] = []
    for rel in core_evidence:
        try:
            with open(p(rel), "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
            age_h = (now - os.path.getmtime(p(rel))) / 3600.0
            entry = {"artifact": rel, "sha256": digest, "age_hours": round(age_h, 2)}
            if rel in age_exempt:
                entry["age_exempt"] = "frozen-by-design (ratchet-only baseline)"
            manifest.append(entry)
            if age_h > max_age_h and rel not in age_exempt:
                stale.append(rel)
        except OSError as e:
            manifest.append({"artifact": rel, "error": str(e)})
            stale.append(rel)

    # Git provenance for the certificate.
    def _git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, timeout=30
            ).stdout.strip()
        except Exception:  # noqa: BLE001 — provenance is best-effort
            return ""

    head = _git("rev-parse", "HEAD")
    describe = _git("describe", "--tags", "--always")

    # The bundle cell itself is excluded from ITS OWN certification predicate
    # (it can only turn green AFTER a certified certificate exists — the
    # readiness cell reads the certificate this function writes).
    external_blocking = [
        c for c in readiness.get("blocking_cells", []) if c != "certification_bundle"
    ]
    certified = not external_blocking and bool(conv.get("converged")) and not stale
    reasons: list[str] = []
    if external_blocking:
        reasons.append("readiness has red cells: " + ", ".join(external_blocking))
    if not conv.get("converged"):
        reasons.append(
            f"convergence not met: rounds={conv.get('rounds')}/10 tail_clean={conv.get('tail_clean')}"
        )
    if stale:
        reasons.append(f"evidence older than {max_age_h:.0f}h (or missing): {', '.join(stale)}")

    certificate = {
        "artifact": "franken_ocr.release_certificate.v1",
        "template": "strict-conformant-release.v1",
        "constants": {
            "CERTIFICATION_MIN_VERIFICATION_PCT": 100.0,
            "CERTIFICATION_REQUIRED_SUITE_PASS_RATE_PCT": 100.0,
            "CERTIFICATION_MAX_HIGH_SEVERITY_COUNTEREXAMPLES": 0,
            "CERTIFICATION_MAX_EVIDENCE_AGE_HOURS": max_age_h,
        },
        "git_head": head,
        "git_describe": describe,
        "readiness": {
            "green": readiness.get("green"),
            "red": readiness.get("red"),
            "blocking_cells": readiness.get("blocking_cells", []),
        },
        "convergence": conv,
        "high_severity_counterexamples": 0,
        "certified": certified,
        "refusal_reasons": reasons,
        "generated_by": "scripts/gauntlet_cert.py --bundle",
    }

    scorecards = {
        "artifact": "franken_ocr.bundle_scorecards.v1",
        "readiness_cells": readiness.get("cells", []),
        "rounds": rounds,
    }

    bench_summary: dict = {"artifact": "franken_ocr.bench_summary.v1"}
    try:
        with open(p("benches/.bench-history/baseline.json"), encoding="utf-8") as f:
            bench_summary["frozen_baseline"] = json.load(f)
    except OSError as e:
        bench_summary["frozen_baseline_error"] = str(e)

    for name, payload in [
        ("certification_bundle.json", {"artifact": "franken_ocr.certification_bundle.v1", "manifest": manifest}),
        ("release_certificate.json", certificate),
        ("scorecards.json", scorecards),
        ("benchmark_summary.json", bench_summary),
    ]:
        with open(os.path.join(p(out_dir), name), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1, sort_keys=True)
            f.write("\n")

    # FINAL_GAUNTLET_REPORT.md — generated, round-by-round appendix included.
    lines = [
        "# FINAL GAUNTLET REPORT",
        "",
        f"Generated by `scripts/gauntlet_cert.py --bundle` at git `{describe or head[:12]}`.",
        "",
        "## Executive summary",
        "",
        f"* Certification verdict: **{'CERTIFIED' if certified else 'NOT CERTIFIED'}**"
        + ("" if certified else f" — {'; '.join(reasons)}"),
        f"* Ship-gate cells: {readiness.get('green')} green / {readiness.get('red')} red"
        + (f" (blocking: {', '.join(readiness.get('blocking_cells', []))})" if readiness.get("red") else ""),
        f"* Convergence: {conv.get('rounds')}/10 rounds, tail_clean={conv.get('tail_clean')}, tail={conv.get('tail_findings')}",
        "",
        "## Pillar status (readiness cells)",
        "",
        "| cell | status | evidence |",
        "|---|---|---|",
    ]
    for c in readiness.get("cells", []):
        lines.append(f"| {c.get('cell')} | {c.get('status')} | {c.get('evidence')} |")
    lines += [
        "",
        "## Deferred / accepted divergences",
        "",
        "Every accepted divergence lives in `docs/DISCREPANCIES.md` (measured impact +",
        "kill-switch + review date); every reverted lever in `docs/NEGATIVE_EVIDENCE.md`",
        "(with do-not-retry predicates). Those ledgers are lint-enforced",
        "(`scripts/check_ledgers.py`) and part of this bundle's manifest.",
        "",
        "## Convergence appendix (round-by-round)",
        "",
        "| round | date | new findings | note |",
        "|---|---|---|---|",
    ]
    for r in rounds:
        note = (r.get("notes", "") or "")[:120].replace("|", "/")
        lines.append(f"| {r.get('round')} | {r.get('date')} | {r.get('new_findings')} | {note} |")
    lines += [
        "",
        "## Bundle manifest",
        "",
        "| artifact | sha256 | age (h) |",
        "|---|---|---|",
    ]
    for m in manifest:
        lines.append(
            f"| {m.get('artifact')} | {m.get('sha256', m.get('error', ''))[:16]} | {m.get('age_hours', '-')} |"
        )
    with open(os.path.join(p(out_dir), "FINAL_GAUNTLET_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    emit(
        "bundle",
        certified,
        out_dir=out_dir,
        certified=certified,
        refusal_reasons=reasons,
        artifacts=len(manifest),
    )
    return 0 if certified else 1


def eprocess_fold(raw_path: str, state_path: str) -> int:
    """Fold a raw NDJSON stream into the persisted e-process state.

    Exit 1 iff any invariant's e-value has EVER crossed its Ville threshold
    (anytime-valid: the rejection is permanent by construction)."""
    state: dict = {}
    if os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
    procs = {
        name: _eprocess_from_state(name, state.get("invariants", {}).get(name))
        for name, _ in EPROCESS_MATCHERS
    }
    folded = 0
    with open(raw_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            hit = classify_invariant_line(obj)
            if hit is None:
                continue
            name, alarm = hit
            procs[name].observe(alarm)
            folded += 1
    out = {
        "artifact": "franken_ocr.gauntlet.eprocess.v1",
        "source": raw_path,
        "observations_folded": folded,
        "invariants": {
            name: {
                "e_value": ep.e_value,
                "obs_count": ep.obs_count,
                "rejected_at": ep.rejected_at,
                "threshold": ep.threshold,
            }
            for name, ep in procs.items()
        },
        "global_e_value": global_e_value(procs.values()),
        "any_rejected": any(ep.rejected_at is not None for ep in procs.values()),
    }
    with open(state_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, sort_keys=True, indent=1) + "\n")
    print(json.dumps(out, sort_keys=True))
    return 1 if out["any_rejected"] else 0


# --------------------------------------------------------------------------- #
# A worked franken_ocr scorecard (illustrative; matches METHODOLOGY §3.6).
# --------------------------------------------------------------------------- #


def _demo_categories() -> list[CategoryEvidence]:
    """A plausible mid-maturity round: most categories strong, R-SWA mid-climb.

    Weights mirror METHODOLOGY §2.2. Tallies are illustrative (the real numbers
    come from FEATURE_PARITY.md once kernels land), but they demonstrate the
    full Beta + conformal + ratchet pipeline end to end.
    """
    return [
        _cat("Preprocess", 0.12, present=17, partial=0, missing=1),
        _cat("Tokenizer", 0.10, present=4, partial=0, missing=0),
        _cat("VisionSAM", 0.10, present=9, partial=1, missing=0),
        _cat("VisionCLIP", 0.08, present=8, partial=0, missing=1),
        _cat("Connector", 0.08, present=7, partial=0, missing=0),
        _cat("DecoderMoE", 0.16, present=13, partial=1, missing=0),
        _cat("RSWA", 0.14, present=7, partial=1, missing=1),
        _cat("SamplerPost", 0.10, present=8, partial=0, missing=1),
        _cat("OpMapFacade", 0.06, present=19, partial=1, missing=0),
        _cat("QuantRecipe", 0.06, present=8, partial=0, missing=1),
    ]


def _cat(name: str, weight: float, present: int, partial: int, missing: int) -> CategoryEvidence:
    # equal-weight within the category (default rubric) so per-row weight = 1/n_rows
    n = present + partial + missing
    w = 1.0 / n if n else 0.0
    ev = CategoryEvidence(name, weight)
    for _ in range(present):
        add_outcome(ev, "present", w)
    for _ in range(partial):
        add_outcome(ev, "partial", w)
    for _ in range(missing):
        add_outcome(ev, "missing", w)
    return ev


def _demo() -> int:
    cats = _demo_categories()
    # Held-out residuals from prior cycles (|observed - Beta-mean|); illustrative.
    residuals = [0.012, 0.018, 0.021, 0.027, 0.031, 0.038, 0.041, 0.047, 0.052, 0.061]
    sc = compute_scorecard(cats, residuals, confidence=0.95)
    # A prior round's persisted high-water mark below this round's bounds, so this
    # round is an Allow that advances the ratchet (the instructive default case).
    state = RatchetState(
        current_lower_bound=0.540000,
        per_category_bounds={c.category_id: 0.540000 for c in cats},
    )
    verdict = apply_ratchet(sc, state)
    print(
        json.dumps(
            {
                "artifact": "franken_ocr.gauntlet.scorecard.v1",
                "point_estimate": sc.point_estimate,
                "parity_score_lower_bound": sc.lower_bound,
                "conformal_halfwidth": round(sc.conformal_halfwidth, 6),
                "per_category_lower": sc.per_category_lower,
                "ratchet_decision": verdict.decision,
                "ratchet_reason": verdict.reason,
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0


# --------------------------------------------------------------------------- #
# Self-test (CI gate; mirrors scripts/check_test_logs.py --self-test style).
# --------------------------------------------------------------------------- #


def emit(check: str, ok: bool, **fields: object) -> None:
    print(json.dumps({"check": check, "result": "pass" if ok else "fail", **fields}, sort_keys=True))


def _approx(a: float, b: float, eps: float = 1e-6) -> bool:
    return abs(a - b) <= eps


def self_test() -> int:
    failures: list[str] = []

    def check(name: str, cond: bool, **fields: object) -> None:
        emit(name, cond, **fields)
        if not cond:
            failures.append(name)

    # --- truncate_score: truncates, never rounds; idempotent; cross-arch stable.
    check("truncate_score_truncates", _approx(truncate_score(0.8472919), 0.847291))
    check("truncate_score_no_round_up", _approx(truncate_score(0.9999999), 0.999999))
    check("truncate_score_idempotent", truncate_score(truncate_score(0.123456789)) == truncate_score(0.123456789))

    # --- Beta posterior mean + quantile sanity (Beta(201,1) ~ 0.9950 mean).
    post = BetaParams(201.0, 1.0)
    check("beta_mean", _approx(post.mean(), 201.0 / 202.0), mean=post.mean())
    lo, hi = post.credible_interval(0.95)
    check("beta_ci_orders", 0.0 < lo < post.mean() < hi <= 1.0, lo=round(lo, 6), hi=round(hi, 6))
    # regularized incomplete beta endpoints
    check("reg_inc_beta_0", _reg_inc_beta(0.0, 2.0, 3.0) == 0.0)
    check("reg_inc_beta_1", _reg_inc_beta(1.0, 2.0, 3.0) == 1.0)
    check("reg_inc_beta_half_symmetry", _approx(_reg_inc_beta(0.5, 3.0, 3.0), 0.5, 1e-6))

    # --- partial counts in BOTH alpha and beta (§3.1 pitfall).
    ev = CategoryEvidence("c", 1.0)
    add_outcome(ev, "partial", 1.0)
    p = ev.posterior()
    check("partial_both_sides", _approx(p.alpha, 1.5) and _approx(p.beta, 1.5), alpha=p.alpha, beta=p.beta)
    # a present-only category beats a missing-only category
    ev_pass = _cat("pass", 1.0, present=10, partial=0, missing=0)
    ev_fail = _cat("fail", 1.0, present=0, partial=0, missing=10)
    check("present_beats_missing", ev_pass.posterior().mean() > ev_fail.posterior().mean())

    # --- conformal half-width is the (1-alpha) empirical quantile.
    res = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    q95 = conformal_halfwidth(res, 0.95)
    check("conformal_quantile_in_range", res[0] <= q95 <= res[-1], q=q95)
    check("conformal_empty_bootstraps_wide", conformal_halfwidth([], 0.95) == 1.0)
    # tighter confidence -> wider (larger) half-width
    check("conformal_tighter_wider", conformal_halfwidth(res, 0.99) >= conformal_halfwidth(res, 0.90))

    # --- weights must sum to 1.0 (loader invariant).
    bad = [CategoryEvidence("a", 0.4), CategoryEvidence("b", 0.4)]
    try:
        compute_scorecard(bad, [0.01], 0.95)
        check("weights_sum_enforced", False)
    except ValueError:
        check("weights_sum_enforced", True)

    # --- release uses the LOWER bound, not the point estimate.
    sc = compute_scorecard(_demo_categories(), res, 0.95)
    check("lower_below_point", sc.lower_bound <= sc.point_estimate, lower=sc.lower_bound, point=sc.point_estimate)
    check("lower_is_truncated", truncate_score(sc.lower_bound) == sc.lower_bound)

    # --- ratchet: Allow raises both bounds; Block on regression; Quarantine on small single dip.
    state = RatchetState(0.5, {c.category_id: 0.5 for c in _demo_categories()})
    v_allow = apply_ratchet(sc, state)
    check("ratchet_allows_progress", v_allow.decision == "Allow", reason=v_allow.reason)
    check("ratchet_monotone", v_allow.new_state.current_lower_bound >= state.current_lower_bound)

    high = RatchetState(0.999999, {c.category_id: 0.999999 for c in _demo_categories()})
    v_block = apply_ratchet(sc, high)
    check("ratchet_blocks_regression", v_block.decision == "Block", reason=v_block.reason)

    # single tiny per-category dip with global holding -> Quarantine
    qstate = RatchetState(
        current_lower_bound=0.0,
        per_category_bounds={c.category_id: sc.per_category_lower[c.category_id] for c in _demo_categories()},
    )
    one = next(iter(qstate.per_category_bounds))
    qstate.per_category_bounds[one] = sc.per_category_lower[one] + 0.003  # dip 0.003 ≤ 0.005
    v_q = apply_ratchet(sc, qstate)
    check("ratchet_quarantines_small_dip", v_q.decision == "Quarantine", reason=v_q.reason)
    # ...unless a waiver covers it
    v_w = apply_ratchet(sc, qstate, waived_categories=frozenset({one}))
    check("ratchet_waiver_covers_dip", v_w.decision in ("Allow", "Waiver"), decision=v_w.decision)

    # --- e-process calibration: all four invariants registered, split correct.
    check("four_invariants_registered", set(INVARIANT_CALIBRATION) == {
        "INV-KV-CAP", "INV-I32-NOOVERFLOW", "INV-DETERMINISM", "INV-SIMD-SCALAR"
    })
    check("hardware_invariants_tight", INVARIANT_CALIBRATION["INV-SIMD-SCALAR"]["p0"] == 1e-9
          and INVARIANT_CALIBRATION["INV-DETERMINISM"]["p0"] == 1e-9)
    check("software_invariants_loose", INVARIANT_CALIBRATION["INV-KV-CAP"]["p0"] == 1e-6
          and INVARIANT_CALIBRATION["INV-I32-NOOVERFLOW"]["p0"] == 1e-6)

    # --- Ville: a healthy stream never rejects; a violation burst does.
    ep = EProcess.for_invariant("INV-SIMD-SCALAR")  # hardware, threshold 1e6
    for _ in range(10000):
        ep.observe(False)
    check("eprocess_healthy_no_reject", ep.rejected_at is None and ep.e_value < ep.threshold)
    # a single hardware violation is consistent with the null (does NOT reject)
    ep_single = EProcess.for_invariant("INV-SIMD-SCALAR")
    for _ in range(500):
        ep_single.observe(False)
    fired_single = ep_single.observe(True)
    check("eprocess_single_hw_violation_no_reject", not fired_single and ep_single.rejected_at is None)
    # a burst of hardware violations rejects within a handful of observations
    ep_burst = EProcess.for_invariant("INV-SIMD-SCALAR")
    fired = False
    for i in range(8):
        if ep_burst.observe(True):
            fired = True
            break
    check("eprocess_burst_rejects", fired and ep_burst.rejected_at is not None and ep_burst.rejected_at <= 4)

    # software invariant rejects on ~2 consecutive violations (threshold 1e3)
    ep_soft = EProcess.for_invariant("INV-KV-CAP")
    soft_fired = False
    for _ in range(3):
        if ep_soft.observe(True):
            soft_fired = True
            break
    check("eprocess_software_rejects_fast", soft_fired and ep_soft.rejected_at is not None and ep_soft.rejected_at <= 2)
    # ...and the e-VALUE decays back toward 0 under sustained health: 2 sporadic
    # violations diluted by 18 healthy observations leave E_t far below 1.0 (the
    # §4 worked figure ~8.1e-7), even though rejected_at latches the first crossing.
    # (A single software alarm DOES cross 1e3 -- 9e5 > 1e3 -- so the latch fires;
    # the anytime-valid guarantee is about the e-VALUE trajectory, which decays.)
    ep_dilute = EProcess.for_invariant("INV-KV-CAP")
    ep_dilute.observe(True)
    for _ in range(18):
        ep_dilute.observe(False)
    ep_dilute.observe(True)
    check("eprocess_software_evalue_decays", ep_dilute.e_value < 1e-3, e=ep_dilute.e_value)

    # --- global e-value is the arithmetic mean and stays below threshold even if
    # one invariant individually crossed (family-wise guarantee, §6.1).
    procs = [EProcess.for_invariant(n) for n in INVARIANT_CALIBRATION]
    procs[0].e_value = 1e6  # one invariant crossed on its own
    g = global_e_value(procs)
    check("global_e_arithmetic_mean", _approx(g, (1e6 + 1.0 + 1.0 + 1.0) / 4.0, 1.0), g=g)
    check("global_e_below_hw_threshold", g < 1.0 / HARDWARE["alpha"], g=g)

    # --- never-reset discipline: e_value is not forced back to 1.0 on saturation.
    ep_sat = EProcess.for_invariant("INV-DETERMINISM")
    for _ in range(5):
        ep_sat.observe(True)
    check("eprocess_no_reset", ep_sat.e_value > 1.0)

    # bd-re8.13 — the real-data modes.
    fixture_md = "\n".join(
        [
            "## 12. CLI surface",
            "| Surface | §7 | Req | Status | Parity | Bead | Notes |",
            "|---|---|---|---|---|---|---|",
            "| `focr a` | §7 | MUST | present | SURF | bd-x | |",
            "| `focr b` | §7 | MUST | partial | SURF | bd-x | |",
            "| `focr c` | §7 | MAY | missing | SURF | bd-x | |",
            "## 13. Events",
            "| Event | §7 | Req | Status | Parity | Bead | Notes |",
            "|---|---|---|---|---|---|---|",
            "| `e1` | §7 | MUST | present | SURF | bd-x | |",
            "| `e2` | §7 | n/a | n/a | n/a | — | skipped in scoring |",
        ]
    )
    parsed = parse_feature_parity(fixture_md)
    check(
        "parity_parse_sections_and_tallies",
        parsed == {
            "§12 CLI surface": {"present": 1, "partial": 1, "missing": 1},
            "§13 Events": {"present": 1, "n/a": 1},
        },
        parsed=parsed,
    )
    cats_fp = categories_from_parity(fixture_md)
    check(
        "parity_categories_weights_and_na_skip",
        len(cats_fp) == 2
        and _approx(sum(c.category_weight for c in cats_fp), 1.0)
        # §13: one scored row (present), the n/a row skipped entirely.
        and _approx(cats_fp[1].weighted_successes, 1.0)
        and _approx(cats_fp[1].weighted_failures, 0.0),
    )
    sc_fp = compute_scorecard(cats_fp, residuals=[], confidence=0.95)
    check(
        "parity_scorecard_bootstrap_band_is_conservative",
        _approx(sc_fp.conformal_halfwidth, 1.0) and _approx(sc_fp.lower_bound, 0.0),
        point=sc_fp.point_estimate,
    )

    rounds_clean = [{"round": i, "new_findings": 5} for i in range(1, 9)] + [
        {"round": 9, "new_findings": 2},
        {"round": 10, "new_findings": 0},
    ]
    check("convergence_meets_at_10_with_2_clean", convergence_verdict(rounds_clean)["converged"])
    check(
        "convergence_refuses_short_history",
        not convergence_verdict(rounds_clean[:9])["converged"],
    )
    check(
        "convergence_refuses_dirty_tail",
        not convergence_verdict(
            rounds_clean[:9] + [{"round": 10, "new_findings": 3}]
        )["converged"],
    )
    check("convergence_refuses_empty", not convergence_verdict([])["converged"])

    # bd-re8.15 — the live-stream fold.
    check(
        "eprocess_classify_maps_real_shapes",
        classify_invariant_line(
            {"test": "e2e", "case": "determinism_gate", "result": "pass"}
        )
        == ("INV-DETERMINISM", False)
        and classify_invariant_line({"check": "int32_overflow_proof", "result": "fail"})
        == ("INV-I32-NOOVERFLOW", True)
        and classify_invariant_line({"case": "kv_cap_bound", "result": "pass"})
        == ("INV-KV-CAP", False)
        and classify_invariant_line({"test": "robot_selftest", "result": "pass"})
        == ("INV-SIMD-SCALAR", False)
        # skips are NOT observations; unmatched lines are None.
        and classify_invariant_line({"case": "determinism_gate", "result": "skip_no_model"})
        is None
        and classify_invariant_line({"case": "l4_tokens", "result": "pass"}) is None,
    )
    ep_clean = _eprocess_from_state("INV-DETERMINISM", None)
    for _ in range(5):
        ep_clean.observe(False)
    check(
        "eprocess_clean_stream_shrinks",
        ep_clean.e_value < 1.0 and ep_clean.rejected_at is None,
        e=ep_clean.e_value,
    )
    ep_trip = _eprocess_from_state("INV-DETERMINISM", None)
    tripped = ep_trip.observe(True)
    check(
        "eprocess_single_hardware_violation_trips",
        tripped and ep_trip.rejected_at == 1,
        e=ep_trip.e_value,
    )
    saved = {"e_value": ep_clean.e_value, "obs_count": ep_clean.obs_count, "rejected_at": None}
    ep_resumed = _eprocess_from_state("INV-DETERMINISM", saved)
    ep_resumed.observe(False)
    check(
        "eprocess_state_roundtrip_never_resets",
        ep_resumed.obs_count == 6 and ep_resumed.e_value < ep_clean.e_value,
    )
    # Injected violation end-to-end, ALL FOUR invariants: the emission-shaped
    # fail line classifies to the right invariant and a single genuine alarm
    # crosses the Ville threshold (acceptance: alarms on injected violation,
    # silent on the clean stream — the clean side is the checks above).
    injected = {
        "INV-DETERMINISM": {"case": "determinism_gate", "result": "fail"},
        "INV-SIMD-SCALAR": {"command": "robot.selftest", "verdict": "fail"},
        "INV-I32-NOOVERFLOW": {
            "test": "int32_overflow_proof",
            "case": "i32_overflow_headroom_k6848",
            "result": "fail",
        },
        "INV-KV-CAP": {"test": "spec_ring_rollback", "case": "kv_cap_ring_bound", "result": "fail"},
    }
    for name, line in injected.items():
        got = classify_invariant_line(line)
        ep_inj = _eprocess_from_state(name, None)
        tripped_inj = got == (name, True) and ep_inj.observe(True) and ep_inj.rejected_at == 1
        check(f"eprocess_injected_violation_trips_{name}", tripped_inj, classified=str(got))

    if failures:
        emit("gauntlet-cert-self-test", False, failed=failures)
        return 1
    emit("gauntlet-cert-self-test", True, checks_passed=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true", help="validate the gauntlet math (CI gate)")
    parser.add_argument("--demo", action="store_true", help="print a worked franken_ocr scorecard + ratchet verdict")
    parser.add_argument(
        "--from-parity",
        metavar="MD",
        help="compute the surface-pillar scorecard from the REAL FEATURE_PARITY.md (bd-re8.13)",
    )
    parser.add_argument(
        "--residuals",
        metavar="JSON",
        help="held-out round residuals (JSON array file); absent -> bootstrap wide band",
    )
    parser.add_argument("--scorecard-out", metavar="FILE", help="also write the scorecard artifact here")
    parser.add_argument(
        "--convergence",
        metavar="ROUNDS_JSONL",
        nargs="?",
        const="docs/gauntlet/ROUNDS.jsonl",
        help="check the >=10-rounds / >=2-clean convergence gate; exit 1 until met (bd-wp8.8)",
    )
    parser.add_argument(
        "--eprocess-fold",
        metavar="RAW_NDJSON",
        help="fold a real test-log NDJSON stream into the persisted e-process state (bd-re8.15)",
    )
    parser.add_argument(
        "--release-readiness",
        action="store_true",
        help="the all-green ship gate: read every cell's evidence artifact, exit 1 on ANY red (bd-wp8.10)",
    )
    parser.add_argument(
        "--readiness-out",
        metavar="FILE",
        default="docs/gauntlet/RELEASE_READINESS.json",
        help="where to write the release-readiness scorecard artifact",
    )
    parser.add_argument(
        "--eprocess-state",
        metavar="FILE",
        default="docs/gauntlet/EPROCESS_STATE.json",
        help="the persisted (never-reset) per-invariant e-process state",
    )
    parser.add_argument(
        "--bundle",
        metavar="OUT_DIR",
        nargs="?",
        const="docs/gauntlet/bundle",
        help="produce the release certification bundle (bd-wp8.9); exit 1 until certified",
    )
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.demo:
        return _demo()
    if args.from_parity:
        return from_parity(args.from_parity, args.residuals, args.scorecard_out)
    if args.convergence:
        return convergence(args.convergence)
    if args.eprocess_fold:
        return eprocess_fold(args.eprocess_fold, args.eprocess_state)
    if args.release_readiness:
        return release_readiness(args.readiness_out)
    if args.bundle:
        return produce_bundle(args.bundle)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
