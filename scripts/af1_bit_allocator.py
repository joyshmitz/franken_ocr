#!/usr/bin/env python3
"""AF-1 — rate-distortion / Lagrangian water-filling per-tensor bit allocator.

Bead: bd-ksps / ALIEN-af1 / P4-af1-bit-allocator.
Spec: docs/alien/AF-1-rate-distortion-bit-allocation.md
Plan: COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md §9.7 (AF-1), §5 (.focrq), §6.3.

Choosing {bf16, int8, int4-g32, int4-g16} per quantizable tensor under a total
footprint budget B is a rate-distortion problem:

    minimize  D(b) = Sum_t D_t(b_t)        (additive layer-output cosine-drop surrogate)
    s.t.      R(b) = Sum_t R_t(b_t) <= B   (total footprint in bytes)

The Lagrangian L(b, lambda) = Sum_t [ D_t(b_t) + lambda * R_t(b_t) ] separates
per-tensor, so at any price `lambda` each tensor independently picks the option
minimizing its RD-cost.  Sweeping `lambda` traces the optimal bit-allocation
frontier (water-filling); we walk it until the footprint just fits B, then spend
the slack with a greedy hull-climb.  An exact bounded-knapsack DP (`--exact-dp`)
closes the duality gap when requested.

This script is the REFERENCE IMPLEMENTATION OF THE ALLOCATION MATH ONLY.  The
per-tensor distortion curves D_t(b) (layer-output cosine drop on a calibration
batch) are measured separately by `focr convert --measure-distortion`; this tool
consumes that JSON and emits the bit_allocation_table.json.

It is deterministic (pure function of inputs; deterministic tie-breaks; no RNG /
clock / map-iteration-order dependence), dependency-free (Python stdlib only),
and `python3 -m py_compile`-clean.

Input  (per-tensor curves JSON, the converter's --measure-distortion output):

    {
      "option_bits": {"bf16": 16.0, "int8": 8.03, "int4-g32": 5.0, "int4-g16": 6.0},
      "tensors": [
        {
          "tensor": "decoder.layer.0.dense.down",
          "numel": 8765440,
          "pin": null,               # or "bf16" to force high precision
          "tier_floor": null,        # or e.g. "int8" (§6.3 _M discipline)
          "curve": {                 # option -> {bits: bpw, distortion: cosine-drop}
            "bf16":     {"bits": 16.0, "distortion": 0.0},
            "int8":     {"bits": 8.03, "distortion": 0.000310},
            "int4-g32": {"bits": 5.0,  "distortion": 0.001740},
            "int4-g16": {"bits": 6.0,  "distortion": 0.000980}
          }
        }
      ]
    }

Output (bit_allocation_table.json, schema_version 1) — see the spec §4.2.

Examples:

    # allocate to a 2.0 GiB budget by water-filling
    python3 scripts/af1_bit_allocator.py curves.json --budget-gib 2.0 -o table.json

    # exact integer-optimal allocation via bounded knapsack DP
    python3 scripts/af1_bit_allocator.py curves.json --budget-gib 2.0 --exact-dp

    # the deterministic fallback: uniform Q4_K_M-class table, no optimization
    python3 scripts/af1_bit_allocator.py curves.json --fallback --uniform-option int4-g32

    # self-test on a built-in synthetic model (no input file needed)
    python3 scripts/af1_bit_allocator.py --selftest
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
GENERATOR = "scripts/af1_bit_allocator.py"

# Default effective bits-per-weight per option (scale overhead included; §1.1 of the spec).
# These are only fallbacks when a curve point omits an explicit `bits` value.
DEFAULT_OPTION_BITS: dict[str, float] = {
    "bf16": 16.0,
    "int8": 8.03,
    "int4-g32": 5.0,
    "int4-g16": 6.0,
}

# Canonical option ordering: index 0 = highest precision.  Used for deterministic
# tie-breaks (prefer the higher-precision option on equal RD-cost) and for the
# "tier floor" comparison (a floor of int8 forbids anything cheaper than int8).
OPTION_ORDER: dict[str, int] = {
    "bf16": 0,
    "int8": 1,
    "int4-g16": 2,
    "int4-g32": 3,
}


class AllocatorError(Exception):
    """Raised on malformed input or an infeasible budget."""


def _option_rank(option: str) -> int:
    """Deterministic precision rank; unknown options sort after known ones by name."""
    if option in OPTION_ORDER:
        return OPTION_ORDER[option]
    return len(OPTION_ORDER) + sum(ord(c) for c in option)


def bytes_for(numel: int, bpw: float) -> int:
    """Footprint in bytes for `numel` weights at `bpw` effective bits-per-weight."""
    return int(math.ceil(numel * bpw / 8.0))


@dataclass(frozen=True)
class Point:
    """One operational rate-distortion point for a tensor."""

    option: str
    bits: float          # effective bits-per-weight (incl. scale overhead)
    distortion: float    # D_t(option): layer-output cosine drop (>= 0, bf16 == 0)
    rate_bytes: int      # R_t(option): footprint in bytes for this tensor


@dataclass
class Tensor:
    """A quantizable tensor with its (pruned, convex) R-D points."""

    name: str
    numel: int
    points: list[Point]          # sorted by increasing bytes; convex lower hull
    pin: Optional[str] = None
    tier_floor: Optional[str] = None
    # Mutable allocation state during the sweep / greedy top-up:
    chosen_idx: int = field(default=0)

    @property
    def chosen(self) -> Point:
        return self.points[self.chosen_idx]

    def cheapest(self) -> Point:
        return self.points[0]

    def costliest(self) -> Point:
        return self.points[-1]


# ----------------------------------------------------------------------------
# Curve ingestion: monotone repair (cummin) + convex-hull prune.
# ----------------------------------------------------------------------------

def _build_points(name: str, numel: int, curve: dict, option_bits: dict[str, float]) -> list[Point]:
    """Turn a raw curve dict into raw (option, bits, distortion, bytes) points."""
    if not curve:
        raise AllocatorError(f"tensor {name!r}: empty curve")
    raw: list[Point] = []
    for option, entry in curve.items():
        if isinstance(entry, dict):
            bits = float(entry.get("bits", option_bits.get(option, DEFAULT_OPTION_BITS.get(option, math.nan))))
            distortion = float(entry["distortion"])
        else:  # bare distortion number, bits from the global table
            bits = float(option_bits.get(option, DEFAULT_OPTION_BITS.get(option, math.nan)))
            distortion = float(entry)
        if math.isnan(bits):
            raise AllocatorError(f"tensor {name!r}: option {option!r} has no known bits-per-weight")
        if distortion < 0.0:
            raise AllocatorError(f"tensor {name!r}: option {option!r} has negative distortion {distortion}")
        raw.append(Point(option, bits, distortion, bytes_for(numel, bits)))
    # Deterministic order: by bytes ascending, then by precision rank, then name.
    raw.sort(key=lambda p: (p.rate_bytes, _option_rank(p.option), p.option))
    return raw


def _monotone_repair(points: list[Point]) -> list[Point]:
    """Enforce D non-increasing as bytes increase (cummin from the cheap end).

    Calibration noise can produce a "more bits but not-less distortion" point; we
    clamp each point's distortion to the running minimum of the cheaper points so
    the curve is monotone (P2 of the spec).  Points are kept (option labels need to
    survive); only the distortion value is repaired.
    """
    if not points:
        return points
    repaired: list[Point] = []
    running_min = math.inf
    for p in points:
        d = min(p.distortion, running_min)
        running_min = d
        repaired.append(Point(p.option, p.bits, d, p.rate_bytes))
    return repaired


def _convex_hull_prune(points: list[Point]) -> list[Point]:
    """Keep only the lower convex hull of the (bytes, distortion) points.

    Walks cheapest -> costliest keeping points whose marginal distortion-per-byte
    return strictly improves on the previous kept edge (diminishing returns).  An
    interior point dominated by a cheaper-and-a-pricier neighbour is dropped: it is
    optimal for no value of lambda, so it can never be selected by water-filling.
    """
    # De-duplicate identical byte sizes, keeping the lowest distortion (then highest precision).
    by_bytes: dict[int, Point] = {}
    for p in points:
        cur = by_bytes.get(p.rate_bytes)
        if cur is None or (p.distortion, _option_rank(p.option)) < (cur.distortion, _option_rank(cur.option)):
            by_bytes[p.rate_bytes] = p
    uniq = sorted(by_bytes.values(), key=lambda p: p.rate_bytes)
    if len(uniq) <= 2:
        return uniq
    hull: list[Point] = []
    for p in uniq:
        # Maintain a lower-convex chain: pop kept points that are not below the
        # line from the previous-kept to the incoming point.
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            # cross product of (b-a) x (p-a) in (bytes, distortion) space; <= 0 => b not below.
            cross = (b.rate_bytes - a.rate_bytes) * (p.distortion - a.distortion) \
                - (b.distortion - a.distortion) * (p.rate_bytes - a.rate_bytes)
            if cross <= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


def _apply_pins_and_floors(name: str, points: list[Point], pin: Optional[str],
                           tier_floor: Optional[str]) -> list[Point]:
    """Restrict the option set per a bf16 pin and/or a tier floor (§6.3 _M discipline)."""
    if pin is not None:
        pinned = [p for p in points if p.option == pin]
        if not pinned:
            raise AllocatorError(f"tensor {name!r}: pin {pin!r} not present in its curve")
        return pinned
    if tier_floor is not None:
        floor_rank = _option_rank(tier_floor)
        kept = [p for p in points if _option_rank(p.option) <= floor_rank]
        if not kept:
            raise AllocatorError(f"tensor {name!r}: tier_floor {tier_floor!r} removes every option")
        return kept
    return points


def load_tensors(doc: dict) -> list[Tensor]:
    """Parse + validate the curves JSON into convex-hull Tensor objects."""
    option_bits = dict(DEFAULT_OPTION_BITS)
    option_bits.update({k: float(v) for k, v in doc.get("option_bits", {}).items()})
    entries = doc.get("tensors")
    if not isinstance(entries, list) or not entries:
        raise AllocatorError("curves JSON must contain a non-empty 'tensors' array")
    tensors: list[Tensor] = []
    seen: set[str] = set()
    for entry in entries:
        name = entry["tensor"]
        if name in seen:
            raise AllocatorError(f"duplicate tensor name {name!r}")
        seen.add(name)
        numel = int(entry["numel"])
        if numel <= 0:
            raise AllocatorError(f"tensor {name!r}: numel must be positive")
        pin = entry.get("pin")
        tier_floor = entry.get("tier_floor")
        raw = _build_points(name, numel, entry["curve"], option_bits)
        # P1: bf16 anchor must be lossless when present.
        for p in raw:
            if p.option == "bf16" and p.distortion != 0.0:
                raise AllocatorError(
                    f"tensor {name!r}: bf16 anchor distortion must be 0, got {p.distortion}")
        raw = _apply_pins_and_floors(name, raw, pin, tier_floor)
        pts = _convex_hull_prune(_monotone_repair(raw))
        if not pts:
            raise AllocatorError(f"tensor {name!r}: no options survive pruning")
        tensors.append(Tensor(name=name, numel=numel, points=pts, pin=pin, tier_floor=tier_floor))
    # Deterministic tensor order (by name) so output is reproducible.
    tensors.sort(key=lambda t: t.name)
    return tensors


# ----------------------------------------------------------------------------
# Lagrangian water-filling.
# ----------------------------------------------------------------------------

def _argmin_at_lambda(tensor: Tensor, lam: float) -> int:
    """Index of the option minimizing D + lambda*R; deterministic tie-break.

    Ties (equal RD-cost) resolve to the HIGHER-precision option (lower hull index,
    i.e. fewer bytes is index 0 here because points are byte-ascending -> we prefer
    the cheaper option only when strictly cheaper; on a tie prefer more precision).
    """
    best_idx = 0
    best_cost = tensor.points[0].distortion + lam * tensor.points[0].rate_bytes
    best_rank = _option_rank(tensor.points[0].option)
    for idx in range(1, len(tensor.points)):
        p = tensor.points[idx]
        cost = p.distortion + lam * p.rate_bytes
        rank = _option_rank(p.option)
        # Strictly lower cost wins; on a tie, prefer higher precision (smaller rank).
        if cost < best_cost - 1e-18 or (abs(cost - best_cost) <= 1e-18 and rank < best_rank):
            best_idx, best_cost, best_rank = idx, cost, rank
    return best_idx


def _footprint(tensors: list[Tensor]) -> int:
    return sum(t.chosen.rate_bytes for t in tensors)


def _distortion(tensors: list[Tensor]) -> float:
    return sum(t.chosen.distortion for t in tensors)


def _assign_at_lambda(tensors: list[Tensor], lam: float) -> None:
    for t in tensors:
        t.chosen_idx = _argmin_at_lambda(t, lam)


def _greedy_topup(tensors: list[Tensor], budget: int) -> None:
    """Spend leftover budget on the globally steepest distortion-per-added-byte upgrade.

    Each candidate is a single hull step (chosen_idx -> a higher-precision point).
    Because every step is a convex-hull edge, the best available dD/dByte is
    monotonically non-increasing, so greedy is the water-filling continuation and
    stays on the lower convex hull of the global R-D curve.
    """
    used = _footprint(tensors)
    while True:
        best_t: Optional[Tensor] = None
        best_to_idx = -1
        best_gain_per_byte = 0.0
        best_added = 0
        for t in tensors:
            cur = t.chosen
            # Higher-precision points are LATER in `points` only if they cost more
            # bytes; our points are byte-ascending so a precision upgrade = a later
            # index with more bytes and less distortion.
            for j in range(t.chosen_idx + 1, len(t.points)):
                cand = t.points[j]
                added = cand.rate_bytes - cur.rate_bytes
                if added <= 0:
                    continue
                if used + added > budget:
                    continue
                drop = cur.distortion - cand.distortion  # >= 0 by monotonicity
                gain_per_byte = drop / added
                # Strictly-better gain wins; tie-break deterministically by name+option.
                if gain_per_byte > best_gain_per_byte + 1e-18 or (
                    abs(gain_per_byte - best_gain_per_byte) <= 1e-18 and best_t is not None
                    and (t.name, cand.option) < (best_t.name, best_t.points[best_to_idx].option)
                ):
                    best_t, best_to_idx, best_gain_per_byte, best_added = t, j, gain_per_byte, added
                break  # only the immediate next hull step is a single edge
        if best_t is None or best_gain_per_byte <= 0.0:
            return
        best_t.chosen_idx = best_to_idx
        used += best_added


def allocate_waterfill(tensors: list[Tensor], budget: int) -> dict:
    """Water-filling allocation: bisection on lambda + greedy hull top-up."""
    cheapest = sum(t.cheapest().rate_bytes for t in tensors)
    costliest = sum(t.costliest().rate_bytes for t in tensors)
    if budget < cheapest:
        raise AllocatorError(
            f"budget {budget} bytes is infeasible: cheapest config needs {cheapest} bytes")
    if budget >= costliest:
        # Whole model fits at top precision; pick costliest everywhere.
        for t in tensors:
            t.chosen_idx = len(t.points) - 1
        return _finish(tensors, budget, method="lagrangian-waterfill", lambda_star=0.0)

    # Bisection on lambda for the LARGEST lambda whose footprint <= budget (the
    # smallest feasible footprint at a single price), then greedy top-up the slack.
    lo, hi = 0.0, 1.0
    # Grow hi until footprint(hi) <= budget (cheap config).  hi large => all-cheapest.
    _assign_at_lambda(tensors, hi)
    while _footprint(tensors) > budget:
        hi *= 2.0
        _assign_at_lambda(tensors, hi)
        if hi > 1e30:
            break
    lam_feasible = hi
    for _ in range(200):  # ~1e-60 resolution; far more than enough
        mid = (lo + hi) / 2.0
        _assign_at_lambda(tensors, mid)
        if _footprint(tensors) <= budget:
            lam_feasible = mid
            hi = mid
        else:
            lo = mid
    _assign_at_lambda(tensors, lam_feasible)
    _greedy_topup(tensors, budget)
    return _finish(tensors, budget, method="lagrangian-waterfill", lambda_star=lam_feasible)


# ----------------------------------------------------------------------------
# Exact bounded-knapsack DP (closes the Lagrangian duality gap when requested).
# ----------------------------------------------------------------------------

def allocate_exact_dp(tensors: list[Tensor], budget: int, grid_bytes: int) -> dict:
    """Exact integer-optimal allocation via 1-D DP over a byte-quantized budget.

    Minimizes Sum D_t subject to Sum R_t <= budget, discretizing bytes to a grid of
    `grid_bytes` cells.  Complexity O(T * |O| * (budget/grid)).  For |O|=4 and a
    moderate grid this is cheap and removes the Lagrangian duality gap (spec §3.4).

    Implementation notes (correctness):
      * Each option's cost is FLOORED to whole cells, so the DP can OVER-estimate how
        much fits (never under-estimate).  After reconstruction we verify the EXACT
        reconstructed footprint <= budget; if floor rounding let a phantom upgrade in
        that overflows, we shrink the grid and retry (bounded retries) so the returned
        allocation is always genuinely feasible.
      * Among cells achieving the global minimum distortion we pick the LARGEST fill,
        so leftover budget is spent on precision (never silently wasted).
    """
    cheapest = sum(t.cheapest().rate_bytes for t in tensors)
    if budget < cheapest:
        raise AllocatorError(
            f"budget {budget} bytes is infeasible: cheapest config needs {cheapest} bytes")

    attempt_grid = max(1, int(grid_bytes))
    for _attempt in range(6):  # progressively finer grid until exactly feasible
        chosen = _dp_once(tensors, budget, attempt_grid)
        # Apply and check the EXACT reconstructed footprint.
        for t, oi in zip(tensors, chosen):
            t.chosen_idx = oi
        if _footprint(tensors) <= budget:
            return _finish(tensors, budget, method="exact-dp", lambda_star=None)
        attempt_grid = max(1, attempt_grid // 4)  # refine and retry
    # Final guard: fall back to the largest-lambda water-filling allocation, which is
    # always exactly feasible.  (Reaches here only if even grid=1 over-rounded, which
    # cannot happen with integer bytes; kept for total robustness.)
    return allocate_waterfill(tensors, budget)


def _dp_once(tensors: list[Tensor], budget: int, grid_bytes: int) -> list[int]:
    """One DP pass at a fixed grid; returns the chosen option index per tensor."""
    cells = max(1, budget // grid_bytes)
    INF = math.inf
    n_opts = max(len(t.points) for t in tensors)
    if n_opts > 255:  # the (c<<8)|oi packing assumes <=255 options per tensor
        raise AllocatorError("DP supports at most 255 options per tensor")
    # dp[c] = min total distortion reachable using exactly c grid-cells of budget.
    dp = [0.0] + [INF] * cells
    choice: list[list[int]] = []
    for t in tensors:
        ndp = [INF] * (cells + 1)
        tchoice = [-1] * (cells + 1)
        # Order options by ascending bytes (fewer bytes preferred on distortion ties).
        opts = sorted(range(len(t.points)),
                      key=lambda i: (t.points[i].rate_bytes, _option_rank(t.points[i].option)))
        for c in range(cells + 1):
            base_d = dp[c]
            if base_d == INF:
                continue
            for oi in opts:
                p = t.points[oi]
                cost_cells = p.rate_bytes // grid_bytes  # FLOOR (never under-fits)
                nc = c + cost_cells
                if nc > cells:
                    continue
                nd = base_d + p.distortion
                if nd < ndp[nc] - 1e-18:
                    ndp[nc] = nd
                    tchoice[nc] = (c << 8) | oi
        dp = ndp
        choice.append(tchoice)
    # Global minimum distortion, then the LARGEST fill achieving it (spend the slack).
    best_d = min(d for d in dp if d < INF)
    best_c = max(c for c in range(cells + 1) if dp[c] <= best_d + 1e-18)
    if best_c < 0 or dp[best_c] == INF:
        raise AllocatorError("DP found no feasible allocation (budget too small)")
    chosen = [0] * len(tensors)
    c = best_c
    for ti in range(len(tensors) - 1, -1, -1):
        packed = choice[ti][c]
        if packed < 0:
            raise AllocatorError("DP backtrack failed (internal inconsistency)")
        prev_c, oi = packed >> 8, packed & 0xFF
        chosen[ti] = oi
        c = prev_c
    return chosen


# ----------------------------------------------------------------------------
# Deterministic fallback: uniform Q4_K_M-class allocation (spec §6).
# ----------------------------------------------------------------------------

def allocate_uniform(tensors: list[Tensor], uniform_option: str, budget: Optional[int]) -> dict:
    """Uniform allocation: every tensor to `uniform_option` (or its tier-floor / pin)."""
    for t in tensors:
        # Respect pins and tier floors; otherwise take the uniform option if available,
        # else the closest higher-precision option present on the hull.
        target_rank = _option_rank(uniform_option)
        # Candidate points at or above the uniform precision (uniform_option or higher).
        eligible = [i for i, p in enumerate(t.points) if _option_rank(p.option) <= target_rank]
        if eligible:
            # Prefer the LOWEST precision >= floor that equals uniform if present.
            exact = [i for i in eligible if t.points[i].option == uniform_option]
            t.chosen_idx = exact[0] if exact else max(eligible, key=lambda i: _option_rank(t.points[i].option))
        else:
            # uniform_option is cheaper than everything kept (e.g. pinned bf16); take cheapest kept.
            t.chosen_idx = min(range(len(t.points)), key=lambda i: t.points[i].rate_bytes)
    eff_budget = budget if budget is not None else _footprint(tensors)
    return _finish(tensors, eff_budget, method=f"uniform-{uniform_option}", lambda_star=None)


# ----------------------------------------------------------------------------
# Result assembly.
# ----------------------------------------------------------------------------

def _uniform_equal_footprint_distortion(tensors: list[Tensor], target_bytes: int) -> dict:
    """The densest uniform option whose footprint <= target_bytes, for the proof pre-check.

    Tries each option from cheapest to most precise; returns the most-precise uniform
    config that still fits `target_bytes`, with its (footprint, distortion).
    """
    # Candidate uniform options = the union of options present across tensors,
    # ordered by ascending bytes (cheapest first).
    all_opts: dict[str, float] = {}
    for t in tensors:
        for p in t.points:
            all_opts[p.option] = p.bits
    ordered = sorted(all_opts.items(), key=lambda kv: (-kv[1], _option_rank(kv[0])))  # least bits first
    best = None
    for option, _bpw in ordered:
        fp = 0
        dd = 0.0
        ok = True
        for t in tensors:
            # nearest available option at or above this precision (respect pins/floors)
            target_rank = _option_rank(option)
            eligible = [p for p in t.points if _option_rank(p.option) <= target_rank]
            if eligible:
                exact = [p for p in eligible if p.option == option]
                p = exact[0] if exact else max(eligible, key=lambda p: _option_rank(p.option))
            else:
                p = min(t.points, key=lambda p: p.rate_bytes)
            fp += p.rate_bytes
            dd += p.distortion
        if fp <= target_bytes:
            cand = {"option": option, "footprint_bytes": fp, "distortion": round(dd, 9)}
            if best is None or fp > best["footprint_bytes"]:
                best = cand
        del ok
    if best is None:
        # Even the cheapest uniform exceeds target; report the cheapest as the reference.
        option, _bpw = ordered[0]
        fp = sum(min(t.points, key=lambda p: p.rate_bytes).rate_bytes for t in tensors)
        dd = sum(min(t.points, key=lambda p: p.rate_bytes).distortion for t in tensors)
        best = {"option": option, "footprint_bytes": fp, "distortion": round(dd, 9)}
    return best


def _finish(tensors: list[Tensor], budget: int, method: str, lambda_star: Optional[float]) -> dict:
    footprint = _footprint(tensors)
    distortion = _distortion(tensors)
    uniform = _uniform_equal_footprint_distortion(tensors, budget if budget else footprint)

    allocation = []
    for t in tensors:
        cur = t.chosen
        # marginal distortion-per-byte of the next available upgrade (hull slope).
        marginal = 0.0
        if t.chosen_idx + 1 < len(t.points):
            nxt = t.points[t.chosen_idx + 1]
            added = nxt.rate_bytes - cur.rate_bytes
            if added > 0:
                marginal = (cur.distortion - nxt.distortion) / added
        if t.pin is not None:
            reason = f"pinned:{t.pin}"
        elif t.tier_floor is not None and cur.option == t.tier_floor:
            reason = f"tier-floored:{t.tier_floor}"
        elif t.chosen_idx == 0:
            reason = "flat: starved to cheapest (return below price)"
        elif t.chosen_idx == len(t.points) - 1:
            reason = "steep: upgraded to highest available precision"
        else:
            reason = "interior: upgraded one or more hull steps"
        allocation.append({
            "tensor": t.name,
            "numel": t.numel,
            "option": cur.option,
            "bits_per_weight": round(cur.bits, 4),
            "bytes": cur.rate_bytes,
            "distortion": round(cur.distortion, 9),
            "marginal_dpb": float(f"{marginal:.6e}"),
            "reason": reason,
        })

    # Surrogate pre-check (spec §5 step 1 / §7 P6): allocated distortion must not
    # exceed the equal-footprint uniform baseline.  By construction it shouldn't.
    surrogate_ok = distortion <= uniform["distortion"] + 1e-12

    table = {
        "schema_version": SCHEMA_VERSION,
        "generator": GENERATOR,
        "method": method,
        "lambda_star": lambda_star,
        "budget_bytes": int(budget),
        "totals": {
            "footprint_bytes": footprint,
            "footprint_gib": round(footprint / (1024 ** 3), 6),
            "distortion": round(distortion, 9),
            "uniform_baseline": uniform,
            "surrogate_precheck_pass": bool(surrogate_ok),
        },
        "n_tensors": len(tensors),
        "allocation": allocation,
    }
    return table


# ----------------------------------------------------------------------------
# Self-test: a small synthetic model proving the core invariants.
# ----------------------------------------------------------------------------

def _synthetic_doc() -> dict:
    """A 6-tensor synthetic model: 2 steep, 2 flat, 1 pinned, 1 tier-floored.

    The two steep and two flat tensors share the SAME numel so that comparing their
    chosen precision rank is a meaningful test of "bits go where distortion-per-bit
    is steepest" (different-sized tensors confound a raw rank comparison because a
    huge tensor may be too big to upgrade even when steep).
    """
    def curve(d_int8: float, d_g32: float, d_g16: float) -> dict:
        return {
            "bf16": {"bits": 16.0, "distortion": 0.0},
            "int8": {"bits": 8.03, "distortion": d_int8},
            "int4-g16": {"bits": 6.0, "distortion": d_g16},
            "int4-g32": {"bits": 5.0, "distortion": d_g32},
        }
    n = 1_146_880  # common size for the steep/flat quartet (1280 x 896, an expert proj)
    return {
        "option_bits": dict(DEFAULT_OPTION_BITS),
        "tensors": [
            # steep tensors: distortion rises fast as bits drop -> should be upgraded.
            {"tensor": "steep.a", "numel": n, "pin": None, "tier_floor": None,
             "curve": curve(0.00030, 0.00500, 0.00210)},
            {"tensor": "steep.b", "numel": n, "pin": None, "tier_floor": None,
             "curve": curve(0.00025, 0.00420, 0.00190)},
            # flat tensors: little distortion even at int4 -> should be starved.
            {"tensor": "flat.a", "numel": n, "pin": None, "tier_floor": None,
             "curve": curve(0.00002, 0.00009, 0.00005)},
            {"tensor": "flat.b", "numel": n, "pin": None, "tier_floor": None,
             "curve": curve(0.00003, 0.00011, 0.00006)},
            # pinned bf16 (high-precision set).
            {"tensor": "router.gate", "numel": 81_920, "pin": "bf16", "tier_floor": None,
             "curve": curve(0.00100, 0.01000, 0.00500)},
            # tier-floored int8 (the _M discipline: v_proj / down_proj never below int8).
            {"tensor": "attn.v", "numel": 163_840, "pin": None, "tier_floor": "int8",
             "curve": curve(0.00040, 0.00800, 0.00300)},
        ],
    }


def _run_selftest() -> int:
    doc = _synthetic_doc()
    tensors = load_tensors(doc)

    # Choose a budget that forces a genuine steep-vs-flat tradeoff: the all-cheapest
    # footprint plus enough slack to upgrade the two steep tensors toward int8 but NOT
    # enough to also upgrade the (equal-sized) flat tensors.  This is the regime where
    # water-filling must actually allocate, not the trivial "everything fits at bf16".
    cheapest = sum(t.cheapest().rate_bytes for t in tensors)
    costliest = sum(t.costliest().rate_bytes for t in tensors)
    # One steep tensor's int4-g32 -> int8 upgrade costs ~ (8.03-5.0)/8 * numel bytes.
    by = {t.name: t for t in tensors}
    steep_upgrade = (by["steep.a"].costliest().rate_bytes - by["steep.a"].cheapest().rate_bytes)
    budget = cheapest + int(2.6 * steep_upgrade)  # room for ~2 steep upgrades, not the flats

    table = allocate_waterfill(load_tensors(doc), budget)
    failures: list[str] = []

    # Invariant 1: footprint within budget.
    if table["totals"]["footprint_bytes"] > budget:
        failures.append(f"footprint {table['totals']['footprint_bytes']} exceeds budget {budget}")

    # Invariant 2: surrogate pre-check passes (allocated <= equal-footprint uniform).
    if not table["totals"]["surrogate_precheck_pass"]:
        failures.append("surrogate pre-check failed (allocated distortion > uniform baseline)")

    # Invariant 3: pins / tier floors respected.
    by_name = {a["tensor"]: a for a in table["allocation"]}
    if by_name["router.gate"]["option"] != "bf16":
        failures.append("pinned tensor not bf16")
    if _option_rank(by_name["attn.v"]["option"]) > _option_rank("int8"):
        failures.append("tier-floored tensor went below int8")

    # Invariant 4: among the equal-sized quartet, water-filling spends bits on the
    # STEEP tensors before the flat ones — every steep tensor sits at >= the precision
    # (more or equal bits, i.e. <= rank) of every flat tensor.  (Equal numel makes the
    # rank comparison meaningful; cf. the docstring of _synthetic_doc.)
    worst_steep_rank = max(_option_rank(by_name["steep.a"]["option"]),
                           _option_rank(by_name["steep.b"]["option"]))
    best_flat_rank = min(_option_rank(by_name["flat.a"]["option"]),
                         _option_rank(by_name["flat.b"]["option"]))
    if worst_steep_rank > best_flat_rank:  # higher rank = lower precision
        failures.append(
            f"water-filling starved a steep tensor (rank {worst_steep_rank}) below a "
            f"flat one (rank {best_flat_rank})")
    # Invariant 4b: the budget actually forced a tradeoff (not everything at bf16).
    if all(a["option"] == "bf16" for a in table["allocation"]):
        failures.append("budget too loose: every tensor reached bf16 (no tradeoff exercised)")

    # Invariant 5: determinism — same input twice yields byte-identical output.
    t1 = json.dumps(allocate_waterfill(load_tensors(doc), budget), sort_keys=True)
    t2 = json.dumps(allocate_waterfill(load_tensors(doc), budget), sort_keys=True)
    if t1 != t2:
        failures.append("non-deterministic allocation (two runs differ)")

    # Invariant 6: exact DP never does worse than water-filling on distortion.
    dp_table = allocate_exact_dp(load_tensors(doc), budget, grid_bytes=4096)
    if dp_table["totals"]["distortion"] > table["totals"]["distortion"] + 1e-9:
        failures.append(
            f"exact-dp distortion {dp_table['totals']['distortion']} worse than "
            f"waterfill {table['totals']['distortion']}")

    # Invariant 7: fallback (uniform) is feasible and >= allocated distortion.
    fb = allocate_uniform(load_tensors(doc), "int4-g32", budget)
    if fb["totals"]["distortion"] + 1e-12 < table["totals"]["distortion"]:
        failures.append("uniform fallback beat the allocator (impossible at equal/looser footprint)")

    result = {
        "selftest": "af1_bit_allocator",
        "budget_bytes": budget,
        "waterfill_footprint": table["totals"]["footprint_bytes"],
        "waterfill_distortion": table["totals"]["distortion"],
        "exact_dp_distortion": dp_table["totals"]["distortion"],
        "uniform_fallback_distortion": fb["totals"]["distortion"],
        "result": "pass" if not failures else "fail",
        "failures": failures,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failures else 1


# ----------------------------------------------------------------------------
# CLI.
# ----------------------------------------------------------------------------

def _parse_budget(args: argparse.Namespace, tensors: list[Tensor]) -> Optional[int]:
    if args.budget_bytes is not None:
        return int(args.budget_bytes)
    if args.budget_gib is not None:
        return int(args.budget_gib * (1024 ** 3))
    if args.budget_gb is not None:
        return int(args.budget_gb * (1000 ** 3))
    return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="af1_bit_allocator.py",
        description=(
            "AF-1 rate-distortion / Lagrangian water-filling per-tensor bit allocator. "
            "Consumes per-tensor {bits:distortion} curves and emits the optimal "
            "bit_allocation_table.json. See docs/alien/AF-1-rate-distortion-bit-allocation.md."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  af1_bit_allocator.py curves.json --budget-gib 2.0 -o table.json\n"
            "  af1_bit_allocator.py curves.json --budget-gib 2.0 --exact-dp\n"
            "  af1_bit_allocator.py curves.json --fallback --uniform-option int4-g32\n"
            "  af1_bit_allocator.py --selftest\n"
        ),
    )
    p.add_argument("curves", nargs="?", default=None,
                   help="per-tensor curves JSON (the converter's --measure-distortion output); "
                        "omit with --selftest")
    b = p.add_argument_group("budget (choose one)")
    b.add_argument("--budget-gib", type=float, default=None, help="footprint budget B in GiB (2^30)")
    b.add_argument("--budget-gb", type=float, default=None, help="footprint budget B in GB (10^9)")
    b.add_argument("--budget-bytes", type=int, default=None, help="footprint budget B in bytes")
    m = p.add_argument_group("method")
    m.add_argument("--exact-dp", action="store_true",
                   help="use the exact bounded-knapsack DP instead of water-filling (closes the duality gap)")
    m.add_argument("--dp-grid-bytes", type=int, default=1_048_576,
                   help="byte granularity for the --exact-dp budget grid (default 1 MiB)")
    m.add_argument("--fallback", action="store_true",
                   help="emit the deterministic uniform Q4_K_M-class fallback table (spec §6)")
    m.add_argument("--uniform-option", default="int4-g32",
                   help="the option for --fallback / the uniform baseline (default: int4-g32)")
    p.add_argument("-o", "--output", default="-",
                   help="write the bit_allocation_table.json here (default: - = stdout)")
    p.add_argument("--source-sha256", default=None,
                   help="sha256 of the source safetensors, copied into the table for provenance")
    p.add_argument("--selftest", action="store_true",
                   help="run the built-in synthetic self-test and exit (no input file needed)")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.selftest:
        return _run_selftest()

    if args.curves is None:
        parser.error("a curves JSON path is required (or pass --selftest)")

    try:
        doc = json.loads(Path(args.curves).read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(json.dumps({"error": f"curves file not found: {args.curves}"}), file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"invalid JSON in {args.curves}: {exc}"}), file=sys.stderr)
        return 2

    try:
        tensors = load_tensors(doc)
        budget = _parse_budget(args, tensors)

        if args.fallback:
            table = allocate_uniform(tensors, args.uniform_option, budget)
        else:
            if budget is None:
                parser.error("a budget is required (--budget-gib / --budget-gb / --budget-bytes) "
                             "unless --fallback")
            if args.exact_dp:
                table = allocate_exact_dp(tensors, budget, args.dp_grid_bytes)
            else:
                table = allocate_waterfill(tensors, budget)
    except AllocatorError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 3

    table["option_set"] = sorted(
        {p.option for t in tensors for p in t.points},
        key=_option_rank,
    )
    if args.source_sha256 is not None:
        table["source_sha256"] = args.source_sha256

    out = json.dumps(table, indent=2, sort_keys=True)
    if args.output == "-":
        print(out)
    else:
        Path(args.output).write_text(out + "\n", encoding="utf-8")
        print(json.dumps({
            "wrote": args.output,
            "method": table["method"],
            "footprint_bytes": table["totals"]["footprint_bytes"],
            "footprint_gib": table["totals"]["footprint_gib"],
            "distortion": table["totals"]["distortion"],
            "surrogate_precheck_pass": table["totals"]["surrogate_precheck_pass"],
        }), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
