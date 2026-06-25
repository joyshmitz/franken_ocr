# AF-4 — Submodular maximization for the high-precision tensor set

**Beads:** `bd-1xfa.4` (SPIKE), `bd-1xfa.4.1` (tests). **Family:** Monotone submodular
maximization under a knapsack constraint. **Tier B · EV medium.**
**Plan refs:** §9.7 AF-4 (recommendation contract), §2.6 (the validated prior-art
heuristic set / OQ-14), §5–§6 (the quant recipe). **Feeds:** E-P4 (the int4 split);
**complementary to** AF-1 (rate-distortion bit-allocation — two views of one budget);
**vetoed by** AF-2 (the CVaR/EVT tail bound). **Doctrine:** AGENTS.md #1 (correctness
outranks speed), #8 (honest measured everything), Alien-Artifact Engineering Contract.

---

## 1. The decision this artifact makes

> *Under a footprint budget `B`, which tensors stay high-precision (BF16/F32) and which
> drop to a quantized tier?*

This is the **dual of AF-1**. AF-1 asks "what bit-width per quantizable tensor under a
total-bit budget" (a continuous rate-distortion / water-filling allocation); AF-4 asks
the discrete set question "which tensors are *kept out* of quantization entirely." The
two are two views of the same footprint budget and must agree on the hard invariants
(§4): the router gate, all norms, and the vision tower are never demoted.

The status-quo answer is the **hand-picked NVFP4/GGUF heuristic** (§2.6): keep the
vision tower, projector, embeddings, MoE router gate, and **all** norms high-precision;
quantize the expert/dense FFN bulk. That heuristic is *folklore* from prior-art quants —
it works, but it is uncertified, and at a tighter-than-prior budget it may be leaving
accuracy-per-byte on the table (or, worse, spending bytes on a tensor that no longer
earns them). AF-4 replaces the hand-pick with a **greedy selection carrying a
`(1 − 1/e)` near-optimality certificate** that should *recover* the folklore (validating
it) and *may refine* it at the margin.

---

## 2. The mathematical structure

### 2.1 State / action / objective (the alien contract)

- **State** `S ⊆ V`: the set of tensors kept high-precision, drawn from the census menu
  `V` (the ~2244 `*_proj.weight` Linears + `lm_head`; CENSUS §(a)). Each tensor `t` has
  a byte cost `c_t` = (high-precision bytes − quantized bytes), the footprint it *spends*
  to stay high-precision.
- **Action**: greedily add the tensor with the largest **marginal accuracy-gain per byte**
  until the budget is exhausted.
- **Objective** `f(S)`: accuracy recovered by keeping `S` high-precision, maximized under
  the **knapsack constraint**

  ```
  max  f(S)   s.t.   Σ_{t ∈ S} c_t ≤ B
  ```

### 2.2 Why `f` is (assumed) monotone submodular

- **Monotone:** keeping *more* tensors high-precision never *reduces* recovered accuracy
  (`S ⊆ T ⟹ f(S) ≤ f(T)`). Adding precision back can only help or be neutral.
- **Submodular (diminishing returns):** the marginal gain of promoting tensor `t` shrinks
  as the kept set grows —

  ```
  f(S ∪ {t}) − f(S)  ≥  f(T ∪ {t}) − f(T)   for all  S ⊆ T,  t ∉ T
  ```

  Intuitively: once you have already recovered most of the accuracy, the *next*
  high-precision tensor recovers less. This is the well-motivated property that licenses
  the greedy guarantee. **It is an ASSUMPTION here, not a theorem** (cross-tensor coupling
  can violate it); §6 says how we check it and why the proof obligation does not depend on
  it holding exactly.

### 2.3 The marginal-gain signal

For a candidate tensor `t` not yet kept, the marginal gain is the **accuracy recovered
when `t` is promoted from its quantized tier back to high precision**, measured as either:

- the **end-to-end CER improvement** on the dense-numeric calibration corpus (the gold
  signal, slower), or
- the **layer-output cosine recovery** on a calibration batch (the cheap proxy, the same
  signal AF-1 uses for its distortion curves `D_t(b)`).

The greedy ranks by **gain per byte** `Δ_t / c_t` (the cost-effective variant for a
knapsack), not raw gain — spend the budget where each byte buys the most accuracy.

---

## 3. The algorithm: cost-effective greedy with a `(1 − 1/e)` guarantee

A plain greedy that maximizes raw marginal gain has **no** constant-factor guarantee
under a *knapsack* (as opposed to a cardinality) constraint. The fix is the
**Khuller–Moss–Naor cost-effective greedy** (the standard knapsack-submodular result):

1. **Forced-in invariants first.** Seed `S` with the hard-locked members (router gate,
   all norms, vision tower — §4); subtract their cost from `B`. These are never
   candidates for demotion regardless of marginal gain.
2. **Cost-effective greedy run `G_ce`.** Repeatedly add the feasible tensor maximizing
   `Δ_t / c_t` (gain per byte) whose cost still fits the remaining budget, until no
   candidate fits.
3. **Single-best-item run `G_sb`.** Compute the best *single* tensor that fits the budget
   on its own (guards the pathological case where one heavy tensor dominates).
4. **Return `argmax(f(G_ce), f(G_sb))`.**

> **Guarantee.** For a monotone submodular `f` under a knapsack constraint, this
> two-candidate cost-effective greedy returns a set with
> `f(S) ≥ (1 − 1/e) · f(OPT) ≈ 0.632 · f(OPT)` — near-optimal, cheap, and **certifiable**.
> (The `(1 − 1/e)` constant is tight for monotone submodular maximization in general.)

### 3.1 Lazy / CELF acceleration

Submodularity (§2.2) means a tensor's marginal gain can **only decrease** as `S` grows.
So a gain computed in an earlier round is a valid **upper bound** in a later round. The
**lazy-greedy / CELF** trick maintains a max-heap of stale `Δ_t / c_t` values: each round,
pop the top, recompute *only that* candidate's current gain, and if it still tops the
heap, accept it without re-evaluating the rest. Over the ~2244-tensor menu this skips the
vast majority of re-evaluations while returning **exactly the same set** as naive greedy
(a unit-tested invariant, §7).

### 3.2 Determinism

Ties in `Δ_t / c_t` are broken **deterministically by tensor name** (lexicographic), so
the chosen set is reproducible and round-trips. The marginal-gain ledger records, per
accepted tensor: the tensor name, its marginal gain, gain-per-byte, the round it was
added, and the running footprint.

---

## 4. Hard invariants (forced-in, non-negotiable)

These are **validated-recipe locks** (§2.6 / OQ-14), not budget items. They are seeded
into `S` before greedy starts and are never demotion candidates, regardless of marginal
gain:

| Locked high-precision | Why (§2.6) |
|-----------------------|------------|
| **All norms** (RMSNorm, LayerNorm, LayerNorm2d) | tiny, sensitive; both GGUF and NVFP4 keep them BF16 |
| **MoE router gate** (`mlp.gate.weight`) | top-6 expert selection; a wrong route is catastrophic, not a small error |
| **Vision tower** (SAM + CLIP-L) | both prior quants keep it unquantized; the GGUF author states quantizing vision *hurts OCR* |
| **Projector + embeddings + `image_newline` / `view_seperator`** | NVFP4 keeps them BF16; small footprint, high leverage |

`lm_head` and attention `q/k/v/o` are **OUR risk** (§2.6) — they sit in the *negotiable*
menu and are gated behind their kill-switches; greedy may or may not keep them. The greedy
must **recover** the locked set as a floor; if greedy *would* demote a locked tensor for
budget, that is a bug (forced-in members are always present — unit-tested, §7).

---

## 5. The artifact

`focr convert --select-high-precision-set --budget <GB>`:

1. Reads the per-tensor byte costs from the census and the marginal gains (CER recovery or
   layer-output-cosine recovery) measured against the f32 oracle on the calibration corpus.
2. Runs the cost-effective greedy (§3) with the forced-in invariants (§4).
3. Writes the chosen high-precision set into the `.focrq` header (complementary to AF-1's
   `bit_allocation_table` — they share the footprint budget and must agree on the locks).
4. Emits the **marginal-gain ledger** to `docs/` (per tensor: gain, gain-per-byte, round,
   running footprint, the empirical-submodularity diagnostic, keep/revert decision).

Without `--select-high-precision-set`, the static NVFP4/GGUF heuristic recipe ships — the
runtime never depends on greedy having run (§8).

---

## 6. Submodularity is an assumption — how we guard it

`f` (accuracy recovered) may **not** be perfectly submodular: cross-tensor coupling can
make one tensor's gain *increase* given another is kept (a violation). We do not assume it
away — we **measure** it:

- **Empirical-submodularity diagnostic.** Along the greedy path, the accepted marginal
  gains (per byte) should be **non-increasing**. Log them; if a later addition has a
  *larger* gain-per-byte than an earlier one, **flag the violation** and treat the
  `(1 − 1/e)` guarantee as *advisory*, not load-bearing.
- **The proof obligation is the real gate (§8), not submodularity.** Whether or not `f` is
  exactly submodular, AF-4 only *keeps* if the greedy set's measured end-to-end CER ≤ the
  heuristic set's at **equal footprint** (and within AF-2's tail bound). The guarantee is a
  reason to *expect* a good set cheaply; the empirical CER comparison is what *certifies*
  it. A grossly violated submodularity assumption with a passing proof obligation is still
  a keep; a satisfied assumption with a failing proof obligation is a **revert**.

---

## 7. Galaxy-brain transparency card

> **Equation.**
> `max f(S) s.t. Σ_{t∈S} c_t ≤ B`, with `f` monotone submodular ⟹ cost-effective greedy
> gives `f(S) ≥ (1 − 1/e)·f(OPT) ≈ 0.632·f(OPT)`. Greedy ranks by **marginal gain per
> byte** `Δ_t / c_t`.
>
> **Substituted values** (filled at convert time from the measured run; ledgered):
> the chosen set, each kept tensor's marginal gain + gain-per-byte + round + running
> footprint, the final footprint, and the greedy-vs-heuristic CER at that footprint.
>
> **Plain English.** "Add the most accuracy-per-byte tensor next, with a proven
> near-optimality certificate. Start from the locked vision/router/norms; greedy should
> rebuild the hand-picked recipe and may refine it under a tight budget."
>
> **Validity assumptions.** `f` is monotone submodular in the kept set — i.e. marginal
> gains are non-increasing along the greedy path (checked empirically, §6); the per-tensor
> byte costs and marginal gains are measured against the same f32 oracle and corpus AF-1
> uses; the budget `B` is the same one AF-1's bit-allocation honors.
>
> **What would flip the decision.** A non-submodular interaction — a tensor whose gain
> *increases* given others are kept — voids the `(1 − 1/e)` guarantee → treat it as
> advisory and lean entirely on the proof obligation. If the proof obligation fails
> (greedy CER > heuristic CER at equal footprint, or AF-2's CVaR is blown) → **revert to
> the heuristic set**, ledger in `docs/NEGATIVE_EVIDENCE.md`.

---

## 8. Deterministic fallback (wired FIRST — AGENTS.md #5)

> The runtime never depends on greedy having run.

The fallback is the **validated NVFP4/GGUF heuristic set** (§2.6): vision tower,
projector, embeddings, MoE router gate, and all norms high-precision; the expert/dense FFN
bulk quantized. Without `--select-high-precision-set`, this static recipe ships. The
fallback engages whenever:

- The selector was not run (no flag).
- The proof obligation (§9) fails — greedy CER > heuristic CER at equal footprint, or
  AF-2's tail bound is violated.
- Submodularity is grossly violated *and* the proof obligation does not independently
  clear the keep-gate.

Because the heuristic set is a strict superset-floor of the locks (§4), the fallback is
always a valid, shippable recipe.

---

## 9. Proof obligation & acceptance (bd-1xfa.4.1)

**Proof obligation (the AF-4 gate, AGENTS.md #1):**

```
CER(greedy set)  ≤  CER(heuristic set)     at EQUAL footprint
```

on the dense-numeric corpus, **and** `CVaR_0.1(greedy)` within AF-2's ledgered budget
(the tail veto). Mean-CER alone is insufficient — OCR fails in the tail (AF-2), so the
comparison runs both sets through the full parity ladder and the tail-risk monitor.

Acceptance criteria:

1. `focr convert --select-high-precision-set --budget` emits a **deterministic**
   high-precision set + marginal-gain ledger.
2. Proof obligation **met**: greedy CER ≤ heuristic CER at equal footprint, within AF-2's
   CVaR/EVT bound.
3. The forced-in invariants (norms / router gate / vision) are honored (always present).
4. This card + the assumptions ledger (§10) committed.
5. **If the proof fails or submodularity is grossly violated: REVERT to the heuristic set,
   ledger in `docs/NEGATIVE_EVIDENCE.md`.**

### Tests (bd-1xfa.4.1)

- **Unit (deterministic, arch-independent):** on a *synthetic monotone-submodular knapsack
  instance* (with a known optimum), cost-effective greedy returns a set within
  `(1 − 1/e)` of OPT; **lazy/CELF greedy == naive greedy** (same chosen set); forced-in
  members are always kept; ties broken by tensor name (round-trip deterministic);
  marginal-gain ledger correctness.
- **E2E (proof obligation; model-gated, skip-with-SUCCESS without weights):** convert at
  equal footprint with the greedy set vs the heuristic set; run both through the parity
  ladder + tail-risk monitor; assert `CER(greedy) ≤ CER(heuristic)` and
  `CVaR_0.1(greedy)` within budget (calls AF-2).
- **Submodularity diagnostic:** log whether accepted marginal gains are non-increasing
  along the greedy path; flag violations.
- **Structured logging** under `tests/artifacts/af4/`: chosen set, per-tensor marginal
  gains / gain-per-byte / round / footprint, greedy-vs-heuristic CER + tail metrics,
  provenance, keep/revert decision.

---

## 10. Assumptions ledger

| # | Assumption | Why it holds / how checked | What breaks it | Fallback if broken |
|---|-----------|----------------------------|----------------|--------------------|
| A1 | `f` (accuracy recovered) is monotone | adding precision back can only help | — (monotonicity is robust) | n/a |
| A2 | `f` is submodular (diminishing returns) | well-motivated; **empirically checked** along the greedy path (§6) | cross-tensor coupling raising a later gain | guarantee → advisory; lean on the proof obligation |
| A3 | Byte costs `c_t` are exact | from the census `weight_map` (CENSUS §a/§b) | a shape/dtype the census missed | census CI guard catches drift |
| A4 | Marginal gains measured vs the SAME f32 oracle + corpus as AF-1 | shared calibration batch / oracle | oracle drift between AF-1 and AF-4 | provenance check; re-measure |
| A5 | The locks (router/norms/vision) are correct | §2.6 prior-art + OQ-14 | OQ-14 reconciliation changes the validated set | re-seed forced-in members; re-run |
| A6 | The budget `B` matches AF-1's | both honor the same footprint target | divergent budgets between AF-1/AF-4 | reconcile; the two must agree |
| A7 | Greedy recovers the heuristic floor | the heuristic is a near-optimal hand-pick | greedy demotes a locked tensor | forced-in invariant prevents it (unit-tested) |

---

## 11. Composition with AF-1 and AF-2

The three quant families compose and share one footprint budget:

- **AF-4 chooses the high-precision SET** (which tensors stay out of quantization).
- **AF-1 chooses the bit-widths** within the quantized remainder (int4-g16 / int4-g32 /
  int8 / bf16 via rate-distortion water-filling). AF-4 and AF-1 are *two views of one
  budget* and must agree on the locks (router / norms / vision never demoted).
- **AF-2 vetoes** any set/allocation whose CVaR/EVT tail bound is blown — the release gate.

The keep order is: AF-4/AF-1 propose a footprint-feasible recipe → AF-2 gates it on the
tail → if it clears, it feeds E-P4 (the int4 split); otherwise revert (§8).

---

## 12. Status

DESIGN complete. Awaiting: the census byte-cost table + the f32 CER oracle (E-P1) and the
quantizer that produces the quantized remainder (E-P2) to measure marginal gains; the
`focr convert --select-high-precision-set` artifact + ledger; and the measured proof
obligation E2E (bd-1xfa.4.1) against AF-2's tail metrics. Until then the deterministic
NVFP4/GGUF heuristic set (§8) ships.
