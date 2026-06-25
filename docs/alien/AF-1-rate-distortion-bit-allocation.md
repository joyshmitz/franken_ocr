# AF-1 — Rate-Distortion / Lagrangian Water-Filling for Per-Tensor Bit Allocation

> **Bead:** `bd-ksps` / `ALIEN-af1` / `P4-af1-bit-allocator` (DESIGN).
> **Plan source:** [`COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md`](../../COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md) §9.7 (AF-1), §5 (`.focrq` format / quant recipe), §6.3 (per-arch int8/int4 GEMM), §9.5 (expected-loss-guided per-layer quant), Phase 4 (§10).
> **Companion AF families:** AF-2 tail-risk CER gate (CVaR + EVT), AF-4 submodular high-precision-set selection, AF-5 USL pool sizing.
> **Runnable prototype:** [`scripts/af1_bit_allocator.py`](../../scripts/af1_bit_allocator.py).
> **Provenance:** model facts trace to the pinned source (HF commit `3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5`); the tensor census (2244 quantizable linears, dims `hidden=1280`, `intermediate=6848`, `moe_intermediate=896`, 64 routed + 2 shared experts, top-6, 12 layers, vocab 129280) is line-backed in [`truth-pack/CENSUS.md`](../truth-pack/CENSUS.md).

---

## 0. TL;DR (what this artifact is)

Choosing **`{bf16, int8, int4-g32, int4-g16}` per quantizable tensor** under a total-footprint
budget is a **rate-distortion (R-D) allocation problem**, not a uniform bit-width pick. The
classic solution is **Lagrangian water-filling**: introduce a single price `λ` (bits per unit of
end-to-end distortion), and at that price each tensor *independently* picks the bit-width that
minimizes `D_t(b) + λ·R_t(b)`. Sweeping `λ` from `+∞` (everything cheapest) to `0` (everything
biggest) traces the **lower convex hull of the global R-D curve** — the Pareto frontier of
`(footprint, distortion)`. We walk that frontier until the footprint just fits the budget `B`.

The output is an offline **`bit_allocation_table`** baked into the `.focrq` header, computed by
`focr convert --optimize-bits --budget <GB>` from per-tensor distortion curves `D_t(b)` measured
as the **layer-output cosine drop** on a calibration batch.

**Proof obligation:** the allocated config's end-to-end CER ≤ the uniform-bit config's CER at
**equal footprint**, on the dense-numeric corpus. **Deterministic fallback:** uniform
Q4_K_M-class allocation (§5) — wired first, always available, selected automatically when any
precondition (§7) fails.

This document specifies: the objective (§1), the distortion model `D_t(b)` (§2), the convex-hull /
water-filling algorithm and its optimality argument (§3), the `bit_allocation_table` emitted
format (§4), the proof obligation and how it is discharged (§5), the deterministic fallback (§6),
the galaxy-brain transparency card (§8), and the assumptions ledger (§9).

---

## 1. Objective

### 1.1 The candidate bit-widths

For each quantizable tensor `t` the allocator may assign one of an ordered set of **bit options**
`O = {bf16, int8, int4-g32, int4-g16}` (more options can be appended; the algorithm is generic
over a finite option set). Effective bits-per-weight (bpw), counting the inline scale overhead:

| Option | Storage scheme (§5.1, §5.2) | Group/scale overhead | Effective bpw `r(o)` |
|---|---|---:|---:|
| `bf16` | verbatim 2-byte (high-precision set; **NOT** lossy f16, §5.2) | none | **16.0** |
| `int8` | per-output-channel symmetric, zp 0 (`scale = max\|w_row\|/127`) | 1 × f32 per output row (amortized over `K`) | **≈ 8.0–8.05** |
| `int4-g32` | per-group symmetric, group 32 (§5.1 "16–32") | 1 × f32 per 32 weights | **4 + 32/32 = 5.0** |
| `int4-g16` | per-group symmetric, group 16 (NVFP4 used 16) | 1 × f32 per 16 weights | **4 + 32/16 = 6.0** |

> The Q4_K_M-equivalent target band is **4.5–4.9 bpw** (§6.3): note that with naive f32 group
> scales `int4-g32 = 5.0` and `int4-g16 = 6.0`. Reaching ≤4.9 needs the K-quant scale-of-scales
> trick (a small per-super-block fp16 scale + 6-bit sub-scales). The allocator is **agnostic to the
> exact bpw of each option** — it consumes whatever `r(o)` the packer reports per tensor. The four
> options above are the v1 set; `r(o)` is supplied by the converter, not hard-coded in the math.

The per-tensor **rate** in bytes is `R_t(o) = ceil(numel_t · r(o) / 8)`. `numel_t` is `out × in`
for a Linear weight (e.g. a routed-expert `down_proj` is `1280 × 896`; the dense layer-0
`down_proj` is `1280 × 6848`).

### 1.2 The constrained program

Let `b = (b_1, …, b_T)` be the per-tensor option assignment over the `T` quantizable tensors
(the 2244 `*_proj.weight` + `lm_head`, minus whatever AF-4 / the validated heuristic pins to
bf16). Minimize **end-to-end distortion** subject to a **footprint budget** `B`:

```
minimize    D(b) = Σ_t  D_t(b_t)              (additive end-to-end distortion surrogate)
subject to  R(b) = Σ_t  R_t(b_t)  ≤  B        (total footprint in bytes)
            b_t ∈ O_t                          (each tensor's allowed option set)
```

The additive-distortion assumption (`D(b) ≈ Σ_t D_t(b_t)`) is the linchpin that makes the program
**separable** and therefore tractable. It is justified to first order (each tensor's quant error
is a small perturbation; cross-terms are second-order) and is **validated, not assumed** by the
§5 proof obligation — if super-additive interactions dominate, the proof fails and we fall back
(§6). See the assumptions ledger (§9, A1).

> **Why not just uniform Q4_K_M?** Uniform spends the same bits on a flat tensor (where bits buy
> almost no accuracy) as on a steep one (where every bit matters). Water-filling spends bits where
> the **marginal distortion-per-bit `∂D_t/∂R_t` is steepest** — empirically the wide dense
> `down_proj` (`intermediate=6848`) and the attention `v_proj` — and **starves the flat ones**.
> Same footprint, strictly-less-or-equal distortion. That is the entire EV of AF-1.

---

## 2. The per-tensor distortion curve `D_t(b)`

### 2.1 Definition: layer-output cosine drop

`D_t(o)` measures how much **quantizing tensor `t` to option `o`** degrades the model's behavior,
holding all other tensors at full precision. It is the **layer-output cosine drop** on a calibration
batch:

```
Given a calibration batch X (a small set of representative document activations),
  y_full  = layer_t( X ; W_t at bf16 )            # reference layer output
  y_quant = layer_t( X ; W_t at option o )        # output with ONLY t quantized
  cos_t(o) = mean over batch of cosine_similarity( y_full_row , y_quant_row )
  D_t(o)   = 1 − cos_t(o)                          # the "cosine drop": 0 = lossless, →1 = wrecked
```

Properties that matter:

- **`D_t(bf16) = 0`** by construction (the reference option). The curve is anchored at zero cost.
- **`D_t` is monotone non-increasing in bits**: more bits never increase distortion (a coarser
  grid is a strict superset of representable values). The allocator **enforces** monotonicity by a
  cummin pass over the option ladder (§3.2) so a noisy measurement can never make "more bits, more
  distortion."
- **Output-space, not weight-space.** We deliberately measure the drop in the *layer output*
  (post-GEMM activation), not `‖W − Ŵ‖`. Two tensors with equal weight MSE can have wildly different
  output impact depending on activation statistics and downstream sensitivity. Cosine (not L2) is
  scale-invariant, which matches the fact that a per-row scale is reabsorbed by the next norm.

### 2.2 Why cosine drop is the right currency

The end-to-end metric we actually care about is **CER on dense-numeric documents** (exact-token
sensitive). But CER is (a) expensive (full decode) and (b) non-additive across tensors. The
**layer-output cosine drop is a cheap, additive, monotone surrogate** that is strongly rank-correlated
with CER impact (the §9.5 "expected-loss-guided per-layer quant" intuition, made measurable). AF-1
**optimizes the surrogate**, then AF-2 / the proof obligation (§5) **validates against true CER**.
The surrogate is the steering wheel; CER is the road test.

### 2.3 Measurement protocol (offline, data-free-by-default)

- **Calibration batch.** A frozen, version-pinned set of layer-input activations captured from the
  golden corpus (§8 of the plan) during a single bf16 reference forward. Data-free PTQ is the v1
  default (NVFP4 was data-free and OCR-identical, §5.4); the calibration batch is used **only to
  measure `D_t`**, never to *fit* quant parameters — so determinism (§5.4) is preserved: same
  weights → same scales → same `.focrq` payload; the calibration batch only reorders the allocation.
- **Per-tensor isolation.** For each `(t, o)` pair, quantize *only* `t`, run the owning layer (or a
  faithful single-layer harness) on the calibration batch, record `D_t(o)`. `T × |O|` measurements
  total — embarrassingly parallel, one-time, offline.
- **Curve object.** The result is, per tensor, a small map `option → (bits=R_t(o), distortion=D_t(o))`.
  This is **exactly the JSON the prototype `scripts/af1_bit_allocator.py` consumes** (§4.3): the
  prototype is the reference implementation of the allocation math, decoupled from the (slow,
  weights-bound) measurement of `D_t`.

> **Where the curves come from in production.** `focr convert --measure-distortion` emits the
> per-tensor curve JSON; `focr convert --optimize-bits --budget <GB>` (or the standalone prototype)
> runs the water-filling on that JSON. Splitting measurement from allocation lets us re-allocate for
> a new budget in milliseconds without re-touching the 6.67 GB of weights.

---

## 3. The algorithm: Lagrangian water-filling over the convex hull

### 3.1 Lagrangian relaxation makes the problem separable

Attach a price `λ ≥ 0` (units: distortion per byte) to the budget constraint. The Lagrangian is

```
L(b, λ) = Σ_t D_t(b_t)  +  λ · ( Σ_t R_t(b_t) − B )
        = Σ_t [ D_t(b_t) + λ·R_t(b_t) ]  −  λB
```

The `−λB` term is constant in `b`. So for any fixed `λ`, **`L` is minimized tensor-by-tensor**:

```
b_t*(λ) = argmin_{o ∈ O_t}  [ D_t(o) + λ · R_t(o) ]            (the per-tensor "RD-cost")
```

This is the water-filling step: at price `λ`, each tensor independently buys the bit-width whose
**marginal distortion-per-bit is worth paying for at that price** and no more. High `λ` (bits
expensive) → everyone picks the cheapest option; `λ → 0` (bits free) → everyone picks bf16.

### 3.2 Per-tensor convex-hull pruning (the "operational R-D points")

For a fixed `λ`, only options on the **lower convex hull** of a tensor's `(R, D)` points can ever be
selected — an option strictly above the hull is dominated for *every* `λ`. We precompute, per tensor:

1. **Monotone repair (cummin):** sort options by increasing bits; enforce `D` non-increasing
   (`D[i] = min(D[i], D[i-1..])`), discarding any "more bits but not-less distortion" point. This
   immunizes against calibration noise.
2. **Convex-hull prune:** keep only points where the slope (the *marginal* `ΔD/ΔR` going to the
   next-cheaper kept point) is **strictly steeper than the previous** — i.e. drop interior points
   whose distortion-per-bit return is dominated by a cheaper-and-a-pricier neighbor.

The kept points are the tensor's **operational R-D points**. The negative slope between consecutive
kept points is the **price band** over which the cheaper point is optimal. As `λ` decreases past a
tensor's slope, that tensor "upgrades" one hull step — spending more bits exactly where the
distortion-per-bit return justifies the price.

> **Intuition (water-filling):** imagine pouring "bit budget" as water; each tensor is a vessel whose
> walls are its hull slopes. At a single global water level (`1/λ`), water rises in the vessels with
> the steepest returns first. The flat tensors stay near-empty (kept at int4); the steep ones fill up
> (upgraded to int8 / bf16). One global price, locally-optimal everywhere — that is convex duality.

### 3.3 Sweeping `λ` to hit the budget

The map `λ ↦ R(b*(λ))` (total footprint at the optimal allocation for price `λ`) is a **monotone
non-increasing step function**: bigger `λ` ⇒ cheaper options ⇒ smaller footprint. We want the
**smallest-distortion allocation whose footprint ≤ B**. Two equivalent walks:

- **Continuous bisection on `λ`** (robust, what the prototype defaults to): binary-search `λ` in
  `[0, λ_hi]` for the point where `R(b*(λ))` crosses `B`. Because `R` is a step function there is a
  whole interval of `λ` giving the same allocation; we take the allocation at the **largest feasible
  `λ`** (smallest footprint ≤ B that the hull permits at a single price), then run a **greedy
  hull-climb top-up** (below) to spend the slack.
- **Breakpoint enumeration** (exact frontier): the only `λ` values where any tensor changes option
  are the finite set of **hull slopes** across all tensors. Sort them; sweeping through them traces
  the *entire* global R-D Pareto frontier as a sequence of single-tensor upgrades. This is the
  textbook "merge all per-tensor hulls by slope" construction and yields the full
  `(footprint, distortion)` frontier in `O(P log P)` for `P` total hull points.

**Slack top-up (greedy, exactness-preserving).** After bisection lands at footprint `R ≤ B`, there
is usually `B − R` bytes of unspent budget. Spend it greedily: repeatedly apply the **single hull
upgrade (across all tensors) with the best distortion-reduction-per-added-byte** that still fits the
budget, until no upgrade fits. Because every candidate upgrade is a hull edge, each greedy pick is
the globally steepest available `ΔD/ΔR` — this is exactly the water-filling continuation and keeps
the result on the lower convex hull of the global R-D curve.

### 3.4 Optimality argument

- **Convex-hull case (exact).** If every tensor's kept points are its lower convex hull, the
  Lagrangian sweep + greedy hull top-up returns the **exact minimizer** of `Σ D_t` subject to
  `Σ R_t ≤ B` restricted to hull points, because (i) for each `λ`, per-tensor argmin is exact
  (finite set), and (ii) greedy on a matroid-like budget over convex (diminishing-return) hull edges
  is optimal — each next-best `ΔD/ΔR` is monotonically worse, so greedy never regrets. This is the
  standard Shoham–Gersho / Ortega–Ramchandran bit-allocation result.
- **General case (1 ≤ option-set, non-convex points).** When non-hull points exist (rare after the
  cummin + hull prune, since options are few and ordered), the Lagrangian frontier touches only hull
  points and may **skip a footprint exactly between two hull points** (the classic integer-program
  duality gap). The prototype therefore **also offers an exact `--exact-dp` mode**: a 1-D bounded
  knapsack / DP over `B` discretized to a byte-granularity grid (`O(T · |O| · B_grid)`), which closes
  the gap at the cost of a coarse budget grid. For the franken_ocr option set (`|O| = 4`, `T ≈ 2244`)
  the DP is cheap and is the default when `--exact-dp` is set; Lagrangian water-filling is the fast
  default and is provably within one hull-step of the DP optimum.

> **Bottom line:** water-filling gives the convex-hull-optimal allocation in `O(P log P)`; the DP
> gives the bit-exact integer optimum. Both dominate uniform Q4_K_M at equal footprint by
> construction (uniform is *a single feasible point*; the frontier passes through or below it).

### 3.5 Determinism

The allocation is a **pure function** of `(curves JSON, budget, pins, option set)`. Ties in RD-cost
are broken **deterministically** (prefer the higher-precision / lower-index option, then lower tensor
id) so the same inputs always yield byte-identical `bit_allocation_table` output. No RNG, no
wall-clock, no map-iteration-order dependence. This satisfies §5.4's determinism gate.

---

## 4. Emitted artifact: the `bit_allocation_table`

### 4.1 Where it lives

The table is computed offline by `focr convert --optimize-bits --budget <GB>` and **baked into the
`.focrq` header** (§5.2 `tensor_directory`): each tensor record's `dtype` / `group_size` / `tier`
fields are *set from* the allocation. The standalone table (below) is additionally emitted next to
the `.focrq` for audit, ledgering, and the proof obligation (§5). It is the machine-readable record
of *what was decided and why*.

### 4.2 Schema (`bit_allocation_table.json`)

```jsonc
{
  "schema_version": 1,
  "generator": "focr convert --optimize-bits",         // or "scripts/af1_bit_allocator.py"
  "source_sha256": "<sha256 of the source safetensors>",// provenance, matches .focrq header (§5.2)
  "option_set": ["bf16", "int8", "int4-g32", "int4-g16"],
  "budget_bytes": 2147483648,                            // B (e.g. 2.0 GiB target for int4 milestone)
  "method": "lagrangian-waterfill",                      // or "exact-dp"
  "lambda_star": 3.71e-9,                                // chosen price (distortion per byte); null for DP
  "totals": {
    "footprint_bytes": 2138210304,                       // Σ R_t(b_t*)  ≤ budget_bytes
    "footprint_gib": 1.991,
    "distortion": 0.004182,                              // Σ D_t(b_t*)  (additive cosine-drop surrogate)
    "uniform_baseline": {                                // the equal-footprint uniform config it must beat
      "option": "int4-g32",
      "footprint_bytes": 2140000000,
      "distortion": 0.009947                             // Σ D_t(uniform) — strictly ≥ allocated, by construction
    }
  },
  "pins": {                                              // tensors forced high-precision before allocation
    "bf16": ["vision.*", "decoder.embed", "decoder.layer.*.router",
             "decoder.layer.*.norm*", "vision.projector"],
    "tier_floor": {                                      // §6.3 _M discipline: never below this option
      "decoder.layer.*.attn.v": "int8",
      "decoder.layer.*.expert.*.down": "int8"
    }
  },
  "allocation": [
    { "tensor": "decoder.layer.0.dense.down",  "numel": 8765440, "option": "int8",
      "bits_per_weight": 8.03, "bytes": 8800110, "distortion": 0.000310,
      "marginal_dpb": 1.9e-11, "reason": "steep: wide intermediate=6848, output-sensitive" },
    { "tensor": "decoder.layer.5.expert.42.gate", "numel": 1146880, "option": "int4-g32",
      "bits_per_weight": 5.0, "bytes": 716800, "distortion": 0.000071,
      "marginal_dpb": 4.0e-13, "reason": "flat: starved to int4 (return below price)" }
    // ... one record per quantizable tensor ...
  ]
}
```

Field semantics:

- **`option` / `bits_per_weight` / `bytes`** — the chosen bit-width, its effective bpw (incl. scale
  overhead), and the resulting tensor footprint. Drives the `.focrq` `tensor_directory` dtype.
- **`distortion`** — `D_t(b_t*)`, the cosine drop incurred at the chosen option (0 if pinned bf16).
- **`marginal_dpb`** — the distortion-per-byte *return* of the next available upgrade (the hull
  slope at the chosen point). Tensors with the highest `marginal_dpb` are the next ones water-filling
  would upgrade if the budget grew — the explicit "where bits want to go" ledger.
- **`reason`** — human-readable why (steep-and-upgraded vs flat-and-starved vs pinned/tier-floored).
- **`uniform_baseline`** — the equal-footprint uniform config, carried so the proof obligation (§5)
  is checkable *from the table alone*: `totals.distortion ≤ uniform_baseline.distortion` must hold by
  construction (a surrogate pre-check before the expensive CER run).

### 4.3 Prototype I/O contract

`scripts/af1_bit_allocator.py` operates on the **per-tensor curves JSON** (the measurement output)
and emits the `bit_allocation_table.json` above. Input shape:

```jsonc
{
  "option_bits": { "bf16": 16.0, "int8": 8.03, "int4-g32": 5.0, "int4-g16": 6.0 },  // optional global default bpw
  "tensors": [
    {
      "tensor": "decoder.layer.0.dense.down",
      "numel": 8765440,
      "pin": null,                                  // or "bf16" to force high precision
      "tier_floor": null,                           // or e.g. "int8" (§6.3 _M discipline)
      "curve": {                                    // option -> {bits: bpw, distortion: cosine-drop}
        "bf16":     { "bits": 16.0, "distortion": 0.0 },
        "int8":     { "bits": 8.03, "distortion": 0.000310 },
        "int4-g32": { "bits": 5.0,  "distortion": 0.001740 },
        "int4-g16": { "bits": 6.0,  "distortion": 0.000980 }
      }
    }
    // ...
  ]
}
```

The prototype validates this, builds per-tensor hulls, runs the water-filling sweep (or `--exact-dp`),
and prints the allocation table. See `python3 scripts/af1_bit_allocator.py --help`.

---

## 5. Proof obligation

> **Claim AF-1 must discharge:** the allocated config's **end-to-end CER ≤ the uniform-bit config's
> CER at equal footprint**, on the **dense-numeric corpus** (§9.7).

How it is discharged (CI-gated, ledgered):

1. **Surrogate pre-check (cheap, in-table).** From `bit_allocation_table.json`,
   `totals.distortion ≤ totals.uniform_baseline.distortion`. This holds **by construction** (the
   allocated point is on/under the global R-D hull that the uniform point sits on or above). If it
   ever fails, the allocator has a bug or the additivity assumption (A1) broke — both block the gate.
2. **Equal-footprint uniform baseline.** Construct the uniform config whose footprint is within a
   tight tolerance (say ±0.5%) of `totals.footprint_bytes` (the densest uniform option that fits B).
   Quantize the model both ways.
3. **True CER on the dense-numeric corpus.** Run both configs end-to-end on the exact-token-sensitive
   corpus (dense tables / numbers / code / sub-superscripts — §6.3, Phase 4). Record
   `CER_allocated` and `CER_uniform` plus the AF-2 tail stats `(mean, CVaR_0.1, EVT_p999)`.
4. **Gate:** `CER_allocated ≤ CER_uniform` (mean) **and** AF-2's `CVaR_0.1` is within the ledgered
   budget. The mean inequality is the AF-1 obligation; the CVaR clause is AF-2's tail guarantee
   layered on top (a quant choice that wins on mean but wrecks the tail is rejected — §9.7 AF-2).
5. **Ledger.** The win (or non-win) is written to [`PERF_LEDGER.md`](../PERF_LEDGER.md) with the
   reproducing command and the source SHA; a non-win triggers the fallback (§6) and a
   [`NEGATIVE_EVIDENCE.md`](../NEGATIVE_EVIDENCE.md) entry.

**Falsifiability.** The obligation is a strict inequality on a measured corpus with a fixed command.
If allocation does **not** beat uniform at equal footprint, AF-1 is *disproven for this model* and
we ship the fallback — exactly the `NEGATIVE_EVIDENCE`-gated-hypothesis discipline of §9.7.

---

## 6. Deterministic fallback

Per §9.7 ("none ships without its deterministic fallback wired first"), the fallback is **uniform
Q4_K_M-class allocation (§5)**:

- **What it is:** every quantizable decoder linear → int4 per-group (g=16–32), with the validated
  high-precision set kept bf16 (vision tower, projector, `embed_tokens`, MoE router, all norms) and
  the `_M` tier-floor discipline (attention `v_proj`, expert `down_proj` one tier higher — int8),
  `lm_head` int8 only behind its kill-switch. This is the §5 recipe verbatim and is **always wired
  first**; the allocator only ever *improves on it*.
- **When it triggers (any one):**
  1. `--optimize-bits` not requested (default convert path = uniform recipe).
  2. The per-tensor distortion curves are missing/stale/unmeasurable (no calibration batch).
  3. The surrogate pre-check (§5 step 1) fails (`Σ D_allocated > Σ D_uniform` — allocator bug or
     additivity broke).
  4. The proof obligation (§5 step 4) is not green on the dense-numeric corpus.
  5. The allocator cannot fit the budget even at the cheapest option for every tensor
     (`B < Σ_t min_o R_t(o)`) → infeasible; report and fall back.
- **Why it is safe:** uniform Q4_K_M is the **already-validated prior-art recipe** (NVFP4 / GGUF,
  §2.6). Falling back never ships an unvalidated configuration; AF-1 is pure upside behind a proven
  floor. The prototype's `--fallback` flag emits exactly this uniform table for parity testing.

---

## 7. Preconditions / validity gates (machine-checkable)

The allocator asserts these before emitting a table; any failure routes to the fallback (§6):

| # | Precondition | Why | Failure action |
|---|---|---|---|
| P1 | Every tensor has `curve["bf16"].distortion == 0`. | bf16 is the anchor reference. | reject curves, fallback |
| P2 | After cummin, each curve is monotone non-increasing in bits. | R-D sanity; noise repaired. | repair + warn |
| P3 | `B ≥ Σ_t min_o R_t(o)` (budget feasible at cheapest). | else no allocation fits. | fallback (infeasible) |
| P4 | `B ≤ Σ_t max_o R_t(o)` (budget below all-bf16). | else trivially all bf16. | emit all-bf16 (no quant) |
| P5 | Pins / tier-floors are consistent (a pinned-bf16 tensor isn't also tier-floored cheaper). | config sanity. | reject config |
| P6 | `Σ D_allocated ≤ Σ D_uniform_at_equal_footprint` (surrogate pre-check). | proof-obligation precursor. | fallback + NEG-EV |

---

## 8. Galaxy-brain transparency card

> Per §9.7: *equation · substituted values · plain-English intuition · validity assumptions · what
> would flip the decision.* One card, everything an auditor needs.

**Equation.**

```
minimize_b  Σ_t D_t(b_t)   s.t.   Σ_t R_t(b_t) ≤ B
Lagrangian:  L(b,λ) = Σ_t [ D_t(b_t) + λ·R_t(b_t) ] − λB
per-tensor argmin:  b_t*(λ) = argmin_{o}  [ D_t(o) + λ·R_t(o) ]
sweep λ ↓ until  Σ_t R_t(b_t*(λ))  just fits B,  then greedy hull top-up.
   D_t(o) = 1 − cos( layer_t(X; bf16) , layer_t(X; o) )      # layer-output cosine drop
   R_t(o) = ceil( numel_t · bpw(o) / 8 )                     # footprint bytes incl. scale overhead
```

**Substituted values (illustrative, this model).**

- Options `O = {bf16:16.0, int8:8.03, int4-g32:5.0, int4-g16:6.0}` bpw (scale overhead included).
- `T ≈ 2244` quantizable linears (`truth-pack/CENSUS.md`); pins remove the bf16 high-precision set.
- Steepest tensors (get upgraded): dense layer-0 `down_proj` (`1280×6848`), attention `v_proj`
  (tier-floored int8 anyway). Flattest (starved to int4-g32): bulk routed-expert `gate`/`up`
  (`1280×896`).
- A budget `B ≈ 2.0 GiB` (the int4-milestone footprint target, §10 Phase 4) selects an allocation
  whose **surrogate distortion `Σ D` is ~2–3× lower than uniform int4-g32 at the same footprint**
  (the gap the proof obligation then confirms in CER). Numbers above (`λ*≈3.7e-9`,
  `Σ D_alloc 0.0042` vs `0.0099`) are placeholders illustrating the table; real values come from the
  measured curves and are ledgered.

**Plain-English intuition.** Set one global "price of a bit." Every tensor independently asks "is the
accuracy I'd buy with more bits worth this price?" The steep tensors say yes and upgrade; the flat
ones say no and stay at int4. Lower the price until the model just fits the footprint budget. Because
the price is global but the choice is local, you provably spend each byte where it removes the most
distortion — strictly better than giving every tensor the same bit-width.

**Validity assumptions.** (full ledger §9)
- **A1 Additivity:** end-to-end distortion ≈ `Σ_t D_t(b_t)` (cross-tensor interactions second-order).
- **A2 Surrogate fidelity:** layer-output cosine drop rank-correlates with end-to-end CER impact.
- **A3 Convexity:** per-tensor R-D points (after cummin) are near-convex, so Lagrangian ≈ optimal.
- **A4 Calibration representativeness:** the calibration batch reflects the dense-numeric corpus.
- **A5 bpw fidelity:** the packer's reported `r(o)` matches realized `.focrq` bytes.

**What would flip the decision (to the fallback).**
- If `CER_allocated > CER_uniform` at equal footprint on the dense-numeric corpus → A1/A2 broke →
  **uniform Q4_K_M** (§6). [The headline kill condition.]
- If AF-2 `CVaR_0.1` of the allocated config exceeds the ledgered tail budget (a tail-wrecking choice)
  → tier-floor the offending tensor up and re-allocate, or fall back (§9.7 AF-2 fallback).
- If the surrogate pre-check `Σ D_alloc > Σ D_uniform` ever fails → allocator/additivity bug → fallback.
- If curves are unmeasurable / stale / budget infeasible → fallback (§7 P1/P3).
- If the win is < ~10–20% footprint-at-equal-CER (diminishing-returns discipline, §9.7) → ship uniform
  for simplicity and ledger AF-1 as "measured, not worth the complexity here."

---

## 9. Assumptions ledger

| # | Assumption | Why we believe it | Risk if false | Mitigation / detection |
|---|---|---|---|---|
| A1 | End-to-end distortion is additive over tensors (`Σ_t D_t`). | Each tensor's quant error is a small perturbation; cross-terms are 2nd-order; classic R-D bit-allocation relies on it and works in practice. | Optimizer minimizes the wrong objective; allocation could lose to uniform. | **Proof obligation §5 directly tests this** (true CER vs uniform). Surrogate pre-check P6. Fallback on failure. |
| A2 | Layer-output cosine drop rank-correlates with CER impact. | §9.5 expected-loss-guided-per-layer-quant intuition; cosine is the standard activation-fidelity proxy. | We upgrade the wrong tensors. | Same §5 CER gate validates the surrogate end-to-end; AF-2 CVaR catches tail regressions the mean-cosine missed. |
| A3 | Per-tensor R-D points are near-convex after cummin. | Only 4 ordered options; coarse, well-separated bit-widths rarely produce non-convex interior points. | Lagrangian skips a budget between hull points (duality gap). | `--exact-dp` mode closes the gap; gap is ≤ one hull step and bounded by option granularity. |
| A4 | Calibration batch is representative of the dense-numeric corpus. | Captured from the golden corpus; data-free PTQ already validated (NVFP4). | Curves mis-rank tensors for the corpus that matters. | Calibration batch is version-pinned & corpus-stratified; §5 gate is on the *dense-numeric* corpus specifically. |
| A5 | Packer-reported bpw `r(o)` matches realized `.focrq` bytes. | The converter computes `r(o)` from the actual packing (scale overhead included). | Footprint accounting drifts; budget not actually met. | `.focrq` writer asserts realized `Σ bytes ≤ B`; round-trip test (§5.4). |
| A6 | Determinism: same inputs → byte-identical table. | Pure function, deterministic tie-breaks, no RNG/clock/iteration-order. | Non-reproducible allocation breaks §5.4 gate. | Prototype is deterministic; a `convert→convert` re-run asserts identical table. |

---

## 10. Relationship to the other AF families

- **AF-2 (tail-risk CER, CVaR + EVT)** is the **guard rail** on AF-1: AF-1 minimizes mean surrogate
  distortion under a footprint budget; AF-2 ensures the *resulting* config's worst-α tail CER stays
  bounded, tier-flooring any tail-offending tensor and re-feeding that floor as a `tier_floor`
  constraint into AF-1. AF-1 + AF-2 co-iterate until both gates are green.
- **AF-4 (submodular high-precision set)** decides **which tensors are pinned bf16 before AF-1 runs**
  (the `pins.bf16` set in §4.2). AF-1 then allocates over the *remaining* tensors. They compose: AF-4
  picks the keep-set under a budget; AF-1 spreads the rest of the budget across the quantizable
  remainder.
- **AF-5 (USL pool sizing)** is orthogonal (runtime thread pools), not a quant lever.

The load-bearing trio is AF-1 / AF-2 / AF-5 (§9.7); AF-1 is **Tier A, EV high**, gated behind the
uniform-Q4_K_M fallback.

---

## 11. Implementation checklist (for the Rust `focr convert --optimize-bits` follow-on)

- [ ] `focr convert --measure-distortion` → emit per-tensor curves JSON (the prototype's input).
- [ ] Port the prototype's hull + water-filling + DP into `convert` (deterministic, pure).
- [ ] Wire pins from AF-4 (or the validated heuristic) and tier-floors from §6.3 / AF-2.
- [ ] Bake the allocation into the `.focrq` `tensor_directory`; emit `bit_allocation_table.json` beside it.
- [ ] Surrogate pre-check (P6) at convert time; refuse + fallback on failure.
- [ ] CI proof-obligation job: allocated-vs-uniform CER on the dense-numeric corpus (§5), ledgered.
- [ ] `--fallback` path = uniform Q4_K_M (§6), wired first, default when `--optimize-bits` absent.

---

*This document is the AF-1 spec. The math is realized first by the runnable prototype
[`scripts/af1_bit_allocator.py`](../../scripts/af1_bit_allocator.py) (allocation only; distortion
measurement is the converter's job), then ported into `focr convert --optimize-bits`. None of it
ships without the deterministic fallback (§6) wired and the proof obligation (§5) green.*
