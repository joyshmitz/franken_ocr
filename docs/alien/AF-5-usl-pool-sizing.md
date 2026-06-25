# AF-5 — Universal Scalability Law for many-core pool sizing

**Beads:** `bd-1xfa.5` (SPIKE), `bd-1xfa.5.1` (tests), `bd-2mo.21` (P3-numa-usl impl),
`bd-2mo.21.1` (transparency card + fallback wiring), `bd-2mo.2` (`robot backends`).
**Family:** Queueing theory — Universal Scalability Law (USL). **Tier A · EV high · low effort.**
**Plan refs:** §6.9 (the detailed treatment), §9.7 AF-5 (the recommendation contract),
§3.3 / §8.5 (capacity certificate). **Doctrine:** AGENTS.md #5 (no oversubscription /
single live forward), #8 (honest measured everything), Alien-Artifact Engineering Contract.
**Fitter:** [`scripts/af5_usl_fit.py`](../../scripts/af5_usl_fit.py).

---

## 1. The decision this artifact makes

> *How many threads should the decode pool and the prefill pool each get on this
> machine?*

The naive answer — `num_cpus`, i.e. a blind `par_iter` over all 64 logical cores —
is **measurably wrong for decode**. A single-token decode GEMV streams the full
~500M active params/token (§2.5, §6.1); it is **memory-bandwidth-bound**. Past the
handful of cores needed to saturate DRAM bandwidth, more threads add no throughput
and *cost* throughput through cache-coherency traffic and scheduler contention. On a
64-core Threadripper, decode peaks at roughly **8–16 effective cores**; oversubscribing
all 64 is a **measured anti-win** — a β-dominated *retrograde* slowdown, not a plateau.

Prefill is the opposite: the prefill GEMM over the vision prefix and the prompt is
**compute-bound** and scales with cores nearly linearly. A single global pool size is
therefore *wrong for one of the two op-classes by construction*. AF-5 fits a separate
scaling model per `(arch, op-class)` and **caps each pool at its own USL peak**.

This is the lowest-effort, highest-certainty of the load-bearing trio (AF-1 / AF-2 /
AF-5): a thread sweep plus a two-parameter least-squares fit.

---

## 2. The model (transcribed from §6.9 / §9.7)

The Universal Scalability Law gives the speedup `C(N)` of a workload run on `N`
parallel workers, relative to `N = 1`:

```
                       N
C(N) = ----------------------------------------
        1 + α·(N − 1) + β·N·(N − 1)
```

| Term | Name | Effect | Physical meaning here |
|------|------|--------|------------------------|
| `α` | **contention** (serialization) | makes `C(N)` *saturate* | the Amdahl fraction: serial setup, the sequential outer page loop, lock-protected shared state |
| `β` | **coherency / crosstalk** | makes `C(N)` *turn over and drop* | pairwise cross-core cost (`N·(N−1)` interactions): cache-line bouncing, the shared DRAM bandwidth ceiling memory-bound decode hits |

**Why USL and not Amdahl.** Amdahl is USL with `β = 0`: it saturates but *never
regresses*. Decode does regress — throughput **drops** past its peak. Only the `β`
term captures that retrograde behavior, so Amdahl would mis-predict (it would say
"plateau at 64", which is exactly the anti-win we are trying to avoid). USL is the
right model precisely because it can be *wrong-shaped* for compute-bound prefill
(β fits ≈ 0, the model degenerates to Amdahl, and the fitter falls back to physical
cores — see §5) and *right-shaped* for bandwidth-bound decode.

### The peak (closed form)

For `β > 0`, `C(N)` is maximized over the reals at

```
N* = sqrt((1 − α) / β)
```

This is the AF-5 pool cap. The fitter takes the better of `floor(N*)` / `ceil(N*)`
under the sampled-and-extrapolated `C(N)`, clamped to `[1, num_cpus]`. **When `β ≤ 0`
(no retrograde term, compute-bound), `N*` is `+∞` → the cap is `num_cpus` → in
practice the deterministic physical-core fallback engages** (the model has nothing to
say, so we do not invent a cap).

---

## 3. The fitting method (`scripts/af5_usl_fit.py`)

The script is **offline tooling, stdlib-only, no NumPy/SciPy** (the same discipline as
`gen_reference_fixtures.py` / `check_ledgers.py`). The runtime never invokes it; it
turns a Rust-measured thread sweep into the `pool_sizing` row the converter bakes.

### 3.1 Linearized seed fit (exact, closed-form)

Substitute the observed speedups `C_i = T_i / T_1` and rearrange the USL into its
*deficiency* form — this is the canonical Gunther linearization, and it is **exact**
(no approximation):

```
N_i / C_i − 1 = α·(N_i − 1) + β·N_i·(N_i − 1)
```

The left side is observed; the right side is linear in `(α, β)` with regressors
`x1 = (N_i − 1)`, `x2 = N_i·(N_i − 1)` and **no intercept** — USL pins `C(1) = 1` by
construction, so we keep the intercept at 0. The fit is then an ordinary least-squares
solve of a 2×2 normal-equation system: closed form, deterministic, dependency-free.

### 3.2 Nonlinear refinement (Gauss-Newton, monotone)

The `1/C_i` transform in §3.1 up-weights noisy high-`N` points. So we refine the seed
with a few **damped Gauss-Newton** steps on the *nonlinear* residual `C_i − C_hat(N_i)`
directly in speedup space (analytic Jacobian, backtracking line search). Every step is
accepted **only if it lowers the nonlinear SSE**; otherwise the step is halved. The
seed is never made worse, so refinement is always safe to run. We report `R²` and
`RMSE` against this nonlinear residual — the honest fit-quality numbers.

### 3.3 I/O contract

Input (`--samples FILE` or stdin) — produced by the Rust bench harness of `bd-2mo.21`:

```json
{
  "arch": "threadripper-7980x", "op_class": "decode_gemv",
  "num_cpus": 64, "physical_cores": 64,
  "samples": [
    {"n": 1,  "throughput": 1.00, "cv_pct": 0.7},
    {"n": 2,  "throughput": 1.92, "cv_pct": 1.0},
    {"n": 4,  "throughput": 3.55, "cv_pct": 1.4},
    {"n": 8,  "throughput": 5.80, "cv_pct": 2.1},
    {"n": 16, "throughput": 7.10, "cv_pct": 3.3},
    {"n": 32, "throughput": 6.40, "cv_pct": 4.8},
    {"n": 64, "throughput": 5.10, "cv_pct": 6.2}
  ]
}
```

`throughput` is any consistent rate (tokens/s, GEMV/s, GFLOP/s) — only the ratio to
`N = 1` matters, so absolute units cancel. `cv_pct` is the coefficient of variation
across the best-of-N timing repeats (§9.3 fairness): any sample above `--cv-max`
(default 5%) flags the run `noisy` and marks the decision advisory.

Output (one JSON object — the `pool_sizing` row, the schema `focr robot backends`
reports, bd-2mo.2):

```json
{
  "schema_version": 1, "arch": "threadripper-7980x", "op_class": "decode_gemv",
  "alpha": 0.0461, "beta": 0.002334,
  "peak_n_real": 20.22, "peak_n": 20,
  "r2": 0.9905, "rmse": 0.21,
  "speedup_at_peak": 7.24, "speedup_at_num_cpus": 4.81,
  "num_cpus": 64, "physical_cores": 64,
  "cap_is_win": true, "predicted_gain_pct": 50.6,
  "noisy": true, "degenerate": false, "fallback_used": false,
  "chosen_pool_n": 20, "decision": "cap-at-usl-peak"
}
```

The self-check (`--selfcheck`, or any no-input invocation) fits a synthetic
decode-shaped instance (true `α = 0.040`, `β = 0.0020`) and asserts the fit recovers
both parameters, that the peak lands below `num_cpus`, that `cap_is_win` is true, and
that `R² > 0.999`. This makes the script `py_compile`-clean **and** smoke-runnable with
no inputs and no weights.

---

## 4. The op-class sweep plan

Sweep `N = 1, 2, 4, 8, 16, 32, 64` (best-of-N, warmup discarded, §9.3) for **each**
op-class **separately** — they have opposite scaling character and therefore very
different `(α, β)`:

| Op-class | Bound by | Shape | Census basis | Expected USL peak (64-core TR) |
|----------|----------|-------|--------------|-------------------------------|
| `decode_gemv` | DRAM **bandwidth** | streams ~500M active params/token; GEMV `M=1` | active-param count §2.5 / CENSUS | **~8–16** (β-dominated) — the cap that matters |
| `prefill_gemm` | **compute** | batched GEMM over the vision prefix + prompt; `hidden=1280`, `inter=6848`, `moe_inter=896` | dims from CENSUS / `config.json` | **near `num_cpus`** (β ≈ 0) → physical-core fallback |

The sweep arch must equal the deploy arch — `(α, β)` is **per-arch**: an M4 P/E sweep,
a Threadripper multi-CCD sweep, and a dual-socket EPYC sweep all differ. Ship a table
per arch; the deterministic fallback (§5) covers un-swept arches.

**Apple Silicon P/E asymmetry (§6.9).** Sweep over **P-cores only**, not the
heterogeneous total — pin the heavy GEMM pool to P-cores (QoS `USER_INTERACTIVE`),
orchestration on E-cores. A sweep over the mixed P+E set fits a meaningless `(α, β)`.

---

## 5. Deterministic fallback (wired FIRST — AGENTS.md #5)

> The runtime never requires the sweep to have run.

The pool size is the **physical-core count** whenever a fitted table is unavailable or
untrustworthy. The fitter emits `fallback_used: true` / `decision:
"fallback-physical-cores"` and the runtime uses physical cores when:

- **No table** for this `(arch, op-class)` — un-swept arch, no baked row.
- **`degenerate`** — `β ≤ 0` (no retrograde term, i.e. compute-bound; the model has no
  interior peak to cap at) or `α ≥ 1` or the normal-equation system is singular.
- **Poor fit** — low `R²` / large residuals (non-USL behavior); ledger and fall back.
- **No env override** — `FOCR_THREADS` always wins (operator/agent override; §6.9).

The fallback engaging is itself a *correct* outcome for prefill: a compute-bound
op-class legitimately wants all physical cores, and "degenerate → physical cores" lands
exactly there. The fallback is **proven to engage** by a degenerate-fit unit test
(`bd-2mo.21.1`, `bd-1xfa.5.1`).

---

## 6. Composition: NUMA + the capacity certificate (§6.9 / §3.3 / §8.5)

AF-5 sets the pool **size**; it must not introduce a second pool or nest one under a
lock. It composes with the rest of §6.9:

- **One global rayon pool** sized to the USL peak; the asupersync blocking pool stays
  tiny; **exactly one live forward fans out at a time** (AGENTS.md #5 — the durable fix
  the frankensearch deadlock saga converged on). The sequential outer page loop is part
  of the `α` (serial) term, by design.
- **NUMA (§6.9).** Pin each page-worker and its rayon sub-pool to one node; first-touch
  or `replicate` the read-only `.focrq` weight blob per node; allocate per-page
  activation buffers node-local. Expose `FOCR_NUMA={replicate|interleave|local}`,
  default `local`. Cross-node weight fetches are the silent killer on EPYC.
- **Two parallelism axes, never mixed (§6.9).** *Latency* (within one forward: prefill
  GEMMs over row-blocks, MoE over experts, attention over heads) vs *throughput* (across
  documents, only with a disjoint thread/NUMA budget per worker). The **capacity
  certificate** (§8.5, gauntlet artifact) must prove

  ```
  workers × per_worker_threads ≤ physical-core budget
  ```

  and **no nested global-pool oversubscription**, holding under the
  `many_pages_without_deadlock` soak (pages ≫ pool; bd-re8.18). The capped pools of AF-5
  are exactly what keeps that certificate true — oversubscribing 64 cores would violate
  it *and* trigger the β-dominated slowdown.

`focr robot backends` (bd-2mo.2) reports the `pool_sizing` table (`α`/`β`/`peak_N` per
op-class) alongside the detected SIMD tier, core count, NUMA mode, and allocator posture
— so an operator/agent can see the chosen `N*`, override it, or catch a regression.

---

## 7. Galaxy-brain transparency card

> **Equation.**
> `C(N) = N / (1 + α·(N−1) + β·N·(N−1))`; peak at `N* = sqrt((1 − α)/β)`.
>
> **Substituted values** (illustrative decode sweep on a 64-core Threadripper; the
> real numbers are baked per arch from the measured sweep):
> `α ≈ 0.046`, `β ≈ 0.0023` → `N* ≈ 20` (integer pool 20), `C(20) ≈ 7.2×` vs
> `C(64) ≈ 4.8×` — capping is a **+50%** predicted decode-throughput win and frees ~44
> cores for page-parallel throughput. Prefill fits `β ≈ 0` → no cap → physical cores.
>
> **Plain English.** Decode is bandwidth-bound: past ~8–20 cores the extra threads just
> fight each other for the same DRAM bus and cache lines, so throughput *drops*. Cap the
> decode pool at the peak; let compute-bound prefill scale to all cores.
>
> **Validity assumptions.** (a) The sweep arch == the deploy arch. (b) The workload mix
> matches the corpus (the GEMV streams the real active-param set; the GEMM uses the real
> prefix length). (c) The host's memory-bandwidth ceiling is the one swept (no THP /
> NUMA-policy change between sweep and deploy). (d) `C(1) = 1` (we pin the intercept).
>
> **What would flip the decision.** A different memory-bandwidth ceiling (faster DRAM,
> more channels, replicated NUMA weights) raises the decode peak — **re-sweep**. A kernel
> change that makes decode compute-bound (e.g. a much heavier per-token op) flips it
> toward prefill's shape — re-sweep. A degenerate / poor fit (`β ≤ 0`, low `R²`) →
> deterministic physical-core fallback, ledgered.

---

## 8. Proof obligation & acceptance (bd-1xfa.5.1)

**Proof obligation (the AF-5 gate).** *Measured* throughput at the chosen `N` ≥
throughput at `num_cpus`, for **both** op-classes (capping is a win or neutral, never a
loss), with §9.3 fairness (best-of-N, warmup discard, `cv_pct` reported, >5% flagged).
This script *predicts* the win (`cap_is_win`, `predicted_gain_pct`); the Rust E2E
harness **measures** it.

Acceptance criteria:

1. `pool_sizing` table per `(arch, op-class)` with fitted `α/β` + chosen `N*`;
   `focr robot backends` reports it as valid versioned JSON.
2. Proof obligation met on the primary arches; `cv_pct` reported.
3. Capacity certificate proves `workers × per_worker_threads ≤ physical-core budget`
   and no nested-pool oversubscription under the `many_pages` soak (composes with the
   deadlock watchdog, bd-re8.18).
4. Deterministic fallback (physical-core count) wired and exercised on an un-swept /
   degenerate arch.
5. This card + the assumptions ledger (§9) committed.
6. **If capping does not help (or hurts): REVERT to physical cores, ledger the result
   in `docs/NEGATIVE_EVIDENCE.md`.**

### Tests (bd-1xfa.5.1)

- **Unit:** USL least-squares recovers known `α/β` on synthetic `C(N)` data
  (`af5_usl_fit.py --selfcheck`); `N* = sqrt((1−α)/β)` matches the sampled argmax; the
  fallback returns physical-core count with no table / degenerate fit; deterministic
  table round-trip.
- **E2E (model-gated; skip-with-SUCCESS without weights):** sweep `decode_gemv` +
  `prefill_gemm` at `N = 1..64`; fit USL; choose `N*`; assert `throughput(N*) ≥
  throughput(num_cpus)` for both; assert `robot backends` reports the table; run
  `many_pages_without_deadlock` with the capped pools (no hang).
- **Structured logging** under `tests/artifacts/af5/`: per-`N` throughput, fitted
  `α/β` + `R²`, `N*`, `throughput(N*)` vs `num_cpus`, `cv_pct`, arch + CPU feature
  string, capacity-certificate result, keep/revert decision.

---

## 9. Assumptions ledger

| # | Assumption | Why it holds / how checked | What breaks it | Fallback if broken |
|---|-----------|----------------------------|----------------|--------------------|
| A1 | Decode is memory-bandwidth-bound (`β > 0`, retrograde) | §6.9 measured reality; ~500M active params streamed/token | a much heavier per-token kernel makes it compute-bound | β fits ≤ 0 → physical-core fallback (correct) |
| A2 | Prefill is compute-bound (`β ≈ 0`) | batched GEMM, arithmetic-intensity high | tiny prefill or extreme NUMA penalty | degenerate fit → physical cores (= full scale, correct) |
| A3 | `(α, β)` is per-arch | distinct memory hierarchies / CCD top/QoS | deploying on an un-swept arch | physical-core fallback for that arch |
| A4 | Sweep arch == deploy arch | the sweep is run on the target | cross-arch table reuse | re-sweep; fallback meanwhile |
| A5 | `C(1) = 1` (pinned intercept) | USL definition; base sample anchors it | base sample is N>1 (anchored proxy) | fit still well-posed; flagged |
| A6 | Timing noise is bounded (`cv_pct ≤ 5%`) | best-of-N, warmup discard (§9.3) | thermal throttling / co-tenant load | `noisy: true` → decision advisory, re-sweep |
| A7 | One global pool, one live forward | AGENTS.md #5; capacity certificate | a nested rayon pool / 2nd runtime sneaks in | `many_pages` watchdog hangs → caught in CI |

---

## 10. Status

DESIGN complete. Fitter (`scripts/af5_usl_fit.py`) implemented, `py_compile`-clean,
self-check passing (recovers `α/β`, caps below `num_cpus`, predicts the win). Awaiting
the Phase-3 SIMD decode-GEMV / prefill-GEMM kernels (bd-2mo) to sweep, the `pool_sizing`
bake into `.focrq` + `robot backends` (bd-2mo.2 / bd-2mo.21), and the measured proof
obligation + capacity certificate (bd-1xfa.5.1 / bd-re8.18).
