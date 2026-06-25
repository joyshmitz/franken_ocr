# AF-2 — Distributionally-Robust Tail-Risk Gate (CVaR + EVT)

> **Family:** AF-2 of the alien-artifact math families (plan
> [§9.7](../../COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md)). **Tier A, EV high.**
> **Bead:** `ALIEN-af2` / `P4-tail-risk-gate` (design), referenced as `bd-3upw`
> / `bd-1xfa.2` in [`PROPOSED_ARCHITECTURE.md`](../PROPOSED_ARCHITECTURE.md).
> **Artifact (offline reference):** [`scripts/af2_tail_risk.py`](../../scripts/af2_tail_risk.py).
> **Artifact (shipping):** the Rust `tail_risk_monitor` (Phase 4 / Phase 5),
> held numerically equivalent to the reference.
> **Provenance:** every CER number this gate consumes is measured against the
> pinned reference oracle (HF `3a7f4db…`, runtime `torch==2.10.0` /
> `transformers==4.57.1`; see [`docs/truth-pack/PINNED_SOURCES.md`](../truth-pack/PINNED_SOURCES.md)).

---

## 1. The problem this family solves

OCR quality is **not** a mean phenomenon. On a representative document corpus the
**Character Error Rate (CER)** distribution is sharply bimodal: the overwhelming
majority of pages — clean printed prose, headings, normal paragraphs — decode at
near-zero CER, while a small minority of pages carry almost all of the error
mass. Those tail documents are exactly the ones we cannot afford to wreck:

- **dense numeric tables** — a single transposed or dropped digit corrupts a cell,
  and tables pack hundreds of exact-token decisions per page;
- **sub- and super-scripts / formulae** — `x²` vs `x2`, `H₂O` vs `H2O`; the model
  must emit the *exact* structural token, and the §8.3 Formula-CDM metric is
  unforgiving;
- **code blocks and long digit runs** — no linguistic redundancy to fall back on,
  so a wrong token is simply wrong;
- **CJK / mixed-script dense layouts** — high token density, low per-token margin.

A weight-quantization choice (int4 on an expert FFN, int8 on attention `q/k/v/o`)
can leave **mean CER essentially unchanged** while **quietly destroying** the
tail: the average page never needed those bits, but the dense-table page did. If
the release gate watches the mean, it green-lights a model that fails precisely
where document parsing matters most.

### Why the mean (and perplexity) lie about the tail

Mean CER and the language-model perplexity used during calibration are both
**expectation-shaped** statistics. They reward getting the easy 95% right and are
nearly blind to a 10× error on the hard 1%. Worse, **perplexity systematically
under-predicts exact-token failure**: a quantized model can stay confidently
fluent (low perplexity) while flipping the one digit or the one `<sub>` token that
breaks the document. The quantity that actually fails in OCR — *exact-token
agreement on dense, low-redundancy content* — lives in the **upper tail** of the
per-document CER distribution, where the mean has no resolution.

The fix is a **distributionally-robust** objective: optimize and gate on the
**worst-case fraction** of documents, not the average document.

---

## 2. The two statistics

Let `c_1, …, c_n` be the **per-document** CER over the frozen golden corpus
(one value in `[0, 1]` per document, larger = worse), produced by the parity
harness against the reference oracle.

### 2.1 CVaR_α — Conditional Value-at-Risk (the worst-α-fraction mean)

`VaR_α` (Value-at-Risk at level α) is the `(1 − α)` quantile of CER — the
threshold the worst `α` fraction of documents exceed. **CVaR_α is the *mean of
that worst α-fraction*** — the average CER over the heaviest tail, not just its
boundary:

```
CVaR_α  =  E[ c | c ≥ VaR_α ]        (the Rockafellar–Uryasev coherent CVaR)
```

We use **α = 0.10** by default: `CVaR_0.1` is the average CER of the worst 10% of
documents. Properties that make it the right gate variable:

- **Coherent risk measure** (sub-additive, monotone, positively homogeneous,
  translation-equivariant) — unlike VaR, it is convex in the decision variables,
  so it can sit inside the AF-1 bit-allocation objective without breaking its
  convex-duality structure.
- **`CVaR_α ≥ VaR_α ≥ mean`** always — it strictly dominates the mean as a
  conservative bound, and it *averages* the tail rather than reading a single
  noisy order statistic, so it is far less jittery than a raw p90/p99.
- **Continuous in α and in the data** — the implementation handles a fractional
  `α·n` exactly (it partially weights the boundary document), so the gate does
  not snap as the corpus grows by one.

**Implementation:** [`scripts/af2_tail_risk.py`](../../scripts/af2_tail_risk.py),
function `cvar()`. For `α·n` non-integer it averages the `⌈α·n⌉` worst documents
with the boundary document down-weighted to the residual fraction — the exact
Rockafellar–Uryasev value, not a rounded approximation.

### 2.2 EVT-p999 — the 99.9th-percentile document via a Generalized-Pareto tail

CVaR summarizes the tail we *observed*. But a golden corpus of a few hundred
documents has **no** empirical p99.9 — the rank simply does not exist, and the
worst observed document is a single noisy draw. To estimate **the document we
have not yet seen but will hit in production**, we fit **Extreme-Value Theory**.

The **Pickands–Balkema–de Haan theorem** states that for a very wide class of
distributions, the conditional distribution of exceedances over a high threshold
`u` converges to a **Generalized-Pareto Distribution (GPD)**:

```
P(c − u ≤ y | c > u)  →  G(y) = 1 − (1 + ξ·y/β)^(−1/ξ)      (ξ ≠ 0)
                                = 1 − exp(−y/β)              (ξ = 0)
```

with **shape** `ξ` (tail heaviness; `ξ > 0` heavy/polynomial tail, `ξ = 0`
exponential, `ξ < 0` bounded) and **scale** `β > 0`. We fit the GPD by
**Peaks-Over-Threshold (POT)**: set `u` to the empirical `(1 − pot_frac)` quantile
(default `pot_frac = 0.15`, i.e. the worst 15% are the exceedances), fit `(ξ, β)`
to those exceedances, then invert to the target quantile:

```
EVT_p999 = u + (β/ξ)·[ ((1 − 0.999)/ζ_u)^(−ξ) − 1 ]          (ξ ≠ 0)
         = u − β·ln( (1 − 0.999)/ζ_u )                        (ξ = 0)
```

where `ζ_u = P(c > u)` is the exceedance rate. This **extrapolates past the
corpus size**: a 300-document corpus has no empirical p99.9, but the GPD fit
yields a principled estimate of the 1-in-1000 document.

**Estimator:** we use the **Hosking & Wallis (1987) probability-weighted-moment
(PWM)** estimator — a *closed-form, optimizer-free* fit that is stable on the
small exceedance counts a golden corpus produces, where maximum-likelihood often
fails to converge. (See `fit_gpd_pwm()` and the explicit derivation in its
docstring.)

**Guardrails on the EVT estimate** (`evt_quantile()`):

- **Domain clamp:** CER ∈ `[0, 1]`, so the EVT quantile is clamped into `[0, 1]`.
  A raw GPD extrapolation legitimately produces a value > 1 (e.g. p99.9 = 1.40 on
  a heavy synthetic tail); that is the math correctly saying "the 1-in-1000
  document is total garbage," and we report the clamped `1.0`.
- **Never under-state observed risk:** the reported tail quantile is never below
  the empirical quantile — the fit may only *raise* the worst-case estimate.
- **Deterministic fallback when the tail is too thin:** if there are fewer than
  `MIN_EXCEEDANCES = 8` exceedances, or the PWM moment relation is degenerate
  (`a0 − 2·a1 ≈ 0`, or a non-finite / non-positive scale), the tool **does not
  invent an extrapolation** — it falls back to the empirical quantile and reports
  `gpd_method = "empirical-fallback"`. No bound is ever fabricated from too little
  data.

---

## 3. The release gate: gate on the tail, NOT the mean

This is the load-bearing rule of AF-2 and it is enforced at the
**release-readiness scorecard** (plan §8.4 / §8.5, the
`/running-the-gauntlet-on-your-rust-port` three-pillar gate):

> **The conformance pillar's accuracy cell is decided by `CVaR_0.1` and
> `EVT_p999` against the f32 baseline — the mean CER is *reported for context but
> never gates*.**

A candidate quantization config (e.g. an int4 expert split) **passes** the
tail-risk cell iff both bounds stay within the **ledgered budget** of the
fp32-reference bounds:

```
CVaR_0.1(candidate)  ≤  CVaR_0.1(f32)  + budget_cvar
EVT_p999(candidate)  ≤  EVT_p999(f32)  + budget_evt
```

The budget is **measured for this model**, not inherited (per AGENTS.md doctrine
#8 and plan §5.4 — frankensearch's `0.055` is a different-model figure). The
target is "int8 within noise; int4 within a small, measured, ledgered budget" (G1,
plan §8). This composes with the gauntlet's other tail discipline: the parity
score is a **conformal lower bound** (§8.5), so even the CVaR/EVT comparison is
read at its lower confidence bound, never the point estimate. A change may only
land if it does not raise the tail bound past budget on *any* category.

The gate is implemented in `apply_gate()` and exposed on the CLI via
`--baseline-cvar` / `--baseline-evt` / `--budget`; a failing gate returns
**exit code 3** and emits the documented fallback (below) in the `gate.fallback`
field.

### Where AF-2 sits relative to AF-1

AF-2 is the **conscience** of AF-1 (the rate-distortion / water-filling bit
allocator, [§9.7](../../COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md)). AF-1 *chooses*
the per-tensor `{bf16, int8, int4-g32, int4-g16}` split to minimize end-to-end
distortion under a footprint budget; AF-2 is the **constraint that split must
satisfy** and the **objective term** that replaces "minimize mean distortion"
with "bound the CVaR/EVT tail." Concretely the AF-1 Lagrangian distortion `D_t`
is read through the CVaR lens, and the AF-2 gate is the hard accept/reject on the
allocator's output. AF-1/AF-2/AF-5 are the load-bearing trio (plan §9.7).

---

## 4. Proof obligation

Per the alien-artifact engineering contract (AGENTS.md), AF-2 ships with an
explicit, falsifiable proof obligation:

> **PO-AF2:** On the dense-numeric / table / formula golden corpus, the
> int4-allocated config's **`CVaR_0.1` is within the ledgered budget of the fp32
> reference's `CVaR_0.1`**, and its **`EVT_p999` is within the ledgered budget of
> the fp32 reference's `EVT_p999`** — both read at the conformal lower bound.

This is discharged by the `tail_risk_monitor` over the frozen corpus, with the
exact command, env, model commit, and fixture hash recorded in
[`docs/PERF_LEDGER.md`](../PERF_LEDGER.md) / [`docs/DISCREPANCIES.md`](../DISCREPANCIES.md)
(the artifact-graph ledger fields of plan §8.4). The obligation **fails closed**:
if it cannot be measured (no weights, no corpus), the tail cell is red and the
release cannot ship (plan §8.4 — "a release cannot ship with a red parity cell").

---

## 5. Deterministic fallback (wired first, per the contract)

> **No adaptive controller ships without a conservative deterministic fallback**
> (AGENTS.md). AF-2's fallback is wired before the gate is trusted.

When the tail gate **fails** for a candidate config, the remediation is the
**llama.cpp `_M` discipline, derived rather than guessed**:

> **Keep the tail-offending tensor one precision tier higher** — `int4 → int8`,
> or `int8 → bf16` — for the specific tensor(s) whose perturbation drives the tail
> documents, then re-measure.

Two levels of fallback:

1. **Per-tensor (the gate-failure remediation):** identify the tail-offending
   tensor (the one whose marginal distortion, AF-1's `∂D_t/∂bits`, dominates on
   the tail documents — typically the dense layer-0 `down_proj` at `K = 6848`, the
   attention `v_proj`, or the wide `intermediate = 6848` expert FFN), promote it
   one tier, and re-run the gate. This is the principled, *derived* form of GGUF's
   `_M`/`_S` mixed-precision variants. The tool surfaces this in `gate.fallback`.
2. **Whole-monitor (the estimator's own fallback):** when the tail is too thin to
   fit a GPD, the EVT estimate degrades gracefully to the empirical quantile
   (`gpd_method = "empirical-fallback"`) — the monitor never fabricates an
   extrapolated bound from insufficient data, so it cannot *falsely pass* a config
   on an over-optimistic fit.

The ultimate conservative fallback for the whole family is the **validated uniform
allocation** (Q4_K_M-class / the NVFP4 heuristic set, plan §5 / §2.6): if the
tail-aware allocation cannot beat it within budget, ship the validated set and
ledger the negative result in [`NEGATIVE_EVIDENCE.md`](../NEGATIVE_EVIDENCE.md).

---

## 6. Galaxy-brain transparency card

| Field | Content |
|-------|---------|
| **Equation** | `CVaR_α = E[c \| c ≥ VaR_α]` (worst-α-fraction mean); EVT tail `P(c−u≤y \| c>u) → 1 − (1+ξy/β)^(−1/ξ)` (GPD, Pickands–Balkema–de Haan), inverted to `EVT_p999`. |
| **Substituted values** | α = 0.10 (worst 10% of docs); EVT target q = 0.999 (p99.9 document); POT exceedance fraction `pot_frac = 0.15`; min exceedances for a trusted fit = 8; GPD fit via Hosking–Wallis PWM (`a0, a1` → `ξ, β`). |
| **Plain-English intuition** | "Don't grade the model on its average page — grade it on its *worst* pages, and on the disaster page we haven't seen yet. Spend bits to protect dense tables and formulae; the mean already takes care of itself." |
| **Validity assumptions** | (1) per-doc CER are comparable draws from one corpus distribution (frozen golden corpus, one oracle stack); (2) the tail is in the GPD max-domain-of-attraction (true for the smooth-ish CER tails we see); (3) the POT threshold sits in the genuine tail (default 15% exceedances, configurable); (4) enough exceedances (≥8) to fit, else empirical fallback; (5) the f32 baseline bounds were measured on the *same* corpus + oracle commit. |
| **What would flip the decision** | A measured int4 `CVaR_0.1` / `EVT_p999` **within budget** of f32 ⇒ keep int4 (cheaper, same tail). A measured **breach** ⇒ promote the tail-offending tensor one tier (§5) and re-measure; if no allocation clears budget, ship the validated uniform set and ledger the negative. A change in corpus or oracle commit ⇒ the old baseline is void, re-measure from scratch. |
| **Failure mode if ignored** | Quant config passes on mean CER, ships, and silently corrupts every dense-table / formula document in production — the exact content document parsing exists to capture. |

---

## 7. Assumptions ledger

| # | Assumption | Why it holds here | Retry / invalidation trigger |
|---|-----------|-------------------|------------------------------|
| A1 | Per-doc CER are i.i.d.-enough draws from one distribution. | Single frozen golden corpus, single pinned oracle stack (`torch==2.10.0`/`transformers==4.57.1`). | Corpus re-sampled or oracle stack changed → re-measure all baselines. |
| A2 | The upper tail lies in the GPD max-domain of attraction. | CER tails are smooth and bounded in `[0,1]`; PBdH applies very broadly. | A pathological multimodal tail (e.g. a second cluster of near-1.0 docs) → inspect; the empirical clamp + "never under-state" guards still bound it. |
| A3 | POT threshold (`1 − pot_frac`) sits in the genuine tail. | Default 15% exceedances balances tail-purity vs fit stability; standard POT practice. | If the fit is unstable across `pot_frac ∈ {0.10, 0.15, 0.20}`, widen the corpus before trusting `EVT_p999`. |
| A4 | ≥ `MIN_EXCEEDANCES` (8) exceedances to trust the GPD. | Small corpora are explicitly handled — `empirical-fallback` below the threshold. | More data lifts a fallback case into a real `pwm` fit; re-run. |
| A5 | The f32 baseline bounds are on the same corpus + commit as the candidate. | The gate compares like-for-like; provenance is recorded with every measurement (truth-pack hashes). | Any provenance mismatch voids the comparison (plan §8.4 artifact-graph fields). |
| A6 | The shipping Rust `tail_risk_monitor` is numerically equivalent to this reference. | Phase 4/5 cross-checks the Rust monitor against `af2_tail_risk.py` on the same CER vectors. | A divergence beyond float-rounding → bug in one implementation; block release until reconciled. |

---

## 8. Artifact interface (`scripts/af2_tail_risk.py`)

Offline reference tool; **never** invoked by the Rust inference binary (no Python
at inference time, AGENTS.md). Stdlib-only. Emits one self-describing NDJSON
record.

```bash
# Compute the three statistics from a JSON array of per-doc CER
python3 scripts/af2_tail_risk.py --input cer.json

# From stdin (one float per line; '#' comments and JSON arrays both accepted)
focr ocr-corpus ... --emit-cer | python3 scripts/af2_tail_risk.py --stdin

# Gate an int4 candidate against the f32 baseline + ledgered budget
python3 scripts/af2_tail_risk.py --input int4_cer.json \
    --alpha 0.1 --baseline-cvar 0.041 --baseline-evt 0.180 --budget 0.01
#   exit 0 = computed (+ gate passed if requested)
#   exit 1 = bad input / usage
#   exit 3 = a release gate was requested and the tail bound FAILED it
```

**Emitted record (keys are parameter-stable, e.g. `cvar_0.1`, `evt_p999`):**

| Key | Meaning |
|-----|---------|
| `mean` | naive average CER — **reported, never gated** |
| `var_0.1` | VaR_α (the `1−α` quantile), for context |
| `cvar_0.1` | CVaR_α — mean of the worst α-fraction (the gate variable) |
| `evt_p999` | EVT/GPD estimate of the p99.9 document (CER-clamped to `[0,1]`) |
| `gpd_shape`, `gpd_scale`, `gpd_threshold` | the fitted `(ξ, β, u)` |
| `gpd_n_exceed`, `gpd_method` | exceedance count; `"pwm"` or `"empirical-fallback"` |
| `max_cer`, `min_cer`, `n` | corpus extremes and size |
| `gate` (when a baseline is given) | `{budget, checks[], verdict, fallback?}` — the release verdict |

The shipping `tail_risk_monitor` (Rust, Phase 4/5) emits the same triple
`(mean_CER, CVaR_0.1, EVT_p999)` into the release scorecard, gating the
conformance pillar on the CVaR/EVT bound exactly as specified here.

---

## 9. References within this repo

- Plan [§9.7 AF-2](../../COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md) — the family spec.
- Plan §8.4 / §8.5 — release scorecard + three-pillar gauntlet (where this gate lives).
- [`PROPOSED_ARCHITECTURE.md`](../PROPOSED_ARCHITECTURE.md) lines on mixed int4/int8
  (AF-1 allocator) and R-SWA int8-attention CVaR gate (`bd-2mo.15`).
- [`AGENTS.md`](../../AGENTS.md) doctrine #8 (honest, measured everything) and the
  Alien-Artifact Engineering Contract (fallback wired first, evidence ledger).
- [`docs/truth-pack/PINNED_SOURCES.md`](../truth-pack/PINNED_SOURCES.md) — the
  oracle stack every CER number is measured against.
