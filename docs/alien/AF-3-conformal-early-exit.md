# AF-3 — Conformal / Sequential-Test Gate for Provably-Safe Speculative & Early-Exit Decode

> **Galaxy-brain transparency card + design spec** for the `speculative_guard` e-process.
> Plan §9.7 **AF-3** · §9.5 (radical-ideas register) · bead `bd-1xfa.3` (`ALIEN-af3`),
> epic `bd-1xfa`. The dedicated test bead is `bd-1xfa.3.1` (`ALIEN-af3-tests`).
>
> **One line.** Speculative/early-exit decode is high-EV (skip the full forward on easy
> printed-text runs) but high-risk for exact-token OCR (one flipped token silently corrupts
> the parse). AF-3 makes it **provably safe**: a cheap *draft* forward proposes the next
> token, and an **anytime-valid sequential test (e-value / e-process)** on the draft-vs-full
> evidence accepts the draft **only while a finite-sample token-flip bound holds at risk
> level α**. By Ville's inequality the lifetime probability of a guard breach is **≤ α**, no
> Bonferroni penalty over the unbounded decode stream — the *same* machinery the §8.5
> gauntlet already runs for its invariant monitors. Default OFF (`α → 0` ⇒ always full
> forward); on only behind a measured A/B and a passing disagreement gate.

**Tier B · EV medium-high · Phase 3+ upside lever (NOT a v1 default).**
AF-1/AF-2/AF-5 are the load-bearing trio; **AF-3/AF-4 are upside levers behind their
guarantees** (plan §9.7 closing paragraph). This document is the DESIGN artifact; the
runtime artifact (`speculative_guard` + calibration fixture) and its proof obligation are
implemented by `bd-1xfa.3` on `src/` and gated by `bd-1xfa.3.1`. **This file writes only to
`docs/`; no `src/` or `Cargo.toml` is touched by this bead.**

---

## 0. Provenance & the numbers this card substitutes

Every model constant below is line-backed in the truth pack @ pinned commits
HF `3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5` / GitHub `7e98affeacba24e95562fbaa234ddb89b856874a`
(`docs/truth-pack/PINNED_SOURCES.md`, `docs/truth-pack/CENSUS.md`, `docs/truth-pack/OQ_INDEX.md`).

| Quantity | Value | Source |
|---|---|---|
| Decoder layers | **12** (`num_hidden_layers`) — layer 0 **dense**, layers 1–11 **MoE** | `config.json:40,108`; `first_k_dense_replace:1` `config.json:28,96` (CENSUS L82–83) |
| Routed experts / shared experts | **64 routed + 2 shared** (`n_routed_experts:64`, `n_shared_experts:2`) | `config.json:36,104` / `:37,105` (CENSUS L80–81) |
| Routing | router gate `1280→64` → softmax → **top-6 greedy** → `norm_topk_prob` (`topk_method:"greedy"`) | plan §3 L119, L134; `modeling_deepseekv2.py` |
| Expert MLP | `1280 ↔ 896` SiLU-gated (`gate`/`up`/`down`); layer-0 dense `intermediate=6848` | plan §3 L134; CENSUS L47–49, L76 |
| Hidden / heads / head_dim | `1280` / `10` / `128` (`use_mla=false`, MHA) | `config.json:29,97 / 38,106`; CENSUS L234–235 |
| Vocab (lm_head `1280→129280`) | **129280** | plan §3 L202–204; CENSUS L69 |
| R-SWA window `W` | **128** (`sliding_window_size`, `config._ring_window`) | `config.json:52`; `modeling_deepseekv2.py:1282`; CENSUS L196–205 |
| Decode regime | **greedy, `temperature=0`** (argmax); `no_repeat_ngram_size=35`, `ngram_window` 128 (single) / 1024 (multi) | plan §3 L166–167, §6.3 L508 |
| Worst-case reference keys `m_max` | `32768 + 128 = 32896` | CENSUS L225, L266 |

These are the substituted values the card promises (plan §9.7: *equation · substituted
values · plain-English intuition · validity assumptions · what would flip the decision*).
α, p₀, λ, the measured acceptance rate and the measured speedup are **calibration outputs**
recorded by `bd-1xfa.3` into `docs/PERF_LEDGER.md` — they are deliberately left as symbols
here because they are measured, not assumed (G1 > G2: no perf number is asserted before it
is measured under §9.3 fairness controls).

---

## 1. The alien-artifact contract (explicit state / action / loss)

Per AGENTS.md *Alien-Artifact Engineering Contract* and the epic's non-negotiable list,
every runtime/adaptive decision ships **(1)** an explicit state space, actions, loss matrix;
**(2)** posterior/confidence terms + a calibration metric; **(3)** a deterministic fallback
trigger **wired first**; **(4)** an evidence-ledger artifact; **(5)** this galaxy-brain card;
**(6)** an assumptions ledger (§9). Filled in for AF-3:

### 1.1 State space
At decode step *t* the controller state is the running **e-process value** `E_t ≥ 0`
(initialised `E_0 = 1`), plus the per-step evidence `e_t` derived from the **draft-vs-full
agreement signal** for token *t* (defined in §3). `E_t` is a deterministic function of the
emitted-token history and the per-step margins — **no RNG**, so the whole controller is
reproducible bit-for-bit (a hard requirement: the decode loop is greedy/deterministic, plan
§6.5 L624).

### 1.2 Actions (per decode step)

| Action | Meaning | Cost |
|---|---|---|
| **ACCEPT_DRAFT** | emit the cheap draft's argmax token; skip the routed top-6 gather + lm_head verify *for that step* | cheap (draft forward only) |
| **FALL_BACK** | run the full forward, emit the **full-model** argmax token, fold the disagreement into `E_t` | full forward (safe) |

> **Critical invariant — the emitted token is ALWAYS the full-model token.** ACCEPT_DRAFT is
> only taken on steps where the draft *provably* agrees with what the full forward would have
> emitted (within the guarded budget); on any step the guard is uncertain, we FALL_BACK and
> emit the verified token. This is why speculation can be made **byte-identical** to full
> decode under greedy `temperature=0` (the determinism gate, §6) and is the load-bearing
> design decision that distinguishes "safe early-exit" from naive speculative decoding.

### 1.3 Loss matrix
The decision-theoretic loss that justifies the gate (state = e-process; the loss the test
controls):

| Outcome | Loss | Why |
|---|---|---|
| ACCEPT, draft == full | `−c_save` (a *win*: cheap step) | the upside we are harvesting |
| ACCEPT, draft ≠ full | `+L_flip` (**catastrophic**) | a silent token flip ⇒ CER-corrupting parity failure on exact-token OCR; `L_flip ≫ c_save` |
| FALL_BACK | `+c_full` (the full-forward cost; **safe**) | the conservative default; correctness preserved |

For exact-token OCR `L_flip` is effectively unbounded relative to `c_save` (dense numerics,
tables, sub/superscripts have **zero** tolerance for a flipped digit). The whole point of
AF-3 is therefore not to *minimise expected loss by guessing* but to **drive the realised
probability of the `ACCEPT, draft ≠ full` event below α with a finite-sample guarantee** —
turning an unacceptable tail into a controlled one. That is exactly what a sequential
test / e-process delivers.

### 1.4 Posterior / confidence + calibration metric
- **Confidence term:** the e-process `E_t` *is* the running confidence — it is the wealth of
  a fair-bet martingale against the null "the draft agrees with the full model." Large `E_t`
  ⇒ accumulated evidence *against* the null ⇒ stop accepting.
- **Calibration metric:** the **measured token-disagreement rate** per corpus slice (printed
  text / tables / dense numerics) versus α; and the **measured guard-breach rate** versus the
  Ville bound α. Both are emitted to `tests/artifacts/af3/` by `bd-1xfa.3.1`.

---

## 2. The math — equation (the card's headline)

### 2.1 The null and the guarantee
Let the **null hypothesis** at step *t* be

```
H0(t):  argmax( draft_logits_t ) == argmax( full_logits_t )
        (the cheap draft would emit the same greedy token as the full model)
```

We build a non-negative **e-process** `(E_t)_{t≥0}` that is a *supermartingale under H0*:

```
E_0 = 1 ,
E_t = E_{t-1} · e_t ,        with   E_{H0}[ e_t | F_{t-1} ] ≤ 1      (the e-value property)
```

so `(E_t)` is a non-negative supermartingale with `E_{H0}[E_t] ≤ 1` for all *t*.
**Ville's inequality** then gives the anytime-valid bound

```
P_{H0} ( sup_{t ≥ 1}  E_t  ≥  1/α )  ≤  α .                          (★)
```

**Decoding rule.** Accept the draft at step *t* iff the guard has *not* tripped:

```
ACCEPT_DRAFT   if   E_t < 1/α
FALL_BACK      if   E_t ≥ 1/α   (and on fall-back, re-base / continue per §3.4)
```

By (★), the probability that the guard *ever* lets the e-process cross `1/α` under the null
— i.e. the probability of a **guard breach over the entire, unbounded decode stream** — is
**≤ α**. There is **no multiple-testing / Bonferroni penalty** for the open-ended sequence;
that anytime-validity is the precise reason an e-process is preferred over a fixed-N test for
a decode loop of unknown length (and is the same property §8.5 relies on for its invariant
e-processes, plan L696).

> **Plain-English intuition.** Picture a gambler who bets, each token, that the cheap draft
> agrees with the full model. The bet is *fair under the null* (it can't make money in
> expectation if the draft really does agree). If the draft is secretly drifting, the gambler
> steadily gets rich. We keep trusting the cheap draft only while the gambler is still poor
> (`E_t < 1/α`); the instant the gambler's wealth crosses `1/α` we stop trusting and pay for
> the full forward. Ville's inequality is the casino's guarantee: a fair gambler multiplies
> their stake 1/α-fold at most an α-fraction of the time. So "draft secretly wrong yet
> accepted" happens with probability ≤ α — a **finite-sample, lifetime token-flip bound**,
> not an asymptotic hope.

### 2.2 The per-step e-value `e_t`
We must turn the draft-vs-full agreement into a concrete `e_t` with `E_{H0}[e_t|F_{t-1}] ≤ 1`.
Two interchangeable, deterministic constructions; the calibration fixture (§4) picks and
parameterises one. **Both are computed from the same forward AF-3 already runs, so there is
no extra model cost beyond the draft itself.**

**(a) Betting (Robbins) e-value on the disagreement indicator (preferred — anytime-valid,
parameter-light).**
Let the per-step *disagreement* indicator be `D_t = 1{draft_t ≠ full_t} ∈ {0,1}`, observed
only on audited/fall-back steps (§3.3). The **null we want to police** is "the draft's
disagreement probability is at most `q0`," i.e. `H0: E[D_t | F_{t-1}] ≤ q0` (where
`q0 = 1 − p0`, and `p0` is the calibrated *agreement* rate AF-3 needs to break even). A
single-step **e-value that grows when disagreements pile up** is the betting wealth update

```
e_t = 1 + λ · ( D_t − q0 ) ,        λ ∈ [0, 1/(1 − q0)) ,        E_t = Π_{audited s ≤ t} e_s
```

Under `H0` (`E[D_t] ≤ q0`) we have `E_{H0}[e_t | F_{t-1}] = 1 + λ·(E[D_t] − q0) ≤ 1` — the
**e-value property** (§2.1) holds, so `E_t` is a non-negative supermartingale and Ville (★)
applies. The bound on `λ` keeps `e_t ≥ 0` even when `D_t = 1` (`e_t = 1 + λ·(1 − q0) ≥ 0`).
**Direction:** an *agreement* (`D_t = 0`) shrinks wealth (`e_t = 1 − λ·q0 < 1`); a
*disagreement* (`D_t = 1`) grows it (`e_t = 1 + λ·(1 − q0) > 1`). So accumulating
disagreements drive `E_t` toward the `1/α` alarm and trip the guard — exactly the alarm
direction we want. The margin `Δ_t = draft_logit[1st] − draft_logit[2nd]` is *not* part of
this e-value; it feeds the **verify policy** (§3.3, low margin ⇒ force a verify), keeping the
e-value's bet purely on the binary agreement event the OCR output actually depends on.

> **Why q0 (= 1 − p0), λ are the only two knobs.** This is the **Robbins / betting**
> e-process: `e_t = 1 + λ(D_t − q0)` is *fair* (an e-value) for **every** true disagreement
> rate `≤ q0`, and it *grows in expectation* whenever the true rate exceeds `q0` (the draft is
> worse than we calibrated for). Calibration chooses `q0` = the disagreement rate AF-3 can
> tolerate at break-even and `λ` to maximise growth-vs-safety on the calibration slice; Ville
> (★) does the rest. `p0 = 1 − q0` and `λ` are exactly the parameters the epic's contract
> names (bd-1xfa.3 step 4: *"e-process parameters (p0, lambda) to the chosen alpha"*).

**(b) Likelihood-ratio / SPRT form (bounded-horizon analog).**
`e_t = q1(x_t) / q0(x_t)` where `q0` is the null agreement model and `q1` an alternative
(draft drifting). `E_t = Π e_t` is the SPRT statistic; the accept boundary is `1/α`. This is
the bounded-horizon analog the bead notes (bd-1xfa.3 step 3); the **e-value/e-process form
(a) is the anytime-valid one and is the default** for the open-ended decode loop.

### 2.3 Substituted values (worked sketch)
Take α = 0.01 ⇒ guard boundary `1/α = 100`. Suppose calibration sets the tolerated
disagreement rate `q0 = 0.03` (so the break-even agreement rate `p0 = 0.97`) and betting
fraction `λ = 0.5` (well inside the cap `1/(1 − q0) ≈ 1.031`). Then per audited step:

- **Agreement** (`D_t = 0`): `e_t = 1 + 0.5·(0 − 0.03) = 0.985` — wealth shrinks ×0.985
  (evidence *for* the null; safe to keep accepting).
- **Disagreement** (`D_t = 1`): `e_t = 1 + 0.5·(1 − 0.03) = 1.485` — wealth grows ×1.485
  (evidence *against* the null; pushing toward the alarm).

Under the calibrated null (`E[D_t] = q0 = 0.03`): `E[e_t] = 1 + 0.5·(0.03 − 0.03) = 1.000` —
exactly fair (the e-value property at the boundary). To *trip* (`E_t ≥ 100 = 10^2`) from a
run of pure disagreements needs `k` with `1.485^k ≥ 100` ⇒ `k ≈ ln100/ln1.485 ≈ 11.6` →
**~12 audited disagreements in a row** fire the guard; interleaved agreements (the common
case) keep `E_t` low and speculation alive. By Ville (★) the lifetime probability that this
ever happens under the true null is **≤ α = 0.01**. The card's commitments are the **boundary
`1/α`**, the **fair update** above, and the **Ville guarantee (★)**; the *fitted* (`q0`, `λ`,
audit cadence, margin threshold) are calibration outputs (§4) and the e-value property +
alarm direction are **unit-tested** under a synthetic null/alternative (§6, bd-1xfa.3.1
step 1).

> The honest statement: AF-3 ships the **e-process recurrence + Ville boundary** as the
> contract; the fitted (`q0`, `λ`) values are a *calibrated, unit-tested* artifact, not
> frozen in prose, so the implementation can pick the tightest fair construction on the real
> corpus. What is non-negotiable is the **e-value property** `E_{H0}[e_t|F_{t-1}] ≤ 1`
> (unit-tested) and the **boundary `1/α`** (Ville). A growth-optimal alternative — the
> **GROW / mixture** e-value that integrates `λ` over a prior instead of fixing it — is a
> drop-in upgrade behind the same boundary if the fixed-`λ` bet leaves acceptance on the
> table.

---

## 3. The cheap draft + the decode-loop wiring

### 3.1 What the draft is
The draft is a **reduced forward** that proposes the next token without paying the full MoE
cost. Two candidates (the bead names both; the calibration fixture picks the one that holds
the bound at the best acceptance rate):

- **Shared-experts-only draft (preferred).** Run only the **2 always-on shared experts**
  (`mlp.shared_experts.{gate,up,down}_proj`, 1280↔896 SiLU-gated) and skip the **top-6 routed
  gather**. Rationale: the shared experts are *already loaded* every step, so this draft
  **streams no extra weights** — and decode is **memory-bandwidth-bound** (plan §6.3), so the
  routed top-6 gather is the dominant decode cost we are trying to skip. This is the lowest
  marginal-cost draft (bead KEY DESIGN DECISION: *"shared-experts-only ... avoids extra weight
  streaming"*).
- **Reduced-top-k routed draft (fallback draft).** Route to the top-`k'` experts with
  `k' < 6` (e.g. top-2). Higher fidelity, higher cost than shared-only.

The draft and the full forward **share the one rayon pool**; AF-3 **must not** spawn a
concurrent draft thread (AGENTS.md doctrine #5: single live forward, never nested rayon
under a lock, never a 2nd asupersync runtime — the draft is computed *inline* on the same
sequential decode step, then optionally verified).

### 3.2 Per-step control flow (the artifact)

```
for each decode step t:                              # greedy, temperature = 0
    draft_logits  = shared_experts_only_forward(h_t) # cheap; no routed gather
    draft_tok     = argmax(draft_logits)             # candidate
    margin Δ_t    = draft_logits[1st] − draft_logits[2nd]

    if speculative_guard.should_verify(t, Δ_t):      # audit / forced-verify policy, §3.3
        full_logits = full_forward(h_t)              # routed top-6 + shared + lm_head
        full_tok    = argmax(full_logits)            # the AUTHORITATIVE token
        D_t         = (draft_tok != full_tok) ? 1 : 0   # disagreement indicator, §2.2
        e_t         = 1 + λ·(D_t − q0)               # betting e-value (q0 = 1 − p0)
        E_t         = E_{t-1} · e_t
        emit full_tok                                # always emit the VERIFIED token
        if E_t ≥ 1/α:  speculative_guard.trip()      # disable further speculation (or re-base, §3.4)
    else:
        emit draft_tok                               # accept the cheap guess (guard holds)
```

The same `no_repeat_ngram_size=35` blocklist (window 128 single / 1024 multi) and the same
EOS/argmax sampler (plan §3 L166, §6.3 L508) wrap **both** branches — the draft must respect
the identical decode contract as the full forward (bead DEPENDENCIES NOTE: E-PM1 sampler/
ngram census), or determinism is lost.

### 3.3 Verify policy (how often we actually pay the full forward)
Pure "accept until breach" cannot ever observe a disagreement (it never runs the full
forward on accepted steps), so the e-process would never update. AF-3 therefore uses a
**forced-audit schedule**: the guard *requires* a full verify (a) on every step where the
draft margin `Δ_t` is below a calibrated threshold (low confidence ⇒ verify), and (b) on a
**deterministic audit cadence** (e.g. every `r`-th accepted step, `r` fixed by calibration)
so the e-process keeps accumulating real evidence even on confident runs. The audit cadence
is what makes the measured disagreement rate a *valid* estimate of the true rate and what
keeps the Ville bound honest (an unaudited accept is an *un-evidenced* bet, which the e-value
construction must treat conservatively). The cadence `r` and the margin threshold are
calibration outputs (§4); they trade acceptance rate against the tightness of the realised
bound. **Conservative default if calibration is unavailable: verify every step (= full
forward = the α→0 fallback).**

> This is the genuinely subtle part and it is where AF-3 differs from textbook speculative
> decoding (which verifies *every* draft token by construction and gets exactness for free
> at the cost of always paying the verify). AF-3's upside comes from *skipping* some verifies
> — which is only sound under the e-process bound. The forced-audit schedule is the bridge:
> enough verifies to keep the bound valid, few enough to net a speedup. The honest proof
> obligation (§5) is what certifies the schedule actually held the bound on the real corpus.

### 3.4 On a guard trip
When `E_t ≥ 1/α` the guard **trips**: AF-3 disables speculation for the remainder of the
document (every subsequent step does the full forward — the safe state). Optionally a
**re-base** policy resets `E_t = 1` and resumes speculation, but **re-basing weakens the
lifetime guarantee** (each re-base is a fresh α-budget); the default is *trip-and-stay-down
per document* so the per-document breach probability is exactly bounded by α. Re-basing, if
ever enabled, is itself behind calibration and ledgered.

---

## 4. The `speculative_guard` artifact + calibration fixture

**Runtime artifact (`bd-1xfa.3`, in `src/` — NOT this bead):** a `speculative_guard`
e-process object carried alongside the decode loop, exposing:
`new(alpha, p0, lambda, audit_cadence, margin_threshold)` · `should_verify(t, margin)` ·
`observe(agree, margin) -> E_t` · `tripped() -> bool`. It holds the running `E_t`, the
forced-audit counter, and the boundary `1/α`. It is **pure / deterministic / no-RNG**.

**Env gate (consistent with the project's `OnceLock`-gated numerics levers,
`FOCR_INT8_ATTN` / `FOCR_INT8_LMHEAD` / `FOCR_VEC_EXP`):**

```
FOCR_SPECULATIVE_DECODE = 0   (DEFAULT — α → 0, speculation OFF, always full forward)
FOCR_SPECULATIVE_DECODE = 1   (opt-in; reads the calibrated (alpha, p0, lambda) profile)
FOCR_SPECULATIVE_ALPHA  = <α> (optional override of the calibrated risk level)
```

**Calibration fixture (`bd-1xfa.3`, under `tests/`/fixtures):** on a **held-out slice** of
the golden corpus, measure the draft-vs-full agreement distribution and the margin→agreement
map, then **solve for (`p0`, `λ`, `audit_cadence`, `margin_threshold`) that hit the target
α** with maximal acceptance. The fixture freezes those parameters and the slice provenance
(transformers==4.57.1, torch==2.10.0, the exact image set), per the §8.6 golden-artifact
discipline. The calibration routine is unit-tested to hit the target α on the calibration
fixture (bead step 4; bd-1xfa.3.1 step 1).

> **Minimum-calibration-corpus precondition (inherited from the conformal-ratchet sibling
> bd-re8.14 addendum).** Too small a calibration slice yields a meaningless (vacuously
> conservative) parameterisation. The calibration routine MUST compute and commit the minimum
> per-slice sample count at which the agreement estimate is not dominated by the prior, and
> **refuse to enable speculation** (fall back to α→0, ledgered) when any slice is below it —
> rather than silently shipping an unvalidated guard. This mirrors the ratchet's "refuse to
> decide → deterministic fallback" rule.

---

## 5. Proof obligation (the gate AF-3 must clear to land)

AF-3 is a `docs/NEGATIVE_EVIDENCE.md`-gated hypothesis until **all** of the following pass on
the **real golden corpus** under §9.3 fairness controls (G1 > G2 — correctness first, every
perf number measured, never assumed):

1. **Token-disagreement bound (the headline).** Measured **token-disagreement rate ≤ α on
   EVERY corpus slice** — printed text, tables, **dense numerics** (the exact-token-sensitive
   tail AF-2 worries about). A single slice over budget ⇒ REVERT + ledger (the do-not-retry
   condition).
2. **Zero CER regression.** End-to-end CER with speculation ON ≤ CER with speculation OFF on
   every slice. AF-3 may never trade accuracy for latency.
3. **Byte-identical determinism.** Decode the golden corpus with `FOCR_SPECULATIVE_DECODE`
   **ON vs OFF** under greedy `temperature=0`: **byte-identical output** (the emitted token is
   always the full-model token; speculation only skips work it would have agreed on). This is
   the determinism gate (bd-1xfa.3.1 step 2) and it is the strongest form of the safety
   claim — it makes AF-3 a *pure latency optimization* with provably no output change.
4. **Ville bound holds (unit).** Under a **synthetic null** the e-process crosses `1/α` on at
   most an α-fraction of streams (the (★) bound, empirically); under a **synthetic
   alternative** it fires reliably; the calibration routine hits the target α on the
   calibration fixture (bd-1xfa.3.1 step 1).
5. **Honest speedup ledger.** The measured **acceptance rate** and **decode speedup** are
   recorded in `docs/PERF_LEDGER.md` with α, the §9.3 fairness controls (same allocator, same
   thread count, `release-perf` profile, `cv_pct` reported), and the keep-gate evidence
   (plan §8.5 keep-gate). A speedup that does not clear the keep-gate is **not** a win.

**Reconstructable decision.** The keep/revert decision must be reconstructable from
`tests/artifacts/af3/` (per-step accept/reject, the `E_t` trajectory, acceptance +
disagreement rates per slice, speedup, provenance) — bd-1xfa.3.1 steps 5–6.

---

## 6. Deterministic fallback (wired FIRST, per the contract)

**No adaptive controller ships without a conservative deterministic fallback** (AGENTS.md
Alien-Artifact Engineering Contract). For AF-3 the fallback is built before the adaptive path
and is the **default**:

- **`α → 0` disables speculation entirely** — every token does the full forward. This is the
  literal plan-§9.7 fallback (*"Fallback: α→0 disables speculation (always full forward)"*).
  In code: `FOCR_SPECULATIVE_DECODE=0` (default) ⇒ the decode loop never constructs a draft;
  it is byte-for-byte the Phase-1/Phase-3 full-forward decode path. The `speculative_guard`
  is simply not on the hot path.
- **Trip ⇒ safe state.** A guard breach (`E_t ≥ 1/α`) falls back to full-forward for the rest
  of the document (§3.4) — the failure mode is *slower*, never *wrong*.
- **Calibration unavailable / slice too small ⇒ fall back to α→0** (the minimum-corpus
  precondition, §4) — refuse to speculate rather than ship an unvalidated guard.
- **Any proof-obligation miss ⇒ REVERT (no source landed) + ledger** in
  `docs/NEGATIVE_EVIDENCE.md` (G1 > G2). AF-3 is opt-in upside; if it cannot be proven safe it
  does not exist in the shipping path.

Because the fallback is the default and is the existing full-forward decode, the **worst case
for AF-3 is "no speedup," never "a flipped OCR token."** That asymmetry is the entire reason
the lever is acceptable for exact-token OCR.

---

## 7. Validity assumptions (when the bound is sound — and when it isn't)

The Ville guarantee (★) is sound **only** under the conditions the calibration measures and
the proof obligation re-checks. The card states them plainly:

1. **The e-value property holds:** `E_{H0}[e_t | F_{t-1}] ≤ 1` for the chosen `e_t`
   construction. This is a *mathematical* property of the construction (unit-tested), not a
   distributional assumption — it is what makes (★) hold for *any* data-generating process
   consistent with the null. **This is the robust core.**
2. **The null is correctly specified / `p0` is a true lower bound on agreement.** The betting
   e-value is fair for every true agreement rate `q ≥ p0`. If the real agreement is *below*
   `p0` (the draft is worse than calibrated), the e-process is *still* a valid supermartingale
   under the *actual* null and (★) still bounds breaches — the test just trips sooner (safe).
   The risk is the *reverse* sloppiness: defining the null around the *wrong event* (e.g.
   agreement on top-1 logit value instead of argmax token) would let a token flip slip the
   bound. The null is **argmax-token agreement** (§2.1), exactly the quantity OCR correctness
   depends on.
3. **Audited evidence is representative (no peeking / no selection bias).** The forced-audit
   schedule (§3.3) must not preferentially audit easy steps; the cadence is deterministic and
   margin-triggered, and the *measured* disagreement rate per slice (proof obligation 1)
   re-validates this on the real corpus. Anytime-validity means **no Bonferroni** is owed for
   continuous monitoring — but it does **not** excuse a biased audit sample.
4. **Calibration ≈ deployment distribution (the one genuinely empirical leg).** `p0`, `λ`,
   the audit cadence and the margin map are calibrated on the golden corpus; a wildly
   out-of-distribution document could have a true agreement rate the calibration never saw.
   Mitigation: (a) the bound is anytime-valid under the *actual* null regardless of
   calibration, so a worse-than-expected draft trips the guard *faster* (the failure is
   conservative); (b) keep α small and report measured acceptance/disagreement **per slice**;
   (c) the determinism gate (proof 3) means even an OOD document yields byte-identical output
   to full decode — the *only* thing OOD can cost is speedup, never correctness.
5. **Greedy / `temperature=0` decode.** The whole "emit the full-model token, speculation
   only skips agreed work" argument assumes a deterministic argmax sampler (plan §6.5 L624).
   Under sampling (not used by this model's default OCR path) the determinism gate would need
   the same seed on both paths; out of scope for v1.
6. **Single live forward / shared rayon pool** (AGENTS.md #5): the draft is inline on the
   sequential decode step, never a concurrent thread — otherwise the deadlock/oversubscription
   discipline is violated and the timing claims are meaningless.

---

## 8. What would FLIP the decision (the card's last required field)

AF-3 is **reverted and ledgered** (no source landed) if any of:

- **Measured token-disagreement rate > α on any corpus slice** — the headline proof
  obligation fails (this is also the `NEGATIVE_EVIDENCE.md` do-not-retry condition).
- **Any CER regression** vs full-forward decode on any slice.
- **ON ≠ OFF output** under greedy `temperature=0` (determinism gate fails) — a correctness
  bug in the guard or the draft, not a tuning issue.
- **No honest speedup** that clears the §8.5 keep-gate after the audit-cadence overhead is
  paid (the forced verifies + the draft cost can, in principle, exceed the routed-gather they
  skip — especially since decode is bandwidth-bound and the shared-experts draft still streams
  the shared weights). If the net is not faster, AF-3 has no EV and is dropped.
- **The calibration corpus is below the minimum** for a meaningful parameterisation on a slice
  that matters — fall back to α→0 (refuse to ship), ledgered.
- **AF-2's tail (CVaR/EVT) worsens** with speculation ON even if mean-CER is flat — exact-token
  OCR fails in the tail (plan AF-2), so the release gate is the **tail** metric, and AF-3 must
  not move it.

---

## 9. Assumptions ledger (the contract's item 6)

| # | Assumption | Status / how it is checked | If violated |
|---|---|---|---|
| A1 | The e-value `e_t` satisfies `E_{H0}[e_t\|F_{t-1}] ≤ 1` | **Proven by construction + unit-tested** (synthetic null, bd-1xfa.3.1) | the bound (★) is void → REVERT |
| A2 | The null is **argmax-token agreement**, the OCR-correctness-relevant event | Fixed in §2.1; re-checked by the determinism gate (ON==OFF) | a flip slips the bound → REVERT |
| A3 | `p0` is a true lower bound on draft-vs-full agreement on each slice | Measured at calibration; below-`p0` reality only trips *sooner* (safe) | conservative; net = lost speedup, not lost correctness |
| A4 | Forced-audit schedule yields a representative disagreement estimate | Deterministic + margin-triggered; re-validated by measured per-slice rate (proof 1) | biased sample → measured rate exposes it → REVERT |
| A5 | Calibration distribution ≈ deployment | Per-slice acceptance/disagreement reported; OOD trips faster + determinism gate holds | conservative; worst case = no speedup |
| A6 | Greedy `temperature=0` decode (deterministic argmax) | Plan §6.5 L624; the model's OCR path | sampling would need seed-matched paths → out of scope v1 |
| A7 | Single live forward, shared rayon pool (no concurrent draft) | AGENTS.md doctrine #5; draft computed inline | nested rayon/runtime → deadlock → forbidden |
| A8 | Calibration slice ≥ committed minimum sample count | Checked precondition (§4, from bd-re8.14 addendum) | refuse to speculate → α→0 fallback, ledgered |
| A9 | `c_full` of the skipped routed top-6 gather > draft + audit overhead | Measured speedup vs §8.5 keep-gate (proof 5) | no EV → drop AF-3 |
| A10 | The bound is **per-document** (trip-and-stay-down); re-basing is OFF by default | §3.4 | re-basing multiplies the α-budget → only behind separate calibration |

---

## 10. EV / recommendation contract (per `/alien-graveyard`)

| Field | Value |
|---|---|
| **Family** | Conformal / sequential testing (SPRT, e-values) for provably-safe early-exit & speculative decode |
| **Tier** | **B** (upside lever behind its guarantee) |
| **EV** | **medium-high** — Impact: real decode-throughput upside on easy printed-text runs; Confidence: medium (exact-token OCR is high-risk, but the e-process *bounds* the risk); Effort/Friction: higher (calibration + audit schedule + proof), hence Tier B |
| **Artifact** | `speculative_guard` e-process per decode step + a calibration fixture (`FOCR_SPECULATIVE_DECODE`) |
| **Proof obligation** | measured token-disagreement rate ≤ α on the golden corpus **AND** zero CER regression **AND** byte-identical ON==OFF under greedy |
| **Deterministic fallback** | **α → 0 disables speculation (always full forward)** — the default; wired first |
| **Fallback trigger** | any proof-obligation miss; calibration below minimum; guard trip ⇒ full forward for rest of document |
| **Phase** | **3+** (lives on the optimized decode path; reuses E-P3's MoE dispatch + decode GEMV for the shared-experts draft). NOT a v1 default. |

---

## 11. Cross-references & dependencies

- **Plan** §9.7 (AF-3), §9.5 (radical-ideas register), §6.3 (memory-bandwidth-bound decode,
  the cost AF-3 skips), §6.5 (greedy determinism / single live forward), §8.5 (the gauntlet's
  e-process invariant machinery this reuses; the keep-gate; the conformal ratchet), §9.3
  (fairness controls for the perf claim).
- **Beads** `bd-1xfa` (epic) · `bd-1xfa.3` = **`ALIEN-af3`** (this design's runtime artifact +
  proof) · `bd-1xfa.3.1` = `ALIEN-af3-tests` (e-process unit + determinism/token-flip E2E,
  depends on this) · `bd-re8.14` (the conformal lower-bound release ratchet — sibling
  conformal family; this card inherits its minimum-calibration-corpus precondition).
- **Sibling AF cards** (same epic, same galaxy-brain format): AF-1 (rate-distortion quant
  allocation), AF-2 (CVaR/EVT tail-risk CER — the release-gate metric AF-3 must not move),
  AF-4 (submodular high-precision set), AF-5 (USL pool sizing).
- **Doctrine** AGENTS.md *Alien-Artifact Engineering Contract* (state/action/loss + posterior
  + deterministic fallback + evidence ledger + this card + assumptions ledger); doctrine #1
  (G1 > G2), #5 (single live forward / no nested rayon), #8 (honest measured everything).

---

*This is a DESIGN artifact (galaxy-brain card). The runtime `speculative_guard`, its
calibration fixture, and the on-corpus proof obligation are implemented and gated by
`bd-1xfa.3` / `bd-1xfa.3.1` on `src/` + `tests/` — out of scope for this `docs/`-only bead.
AF-3 does not land until its proof obligation passes; until then it is a
`docs/NEGATIVE_EVIDENCE.md`-gated hypothesis, default OFF (α→0).*
