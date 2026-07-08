# RELEASE CERTIFICATION TEMPLATE — strict-conformant-release.v1 (bd-wp8.9)

The contract `scripts/gauntlet_cert.py --bundle` certifies against. A release
is CERTIFIED only when every predicate below holds at generation time; a
single red cell blocks certification (the bundle is still written, honestly
marked `certified: false` with the refusal reasons).

## Required-pass constants

| constant | value | meaning |
|---|---|---|
| `CERTIFICATION_MIN_VERIFICATION_PCT` | 100.0 | every readiness cell green — no partial credit |
| `CERTIFICATION_REQUIRED_SUITE_PASS_RATE_PCT` | 100.0 | the full gate (`scripts/check.sh`) exits 0 |
| `CERTIFICATION_MAX_HIGH_SEVERITY_COUNTEREXAMPLES` | 0 | no open high-severity finding anywhere in the ledgers |
| `CERTIFICATION_MAX_EVIDENCE_AGE_HOURS` | 24 | every core evidence artifact regenerated within a day of the bundle |

## Evidence-bundle classes

1. **Parity receipts** — `tests/fixtures/ladder_scorecard/scorecard_armed.json`
   (L0–L5), the armed multi-page rung logs, `focr robot selftest` per-model
   verdicts.
2. **Convergence record** — `docs/gauntlet/ROUNDS.jsonl` (≥10 rounds, last 2
   with <3 new genuine findings each; every finding a real find-fix cycle).
3. **Statistical state** — `docs/gauntlet/EPROCESS_STATE.json` (never-reset
   Ville e-processes; no invariant rejected), the conformal ratchet state in
   `docs/gauntlet/RELEASE_SCORECARD.json`.
4. **Ledgers** — `docs/DISCREPANCIES.md` (accepted divergences: measured
   impact + kill-switch + review date), `docs/NEGATIVE_EVIDENCE.md` (reverted
   levers + do-not-retry predicates), `docs/PERF_LEDGER.md` (fairness-pinned
   measured rows) — all lint-enforced by `scripts/check_ledgers.py`.
5. **Perf baseline** — `benches/.bench-history/baseline.json` (the frozen
   guardrail baseline; ratchet-only advances).

## Gate / ratchet spec

* The ship gate is `scripts/gauntlet_cert.py --release-readiness`: every cell
  reads its evidence artifact live; ANY red exits 1.
* The conformal ratchet (`--from-parity`) uses the Jeffreys×Hoeffding LOWER
  bound, truncated to 6 dp; the persisted baseline may only move up
  (`--ratchet`), never down, and `MIN_CALIBRATION_N = 20`.
* Bench regressions gate at +10% vs the frozen baseline with cv% > 5 runs
  ineligible as evidence (`scripts/bench_guardrail.py`).

## Certification flow

```
bash scripts/check.sh                                  # 100% suite pass
python3 scripts/gauntlet_cert.py --release-readiness   # all cells green
python3 scripts/gauntlet_cert.py --bundle              # writes docs/gauntlet/bundle/, exit 0 = CERTIFIED
```

The signed record is `docs/gauntlet/bundle/release_certificate.json`
(git head + describe + constants + verdicts); the human-readable twin is
`FINAL_GAUNTLET_REPORT.md` beside it.
