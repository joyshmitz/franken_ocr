# franken_ocr — Known Conformance Discrepancies

This document is the honest-divergence ledger: every place where `franken_ocr`'s
output or behavior **intentionally or measurably differs** from the reference
Baidu Unlimited-OCR model (the PyTorch `transformers` oracle pinned by
`scripts/gen_reference_fixtures.py`).

A discrepancy is only recorded once its impact has been **measured** against the
reference. Speculation does not belong here; the cost of a divergence must be a
real number tied to a real test before it is accepted. Every accepted divergence
carries a **kill-switch** (an environment variable that restores reference
behavior) so it can be toggled off for bit-exact comparison.

## Per-entry schema

```
## DISC-NNN: <short title>
- Reference behavior: <what the torch/transformers oracle does>
- Our impl: <what franken_ocr does, and where (file)>
- Measured impact: <real numbers vs reference — CER / token diff / TEDS / timing>
- Kill switch: <FOCR_* env var that restores reference behavior>
- Resolution: ACCEPTED / INVESTIGATING / REVERT
- Tests affected: <test names / fixture corpus>
- Review date: <YYYY-MM-DD>
```

Quantization-induced divergences (int8, then int4) are the expected source of
most future entries: each will record the per-bit-width measured accuracy delta
against the bf16 reference, the kill switch (e.g. forcing a layer back to higher
precision), and the corpus slice (dense text / tables / formulas / numbers) where
the impact was measured.

---

_No discrepancies recorded yet. Nothing has been measured against the reference
oracle — the inference path does not exist. This stays empty until a real,
measured divergence appears; no placeholder or fabricated entries._
