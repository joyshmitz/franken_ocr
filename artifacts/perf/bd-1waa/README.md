# bd-1waa — decode-attention levers evidence bundle

Lever 1 of the M4 decode perf sweep: replace the bit-exact scalar
`rswa::decode_attention` with faster but **non-bit-exact** variants.

Two variants tested, both gated behind kill-switches (default OFF):

- `FOCR_ATTN_GEMM` — per-head batched GEMM for QKᵀ / softmax / probs@V
  (reorders f32 accumulation → not bit-exact).
- `FOCR_INT8_KV` — int8-quantized KV cache + int8 QK (lossy by construction).

A third, separate lever in the same sweep — `FOCR_QKV_FUSED` (fuse the three
q/k/v projections into one `[3*1280,1280]` GEMV) — is **byte-identical** (unit
test `fused_qkv_gemv_is_byte_identical_to_three_calls`) and is KEPT as a lossless
win; it is NOT in this negative bundle.

## Files

- `perf_sweep_m4_page0023.txt` — M4 decode timings, all five configs
  (base / qkv / gemm / gemmqkv / int8kv) on page_0023 (821 decode tokens).
- `gemmqkv_perpage_cer.txt` — span-stripped content CER, gemmqkv vs baidu
  reference, all 20 pages. 19/20 pages bit-near-exact (4 byte-exact, rest
  CER < 0.04); page_0590 alone drives the 1.30 aggregate.
- `page_0590_runaway_tail.txt` — head + tail of the gemmqkv page_0590 output:
  a correct table header that degenerates into an infinite
  `<tr><td>…Hornet…David Comin…</td></tr>` row-repeat (never emits EOS).
- `page_0590_sizes.txt` — page_0590 output length per path (ref / base / qkv /
  gemm / int8kv) — the bounded-truncation-vs-runaway proof.

## Headline

- **base (scalar, bit-exact):** page_0590 → 8755 chars (graceful truncation),
  aggregate 20-page CER 0.2116.
- **gemm:** page_0590 → 91,243 chars (runaway, CER 4.23 on that page),
  aggregate CER 1.3030. **REJECT.**
- **qkv (bit-identical):** page_0590 → 8756 chars (== base), aggregate CER
  unchanged. **KEEP** (lossless).

The GEMM accumulation drift is harmless on 19/20 pages but, on the longest
repetitive-table page, tips the autoregressive sampler past the EOS-emission
tipping point → degenerate repetition. Same f32-reorder drift was KEPT for the
SAM *vision* attention (bd-3n16) because there it feeds a projector, not a token
sampler. Non-bit-exactness is acceptable in vision, disqualifying in decode.
