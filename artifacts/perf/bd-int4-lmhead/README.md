# bd-int4-lmhead — int4 `lm_head` decode GEMV (REJECTED, perf regression)

**Claim under test:** `lm_head` is the one bandwidth-bound decode GEMV
(129280×1280 = 165 MB int8 weight read/token, measured ~62 GB/s), so storing +
running it as packed int4 (half the bytes) should reduce its ~2130 ms of the
15.75 s decode. Argmax is robust to per-group int4 error, so accuracy was
expected to hold.

**Lever:** new `decoder::gemv_i4` (mirror of `gemv_i8`) calling
`simd::int4::igemm_s4s8`, an opt-in packed-int4 `lm_head` cache built when
`FOCR_LMHEAD_INT4` is set (value selects group size 16 or 32), routed through
`lm_head_cached_i8`. Activation int8-quantized once; weight per-group int4.

**Measured (Apple M4, aarch64+neon+dotprod, page_0023, 821 decode tokens,
`FOCR_DECODE_INT8=1`, profiling on):**

| config | decode (s) | s/tok | lm_head phase (ms) |
|---|---|---|---|
| int8 baseline (`int8_baseline.perf`) | 15.75 | 0.019 | **2130** |
| int4 lm_head g32 (`int4_lmhead_g32.perf`) | 25.85 | 0.031 | **12278** |
| int4 lm_head g16 (`int4_lmhead_g16.perf`) | 26.12 | 0.032 | **12720** |

The lm_head phase **regressed 5.8×** (2130 → 12278 ms) and whole-decode wall time
went 15.75 → 25.85 s. Not a measurement artifact: wall time and the phase counter
moved together by ~+10 s.

**Root cause — `igemm_s4s8` unpacks to int8 in memory before the dot.** The int4
kernel has SIMD only for *nibble unpack* (`unpack_nibbles_{scalar,neon,avx2}` →
`&mut [i8]`), then calls the int8 SDOT on the materialized buffer. So per GEMV the
traffic is `read 0.5 B/weight packed → write 1 B/weight int8 → read 1 B/weight
int8` ≈ **2.5× the bytes** of the int8 path (which reads 1 B/weight once), plus a
per-64-row-block allocation × 2020 blocks. For a bandwidth-bound GEMV that is
strictly worse than int8 — the packed form saves disk/RAM footprint but the decode
kernel pays *more* memory traffic, not less.

**Generalization (why the int4-experts arm of the blend is closed by this too):**
the experts GEMVs use the SAME `igemm_s4s8` unpack kernel and are *overhead/
dispatch-bound* (~3.2 GB/s, far below bandwidth), so halving stored bytes cannot
help them either while the unpack adds the identical 2.5× traffic penalty. Testing
int4 on the single *most-favorable* (bandwidth-bound) tensor and seeing a 5.8×
regression closes the int4-via-unpack blend for the whole decoder.

**Disposition:** REVERT (source wiring removed; kill-switch dormant code not
landed). Accuracy (20-page CER) intentionally NOT run — rejected on perf alone.

**Retry condition:** do not retry int4 for any decode GEMV until a **native
packed-int4 dot** exists that consumes nibbles directly into the SDOT/SMMLA MAC
*without materializing an int8 buffer in memory* (fused unpack-in-register). Even
then, the ceiling is small: lm_head is ~14% of decode, so a perfect int4 lm_head
saves ≤ ~4% of a full page; the other 82% of decode (experts 45% + attn 37%) is
overhead-bound, where int4 (a bytes lever) does not apply. The decode lever is
dispatch/alloc-overhead reduction, not weight-byte reduction.
