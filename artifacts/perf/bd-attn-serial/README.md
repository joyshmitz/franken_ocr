# bd-attn-serial — serial/quantize-once decode attn projections (REJECTED, perf regression)

**Claim under test:** the decode "attn" phase (5436 ms of a 15.15 s decode, ~36%)
looked overhead/dispatch-bound, so running the four projections (q/k/v/o) as
SERIAL `gemv_i8_serial` calls — with the input row quantized ONCE and shared
across q/k/v — should cut per-token rayon dispatch (~960 tiny tasks/token) and the
two redundant re-quantizations, speeding the phase.

**Lever:** `FOCR_ATTN_SERIAL` branch in `decode_step_with_cache_i8`: quantize
`nrow` once → three `gemv_i8_serial` (q/k/v) + one for o_proj, vs the default four
`gemv_i8` (each re-quantizes and fans the 1280-wide m=1 GEMV across ~20 rayon
blocks). Provably **bit-identical** to the default (same `quantize_row_i8`, same
exact i32 SDOT accumulation regardless of block split) — so a perf win would have
been free/lossless.

**Measured (Apple M4, aarch64+neon+dotprod, page_0023, 821 decode tokens,
`FOCR_DECODE_INT8=1`, profiling on):**

| config | decode (s) | s/tok | attn phase (ms) |
|---|---|---|---|
| int8 default (`int8_baseline.perf`) | 15.15 | 0.018 | **5436** |
| serial/quantize-once (`attn_serial.perf`) | 15.73 | 0.019 | **6456 (+19%)** |

The attn phase **regressed +1020 ms (+19%)**; whole-decode went 15.15 → 15.73 s.
(lm_head/experts/route drifted ±370 ms run-to-run noise in the untouched phases;
the +1020 ms attn move is ~2.7× that band — a real regression.)

**Root cause / lesson:** the q/k/v/o projections are NOT dispatch-bound — the
rayon block-parallel `gemv_i8` (a 1280-wide m=1 GEMV split across ~20 cores) is
genuinely faster than a single-core serial SDOT, so serializing them loses far
more to single-threading than it saves on dispatch + the one extra quantize. The
"~0.9 GB/s effective" figure that motivated this was a MIS-ATTRIBUTION: the attn
phase's time is dominated by the **f32 `decode_attention`** (R-SWA over ~277
reference + 128 ring keys × 10 heads × 12 layers), NOT by the int8 projections.
The projection bytes are small and already parallel; the real attn lever is
`rswa::decode_attention` itself (f32 scalar-dot attention), not the projections.

**Disposition:** REVERT (flag + branch removed; `decode_step_with_cache_i8`
restored byte-for-byte to the committed baseline). No CER run needed — the path
was bit-identical, and it lost on perf.

**Retry condition:** do not re-serialize the decode attn projections (q/k/v/o)
unless they are first FUSED into one wide GEMV (e.g. a stacked `[3·1280, 1280]`
qkv weight so ONE parallel dispatch covers q+k+v) — i.e. reduce dispatch COUNT
while KEEPING the block-parallelism, never trade the parallelism for single-core
serial. The attn-phase perf budget should target `rswa::decode_attention` (f32),
not the projections.
