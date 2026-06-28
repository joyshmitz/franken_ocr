//! `int4_packed_tier` — model-free A/B microbench of the THREE int4/int8 GEMM
//! paths at the shapes the model's int4 expert/dense weights actually use
//! (bd-1azu.22). This file MEASURES; the numbers come out of `cargo bench`, not
//! out of this file — and the honest question it answers is "does consuming the
//! packed nibbles directly beat int8 (and beat the unpack-then-int8 path)?".
//!
//! Three kernels, identical (M,K,N), same per-iter cost model:
//!   * `int4::igemm_s4s8_packed` — NATIVE PACKED: nibbles masked/shifted
//!     in-register and fed straight to SDOT/SMMLA, B never materialized.
//!   * `int4::igemm_s4s8`        — UNPACK-THEN-INT8: materialize the whole dense
//!     `[n,k]` int8 B, then run the int8 GEMM (the ~5.8x-slower ledger entry).
//!   * `dispatch::igemm_s8s8`    — INT8 baseline (the bandwidth/throughput floor
//!     the int4 paths are trying to beat by halving the streamed weight bytes).
//!
//! ```text
//! FOCR_FORCE_ARCH=sdot  cargo bench --bench int4_packed_tier -- --nocapture
//! FOCR_FORCE_ARCH=smmla cargo bench --bench int4_packed_tier -- --nocapture
//! ```
//!
//! Each process reads `FOCR_FORCE_ARCH` ONCE (cached in a OnceLock), so every
//! `#[bench]` in a run dispatches to the same forced tier; the
//! `[int4_packed_tier] DISPATCHED tier = …` line (under `--nocapture`) records it.
//! Every path is BIT-IDENTICAL (held by `tests/int4_packed_parity.rs` and the
//! `simd::arm` unit tests), so this measures pure throughput.
//!
//! SHAPE RATIONALE (real dims; cite CENSUS.md / config.json):
//!   * hidden_size = 1280; moe_intermediate_size = 896 (routed-expert GEMMs, the
//!     int4 bandwidth target). Expert down_proj: K=896→N=1280; gate/up: K=1280→
//!     N=896. Dense down_proj K=6848 is the doctrine-#6 worst-case K.
//!   * M=1 = the decode GEMV (latency-critical); M=16 = a small prefill batch.
//!   * group_size = 16 (`Int4G16`, the default).

#![feature(test)]
extern crate test;

use franken_ocr::simd::{dispatch, int4};
use std::sync::Once;
use test::{Bencher, black_box};

const HIDDEN: usize = 1280; // hidden_size
const EXPERT_INTER: usize = 896; // moe_intermediate_size (routed expert GEMMs)
const DENSE_INTER: usize = 6848; // dense layer-0 intermediate_size (worst-case K)
const GROUP: usize = 16; // default int4 group size (Int4G16)

static TIER_ONCE: Once = Once::new();
fn report_tier() {
    TIER_ONCE.call_once(|| {
        eprintln!(
            "[int4_packed_tier] DISPATCHED tier = {} ({})  [override via FOCR_FORCE_ARCH]",
            dispatch::detected_tier().tag(),
            dispatch::tier_string()
        );
    });
}

// Deterministic fillers (cheap LCG, built ONCE per bench).
fn filler_i8(n: usize, seed: u64) -> Vec<i8> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (((s >> 40) % 255) as i64 - 127) as i8
        })
        .collect()
}
fn filler_u8(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(3);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 40) & 0xFF) as u8
        })
        .collect()
}
fn filler_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(7);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (((s >> 40) % 4096) as f32 / 4096.0 - 0.5) * 0.5
        })
        .collect()
}

// Native packed-int4 GEMM at (m, k, n), group=16. `out` (f32) is OVERWRITTEN, so
// no per-iter reset is needed (unlike the int8 `+=` kernel).
fn bench_packed(b: &mut Bencher, m: usize, k: usize, n: usize) {
    let a = filler_i8(m * k, 0x11);
    let bp = filler_u8(n * (k / 2), 0x22);
    let scales = filler_f32(n * (k / GROUP), 0x33);
    let mut out = vec![0.0f32; m * n];
    int4::igemm_s4s8_packed(&a, &bp, &scales, GROUP, m, k, n, &mut out); // pre-roll
    report_tier();
    b.iter(|| {
        int4::igemm_s4s8_packed(
            black_box(&a),
            black_box(&bp),
            black_box(&scales),
            GROUP,
            m,
            k,
            n,
            black_box(&mut out),
        );
        black_box(&out);
    });
}

// Unpack-then-int8 int4 GEMM (the dense-materialization path) at (m, k, n).
fn bench_unpack(b: &mut Bencher, m: usize, k: usize, n: usize) {
    let a = filler_i8(m * k, 0x11);
    let bp = filler_u8(n * (k / 2), 0x22);
    let scales = filler_f32(n * (k / GROUP), 0x33);
    let mut out = vec![0.0f32; m * n];
    int4::igemm_s4s8(&a, &bp, &scales, GROUP, m, k, n, &mut out); // pre-roll
    report_tier();
    b.iter(|| {
        int4::igemm_s4s8(
            black_box(&a),
            black_box(&bp),
            black_box(&scales),
            GROUP,
            m,
            k,
            n,
            black_box(&mut out),
        );
        black_box(&out);
    });
}

// int8 baseline at (m, k, n); the i32 out is re-zeroed each iter (the kernel ADDS).
fn bench_s8s8(b: &mut Bencher, m: usize, k: usize, n: usize) {
    let a = filler_i8(m * k, 0x11);
    let w = filler_i8(n * k, 0x22);
    let mut out = vec![0i32; m * n];
    dispatch::igemm_s8s8(&a, &w, m, k, n, &mut out); // pre-roll
    report_tier();
    b.iter(|| {
        out.fill(0);
        dispatch::igemm_s8s8(black_box(&a), black_box(&w), m, k, n, black_box(&mut out));
        black_box(&out);
    });
}

// ── expert down_proj  (K = 896, N = 1280) ──────────────────────────────────────
// PROV: int4::igemm_s4s8_packed · expert down_proj [M=1,K=896,N=1280] · decode step
#[bench]
fn packed_expert_down_decode_m1(b: &mut Bencher) {
    bench_packed(b, 1, EXPERT_INTER, HIDDEN);
}
// PROV: int4::igemm_s4s8 (unpack→int8) · expert down_proj [M=1,K=896,N=1280] · decode
#[bench]
fn unpack_expert_down_decode_m1(b: &mut Bencher) {
    bench_unpack(b, 1, EXPERT_INTER, HIDDEN);
}
// PROV: simd::igemm_s8s8 · expert down_proj [M=1,K=896,N=1280] · decode step
#[bench]
fn s8s8_expert_down_decode_m1(b: &mut Bencher) {
    bench_s8s8(b, 1, EXPERT_INTER, HIDDEN);
}

// ── expert gate/up  (K = 1280, N = 896) ────────────────────────────────────────
// PROV: int4::igemm_s4s8_packed · expert gate/up [M=1,K=1280,N=896] · decode step
#[bench]
fn packed_expert_gateup_decode_m1(b: &mut Bencher) {
    bench_packed(b, 1, HIDDEN, EXPERT_INTER);
}
// PROV: int4::igemm_s4s8 (unpack→int8) · expert gate/up [M=1,K=1280,N=896] · decode
#[bench]
fn unpack_expert_gateup_decode_m1(b: &mut Bencher) {
    bench_unpack(b, 1, HIDDEN, EXPERT_INTER);
}
// PROV: simd::igemm_s8s8 · expert gate/up [M=1,K=1280,N=896] · decode step
#[bench]
fn s8s8_expert_gateup_decode_m1(b: &mut Bencher) {
    bench_s8s8(b, 1, HIDDEN, EXPERT_INTER);
}

// ── prefill (M=16) at expert down_proj ─────────────────────────────────────────
// PROV: int4::igemm_s4s8_packed · expert down_proj [M=16,K=896,N=1280] · prefill16
#[bench]
fn packed_expert_down_prefill_m16(b: &mut Bencher) {
    bench_packed(b, 16, EXPERT_INTER, HIDDEN);
}
// PROV: int4::igemm_s4s8 (unpack→int8) · expert down_proj [M=16,K=896,N=1280] · prefill16
#[bench]
fn unpack_expert_down_prefill_m16(b: &mut Bencher) {
    bench_unpack(b, 16, EXPERT_INTER, HIDDEN);
}
// PROV: simd::igemm_s8s8 · expert down_proj [M=16,K=896,N=1280] · prefill16
#[bench]
fn s8s8_expert_down_prefill_m16(b: &mut Bencher) {
    bench_s8s8(b, 16, EXPERT_INTER, HIDDEN);
}

// ── dense down_proj worst-case K=6848 (the doctrine-#6 accumulator bound) ──────
// PROV: int4::igemm_s4s8_packed · dense down_proj [M=1,K=6848,N=1280] · decode step
#[bench]
fn packed_dense_down_decode_m1(b: &mut Bencher) {
    bench_packed(b, 1, DENSE_INTER, HIDDEN);
}
// PROV: simd::igemm_s8s8 · dense down_proj [M=1,K=6848,N=1280] · decode step
#[bench]
fn s8s8_dense_down_decode_m1(b: &mut Bencher) {
    bench_s8s8(b, 1, DENSE_INTER, HIDDEN);
}
