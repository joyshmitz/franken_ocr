//! bd-1azu.17 — GATE: batched-decode bit-identity (M=B GEMM == sequential per-row
//! GEMV) at the REAL decode linear shapes, plus the *negative* self-tests that
//! prove this gate is actually sensitive to a batching bug.
//!
//! ## What proves what (no duplication)
//! The POSITIVE bit-identity of the batched decode path is already proven, byte
//! for byte, by the shipped harnesses — this file cites and *spot-re-affirms*
//! them rather than re-deriving them:
//! - `tests/batched_igemm_parity.rs` — the int8 micro-kernel: `igemm(M=B)` row `r`
//!   == `igemm(m=1)` row `r` for M 1→256, the i64 absolute oracle, and the
//!   K=6848 Doctrine-#6 overflow worst case, across every `FOCR_FORCE_ARCH` tier
//!   the host can dispatch.
//! - `tests/int32_overflow_proof.rs` — i32 accumulator never overflows at K=6848.
//! - `tests/batched_forward_parity.rs` / `_attention_` / `_moe_` / `_sampler_` —
//!   the per-layer projection driver, R-SWA attention, grouped MoE, and sampler
//!   each batched == sequential per stream.
//!
//! ## What THIS file adds
//! 1. A decode-shape spot-check: the actual decoder linear shapes (qkv-fused,
//!    o_proj, expert gate/up + down, dense-0 down at K=6848, an lm_head slice)
//!    are byte-identical M=B vs m=1 and i64-oracle-clean — confirming the
//!    M-independence holds at the shapes the spine actually runs.
//! 2. The **perturbed-batcher self-tests** (the bd-1azu.17 mandate): a deliberately
//!    wrong row-scatter MUST make the equality assertion FAIL. Without this, a
//!    green positive test could hide a batcher that happens to be wrong in a way
//!    two paths share — these prove the gate has teeth.
//!
//! ## Tier coverage
//! Runs on the host's detected tier; the cross-tier `FOCR_FORCE_ARCH` sweep
//! (scalar / sdot / smmla on Apple Silicon; avx2 / avxvnni / avx512vnni on x86)
//! is driven by `tests/batched_igemm_parity.rs`. The VNNI/AMX uarch tiers
//! (bd-1azu.52 / .53) are skip-with-SUCCESS where the hardware is absent.

use franken_ocr::simd::{detected_tier, igemm_s8s8};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        // SplitMix64.
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn i8(&mut self) -> i8 {
        (self.next() & 0xff) as u8 as i8
    }
    fn fill_i8(&mut self, n: usize) -> Vec<i8> {
        (0..n).map(|_| self.i8()).collect()
    }
}

/// True S8S8 dot products in i64 — the absolute oracle, independent of tiling.
fn oracle_i64_s8s8(a: &[i8], b: &[i8], m: usize, k: usize, n: usize) -> Vec<i64> {
    let mut out = vec![0i64; m * n];
    for (row, out_row) in out.chunks_mut(n).enumerate() {
        for (col, cell) in out_row.iter_mut().enumerate() {
            let mut acc = 0i64;
            for kk in 0..k {
                acc += i64::from(a[row * k + kk]) * i64::from(b[col * k + kk]);
            }
            *cell = acc;
        }
    }
    out
}

/// Assert every oracle cell fits in i32 and the kernel matches it exactly.
fn assert_fits_and_matches(got: &[i32], oracle: &[i64], label: &str) {
    for (idx, (&g, &o)) in got.iter().zip(oracle.iter()).enumerate() {
        assert!(
            i32::try_from(o).is_ok(),
            "{label}: oracle cell {idx} = {o} overflows i32 (Doctrine-#6 proof obligation violated)"
        );
        assert_eq!(
            i64::from(g),
            o,
            "{label}: kernel cell {idx} = {g} != i64-oracle {o}"
        );
    }
}

/// The decoder's real linear shapes as `(label, n, k)` — `n` output features,
/// `k` contraction. `lm_head` is a representative column slice (M-independence is
/// per-output-column, so a slice proves the same property the full
/// `[129280, 1280]` does — which `batched_lm_head_i8` exercises end to end).
const DECODE_SHAPES: &[(&str, usize, usize)] = &[
    ("qkv_fused", 3840, 1280),
    ("o_proj", 1280, 1280),
    ("expert_gate_up", 896, 1280),
    ("expert_down", 1280, 896),
    ("dense0_down_K6848", 1280, 6848),
    ("lm_head_slice", 512, 1280),
];

/// M=B batched GEMM row `r` is BYTE-identical to the standalone m=1 GEMV for row
/// `r`, at every real decode shape and several batch widths — the lossless-spine
/// invariant at the shapes the decoder actually runs.
#[test]
fn decode_shapes_batched_equals_per_row_gemv() {
    let mut rng = Rng(0xa11c_e0ba_df00_d042);
    for &(label, n, k) in DECODE_SHAPES {
        // K=6848 is the heaviest; keep its widths small (the kernel is fast but
        // the i64 oracle is O(B*K*N)). Smaller-K shapes sweep wider.
        let widths: &[usize] = if k > 4096 {
            &[1, 2, 16]
        } else {
            &[1, 2, 16, 64]
        };
        let b = rng.fill_i8(n * k); // weight panel [n, k], shared across rows.
        for &m in widths {
            let a = rng.fill_i8(m * k); // m stacked activation rows [m, k].
            let mut batched = vec![0i32; m * n];
            igemm_s8s8(&a, &b, m, k, n, &mut batched);

            // Per-row m=1 GEMV must reproduce each batched row byte-for-byte.
            for r in 0..m {
                let row = &a[r * k..(r + 1) * k];
                let mut single = vec![0i32; n];
                igemm_s8s8(row, &b, 1, k, n, &mut single);
                assert_eq!(
                    &batched[r * n..(r + 1) * n],
                    single.as_slice(),
                    "{label} B={m}: batched row {r} != standalone m=1 GEMV (M-independence broken)"
                );
            }

            // Absolute correctness vs the i64 oracle at the smallest non-trivial
            // width (cheap, and the overflow proof for K=6848 lives here too).
            if m == 2 {
                let oracle = oracle_i64_s8s8(&a, &b, m, k, n);
                assert_fits_and_matches(&batched, &oracle, &format!("{label} B=2 oracle"));
            }
        }
    }
}

/// SELF-TEST (sensitivity): a deliberately wrong row-scatter MUST diverge from
/// the per-row oracle. Proves the positive gate above would CATCH a batching bug
/// (transposed/swapped scatter index — the bd-1waa silent-drift failure class),
/// rather than passing vacuously.
#[test]
fn perturbed_row_scatter_is_caught() {
    let mut rng = Rng(0x0ddb_a11c_0ffe_e099);
    let (n, k, m) = (256usize, 1280usize, 4usize);
    let b = rng.fill_i8(n * k);
    let a = rng.fill_i8(m * k);

    let mut correct = vec![0i32; m * n];
    igemm_s8s8(&a, &b, m, k, n, &mut correct);

    // Build the per-row oracle the real gate compares against.
    let mut oracle_rows: Vec<Vec<i32>> = Vec::with_capacity(m);
    for r in 0..m {
        let mut single = vec![0i32; n];
        igemm_s8s8(&a[r * k..(r + 1) * k], &b, 1, k, n, &mut single);
        oracle_rows.push(single);
    }
    // Sanity: the correct batched output matches the oracle row-by-row.
    for (r, orow) in oracle_rows.iter().enumerate() {
        assert_eq!(
            &correct[r * n..(r + 1) * n],
            orow.as_slice(),
            "row {r} oracle mismatch"
        );
    }

    // Perturb: swap output rows 0 and 1 (a classic mis-scatter). The rows differ
    // (distinct random activations), so the perturbed buffer MUST now disagree
    // with the oracle — i.e., the equality gate fails on the bug, as required.
    let mut perturbed = correct.clone();
    for c in 0..n {
        perturbed.swap(c, n + c);
    }
    let row0_ok = perturbed[0..n] == oracle_rows[0][..];
    let row1_ok = perturbed[n..2 * n] == oracle_rows[1][..];
    assert!(
        !(row0_ok && row1_ok),
        "perturbed (row-swapped) output still matched the oracle — the gate is NOT sensitive to a scatter bug"
    );
}

/// SELF-TEST (sensitivity): corrupting a single accumulator cell must be caught
/// by the i64-oracle assertion path (guards against an oracle that is trivially
/// satisfiable).
#[test]
fn perturbed_single_cell_diverges_from_oracle() {
    let mut rng = Rng(0x5eed_1a2b_3c4d_5e6f);
    let (n, k, m) = (128usize, 1280usize, 3usize);
    let b = rng.fill_i8(n * k);
    let a = rng.fill_i8(m * k);
    let mut got = vec![0i32; m * n];
    igemm_s8s8(&a, &b, m, k, n, &mut got);
    let oracle = oracle_i64_s8s8(&a, &b, m, k, n);
    // Correct output matches.
    assert_fits_and_matches(&got, &oracle, "baseline");
    // Corrupt one cell; the comparison must now find a mismatch.
    got[m * n / 2] = got[m * n / 2].wrapping_add(1);
    let any_mismatch = got
        .iter()
        .zip(oracle.iter())
        .any(|(&g, &o)| i64::from(g) != o);
    assert!(
        any_mismatch,
        "a corrupted accumulator cell slipped past the oracle comparison"
    );
}

/// Report the host tier so the gate's coverage is auditable in the test log.
#[test]
fn reports_host_tier() {
    let tier = detected_tier();
    // Always passes; the dispatched tier is recorded for the audit trail and the
    // cross-tier sweep is owned by batched_igemm_parity (FOCR_FORCE_ARCH).
    println!("batched_decode_bit_identity: host tier = {tier:?}");
}
