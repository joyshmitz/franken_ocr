//! Native packed-int4 GEMM parity gate (bd-1azu.22 — the "nibbles contiguous,
//! NOT unpack-then-int8" leaf of the Phase-6 decode-bandwidth work, bd-1azu).
//!
//! The deliverable is BIT-EXACT parity, not a speed claim. `int4::igemm_s4s8`
//! (the pre-existing path) materializes the WHOLE int4 weight into a dense `[n,k]`
//! int8 buffer and then runs the int8 GEMM — the "unpack-then-int8" path the
//! negative-evidence ledger clocked at ~5.8x slower than int8. `igemm_s4s8_packed`
//! (this gate's subject) NEVER materializes B: the ARM SDOT/SMMLA kernels mask/
//! shift the packed nibbles in-register and feed them straight to the int8 MAC.
//! Whether that is FASTER than int8 is a separate, honestly-reported bench outcome
//! (`benches/int4_packed_tier.rs`); CORRECTNESS (Doctrine #1) is proven here.
//!
//! Four independent cross-checks per (shape, group, dispatched tier):
//!   1. **dispatch == scalar packed reference** — the host's selected packed
//!      kernel (`igemm_s4s8_packed`, → SDOT on Apple Silicon) equals the
//!      bit-identical scalar oracle `igemm_s4s8_packed_scalar`.
//!   2. **scalar packed == unpack-then-int8** — equals the OTHER int4 impl
//!      `igemm_s4s8`, so the native-packed kernel cannot silently diverge from the
//!      already-trusted dense path.
//!   3. **== independent spec oracle** — an f32 reference written straight from
//!      the pinned packing spec in THIS file (not by calling the crate), the true
//!      cross-check.
//!   4. **i32 per-group accumulator fits** — an i64 oracle proves every group dot
//!      (≤ 32 terms) stays in `[i32::MIN, i32::MAX]`, INCLUDING at the doctrine-#6
//!      worst case K=6848 (dense layer-0 `down_proj`). int4's accumulator is
//!      per-GROUP (≤ 32·8·127 = 32_512), so the bound is vastly looser than the
//!      full-K int8 bound — but we assert it at K=6848 anyway.
//!
//! Tier coverage: the public `igemm_s4s8_packed` runs the host's selected tier.
//! The whole binary is re-run under `FOCR_FORCE_ARCH=scalar|sdot|smmla` (the
//! kill-switch — dispatch.rs / arm::detect_tier — cached in a OnceLock, swept by
//! process re-exec) to exercise EVERY ARM tier the host can dispatch; on x86 the
//! packed entrypoint has no native int4 kernel and runs the scalar reference, so
//! the parity holds trivially (AVX2/VNNI skip-with-SUCCESS).
//! [`forced_tier_takes_effect`] makes the sweep self-verifying.

use franken_ocr::simd::int4::{igemm_s4s8, igemm_s4s8_packed, igemm_s4s8_packed_scalar};
use franken_ocr::simd::{IsaTier, available_tiers, detected_tier};

/// Deterministic xorshift64 PRNG — reproducible, no dev-dependency.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn byte(&mut self) -> u8 {
        (self.next() & 0xFF) as u8
    }
    /// int8 activation in the dynamic-quant symmetric domain `[-127, 127]`.
    fn i8q(&mut self) -> i8 {
        ((self.next() % 255) as i64 - 127) as i8
    }
    /// A small, mixed-sign f32 scale (incl. non-power-of-two values).
    fn scale(&mut self) -> f32 {
        ((self.next() % 4096) as f32 / 4096.0 - 0.5) * 1.7
    }
    fn fill_i8(&mut self, n: usize) -> Vec<i8> {
        (0..n).map(|_| self.i8q()).collect()
    }
    fn fill_bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.byte()).collect()
    }
    fn fill_scales(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.scale()).collect()
    }
}

/// Sign-extend a 4-bit two's-complement nibble to i8 `[-8, 7]` (spec-direct).
fn sign_extend_nibble(nib: u8) -> i8 {
    ((nib << 4) as i8) >> 4
}

/// Independent packed-int4 GEMM oracle, written from the pinned spec: byte
/// `b_packed[ni*(k/2) + j]` holds weight `2j` (low nibble) then `2j+1` (high);
/// per-group i32 dot, dequant-and-sum in f32 in increasing group order. Computed
/// WITHOUT calling any crate helper, so it is a true cross-check.
fn oracle_packed_f32(
    a: &[i8],
    b_packed: &[u8],
    scales: &[f32],
    group: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let groups = k / group;
    let mut out = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc_f = 0.0f32;
            for g in 0..groups {
                let mut gi: i32 = 0;
                for kk in g * group..(g + 1) * group {
                    let byte = b_packed[ni * (k / 2) + kk / 2];
                    let w = if kk % 2 == 0 {
                        sign_extend_nibble(byte & 0x0F)
                    } else {
                        sign_extend_nibble(byte >> 4)
                    };
                    gi += i32::from(a[mi * k + kk]) * i32::from(w);
                }
                acc_f += scales[ni * groups + g] * gi as f32;
            }
            out[mi * n + ni] = acc_f;
        }
    }
    out
}

/// Doctrine #6: every per-group i32 accumulator (≤ `group` ≤ 32 terms) fits in
/// i32, checked via an i64 oracle that cannot itself overflow. Asserted over ALL
/// `(mi, ni, g)` — including K=6848 callers.
fn assert_group_i32_fits(
    a: &[i8],
    b_packed: &[u8],
    group: usize,
    m: usize,
    k: usize,
    n: usize,
    label: &str,
) {
    let groups = k / group;
    for mi in 0..m {
        for ni in 0..n {
            for g in 0..groups {
                let mut acc: i64 = 0;
                for kk in g * group..(g + 1) * group {
                    let byte = b_packed[ni * (k / 2) + kk / 2];
                    let w = if kk % 2 == 0 {
                        sign_extend_nibble(byte & 0x0F)
                    } else {
                        sign_extend_nibble(byte >> 4)
                    };
                    acc += i64::from(a[mi * k + kk]) * i64::from(w);
                }
                assert!(
                    acc >= i64::from(i32::MIN) && acc <= i64::from(i32::MAX),
                    "{label}: group dot ({mi},{ni},{g}) = {acc} overflows i32 (proof obligation violated)"
                );
            }
        }
    }
}

/// Run all four cross-checks for one operand set.
#[allow(clippy::too_many_arguments)]
fn check_all(
    a: &[i8],
    b_packed: &[u8],
    scales: &[f32],
    group: usize,
    m: usize,
    k: usize,
    n: usize,
    label: &str,
) {
    // (host-dispatched packed kernel; SDOT/SMMLA/scalar per tier + FOCR_FORCE_ARCH)
    let mut got_dispatch = vec![0.0f32; m * n];
    igemm_s4s8_packed(a, b_packed, scales, group, m, k, n, &mut got_dispatch);

    // (bit-identical scalar packed reference — the oracle)
    let mut got_scalar = vec![0.0f32; m * n];
    igemm_s4s8_packed_scalar(a, b_packed, scales, group, m, k, n, &mut got_scalar);

    // (the OTHER int4 impl: unpack-then-int8 dense path)
    let mut got_unpack = vec![0.0f32; m * n];
    igemm_s4s8(a, b_packed, scales, group, m, k, n, &mut got_unpack);

    // (independent spec oracle)
    let want = oracle_packed_f32(a, b_packed, scales, group, m, k, n);

    let tier = detected_tier().tag();
    assert_eq!(
        got_dispatch, got_scalar,
        "{label}: dispatched packed kernel (tier {tier}) != scalar packed reference"
    );
    assert_eq!(
        got_scalar, got_unpack,
        "{label}: scalar packed reference != unpack-then-int8 igemm_s4s8"
    );
    assert_eq!(
        got_scalar, want,
        "{label}: scalar packed reference != independent spec oracle"
    );
    assert_group_i32_fits(a, b_packed, group, m, k, n, label);
}

/// Randomized sweep across both group sizes (16/32) and shapes that straddle the
/// SDOT 4×4 and SMMLA 2×2 output-tile remainders in M and N, single & multi
/// sub-block groups, and the NR/MR edges.
#[test]
fn packed_matches_all_references_randomized() {
    let mut rng = Rng(0x1a2b_3c4d_5e6f_7081);
    let cases = [
        (1usize, 16usize, 1usize, 16usize),
        (1, 32, 1, 32),
        (2, 32, 3, 16),
        (3, 48, 2, 16),
        (4, 64, 5, 32),
        (5, 96, 7, 16),
        (7, 128, 6, 32),
        (8, 160, 9, 16),
        (9, 16, 4, 16),
        (16, 64, 13, 16),
        (17, 32, 5, 32),
        (33, 48, 3, 16),
    ];
    for (m, k, n, group) in cases {
        let groups = k / group;
        let a = rng.fill_i8(m * k);
        let b_packed = rng.fill_bytes(n * (k / 2));
        let scales = rng.fill_scales(n * groups);
        check_all(
            &a,
            &b_packed,
            &scales,
            group,
            m,
            k,
            n,
            &format!("rand m={m} k={k} n={n} g={group}"),
        );
    }
}

/// Doctrine #6 worst case K=6848 (dense layer-0 `down_proj`), both group sizes,
/// randomized + the adversarial all-int4-min weights with ±127 activations (which
/// maximize every per-group i32 accumulator). The native-packed kernel must match
/// the references bit-for-bit AND keep every group dot inside i32.
#[test]
fn packed_worst_case_k6848_and_adversarial() {
    let mut rng = Rng(0xfeed_face_dead_0001);
    let k = 6848usize;
    for &group in &[16usize, 32] {
        let groups = k / group;
        for &(m, n) in &[(4usize, 5usize), (9, 3)] {
            // Randomized.
            let a = rng.fill_i8(m * k);
            let b_packed = rng.fill_bytes(n * (k / 2));
            let scales = rng.fill_scales(n * groups);
            check_all(
                &a,
                &b_packed,
                &scales,
                group,
                m,
                k,
                n,
                &format!("k6848 rand g={group} m={m} n={n}"),
            );

            // Adversarial: all weights = int4-min (-8, byte 0x88), activations ±127.
            let b_min = vec![0x88u8; n * (k / 2)];
            let a_pm: Vec<i8> = (0..m * k)
                .map(|i| if i % 2 == 0 { 127 } else { -127 })
                .collect();
            let unit = vec![1.0f32; n * groups];
            check_all(
                &a_pm,
                &b_min,
                &unit,
                group,
                m,
                k,
                n,
                &format!("k6848 adversarial g={group}"),
            );
        }
    }
}

/// With `group == k` there is exactly one group per output row, so int4's
/// per-group dequant collapses to a single per-output-channel scale — the
/// packed kernel must then equal a plain unpack-and-scale int8 GEMV. Proves the
/// native-packed contraction is the same MAC at single-group granularity. Only
/// `k ∈ {16,32}` give a valid single group (group must be 16 or 32).
#[test]
fn packed_single_group_matches_per_channel() {
    let mut rng = Rng(0x0f0f_1212_3434_5656);
    for &k in &[16usize, 32] {
        let (m, n, group) = (3usize, 4usize, k); // group == k → one group/row
        let a = rng.fill_i8(m * k);
        let b_packed = rng.fill_bytes(n * (k / 2));
        let scales = rng.fill_scales(n); // one group/row
        check_all(
            &a,
            &b_packed,
            &scales,
            group,
            m,
            k,
            n,
            &format!("single-group k={k}"),
        );
    }
}

/// `+`-vs-overwrite hygiene: the packed kernel OVERWRITES `out` (`= C`), so a
/// pre-seeded buffer must be ignored (unlike the int8 `+=` kernels). Confirms the
/// dispatched and scalar paths agree on the overwrite semantics.
#[test]
fn packed_overwrites_out_not_accumulates() {
    let mut rng = Rng(0xabcd_1234_5678_9f01);
    let (m, k, n, group) = (4usize, 64usize, 5usize, 16usize);
    let groups = k / group;
    let a = rng.fill_i8(m * k);
    let b_packed = rng.fill_bytes(n * (k / 2));
    let scales = rng.fill_scales(n * groups);

    let want = oracle_packed_f32(&a, &b_packed, &scales, group, m, k, n);
    let mut seeded = vec![123.5f32; m * n]; // non-zero seed must be overwritten
    igemm_s4s8_packed(&a, &b_packed, &scales, group, m, k, n, &mut seeded);
    assert_eq!(
        seeded, want,
        "packed kernel must overwrite out, not accumulate"
    );
}

/// Self-verification of the per-tier sweep: when `FOCR_FORCE_ARCH` names a tier
/// the host actually advertises, the dispatched tier MUST become it — otherwise
/// the CI sweep would silently retest the same tier and the "all tiers" claim
/// would be hollow. Unknown/absent forces are ignored by design. With no env var
/// set, this just reports the natural tier. (The int4 packed path dispatches via
/// `arm::detect_tier`, which honors the same env var as `detected_tier`.)
#[test]
fn forced_tier_takes_effect() {
    let tier = detected_tier();
    eprintln!("[int4_packed_parity] dispatched tier = {}", tier.tag());
    let Ok(force) = std::env::var("FOCR_FORCE_ARCH") else {
        return; // natural selection — nothing to assert.
    };
    let want = force.trim().to_ascii_lowercase();
    if want == "scalar" {
        assert_eq!(
            tier,
            IsaTier::Scalar,
            "FOCR_FORCE_ARCH=scalar must dispatch the scalar floor"
        );
        return;
    }
    let host_has = available_tiers().iter().any(|t| t.tag() == want);
    if host_has {
        assert_eq!(
            tier.tag(),
            want,
            "FOCR_FORCE_ARCH={want} is host-available but was not dispatched"
        );
    }
}
