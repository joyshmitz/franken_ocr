//! Batched R-SWA decode-attention parity gate (bd-1azu.5 — the Phase-6 batched
//! attention stage of the continuous-batch decode spine, bd-1azu).
//!
//! The batched attention stage MUST be LOSSLESS (Doctrine #1): for every
//! in-flight page-stream `s` and layer `L`, the batched call's context
//! `result[s]` is **byte-for-byte** the single-page
//! `decode_attention(cache.layer(s, L), q_s)`. The attention is run PER STREAM
//! over its OWN `reference ++ ring` union (never key-batched across streams), so
//! it is lossless by construction — this file is the executing proof, the parity
//! gate Doctrine #1 demands BEFORE any forward driver batches the attention stage
//! on top of it (the cross-stream query-batching SEAM in `rswa.rs` stays OFF and
//! is what a future bd-1waa-safe optimization must keep bit-exact against this).
//!
//! The oracle is the SAME [`RingCache`] the batched call reads
//! (`cache.layer(s, L)`), so parity is airtight regardless of the per-stream
//! env-gated dispatch (`FOCR_ATTN_GEMM` / `FOCR_INT8_KV`): both sides funnel
//! through the identical `decode_attention` path on the identical cache.
//!
//! Coverage: `B = 8` streams with DISTINCT prefill caps (so each stream's
//! effective key-set length differs inside one batched call) × 2 layers, sampled
//! across the warm-up ring (`ring_len < RING_WINDOW`) and steady-state overwrite
//! (`ring_len == W`) regimes; a `B = 1` edge case; and the two argument-validation
//! errors (bad layer, bad `queries` length).

use franken_ocr::native_engine::rswa::{
    BatchedRingCache, HEAD_DIM, NUM_HEADS, RING_WINDOW, batched_decode_attention, decode_attention,
};

/// Deterministic value in `[-1, 1)` per `(stream, layer, head, row, dim)` —
/// distinct per stream yet reproducible, no dev-dependency (splitmix64-style
/// finalizer over a mixed key).
fn val(stream: usize, layer: usize, h: usize, r: usize, d: usize) -> f32 {
    let mut x = (stream as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (layer as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ (h as u64).wrapping_mul(0x1656_67B1_9E37_79F9)
        ^ (r as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93)
        ^ (d as u64).wrapping_mul(0x27D4_EB2F_1656_67C5);
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

/// Head-major `[NUM_HEADS, seq, HEAD_DIM]` prefill K and V (V offset by one dim so
/// K != V), matching the head-major layout [`BatchedRingCache::record_prefill`]
/// expects.
fn build_prefill(stream: usize, layer: usize, seq: usize) -> (Vec<f32>, Vec<f32>) {
    let mut k = vec![0.0f32; NUM_HEADS * seq * HEAD_DIM];
    let mut v = vec![0.0f32; NUM_HEADS * seq * HEAD_DIM];
    for h in 0..NUM_HEADS {
        for r in 0..seq {
            for d in 0..HEAD_DIM {
                let idx = h * seq * HEAD_DIM + r * HEAD_DIM + d;
                k[idx] = val(stream, layer, h, r, d);
                v[idx] = val(stream, layer, h, r, d + 1);
            }
        }
    }
    (k, v)
}

/// One decode token's `[NUM_HEADS, HEAD_DIM]` K/V at logical step `t` (row space
/// offset so it never collides with prefill rows).
fn build_step(stream: usize, layer: usize, t: usize) -> (Vec<f32>, Vec<f32>) {
    let mut k = vec![0.0f32; NUM_HEADS * HEAD_DIM];
    let mut v = vec![0.0f32; NUM_HEADS * HEAD_DIM];
    for h in 0..NUM_HEADS {
        for d in 0..HEAD_DIM {
            let idx = h * HEAD_DIM + d;
            k[idx] = val(stream, layer, h, 1_000 + t, d);
            v[idx] = val(stream, layer, h, 1_000 + t, d + 1);
        }
    }
    (k, v)
}

/// One decode query `[NUM_HEADS, HEAD_DIM]` (row space offset from K/V and
/// prefill so the QK scores are non-degenerate).
fn build_q(stream: usize, layer: usize, t: usize) -> Vec<f32> {
    let mut q = vec![0.0f32; NUM_HEADS * HEAD_DIM];
    for h in 0..NUM_HEADS {
        for d in 0..HEAD_DIM {
            q[h * HEAD_DIM + d] = val(stream, layer, h, 2_000 + t, d);
        }
    }
    q
}

/// Stack the `B` streams' decode queries for `(layer, t)` into the
/// `[B, NUM_HEADS * HEAD_DIM]` flat layout [`batched_decode_attention`] expects
/// (row `s` is stream `s`'s query).
fn stack_queries(b: usize, layer: usize, t: usize) -> Vec<f32> {
    let per_stream = NUM_HEADS * HEAD_DIM;
    let mut stacked = vec![0.0f32; b * per_stream];
    for s in 0..b {
        let q = build_q(s, layer, t);
        stacked[s * per_stream..(s + 1) * per_stream].copy_from_slice(&q);
    }
    stacked
}

/// THE invariant: each stream's batched context is byte-for-byte the single-page
/// [`decode_attention`] on that stream's own `(layer)` cache, across distinct
/// prefill caps and both ring regimes.
#[test]
fn batched_attention_matches_per_stream_decode_attention() {
    let n_layers = 2usize;
    let caps = [5usize, 9, 16, 4, 7, 20, 3, 11]; // B=8, distinct prefill caps.
    let mut bc = BatchedRingCache::new(&caps, n_layers);
    assert_eq!(bc.num_streams(), caps.len());
    assert_eq!(bc.num_layers(), n_layers);

    for (s, &cap) in caps.iter().enumerate() {
        for l in 0..n_layers {
            let (k, v) = build_prefill(s, l, cap);
            bc.record_prefill(s, l, &k, &v, cap).expect("prefill");
        }
    }

    let steps = 200usize; // crosses RING_WINDOW=128 into steady state.
    let sampled = |t: usize| t == 0 || t == RING_WINDOW - 1 || t == RING_WINDOW || t == steps - 1;

    for t in 0..steps {
        // Advance EVERY stream/layer one decode step (writes into each ring).
        for (s, _) in caps.iter().enumerate() {
            for l in 0..n_layers {
                let (k, v) = build_step(s, l, t);
                bc.write_decode_step(s, l, &k, &v).expect("step");
            }
        }
        if !sampled(t) {
            continue;
        }

        for l in 0..n_layers {
            let stacked = stack_queries(caps.len(), l, t);
            let batched = batched_decode_attention(&bc, l, &stacked).expect("batched attn");
            assert_eq!(
                batched.len(),
                caps.len(),
                "one context per stream l{l} t{t}"
            );

            for (s, _) in caps.iter().enumerate() {
                // Oracle: the SAME cache the batched call reads, single-stream.
                let q = build_q(s, l, t);
                let single = decode_attention(bc.layer(s, l), &q).expect("single attn");
                assert_eq!(batched[s].rows, single.rows, "rows s{s} l{l} t{t}");
                assert_eq!(batched[s].cols, single.cols, "cols s{s} l{l} t{t}");
                assert_eq!(batched[s].rows, 1, "context is [1, N] s{s} l{l} t{t}");
                assert_eq!(
                    batched[s].cols,
                    NUM_HEADS * HEAD_DIM,
                    "ctx cols s{s} l{l} t{t}"
                );
                assert_eq!(
                    batched[s].data, single.data,
                    "stream {s} layer {l} step {t}: batched context != \
                     single-stream decode_attention (LOSSLESS invariant broken)"
                );
            }
        }
    }
}

/// Warm-up-only regime (every step keeps `ring_len < RING_WINDOW`): the batched
/// stage is still byte-for-byte the per-stream path with the ring still growing.
#[test]
fn batched_attention_warmup_regime_bit_exact() {
    let n_layers = 3usize;
    let caps = [2usize, 6, 13]; // B=3.
    let mut bc = BatchedRingCache::new(&caps, n_layers);
    for (s, &cap) in caps.iter().enumerate() {
        for l in 0..n_layers {
            let (k, v) = build_prefill(s, l, cap);
            bc.record_prefill(s, l, &k, &v, cap).expect("prefill");
        }
    }

    let steps = RING_WINDOW / 2; // stays strictly inside warm-up.
    for t in 0..steps {
        for (s, _) in caps.iter().enumerate() {
            for l in 0..n_layers {
                let (k, v) = build_step(s, l, t);
                bc.write_decode_step(s, l, &k, &v).expect("step");
            }
        }
    }
    // Still warming: ring has grown but not saturated.
    for (s, _) in caps.iter().enumerate() {
        assert!(
            !bc.layer(s, 0).is_warm(),
            "stream {s} should still be warming"
        );
        assert_eq!(bc.layer(s, 0).ring_len(), steps, "stream {s} ring_len");
    }

    let t = steps - 1;
    for l in 0..n_layers {
        let stacked = stack_queries(caps.len(), l, t);
        let batched = batched_decode_attention(&bc, l, &stacked).expect("batched attn");
        for (s, _) in caps.iter().enumerate() {
            let q = build_q(s, l, t);
            let single = decode_attention(bc.layer(s, l), &q).expect("single attn");
            assert_eq!(
                batched[s].data, single.data,
                "warm-up: stream {s} layer {l} batched != single"
            );
        }
    }
}

/// `B = 1` edge case: the batched stage degenerates to exactly one
/// [`decode_attention`] call and returns it verbatim.
#[test]
fn batched_attention_single_stream_identity() {
    let mut bc = BatchedRingCache::new(&[7usize], 1);
    let (k, v) = build_prefill(0, 0, 7);
    bc.record_prefill(0, 0, &k, &v, 7).expect("prefill");
    for t in 0..3 {
        let (ks, vs) = build_step(0, 0, t);
        bc.write_decode_step(0, 0, &ks, &vs).expect("step");
    }
    let q = build_q(0, 0, 2);
    let batched = batched_decode_attention(&bc, 0, &q).expect("batched attn");
    assert_eq!(batched.len(), 1);
    let single = decode_attention(bc.layer(0, 0), &q).expect("single attn");
    assert_eq!(batched[0].data, single.data, "B=1 batched != single");
}

/// Argument validation: an out-of-range `layer` is a clear error, not a panic.
#[test]
fn batched_attention_rejects_bad_layer() {
    let bc = BatchedRingCache::new(&[4usize, 4], 2);
    let stacked = vec![0.0f32; 2 * NUM_HEADS * HEAD_DIM];
    let err = batched_decode_attention(&bc, 2, &stacked)
        .expect_err("layer 2 of a 2-layer cache must error");
    assert!(
        err.to_string().contains("layer 2 >= num_layers 2"),
        "unexpected error: {err}"
    );
}

/// Argument validation: a `queries` buffer that is not exactly
/// `B * NUM_HEADS * HEAD_DIM` long is rejected before any per-stream work.
#[test]
fn batched_attention_rejects_bad_queries_len() {
    let bc = BatchedRingCache::new(&[4usize, 4, 4], 1); // B=3.
    let short = vec![0.0f32; 3 * NUM_HEADS * HEAD_DIM - 1];
    let err =
        batched_decode_attention(&bc, 0, &short).expect_err("short queries buffer must error");
    assert!(
        err.to_string().contains("queries len") && err.to_string().contains("B=3"),
        "unexpected error: {err}"
    );
}
