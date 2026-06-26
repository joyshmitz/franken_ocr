//! `perf_harness` — the pillar-(a) PERFORMANCE GAUNTLET infrastructure.
//!
//! This module is **pure infra**: it has *zero* dependency on the `franken_ocr`
//! crate, on `ft-kernel-cpu`, or on any serde — it compiles standalone inside
//! the `benches/gauntlet_harness.rs` bench target (declared via
//! `#[path = "support/perf_harness.rs"] mod perf_harness;`). Everything that
//! *measures a kernel* lives in the bench runner; everything here only
//! **records, serializes, and ratchets** a measurement. Keeping the two apart is
//! deliberate: the math below (the ratchet gates, the percentile/CoV stats, the
//! JSON shape) is the part we unit-test and freeze, and it must be testable
//! without a 6.67 GB model on disk.
//!
//! It provides, exactly as `docs/gauntlet/METHODOLOGY.md` §5 and
//! `COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md` §9.3 specify:
//!
//! * [`BenchRecord`] — a comprehensive-bench-style result row: bench name,
//!   shape, p50 / p90, throughput, thread count, allocator, precision, `cv_pct`,
//!   plus the head-to-head `ref_ms` / `ratio` when a baseline ran.
//! * Hand-rolled JSON serialization ([`BenchRecord::to_json`], [`History`] →
//!   `latest.json`) — **no serde dependency added** (the bench target stays
//!   dep-free so it can never drift the crate's dependency graph; AGENTS.md
//!   "do NOT edit Cargo.toml"). A minimal embedded JSON value parser reads a
//!   prior `latest.json` back.
//! * [`Ratchet`] — the `.bench-history` pass-over-pass comparator implementing
//!   the **five** gates from METHODOLOGY §5 / plan §9.2:
//!   primary regression ≥ −3 %, geomean ≥ −5 %, per-category geomean ≥ −10 %,
//!   p90 ≥ −15 %, throughput ≥ −5 %. It reads/writes a `latest.json` high-water
//!   mark and returns an `Allow | Block` verdict mirroring `gauntlet_cert.py`.
//! * The **fairness-control knobs** ([`Fairness`]) — pin the thread budget,
//!   record the allocator posture and the precision tag — so every ratio is
//!   apples-to-apples (plan §9.3; **never bench torch @64**).
//!
//! Nothing here is ever linked into the shipping `focr` binary — like the PyO3
//! oracle bridge this is a verification-only artifact (G3's no-FFI runtime claim
//! is preserved: it lives under `benches/`).
//!
//! ## Why hand-rolled stats and not criterion?
//! The bench target is auto-discovered on nightly via `#![feature(test)]` +
//! `#[bench]`; there is no `[[bench]]` manifest entry and no criterion (adding
//! either would edit `Cargo.toml`, which this agent must not do). So we own the
//! percentile / CoV math here and unit-test it directly. If a `[[bench]]` slot
//! or criterion is later wanted, that is a `deps_wanted` note, not a silent edit.

#![allow(dead_code)] // infra surface; not every field is read by every bench yet.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;
use std::time::Duration;

// ─────────────────────────────────────────────────────────────────────────────
// Fairness controls (plan §9.3 — ALL mandatory for a meaningful ratio).
// ─────────────────────────────────────────────────────────────────────────────

/// The numeric precision a row was measured at. A raw `reference/focr` ratio is
/// meaningless without this — `focr-int8` vs `torch-bf16` is a different claim
/// than `focr-int8` vs `torch-int8` (plan §9.3, METHODOLOGY §5.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// f32 parity spine (the vision tower / projector default rail).
    F32,
    /// bf16 — the reference model's native weight precision.
    Bf16,
    /// int8 dynamic-quantized decoder FFN/expert GEMMs (the validated quant set).
    Int8,
    /// int4-group decode-bandwidth wedge (Phase 4; `g32`/`g16` recorded in the tag).
    Int4,
}

impl Precision {
    /// The stable string used in JSON / the `precision` ledger column.
    #[must_use]
    pub fn tag(self) -> &'static str {
        match self {
            Precision::F32 => "f32",
            Precision::Bf16 => "bf16",
            Precision::Int8 => "int8",
            Precision::Int4 => "int4",
        }
    }
}

/// Allocator posture of the *measured binary* (plan §9.3 — must be wired in, not
/// merely mentioned; mimalloc lives behind a cargo feature, §9.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Allocator {
    /// The platform system allocator (the default measured posture).
    System,
    /// mimalloc, enabled via the crate's allocator feature for the claim.
    Mimalloc,
}

impl Allocator {
    /// The stable string used in JSON / the `allocator` ledger column.
    #[must_use]
    pub fn tag(self) -> &'static str {
        match self {
            Allocator::System => "system",
            Allocator::Mimalloc => "mimalloc",
        }
    }

    /// Resolve the posture from how the bench binary was actually built.
    ///
    /// This reads the *compiled* configuration, not an env var — so the tag can
    /// never claim mimalloc on a binary that did not link it (the §9.3 trap:
    /// "wired into the measured binary, not merely mentioned"). The bench crate
    /// has no allocator feature of its own; if/when one is added the `cfg` here
    /// flips. Until then a measured binary is honestly `system`.
    #[must_use]
    pub fn from_build() -> Self {
        // No `mimalloc` feature is wired into this dep-free bench target yet, so
        // the only honest answer the *binary* can give is `system`. A future
        // `--features mimalloc` claim flips this via a real `cfg!`, never a knob.
        #[cfg(feature = "mimalloc")]
        {
            Allocator::Mimalloc
        }
        #[cfg(not(feature = "mimalloc"))]
        {
            Allocator::System
        }
    }
}

/// The fairness knobs that make a head-to-head ratio honest (plan §9.3).
///
/// All three are *recorded on every row*; the thread budget is additionally the
/// number the reference baseline MUST be pinned to (`OMP_NUM_THREADS` /
/// `torch.set_num_threads(N)`). The cardinal sin this guards is benching torch
/// at @64 while focr runs at @8 — oversubscription inflates a fake "win".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fairness {
    /// focr's thread budget for the row; the reference is pinned to the SAME N.
    pub threads: usize,
    /// Allocator posture of the measured binary.
    pub allocator: Allocator,
    /// Numeric precision the focr side ran at.
    pub precision: Precision,
}

/// The hard ceiling above which a torch/reference baseline must NEVER be pinned
/// in this harness (plan §9.3 / METHODOLOGY §5.1 — the @64 oversubscription
/// trap). `assert_thread_parity` enforces equality at the focr budget; this
/// constant documents *why* the budgets are pinned at @8/@32, not `num_cpus`.
pub const REFERENCE_THREAD_HARD_CAP_NOTE: &str =
    "pin the reference to focr's thread budget (measure @8/@32); NEVER @64.";

impl Fairness {
    /// Construct with explicit knobs, reading the allocator posture from the
    /// build so it can never be over-claimed.
    #[must_use]
    pub fn new(threads: usize, precision: Precision) -> Self {
        Self {
            threads,
            allocator: Allocator::from_build(),
            precision,
        }
    }

    /// Assert the reference baseline was pinned to **exactly** focr's thread
    /// budget. Returns `Err` (rather than panicking) so the bench runner can log
    /// a fairness violation and downgrade the row to "warn" instead of dying.
    ///
    /// # Errors
    /// Returns the mismatch message if `reference_threads != self.threads`.
    pub fn assert_thread_parity(&self, reference_threads: usize) -> Result<(), String> {
        if reference_threads != self.threads {
            return Err(format!(
                "FAIRNESS VIOLATION: focr@{} vs reference@{} — thread parity broken \
                 (plan §9.3: {REFERENCE_THREAD_HARD_CAP_NOTE})",
                self.threads, reference_threads,
            ));
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sample statistics (p50/p90/CoV) — owned here so the bench needs no criterion.
// ─────────────────────────────────────────────────────────────────────────────

/// Reduced statistics over a set of per-iteration wall-clock samples.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleStats {
    /// Number of *kept* samples (after warmup discard).
    pub n: usize,
    /// 50th-percentile (median) seconds.
    pub p50_s: f64,
    /// 90th-percentile seconds.
    pub p90_s: f64,
    /// Minimum seconds (best-of-N; plan §9.3 "report the min").
    pub min_s: f64,
    /// Arithmetic mean seconds.
    pub mean_s: f64,
    /// Coefficient of variation, percent (`100 * stddev/mean`). `> 5` is noise
    /// and the row is ineligible for the keep-gate (METHODOLOGY §5).
    pub cv_pct: f64,
}

impl SampleStats {
    /// Reduce a slice of per-iteration [`Duration`]s after discarding the first
    /// `warmup` samples (plan §9.3 "best-of-N with warmup discard").
    ///
    /// Percentiles use the nearest-rank method on the sorted kept samples, which
    /// is deterministic and dependency-free (no interpolation ambiguity across
    /// platforms — matching the `truncate_score` cross-arch-determinism spirit
    /// of METHODOLOGY §3.4).
    ///
    /// # Panics
    /// Panics if every sample is discarded by `warmup` (a misuse — there is
    /// nothing to summarize).
    #[must_use]
    pub fn from_durations(samples: &[Duration], warmup: usize) -> Self {
        let kept: Vec<f64> = samples
            .iter()
            .skip(warmup)
            .map(Duration::as_secs_f64)
            .collect();
        assert!(
            !kept.is_empty(),
            "SampleStats: all {} samples discarded by warmup={}",
            samples.len(),
            warmup
        );
        Self::from_secs(&kept)
    }

    /// Reduce already-extracted second-valued samples (the testable core).
    #[must_use]
    pub fn from_secs(secs: &[f64]) -> Self {
        assert!(!secs.is_empty(), "SampleStats::from_secs: empty");
        let n = secs.len();
        let mut sorted = secs.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let p50_s = percentile_nearest_rank(&sorted, 0.50);
        let p90_s = percentile_nearest_rank(&sorted, 0.90);
        let min_s = sorted[0];
        let mean_s = secs.iter().sum::<f64>() / n as f64;
        let var = if n > 1 {
            secs.iter().map(|x| (x - mean_s).powi(2)).sum::<f64>() / (n as f64 - 1.0)
        } else {
            0.0
        };
        let std = var.sqrt();
        let cv_pct = if mean_s > 0.0 {
            100.0 * std / mean_s
        } else {
            0.0
        };

        Self {
            n,
            p50_s,
            p90_s,
            min_s,
            mean_s,
            cv_pct,
        }
    }

    /// Whether this row is keep-gate-eligible: `cv_pct ≤ 5` (METHODOLOGY §5 —
    /// "`cv_pct > 5` is noise and ineligible for keep").
    #[must_use]
    pub fn is_keep_eligible(&self) -> bool {
        self.cv_pct <= CV_NOISE_PCT
    }
}

/// `cv_pct` above this is noise; the row does not enter the ratchet (METHODOLOGY §5).
pub const CV_NOISE_PCT: f64 = 5.0;

/// Nearest-rank percentile on an ascending-sorted slice. `q ∈ [0,1]`.
#[must_use]
pub fn percentile_nearest_rank(sorted_asc: &[f64], q: f64) -> f64 {
    assert!(!sorted_asc.is_empty());
    let n = sorted_asc.len();
    // Nearest-rank: rank = ceil(q * n), clamped to [1, n]; index = rank-1.
    let rank = (q * n as f64).ceil().max(1.0) as usize;
    sorted_asc[rank.min(n) - 1]
}

// ─────────────────────────────────────────────────────────────────────────────
// BenchRecord — the comprehensive-bench-style result row.
// ─────────────────────────────────────────────────────────────────────────────

/// One measured row in the gauntlet. The field set mirrors the PERF_LEDGER /
/// guardrail artifact shape (LOGGING_AND_E2E §6) so a row drops straight into
/// the evidence graph.
#[derive(Debug, Clone, PartialEq)]
pub struct BenchRecord {
    /// Stable bench id, e.g. `"decode_token"` or `"linear_int8_dynamic"`.
    pub name: String,
    /// Coarse bucket for the per-category geomean gate, e.g. `"decode"`,
    /// `"prefill"`, `"vision"`, `"kernel"`.
    pub category: String,
    /// Human/agent-readable problem shape, e.g. `"m=1,k=6848,n=1280"`.
    pub shape: String,
    /// Reduced sample statistics for the focr side.
    pub stats: SampleStats,
    /// Throughput in the row's natural unit (e.g. tokens/s, GFLOP/s). `None` when
    /// a throughput is not meaningful for the row.
    pub throughput: Option<f64>,
    /// Unit string for `throughput` (e.g. `"tok/s"`, `"GFLOP/s"`).
    pub throughput_unit: Option<String>,
    /// The fairness controls under which this row was measured.
    pub fairness: Fairness,
    /// Head-to-head: the reference baseline's p50 seconds, when a baseline ran.
    /// `None` ⇒ self-relative microbench (no `reference/focr` ratio claimed).
    pub reference_p50_s: Option<f64>,
    /// Which CPU reference backend ran (e.g. `onnx`, `hf`, `gguf`, `mlas`).
    /// `None` ⇒ self-relative microbench or reference-only probe without a ratio.
    pub reference_backend: Option<String>,
    /// Precision tag the *reference* ran at (e.g. `"bf16"`); pairs with `ratio`.
    pub reference_precision: Option<String>,
    /// Free-form note (e.g. a skip reason, an isomorphism/golden citation).
    pub note: Option<String>,
}

impl BenchRecord {
    /// The honest `reference/focr` ratio for this stage, or `None` when no
    /// baseline ran. `> 1.0` ⇒ focr is faster; `< 1.0` ⇒ slower (PERF_LEDGER).
    #[must_use]
    pub fn ratio(&self) -> Option<f64> {
        self.reference_p50_s.map(|r| r / self.stats.p50_s)
    }

    /// Tag the ratio OK / warn / slower / "focr faster" (plan §9.3 row tags).
    /// `None` when the row is self-relative (no reference).
    #[must_use]
    pub fn ratio_tag(&self) -> Option<&'static str> {
        self.ratio().map(|r| {
            if r > 1.03 {
                "focr_faster"
            } else if r >= 0.95 {
                "ok"
            } else if r >= 0.80 {
                "warn"
            } else {
                "slower"
            }
        })
    }

    /// Hand-rolled JSON object (one line; data-only — the robot/NDJSON style).
    ///
    /// No serde: this is a closed, known shape, so a tiny manual serializer is
    /// simpler than wiring a dep into a dep-free bench target. Keys are emitted
    /// in a fixed order for byte-stable, diffable output (the `.bench-history`
    /// discipline relies on stable bytes).
    #[must_use]
    pub fn to_json(&self) -> String {
        let mut s = String::with_capacity(512);
        s.push('{');
        json_kv_str(&mut s, "name", &self.name, true);
        json_kv_str(&mut s, "category", &self.category, false);
        json_kv_str(&mut s, "shape", &self.shape, false);
        json_kv_num(&mut s, "p50_s", self.stats.p50_s, false);
        json_kv_num(&mut s, "p90_s", self.stats.p90_s, false);
        json_kv_num(&mut s, "min_s", self.stats.min_s, false);
        json_kv_num(&mut s, "cv_pct", self.stats.cv_pct, false);
        json_kv_int(&mut s, "n_samples", self.stats.n as i64, false);
        json_kv_bool(
            &mut s,
            "keep_eligible",
            self.stats.is_keep_eligible(),
            false,
        );
        match self.throughput {
            Some(t) => json_kv_num(&mut s, "throughput", t, false),
            None => json_kv_null(&mut s, "throughput", false),
        }
        match &self.throughput_unit {
            Some(u) => json_kv_str(&mut s, "throughput_unit", u, false),
            None => json_kv_null(&mut s, "throughput_unit", false),
        }
        json_kv_int(&mut s, "threads", self.fairness.threads as i64, false);
        json_kv_str(&mut s, "allocator", self.fairness.allocator.tag(), false);
        json_kv_str(&mut s, "precision", self.fairness.precision.tag(), false);
        match self.reference_p50_s {
            Some(r) => json_kv_num(&mut s, "reference_p50_s", r, false),
            None => json_kv_null(&mut s, "reference_p50_s", false),
        }
        match &self.reference_backend {
            Some(b) => json_kv_str(&mut s, "reference_backend", b, false),
            None => json_kv_null(&mut s, "reference_backend", false),
        }
        match &self.reference_precision {
            Some(p) => json_kv_str(&mut s, "reference_precision", p, false),
            None => json_kv_null(&mut s, "reference_precision", false),
        }
        match self.ratio() {
            Some(r) => json_kv_num(&mut s, "ratio", r, false),
            None => json_kv_null(&mut s, "ratio", false),
        }
        match self.ratio_tag() {
            Some(t) => json_kv_str(&mut s, "ratio_tag", t, false),
            None => json_kv_null(&mut s, "ratio_tag", false),
        }
        match &self.note {
            Some(note) => json_kv_str(&mut s, "note", note, false),
            None => json_kv_null(&mut s, "note", false),
        }
        s.push('}');
        s
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// History — the persisted `.bench-history` / latest.json high-water mark.
// ─────────────────────────────────────────────────────────────────────────────

/// A serializable snapshot of one gauntlet round: per-bench p50/p90/throughput,
/// keyed by bench name. This is what `latest.json` stores and what the ratchet
/// compares the next round against.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct History {
    /// `bench name -> snapshot` for every keep-eligible row.
    pub rows: BTreeMap<String, HistoryRow>,
}

/// The minimal per-bench snapshot the ratchet needs (a subset of [`BenchRecord`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HistoryRow {
    /// p50 seconds (lower is better).
    pub p50_s: f64,
    /// p90 seconds (lower is better).
    pub p90_s: f64,
    /// throughput in the row's unit (higher is better); `0.0` ⇒ not tracked.
    pub throughput: f64,
}

impl History {
    /// Build a history snapshot from the current round's keep-eligible records.
    /// Non-eligible (`cv_pct > 5`) rows are excluded — they never enter the
    /// ratchet (METHODOLOGY §5).
    #[must_use]
    pub fn from_records(records: &[BenchRecord]) -> Self {
        let mut rows = BTreeMap::new();
        for r in records {
            if !r.stats.is_keep_eligible() {
                continue;
            }
            rows.insert(
                r.name.clone(),
                HistoryRow {
                    p50_s: r.stats.p50_s,
                    p90_s: r.stats.p90_s,
                    throughput: r.throughput.unwrap_or(0.0),
                },
            );
        }
        Self { rows }
    }

    /// Map of `bench name -> category` for the per-category geomean gate. Built
    /// from the records (history rows alone do not carry the category).
    #[must_use]
    pub fn categories(records: &[BenchRecord]) -> BTreeMap<String, String> {
        records
            .iter()
            .filter(|r| r.stats.is_keep_eligible())
            .map(|r| (r.name.clone(), r.category.clone()))
            .collect()
    }

    /// Serialize to a stable, diffable `latest.json` (sorted keys via `BTreeMap`).
    #[must_use]
    pub fn to_json(&self) -> String {
        let mut s = String::with_capacity(256);
        s.push_str("{\n  \"artifact\": \"franken_ocr.bench-history.v1\",\n  \"rows\": {");
        let mut first = true;
        for (name, row) in &self.rows {
            if !first {
                s.push(',');
            }
            first = false;
            let _ = write!(
                s,
                "\n    {}: {{\"p50_s\": {}, \"p90_s\": {}, \"throughput\": {}}}",
                json_str(name),
                fmt_num(row.p50_s),
                fmt_num(row.p90_s),
                fmt_num(row.throughput),
            );
        }
        s.push_str("\n  }\n}\n");
        s
    }

    /// Parse a prior `latest.json` produced by [`History::to_json`]. Uses the
    /// embedded minimal JSON value parser ([`json::parse`]) — no serde.
    ///
    /// # Errors
    /// Returns a message on malformed JSON or a missing/ill-typed `rows` object.
    pub fn from_json(text: &str) -> Result<Self, String> {
        let v = json::parse(text)?;
        let obj = v.as_object().ok_or("top-level value is not an object")?;
        let rows_v = obj.get("rows").ok_or("missing `rows`")?;
        let rows_obj = rows_v.as_object().ok_or("`rows` is not an object")?;
        let mut rows = BTreeMap::new();
        for (name, rv) in rows_obj {
            let ro = rv
                .as_object()
                .ok_or_else(|| format!("row {name:?} is not an object"))?;
            let get = |k: &str| -> Result<f64, String> {
                ro.get(k)
                    .and_then(json::Value::as_f64)
                    .ok_or_else(|| format!("row {name:?} missing numeric {k:?}"))
            };
            rows.insert(
                name.clone(),
                HistoryRow {
                    p50_s: get("p50_s")?,
                    p90_s: get("p90_s")?,
                    throughput: get("throughput")?,
                },
            );
        }
        Ok(Self { rows })
    }

    /// Read a prior history from `path`. A missing file is **not** an error — it
    /// is the first-ever round; returns an empty history (every gate vacuously
    /// passes and the round seeds the baseline).
    ///
    /// # Errors
    /// Returns a message on an unreadable (present-but-broken) file or malformed
    /// JSON.
    pub fn read_or_empty(path: &Path) -> Result<Self, String> {
        match std::fs::read_to_string(path) {
            Ok(text) => Self::from_json(&text),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(format!("reading {}: {e}", path.display())),
        }
    }

    /// Write this history to `path` as `latest.json` (the new high-water mark).
    ///
    /// # Errors
    /// Returns a message on an IO failure.
    pub fn write(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("creating {}: {e}", parent.display()))?;
        }
        std::fs::write(path, self.to_json()).map_err(|e| format!("writing {}: {e}", path.display()))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Ratchet — the five pass-over-pass gates (METHODOLOGY §5 / plan §9.2).
// ─────────────────────────────────────────────────────────────────────────────

/// The five `.bench-history` regression thresholds. A change past any of these
/// **blocks** the bench gate (a faster path that drifts is reverted; a regression
/// past the floor cannot land). Signs: a regression is a *worsening* — for
/// latency that is an increase, for throughput a decrease — so the thresholds
/// are the worst tolerated *relative* move.
///
/// From METHODOLOGY §5 / plan §9.2 verbatim:
/// * `primary` p50 regression ≥ −3 % blocks (the named primary bench, decode-per-token).
/// * `geomean` of all p50 ≥ −5 % blocks.
/// * `per_category` geomean of p50 ≥ −10 % blocks (per coarse bucket).
/// * `p90` regression ≥ −15 % blocks (tail).
/// * `throughput` regression ≥ −5 % blocks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RatchetGates {
    /// Max tolerated p50 regression on the primary bench (e.g. `0.03` = 3 %).
    pub primary_pct: f64,
    /// Max tolerated geomean p50 regression across all benches.
    pub geomean_pct: f64,
    /// Max tolerated per-category geomean p50 regression.
    pub per_category_pct: f64,
    /// Max tolerated p90 regression.
    pub p90_pct: f64,
    /// Max tolerated throughput regression.
    pub throughput_pct: f64,
}

impl Default for RatchetGates {
    /// The frozen gate values from METHODOLOGY §5 / plan §9.2.
    fn default() -> Self {
        Self {
            primary_pct: 0.03,
            geomean_pct: 0.05,
            per_category_pct: 0.10,
            p90_pct: 0.15,
            throughput_pct: 0.05,
        }
    }
}

/// One gate's outcome line (for the report and the unit tests).
#[derive(Debug, Clone, PartialEq)]
pub struct GateOutcome {
    /// Which gate (e.g. `"primary"`, `"geomean"`, `"per_category:decode"`).
    pub gate: String,
    /// The observed relative regression (positive ⇒ a worsening move).
    pub observed_regression: f64,
    /// The gate's allowed maximum regression.
    pub allowed: f64,
    /// `true` when the gate passed (`observed ≤ allowed`).
    pub pass: bool,
}

/// The verdict of one ratchet comparison.
#[derive(Debug, Clone, PartialEq)]
pub struct RatchetVerdict {
    /// `"Allow"` ⇒ the round may land and advance the high-water mark;
    /// `"Block"` ⇒ a gate regressed past its floor.
    pub decision: String,
    /// Every gate's outcome (passing and failing), for the report.
    pub gates: Vec<GateOutcome>,
    /// Human/agent-readable summary of why.
    pub reason: String,
}

impl RatchetVerdict {
    /// Whether the round is allowed to land.
    #[must_use]
    pub fn allowed(&self) -> bool {
        self.decision == "Allow"
    }
}

/// The pass-over-pass comparator. Holds the gate thresholds and the name of the
/// primary bench whose p50 carries the strictest (−3 %) gate.
#[derive(Debug, Clone)]
pub struct Ratchet {
    /// The frozen gate thresholds.
    pub gates: RatchetGates,
    /// The primary bench name (decode-per-token on the primary arch). When the
    /// current round lacks it, the primary gate is reported `n/a` (not a block).
    pub primary_bench: String,
}

impl Ratchet {
    /// Construct with the default frozen gates and a named primary bench.
    #[must_use]
    pub fn new(primary_bench: impl Into<String>) -> Self {
        Self {
            gates: RatchetGates::default(),
            primary_bench: primary_bench.into(),
        }
    }

    /// Compare the current round (`records`) against the persisted `baseline`
    /// history, returning the five-gate verdict.
    ///
    /// Only keep-eligible rows (`cv_pct ≤ 5`) participate ([`History::from_records`]
    /// already filters). A bench present now but absent from the baseline is a
    /// *new* bench: it cannot regress, so it never blocks (it seeds its own
    /// floor on the next write).
    #[must_use]
    pub fn compare(&self, records: &[BenchRecord], baseline: &History) -> RatchetVerdict {
        let current = History::from_records(records);
        let categories = History::categories(records);
        let mut gates: Vec<GateOutcome> = Vec::new();

        // ── Gate 1: primary p50 (−3 %). ─────────────────────────────────────
        if let (Some(cur), Some(base)) = (
            current.rows.get(&self.primary_bench),
            baseline.rows.get(&self.primary_bench),
        ) {
            let reg = rel_regression_latency(base.p50_s, cur.p50_s);
            gates.push(GateOutcome {
                gate: "primary".into(),
                observed_regression: reg,
                allowed: self.gates.primary_pct,
                pass: reg <= self.gates.primary_pct + EPS,
            });
        }

        // ── Gate 2: geomean of all paired p50 (−5 %). ───────────────────────
        if let Some(reg) = geomean_latency_regression(&current, baseline, |_| true) {
            gates.push(GateOutcome {
                gate: "geomean".into(),
                observed_regression: reg,
                allowed: self.gates.geomean_pct,
                pass: reg <= self.gates.geomean_pct + EPS,
            });
        }

        // ── Gate 3: per-category geomean of p50 (−10 %). ────────────────────
        let mut cats: Vec<&String> = categories.values().collect();
        cats.sort();
        cats.dedup();
        for cat in cats {
            if let Some(reg) = geomean_latency_regression(&current, baseline, |name| {
                categories.get(name).map(String::as_str) == Some(cat.as_str())
            }) {
                gates.push(GateOutcome {
                    gate: format!("per_category:{cat}"),
                    observed_regression: reg,
                    allowed: self.gates.per_category_pct,
                    pass: reg <= self.gates.per_category_pct + EPS,
                });
            }
        }

        // ── Gate 4: p90 — worst single-bench tail regression (−15 %). ───────
        if let Some((worst_bench, reg)) = worst_p90_regression(&current, baseline) {
            gates.push(GateOutcome {
                gate: format!("p90:{worst_bench}"),
                observed_regression: reg,
                allowed: self.gates.p90_pct,
                pass: reg <= self.gates.p90_pct + EPS,
            });
        }

        // ── Gate 5: throughput — worst single-bench drop (−5 %). ────────────
        if let Some((worst_bench, reg)) = worst_throughput_regression(&current, baseline) {
            gates.push(GateOutcome {
                gate: format!("throughput:{worst_bench}"),
                observed_regression: reg,
                allowed: self.gates.throughput_pct,
                pass: reg <= self.gates.throughput_pct + EPS,
            });
        }

        let failed: Vec<&GateOutcome> = gates.iter().filter(|g| !g.pass).collect();
        if failed.is_empty() {
            RatchetVerdict {
                decision: "Allow".into(),
                reason: format!("all {} gate(s) within thresholds", gates.len()),
                gates,
            }
        } else {
            let names: Vec<String> = failed
                .iter()
                .map(|g| {
                    format!(
                        "{} (regressed {:.2}% > allowed {:.2}%)",
                        g.gate,
                        g.observed_regression * 100.0,
                        g.allowed * 100.0
                    )
                })
                .collect();
            RatchetVerdict {
                decision: "Block".into(),
                reason: format!("gate(s) regressed past floor: {}", names.join("; ")),
                gates,
            }
        }
    }

    /// The monotone advance: merge the current round into the baseline, keeping
    /// the *best* seen value per bench (min latency, max throughput) so the
    /// high-water mark never silently slides backward. New benches are added.
    ///
    /// This mirrors the `gauntlet_cert.py` ratchet's `max(current, floor)` move
    /// (here `min` for latency, since lower is better) — the floor only ever
    /// tightens.
    #[must_use]
    pub fn advance(&self, records: &[BenchRecord], baseline: &History) -> History {
        let current = History::from_records(records);
        let mut merged = baseline.clone();
        for (name, cur) in &current.rows {
            merged
                .rows
                .entry(name.clone())
                .and_modify(|b| {
                    b.p50_s = b.p50_s.min(cur.p50_s);
                    b.p90_s = b.p90_s.min(cur.p90_s);
                    b.throughput = b.throughput.max(cur.throughput);
                })
                .or_insert(*cur);
        }
        merged
    }
}

/// Floating-point slack so an exactly-at-threshold move is allowed, not blocked.
const EPS: f64 = 1e-12;

/// Relative *regression* for a lower-is-better metric (latency): positive when
/// `current > base`. `(current - base) / base`.
#[must_use]
pub fn rel_regression_latency(base: f64, current: f64) -> f64 {
    if base <= 0.0 {
        return 0.0;
    }
    (current - base) / base
}

/// Relative *regression* for a higher-is-better metric (throughput): positive
/// when `current < base`. `(base - current) / base`.
#[must_use]
pub fn rel_regression_throughput(base: f64, current: f64) -> f64 {
    if base <= 0.0 {
        return 0.0;
    }
    (base - current) / base
}

/// Geomean of the per-bench latency *ratios* `current/base` over the paired
/// benches matching `filter`, returned as a regression (`geomean_ratio - 1`).
/// `None` when no bench pairs match (the gate is then `n/a`, never a block).
fn geomean_latency_regression(
    current: &History,
    baseline: &History,
    filter: impl Fn(&str) -> bool,
) -> Option<f64> {
    let mut log_sum = 0.0;
    let mut count = 0usize;
    for (name, cur) in &current.rows {
        if !filter(name) {
            continue;
        }
        if let Some(base) = baseline.rows.get(name)
            && base.p50_s > 0.0
            && cur.p50_s > 0.0
        {
            log_sum += (cur.p50_s / base.p50_s).ln();
            count += 1;
        }
    }
    if count == 0 {
        return None;
    }
    let geomean_ratio = (log_sum / count as f64).exp();
    Some(geomean_ratio - 1.0)
}

/// The worst single-bench p90 regression across paired benches (the tail gate is
/// per-bench, not a geomean — one bad tail blocks). `None` when no pairs.
fn worst_p90_regression(current: &History, baseline: &History) -> Option<(String, f64)> {
    let mut worst: Option<(String, f64)> = None;
    for (name, cur) in &current.rows {
        if let Some(base) = baseline.rows.get(name) {
            let reg = rel_regression_latency(base.p90_s, cur.p90_s);
            if worst.as_ref().is_none_or(|(_, w)| reg > *w) {
                worst = Some((name.clone(), reg));
            }
        }
    }
    worst
}

/// The worst single-bench throughput drop across paired benches with a tracked
/// throughput (`> 0`). `None` when no pairs track throughput.
fn worst_throughput_regression(current: &History, baseline: &History) -> Option<(String, f64)> {
    let mut worst: Option<(String, f64)> = None;
    for (name, cur) in &current.rows {
        if cur.throughput <= 0.0 {
            continue;
        }
        if let Some(base) = baseline.rows.get(name) {
            if base.throughput <= 0.0 {
                continue;
            }
            let reg = rel_regression_throughput(base.throughput, cur.throughput);
            if worst.as_ref().is_none_or(|(_, w)| reg > *w) {
                worst = Some((name.clone(), reg));
            }
        }
    }
    worst
}

// ─────────────────────────────────────────────────────────────────────────────
// Tiny hand-rolled JSON — serializer helpers + a minimal value parser.
// ─────────────────────────────────────────────────────────────────────────────

/// Format an f64 for JSON: finite ⇒ shortest round-trippable; non-finite ⇒
/// `null` (JSON has no NaN/Inf, and a non-finite timing is a bug we surface).
#[must_use]
fn fmt_num(x: f64) -> String {
    if x.is_finite() {
        // `{}` on f64 is shortest round-trippable in Rust; good enough and stable.
        format!("{x}")
    } else {
        "null".into()
    }
}

/// JSON-escape a string and wrap in quotes.
#[must_use]
fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn json_kv_str(s: &mut String, key: &str, val: &str, first: bool) {
    if !first {
        s.push(',');
    }
    let _ = write!(s, "{}:{}", json_str(key), json_str(val));
}
fn json_kv_num(s: &mut String, key: &str, val: f64, first: bool) {
    if !first {
        s.push(',');
    }
    let _ = write!(s, "{}:{}", json_str(key), fmt_num(val));
}
fn json_kv_int(s: &mut String, key: &str, val: i64, first: bool) {
    if !first {
        s.push(',');
    }
    let _ = write!(s, "{}:{}", json_str(key), val);
}
fn json_kv_bool(s: &mut String, key: &str, val: bool, first: bool) {
    if !first {
        s.push(',');
    }
    let _ = write!(s, "{}:{}", json_str(key), val);
}
fn json_kv_null(s: &mut String, key: &str, first: bool) {
    if !first {
        s.push(',');
    }
    let _ = write!(s, "{}:null", json_str(key));
}

/// A minimal, allocation-light JSON *value* parser — just enough to read back a
/// `latest.json` this module wrote (objects, strings, numbers, bool, null). It
/// is intentionally tiny (no arrays needed for the history shape) and rejects
/// anything it does not understand, so a malformed history is a loud error, not
/// a silent empty baseline.
pub mod json {
    use std::collections::BTreeMap;

    /// A parsed JSON value (the subset the history needs).
    #[derive(Debug, Clone, PartialEq)]
    pub enum Value {
        /// A JSON null.
        Null,
        /// A JSON boolean.
        Bool(bool),
        /// A JSON number (always stored as f64).
        Num(f64),
        /// A JSON string.
        Str(String),
        /// A JSON object with insertion-independent (sorted) keys.
        Object(BTreeMap<String, Value>),
    }

    impl Value {
        /// Borrow as an object, or `None`.
        #[must_use]
        pub fn as_object(&self) -> Option<&BTreeMap<String, Value>> {
            match self {
                Value::Object(m) => Some(m),
                _ => None,
            }
        }
        /// Read as an f64, or `None`.
        #[must_use]
        pub fn as_f64(&self) -> Option<f64> {
            match self {
                Value::Num(n) => Some(*n),
                _ => None,
            }
        }
        /// Borrow as a `&str`, or `None`.
        #[must_use]
        pub fn as_str(&self) -> Option<&str> {
            match self {
                Value::Str(s) => Some(s),
                _ => None,
            }
        }
    }

    /// Parse a full JSON document into a [`Value`].
    ///
    /// # Errors
    /// Returns a message on any syntax error or trailing garbage.
    pub fn parse(text: &str) -> Result<Value, String> {
        let mut p = Parser {
            b: text.as_bytes(),
            i: 0,
        };
        p.skip_ws();
        let v = p.value()?;
        p.skip_ws();
        if p.i != p.b.len() {
            return Err(format!("trailing data at byte {}", p.i));
        }
        Ok(v)
    }

    struct Parser<'a> {
        b: &'a [u8],
        i: usize,
    }

    impl Parser<'_> {
        fn skip_ws(&mut self) {
            while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
                self.i += 1;
            }
        }

        fn value(&mut self) -> Result<Value, String> {
            self.skip_ws();
            match self.b.get(self.i) {
                Some(b'{') => self.object(),
                Some(b'"') => Ok(Value::Str(self.string()?)),
                Some(b't') | Some(b'f') => self.boolean(),
                Some(b'n') => self.null(),
                Some(c) if *c == b'-' || c.is_ascii_digit() => self.number(),
                Some(c) => Err(format!("unexpected byte {:?} at {}", *c as char, self.i)),
                None => Err("unexpected end of input".into()),
            }
        }

        fn object(&mut self) -> Result<Value, String> {
            self.expect_byte(b'{')?;
            let mut map = BTreeMap::new();
            self.skip_ws();
            if self.b.get(self.i) == Some(&b'}') {
                self.i += 1;
                return Ok(Value::Object(map));
            }
            loop {
                self.skip_ws();
                let key = self.string()?;
                self.skip_ws();
                self.expect_byte(b':')?;
                let val = self.value()?;
                map.insert(key, val);
                self.skip_ws();
                match self.b.get(self.i) {
                    Some(b',') => {
                        self.i += 1;
                    }
                    Some(b'}') => {
                        self.i += 1;
                        break;
                    }
                    _ => return Err(format!("expected ',' or '}}' at {}", self.i)),
                }
            }
            Ok(Value::Object(map))
        }

        fn string(&mut self) -> Result<String, String> {
            self.expect_byte(b'"')?;
            let mut out = String::new();
            let mut raw = Vec::new();
            while let Some(&c) = self.b.get(self.i) {
                self.i += 1;
                match c {
                    b'"' => {
                        Self::flush_string_bytes(&mut out, &mut raw, self.i)?;
                        return Ok(out);
                    }
                    b'\\' => {
                        Self::flush_string_bytes(&mut out, &mut raw, self.i)?;
                        let e = *self.b.get(self.i).ok_or("bad escape")?;
                        self.i += 1;
                        match e {
                            b'"' => out.push('"'),
                            b'\\' => out.push('\\'),
                            b'/' => out.push('/'),
                            b'n' => out.push('\n'),
                            b'r' => out.push('\r'),
                            b't' => out.push('\t'),
                            b'u' => {
                                let hex = self.b.get(self.i..self.i + 4).ok_or("bad \\u escape")?;
                                let code = u32::from_str_radix(
                                    std::str::from_utf8(hex).map_err(|_| "bad \\u hex")?,
                                    16,
                                )
                                .map_err(|_| "bad \\u hex")?;
                                self.i += 4;
                                out.push(char::from_u32(code).unwrap_or('\u{FFFD}'));
                            }
                            other => return Err(format!("bad escape \\{}", other as char)),
                        }
                    }
                    c if c < 0x20 => {
                        return Err(format!(
                            "unescaped control byte in string at {}",
                            self.i - 1
                        ));
                    }
                    _ => raw.push(c),
                }
            }
            Err("unterminated string".into())
        }

        fn flush_string_bytes(
            out: &mut String,
            raw: &mut Vec<u8>,
            at: usize,
        ) -> Result<(), String> {
            if raw.is_empty() {
                return Ok(());
            }
            let text = std::str::from_utf8(raw)
                .map_err(|_| format!("invalid utf-8 in string ending before byte {at}"))?;
            out.push_str(text);
            raw.clear();
            Ok(())
        }

        fn number(&mut self) -> Result<Value, String> {
            let start = self.i;
            if self.b.get(self.i) == Some(&b'-') {
                self.i += 1;
            }
            while let Some(&c) = self.b.get(self.i) {
                if c.is_ascii_digit() || matches!(c, b'.' | b'e' | b'E' | b'+' | b'-') {
                    self.i += 1;
                } else {
                    break;
                }
            }
            let s = std::str::from_utf8(&self.b[start..self.i]).map_err(|_| "bad number utf8")?;
            s.parse::<f64>()
                .map(Value::Num)
                .map_err(|_| format!("bad number {s:?}"))
        }

        fn boolean(&mut self) -> Result<Value, String> {
            if self.b[self.i..].starts_with(b"true") {
                self.i += 4;
                Ok(Value::Bool(true))
            } else if self.b[self.i..].starts_with(b"false") {
                self.i += 5;
                Ok(Value::Bool(false))
            } else {
                Err(format!("bad literal at {}", self.i))
            }
        }

        fn null(&mut self) -> Result<Value, String> {
            if self.b[self.i..].starts_with(b"null") {
                self.i += 4;
                Ok(Value::Null)
            } else {
                Err(format!("bad literal at {}", self.i))
            }
        }

        fn expect_byte(&mut self, c: u8) -> Result<(), String> {
            if self.b.get(self.i) == Some(&c) {
                self.i += 1;
                Ok(())
            } else {
                Err(format!("expected {:?} at {}", c as char, self.i))
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests — the ratchet math with synthetic histories (the key deliverable).
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn stats(p50: f64, p90: f64, cv: f64) -> SampleStats {
        SampleStats {
            n: 30,
            p50_s: p50,
            p90_s: p90,
            min_s: p50 * 0.98,
            mean_s: p50,
            cv_pct: cv,
        }
    }

    fn rec(
        name: &str,
        category: &str,
        p50: f64,
        p90: f64,
        throughput: Option<f64>,
        cv: f64,
    ) -> BenchRecord {
        BenchRecord {
            name: name.into(),
            category: category.into(),
            shape: "synthetic".into(),
            stats: stats(p50, p90, cv),
            throughput,
            throughput_unit: throughput.map(|_| "tok/s".into()),
            fairness: Fairness::new(8, Precision::Int8),
            reference_p50_s: None,
            reference_backend: None,
            reference_precision: None,
            note: None,
        }
    }

    // ── SampleStats ─────────────────────────────────────────────────────────

    #[test]
    fn sample_stats_percentiles_and_cv() {
        // 1..=10 ms; median by nearest-rank is the 5th value (0.5*10=5 -> idx 4),
        // p90 is the 9th value (0.9*10=9 -> idx 8).
        let secs: Vec<f64> = (1..=10).map(|i| i as f64 * 1e-3).collect();
        let s = SampleStats::from_secs(&secs);
        assert_eq!(s.n, 10);
        assert!((s.p50_s - 5e-3).abs() < 1e-12, "p50={}", s.p50_s);
        assert!((s.p90_s - 9e-3).abs() < 1e-12, "p90={}", s.p90_s);
        assert!((s.min_s - 1e-3).abs() < 1e-12);
        assert!(s.cv_pct > 0.0);
    }

    #[test]
    fn sample_stats_warmup_discard() {
        let samples = vec![
            Duration::from_millis(100), // warmup, discarded
            Duration::from_millis(10),
            Duration::from_millis(10),
            Duration::from_millis(10),
        ];
        let s = SampleStats::from_durations(&samples, 1);
        assert_eq!(s.n, 3);
        assert!((s.p50_s - 0.010).abs() < 1e-9);
        assert!(s.cv_pct < 1e-6, "stable samples -> ~0 cv, got {}", s.cv_pct);
    }

    #[test]
    fn cv_gate_eligibility() {
        assert!(stats(1.0, 1.0, 4.9).is_keep_eligible());
        assert!(stats(1.0, 1.0, 5.0).is_keep_eligible());
        assert!(!stats(1.0, 1.0, 5.1).is_keep_eligible());
    }

    // ── JSON round-trip ─────────────────────────────────────────────────────

    #[test]
    fn history_json_roundtrips() -> Result<(), String> {
        let records = vec![
            rec("decode_token", "decode", 0.002, 0.0025, Some(500.0), 2.0),
            rec("vision_encode", "vision", 0.05, 0.06, None, 3.0),
        ];
        let h = History::from_records(&records);
        let json = h.to_json();
        let back = History::from_json(&json)?;
        assert_eq!(h, back);
        assert_eq!(back.rows.len(), 2);
        assert!((back.rows["decode_token"].p50_s - 0.002).abs() < 1e-12);
        assert!((back.rows["decode_token"].throughput - 500.0).abs() < 1e-9);
        Ok(())
    }

    #[test]
    fn history_json_preserves_utf8_names_and_escaped_controls() -> Result<(), String> {
        let name = "decode_\u{00e9}_\u{6f22}_\u{1f9ea}\nunit\u{001f}";
        let h = History::from_records(&[rec(name, "decode", 0.002, 0.0025, Some(500.0), 2.0)]);
        let json = h.to_json();
        let back = History::from_json(&json)?;
        assert!(back.rows.contains_key(name));
        assert!(
            History::from_json(
                "{\"rows\":{\"bad\nkey\":{\"p50_s\":1,\"p90_s\":1,\"throughput\":1}}}"
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn bench_record_json_is_data_only_object() {
        let mut r = rec("k", "kernel", 0.001, 0.0012, Some(1234.0), 1.5);
        r.reference_p50_s = Some(0.0011);
        r.reference_backend = Some("onnx".into());
        r.reference_precision = Some("bf16".into());
        let j = r.to_json();
        assert!(j.starts_with('{') && j.ends_with('}'));
        assert!(j.contains("\"precision\":\"int8\""));
        assert!(j.contains("\"reference_backend\":\"onnx\""));
        assert!(j.contains("\"reference_precision\":\"bf16\""));
        assert!(j.contains("\"ratio\":"));
        // parse it back through the embedded parser to prove it is valid JSON.
        let v = json::parse(&j).expect("valid json");
        assert_eq!(
            v.as_object().unwrap().get("name").unwrap().as_str(),
            Some("k")
        );
    }

    #[test]
    fn malformed_history_is_loud_not_silent() {
        assert!(History::from_json("{not json").is_err());
        assert!(History::from_json("{\"rows\": 5}").is_err());
        assert!(History::from_json("{\"rows\": {\"a\": {\"p50_s\": \"x\"}}}").is_err());
    }

    #[test]
    fn missing_history_file_is_empty_not_error() {
        let p = std::env::temp_dir().join("focr_no_such_history_xyz.json");
        let _ = std::fs::remove_file(&p);
        let h = History::read_or_empty(&p).expect("missing -> empty");
        assert!(h.rows.is_empty());
    }

    // ── Ratchet gates (the central deliverable) ─────────────────────────────

    fn baseline_two() -> History {
        // decode @2ms p50 / 2.5ms p90 / 500 tok/s; vision @50ms / 60ms.
        History::from_records(&[
            rec("decode_token", "decode", 0.002, 0.0025, Some(500.0), 1.0),
            rec("vision_encode", "vision", 0.050, 0.060, None, 1.0),
        ])
    }

    #[test]
    fn ratchet_allows_an_improvement() {
        let base = baseline_two();
        let r = Ratchet::new("decode_token");
        // both faster + higher throughput.
        let now = vec![
            rec("decode_token", "decode", 0.0018, 0.0022, Some(560.0), 1.0),
            rec("vision_encode", "vision", 0.048, 0.058, None, 1.0),
        ];
        let v = r.compare(&now, &base);
        assert!(v.allowed(), "{}", v.reason);
        assert!(v.gates.iter().all(|g| g.pass));
    }

    #[test]
    fn ratchet_allows_a_within_threshold_wobble() {
        let base = baseline_two();
        let r = Ratchet::new("decode_token");
        // decode p50 +2% (< 3% primary), vision p50 +4% (geomean still < 5%),
        // p90 +5% (< 15%), throughput -3% (< 5%).
        let now = vec![
            rec(
                "decode_token",
                "decode",
                0.00204,
                0.002625,
                Some(485.0),
                1.0,
            ),
            rec("vision_encode", "vision", 0.0520, 0.0630, None, 1.0),
        ];
        let v = r.compare(&now, &base);
        assert!(v.allowed(), "{}", v.reason);
    }

    #[test]
    fn ratchet_blocks_primary_regression() {
        let base = baseline_two();
        let r = Ratchet::new("decode_token");
        // decode p50 +4% > 3% primary gate.
        let now = vec![
            rec("decode_token", "decode", 0.00208, 0.0025, Some(500.0), 1.0),
            rec("vision_encode", "vision", 0.050, 0.060, None, 1.0),
        ];
        let v = r.compare(&now, &base);
        assert!(!v.allowed(), "should block; {}", v.reason);
        let primary = v.gates.iter().find(|g| g.gate == "primary").unwrap();
        assert!(!primary.pass);
        assert!(v.reason.contains("primary"));
    }

    #[test]
    fn ratchet_blocks_geomean_regression() {
        let base = baseline_two();
        // Primary (decode) within 3% but the geomean across both p50 > 5%:
        // decode +2.9%, vision +8% -> geomean ratio ~ sqrt(1.029*1.08)=1.054 > 1.05.
        let r = Ratchet::new("decode_token");
        let now = vec![
            rec(
                "decode_token",
                "decode",
                0.0020579,
                0.0025,
                Some(500.0),
                1.0,
            ),
            rec("vision_encode", "vision", 0.054, 0.060, None, 1.0),
        ];
        let v = r.compare(&now, &base);
        let geo = v.gates.iter().find(|g| g.gate == "geomean").unwrap();
        assert!(!geo.pass, "geomean reg={}", geo.observed_regression);
        assert!(!v.allowed());
    }

    #[test]
    fn ratchet_blocks_per_category_regression() {
        // A category-local regression that the global geomean dilutes below 5%.
        // Two decode benches each +12% (per-category geomean +12% > 10%), while a
        // big fast vision bench keeps the GLOBAL geomean under 5%.
        let base = History::from_records(&[
            rec("decode_a", "decode", 0.002, 0.0025, None, 1.0),
            rec("decode_b", "decode", 0.002, 0.0025, None, 1.0),
            rec("vision_encode", "vision", 0.050, 0.060, None, 1.0),
        ]);
        let r = Ratchet::new("decode_a");
        let now = vec![
            rec("decode_a", "decode", 0.00224, 0.0025, None, 1.0), // +12%
            rec("decode_b", "decode", 0.00224, 0.0025, None, 1.0), // +12%
            rec("vision_encode", "vision", 0.0475, 0.060, None, 1.0), // -5%
        ];
        let v = r.compare(&now, &base);
        let cat = v
            .gates
            .iter()
            .find(|g| g.gate == "per_category:decode")
            .expect("decode category gate");
        assert!(!cat.pass, "per-cat decode reg={}", cat.observed_regression);
        // global geomean: (1.12*1.12*0.95)^(1/3) ~ 1.061... actually verify it is
        // the per-category gate that is doing the blocking semantics here.
        assert!(!v.allowed());
    }

    #[test]
    fn ratchet_blocks_p90_tail_regression() {
        let base = baseline_two();
        let r = Ratchet::new("decode_token");
        // p50 flat, but decode p90 +20% > 15% tail gate.
        let now = vec![
            rec("decode_token", "decode", 0.002, 0.0030, Some(500.0), 1.0),
            rec("vision_encode", "vision", 0.050, 0.060, None, 1.0),
        ];
        let v = r.compare(&now, &base);
        let p90 = v.gates.iter().find(|g| g.gate.starts_with("p90:")).unwrap();
        assert!(!p90.pass, "p90 reg={}", p90.observed_regression);
        assert!(!v.allowed());
    }

    #[test]
    fn ratchet_blocks_throughput_regression() {
        let base = baseline_two();
        let r = Ratchet::new("decode_token");
        // everything flat except throughput -8% > 5%.
        let now = vec![
            rec("decode_token", "decode", 0.002, 0.0025, Some(460.0), 1.0),
            rec("vision_encode", "vision", 0.050, 0.060, None, 1.0),
        ];
        let v = r.compare(&now, &base);
        let tp = v
            .gates
            .iter()
            .find(|g| g.gate.starts_with("throughput:"))
            .unwrap();
        assert!(!tp.pass, "tp reg={}", tp.observed_regression);
        assert!(!v.allowed());
    }

    #[test]
    fn ratchet_new_bench_never_blocks() {
        let base = baseline_two();
        let r = Ratchet::new("decode_token");
        // a brand-new bench absent from baseline cannot regress.
        let now = vec![
            rec("decode_token", "decode", 0.002, 0.0025, Some(500.0), 1.0),
            rec("vision_encode", "vision", 0.050, 0.060, None, 1.0),
            rec("brand_new_kernel", "kernel", 999.0, 999.0, Some(1.0), 1.0),
        ];
        let v = r.compare(&now, &base);
        assert!(v.allowed(), "{}", v.reason);
    }

    #[test]
    fn ratchet_excludes_noisy_rows() {
        let base = baseline_two();
        let r = Ratchet::new("decode_token");
        // decode regressed badly but cv=9% (>5) -> excluded, not a block.
        let now = vec![
            rec("decode_token", "decode", 0.010, 0.012, Some(100.0), 9.0),
            rec("vision_encode", "vision", 0.050, 0.060, None, 1.0),
        ];
        let v = r.compare(&now, &base);
        assert!(v.allowed(), "noisy row excluded; {}", v.reason);
        // and it does not appear in the advanced history either.
        let adv = r.advance(&now, &base);
        assert!((adv.rows["decode_token"].p50_s - 0.002).abs() < 1e-12);
    }

    #[test]
    fn ratchet_advance_is_monotone_best_of() {
        let base = baseline_two();
        let r = Ratchet::new("decode_token");
        // a faster decode tightens the floor; a slower vision does NOT loosen it.
        let now = vec![
            rec("decode_token", "decode", 0.0015, 0.0020, Some(620.0), 1.0),
            rec("vision_encode", "vision", 0.070, 0.080, None, 1.0),
        ];
        let adv = r.advance(&now, &base);
        assert!(
            (adv.rows["decode_token"].p50_s - 0.0015).abs() < 1e-12,
            "tightened"
        );
        assert!((adv.rows["decode_token"].throughput - 620.0).abs() < 1e-9);
        assert!(
            (adv.rows["vision_encode"].p50_s - 0.050).abs() < 1e-12,
            "not loosened"
        );
    }

    #[test]
    fn first_round_against_empty_history_allows_and_seeds() {
        let empty = History::default();
        let r = Ratchet::new("decode_token");
        let now = vec![rec(
            "decode_token",
            "decode",
            0.002,
            0.0025,
            Some(500.0),
            1.0,
        )];
        let v = r.compare(&now, &empty);
        assert!(v.allowed(), "first round vacuously passes");
        let seeded = r.advance(&now, &empty);
        assert_eq!(seeded.rows.len(), 1);
    }

    // ── Fairness controls ───────────────────────────────────────────────────

    #[test]
    fn fairness_thread_parity_enforced() {
        let f = Fairness::new(8, Precision::Int8);
        assert!(f.assert_thread_parity(8).is_ok());
        let err = f.assert_thread_parity(64).unwrap_err();
        assert!(err.contains("FAIRNESS VIOLATION"));
        assert!(err.contains("NEVER @64"));
    }

    #[test]
    fn fairness_allocator_from_build_is_honest() {
        // Without the (unwired) mimalloc feature, the measured binary is system.
        assert_eq!(
            Fairness::new(8, Precision::F32).allocator,
            Allocator::System
        );
    }

    #[test]
    fn precision_and_allocator_tags_stable() {
        assert_eq!(Precision::Int8.tag(), "int8");
        assert_eq!(Precision::Bf16.tag(), "bf16");
        assert_eq!(Allocator::System.tag(), "system");
        assert_eq!(Allocator::Mimalloc.tag(), "mimalloc");
    }

    // ── ratio tagging ───────────────────────────────────────────────────────

    #[test]
    fn ratio_tags_match_plan_buckets() {
        let mk = |focr: f64, refr: f64| {
            let mut r = rec("x", "decode", focr, focr, None, 1.0);
            r.reference_p50_s = Some(refr);
            r
        };
        assert_eq!(mk(0.8, 1.0).ratio(), Some(1.25));
        assert_eq!(mk(0.8, 1.0).ratio_tag(), Some("focr_faster"));
        assert_eq!(mk(1.0, 1.0).ratio(), Some(1.0));
        assert_eq!(mk(1.0, 1.0).ratio_tag(), Some("ok"));
        assert_eq!(mk(1.2, 1.0).ratio(), Some(1.0 / 1.2));
        assert_eq!(mk(1.2, 1.0).ratio_tag(), Some("warn"));
        assert_eq!(mk(1.5, 1.0).ratio(), Some(1.0 / 1.5));
        assert_eq!(mk(1.5, 1.0).ratio_tag(), Some("slower"));
        assert_eq!(rec("x", "decode", 1.0, 1.0, None, 1.0).ratio_tag(), None);
    }
}
