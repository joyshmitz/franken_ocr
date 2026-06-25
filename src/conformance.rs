//! Phase-0 conformance contract seed.
//!
//! This module declares the stable vocabulary that later L0-L5 parity gates,
//! rollout receipts, and invariant monitors plug into. It intentionally does
//! not compare real model tensors yet: the real numeric budgets are derived
//! from the oracle nondeterminism floor before Phase-1+ gates claim parity.

/// Ordered parity ladder levels from plan §8.2.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ParityLevel {
    /// L0: preprocessing, tile geometry, and image-token stream are exact.
    L0Preprocess,
    /// L1: per-op activations compared with cosine / ULP budgets.
    L1PerOp,
    /// L2: per-layer hidden states, max-abs-diff ledgered.
    L2PerLayer,
    /// L3: logits within the measured budget, argmax exact when deterministic.
    L3Logits,
    /// L4: decoded tokens exact over the oracle reproducible prefix.
    L4Tokens,
    /// L5: end-to-end OCR text / tables / formula metrics within budget.
    L5EndToEnd,
}

impl ParityLevel {
    /// The complete ordered ladder. Integration runners execute this order and
    /// short-circuit lower-rung failures before higher-rung claims.
    pub const ALL: [Self; 6] = [
        Self::L0Preprocess,
        Self::L1PerOp,
        Self::L2PerLayer,
        Self::L3Logits,
        Self::L4Tokens,
        Self::L5EndToEnd,
    ];

    /// Stable machine label for scorecards and structured logs.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::L0Preprocess => "L0_preprocess",
            Self::L1PerOp => "L1_per_op",
            Self::L2PerLayer => "L2_per_layer",
            Self::L3Logits => "L3_logits",
            Self::L4Tokens => "L4_tokens",
            Self::L5EndToEnd => "L5_end_to_end",
        }
    }

    /// Zero-based rung index, useful for ordered scorecard rows.
    #[must_use]
    pub const fn index(self) -> u8 {
        match self {
            Self::L0Preprocess => 0,
            Self::L1PerOp => 1,
            Self::L2PerLayer => 2,
            Self::L3Logits => 3,
            Self::L4Tokens => 4,
            Self::L5EndToEnd => 5,
        }
    }
}

/// How a gate's tolerance is interpreted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ToleranceKind {
    /// Discrete value or geometry must match exactly.
    Exact,
    /// Continuous activation is compared by cosine and/or ULP.
    Cosine,
    /// Budget is derived from measured oracle variance.
    Measured,
    /// Exactness applies only to the oracle reproducible prefix.
    ExactPrefix,
    /// Aggregate document metric within a ledgered budget.
    Budget,
}

/// Source of a tolerance value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ToleranceSource {
    /// The value is a fixed structural contract, not a statistical budget.
    StructuralSpec,
    /// Placeholder pending the measured oracle nondeterminism floor.
    TodoDeriveFromOracleFloor,
    /// Placeholder pending a ledgered corpus / release budget.
    TodoLedgerBudget,
}

/// One rung's tolerance contract.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GateTolerance {
    /// Ladder level this row applies to.
    pub level: ParityLevel,
    /// Tolerance interpretation.
    pub kind: ToleranceKind,
    /// Minimum cosine similarity for continuous f32 activations.
    pub cosine_min: Option<f64>,
    /// Optional max-absolute-difference budget. `None` means "ledger only" or
    /// "derive before use"; do not silently substitute the old `0.055` precedent.
    pub max_abs_diff: Option<f64>,
    /// Optional logit tolerance. L3 leaves this unset until the oracle floor is
    /// measured for this model.
    pub logit_tolerance: Option<f64>,
    /// Whether the argmax index must match on deterministic positions.
    pub argmax_must_match: bool,
    /// Whether the compared payload must be exactly equal.
    pub exact_match: bool,
    /// Optional CER budget for L5.
    pub cer_budget: Option<f64>,
    /// Optional TEDS budget for L5.
    pub teds_budget: Option<f64>,
    /// Optional Formula-CDM budget for L5.
    pub formula_cdm_budget: Option<f64>,
    /// Where the tolerance came from.
    pub source: ToleranceSource,
    /// Short human note for scorecards and reviewers.
    pub note: &'static str,
}

impl GateTolerance {
    const fn exact(level: ParityLevel, note: &'static str) -> Self {
        Self {
            level,
            kind: ToleranceKind::Exact,
            cosine_min: None,
            max_abs_diff: Some(0.0),
            logit_tolerance: None,
            argmax_must_match: false,
            exact_match: true,
            cer_budget: None,
            teds_budget: None,
            formula_cdm_budget: None,
            source: ToleranceSource::StructuralSpec,
            note,
        }
    }

    const fn cosine(level: ParityLevel, note: &'static str) -> Self {
        Self {
            level,
            kind: ToleranceKind::Cosine,
            cosine_min: Some(0.9999),
            max_abs_diff: None,
            logit_tolerance: None,
            argmax_must_match: false,
            exact_match: false,
            cer_budget: None,
            teds_budget: None,
            formula_cdm_budget: None,
            source: ToleranceSource::StructuralSpec,
            note,
        }
    }
}

/// Phase-0 seed of the L0-L5 tolerance table.
#[derive(Clone, Debug, PartialEq)]
pub struct Tolerances {
    /// L0 preprocessing exactness.
    pub l0: GateTolerance,
    /// L1 per-op activation tolerance.
    pub l1: GateTolerance,
    /// L2 per-layer activation tolerance.
    pub l2: GateTolerance,
    /// L3 logit tolerance, derived from oracle floor before use.
    pub l3: GateTolerance,
    /// L4 token exactness over reproducible prefix.
    pub l4: GateTolerance,
    /// L5 aggregate OCR metric budget.
    pub l5: GateTolerance,
}

impl Tolerances {
    /// Iterate the table in L0-L5 order.
    pub fn ordered(&self) -> [GateTolerance; 6] {
        [self.l0, self.l1, self.l2, self.l3, self.l4, self.l5]
    }
}

impl Default for Tolerances {
    fn default() -> Self {
        default_tolerances()
    }
}

/// Default Phase-0 tolerance seed.
///
/// The L3/L4/L5 rows deliberately carry TODO sources. The old frankensearch
/// `0.055` logit-diff precedent is not meaningful for this model and is not
/// encoded here; Phase 1 fills these rows from the committed oracle floor and
/// corpus budget artifacts.
#[must_use]
pub fn default_tolerances() -> Tolerances {
    Tolerances {
        l0: GateTolerance::exact(
            ParityLevel::L0Preprocess,
            "exact gray pad, [-1,1] normalize, ratio selection, and tile geometry",
        ),
        l1: GateTolerance::cosine(
            ParityLevel::L1PerOp,
            "cosine >= 0.9999 f32; bridge path applies the per-op ULP table",
        ),
        l2: GateTolerance::cosine(
            ParityLevel::L2PerLayer,
            "cosine ~= 1.0 with per-layer max-abs-diff ledgered",
        ),
        l3: GateTolerance {
            level: ParityLevel::L3Logits,
            kind: ToleranceKind::Measured,
            cosine_min: None,
            max_abs_diff: None,
            logit_tolerance: None,
            argmax_must_match: true,
            exact_match: false,
            cer_budget: None,
            teds_budget: None,
            formula_cdm_budget: None,
            source: ToleranceSource::TodoDeriveFromOracleFloor,
            note: "TODO derive logit budget from oracle nondeterminism floor before use",
        },
        l4: GateTolerance {
            level: ParityLevel::L4Tokens,
            kind: ToleranceKind::ExactPrefix,
            cosine_min: None,
            max_abs_diff: None,
            logit_tolerance: None,
            argmax_must_match: false,
            exact_match: true,
            cer_budget: None,
            teds_budget: None,
            formula_cdm_budget: None,
            source: ToleranceSource::TodoDeriveFromOracleFloor,
            note: "TODO set reproducible prefix lengths from oracle nondeterminism floor",
        },
        l5: GateTolerance {
            level: ParityLevel::L5EndToEnd,
            kind: ToleranceKind::Budget,
            cosine_min: None,
            max_abs_diff: None,
            logit_tolerance: None,
            argmax_must_match: false,
            exact_match: false,
            cer_budget: None,
            teds_budget: None,
            formula_cdm_budget: None,
            source: ToleranceSource::TodoLedgerBudget,
            note: "TODO fill CER/TEDS/Formula-CDM budgets from corpus and release ledger",
        },
    }
}

/// Rollout stage that a parity receipt can be attached to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RolloutStage {
    /// Phase 1: framework-free f32 path.
    Fp32Reference,
    /// Phase 2a: decoder FFN / expert GEMMs only.
    Int8FfnExpertsOnly,
    /// Phase 2b: attention projections int8 behind accuracy gate.
    Int8Attention,
    /// Phase 2c: `lm_head` int8 behind argmax/token gate.
    Int8LmHead,
    /// Phase 3: SIMD/native kernel rollout.
    SimdKernels,
    /// Phase 4: int4 expert bulk under measured budget.
    Int4Experts,
}

impl RolloutStage {
    /// Stable label for parity receipts.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Fp32Reference => "P1_fp32_reference",
            Self::Int8FfnExpertsOnly => "P2a_int8_ffn_experts",
            Self::Int8Attention => "P2b_int8_attention",
            Self::Int8LmHead => "P2c_int8_lm_head",
            Self::SimdKernels => "P3_simd_kernels",
            Self::Int4Experts => "P4_int4_experts",
        }
    }
}

/// Invariant families tracked by the conformance/gauntlet machinery.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InvariantKind {
    /// Generated-token KV remains bounded by the R-SWA reference + window contract.
    KvCacheBound,
    /// int8 i32 accumulation cannot overflow at the model's worst-case K.
    Int8AccumulatorOverflow,
    /// Same input and deterministic settings produce byte-identical output.
    Determinism,
    /// SIMD and scalar fallbacks are bit-identical where required.
    SimdScalarBitIdentical,
}

impl InvariantKind {
    /// Stable label for structured logs.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::KvCacheBound => "kv_cache_bound",
            Self::Int8AccumulatorOverflow => "int8_accumulator_overflow",
            Self::Determinism => "determinism",
            Self::SimdScalarBitIdentical => "simd_scalar_bit_identical",
        }
    }
}

/// Minimal result row emitted by placeholder validators and future real gates.
#[derive(Clone, Debug, PartialEq)]
pub struct GateResult {
    /// Gate or invariant name.
    pub name: &'static str,
    /// Optional ladder level for parity gates.
    pub level: Option<ParityLevel>,
    /// Whether the check passed.
    pub passed: bool,
    /// Measured value, if numeric.
    pub measured: Option<f64>,
    /// Tolerance value, if numeric.
    pub tolerance: Option<f64>,
    /// Short diagnostic suitable for structured logs.
    pub message: &'static str,
}

impl GateResult {
    /// Construct a passing placeholder / structural result.
    #[must_use]
    pub const fn pass(
        name: &'static str,
        level: Option<ParityLevel>,
        message: &'static str,
    ) -> Self {
        Self {
            name,
            level,
            passed: true,
            measured: None,
            tolerance: None,
            message,
        }
    }

    /// Construct a failing result.
    #[must_use]
    pub const fn fail(
        name: &'static str,
        level: Option<ParityLevel>,
        message: &'static str,
    ) -> Self {
        Self {
            name,
            level,
            passed: false,
            measured: None,
            tolerance: None,
            message,
        }
    }
}

/// Validator trait for L0-L5 parity gates.
pub trait ParityGate {
    /// Stable gate name.
    fn name(&self) -> &'static str;
    /// Ladder level.
    fn level(&self) -> ParityLevel;
    /// Validate subject bytes against oracle bytes.
    fn validate(&self, subject: &[u8], oracle: &[u8]) -> GateResult;
}

/// Validator trait for non-ladder invariants.
pub trait Invariant {
    /// Stable invariant name.
    fn name(&self) -> &'static str;
    /// Invariant family.
    fn kind(&self) -> InvariantKind;
    /// Validate the invariant in the current context.
    fn validate(&self) -> GateResult;
}

/// Phase-0 no-op parity gate used to prove the trait shape is object-safe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaceholderParityGate {
    name: &'static str,
    level: ParityLevel,
}

impl PlaceholderParityGate {
    /// Create a placeholder gate for a future real validator.
    #[must_use]
    pub const fn new(name: &'static str, level: ParityLevel) -> Self {
        Self { name, level }
    }
}

impl ParityGate for PlaceholderParityGate {
    fn name(&self) -> &'static str {
        self.name
    }

    fn level(&self) -> ParityLevel {
        self.level
    }

    fn validate(&self, _subject: &[u8], _oracle: &[u8]) -> GateResult {
        GateResult::pass(
            self.name,
            Some(self.level),
            "placeholder validator declared",
        )
    }
}

/// Phase-0 no-op invariant used to prove the trait shape is object-safe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaceholderInvariant {
    name: &'static str,
    kind: InvariantKind,
}

impl PlaceholderInvariant {
    /// Create a placeholder invariant for a future real monitor.
    #[must_use]
    pub const fn new(name: &'static str, kind: InvariantKind) -> Self {
        Self { name, kind }
    }
}

impl Invariant for PlaceholderInvariant {
    fn name(&self) -> &'static str {
        self.name
    }

    fn kind(&self) -> InvariantKind {
        self.kind
    }

    fn validate(&self) -> GateResult {
        GateResult::pass(self.name, None, "placeholder invariant declared")
    }
}

/// Same-input-twice determinism gate seed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeterminismGate;

impl DeterminismGate {
    /// Compare two serialized outputs from identical inputs and deterministic
    /// settings.
    #[must_use]
    pub fn validate_bytes(&self, first: &[u8], second: &[u8]) -> GateResult {
        if first == second {
            GateResult::pass(
                "same_input_twice_byte_identical",
                None,
                "same input produced byte-identical output",
            )
        } else {
            GateResult::fail(
                "same_input_twice_byte_identical",
                None,
                "same input produced divergent output",
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn log_result(check: &str, passed: bool) {
        println!(
            "{{\"check\":\"{check}\",\"result\":\"{}\"}}",
            if passed { "pass" } else { "fail" }
        );
    }

    #[test]
    fn tolerances_default_constructs() {
        let tolerances = default_tolerances();
        let ordered = tolerances.ordered();
        let ok = ordered.len() == ParityLevel::ALL.len()
            && tolerances.l0.exact_match
            && tolerances.l1.cosine_min == Some(0.9999)
            && tolerances.l2.cosine_min == Some(0.9999)
            && tolerances.l3.source == ToleranceSource::TodoDeriveFromOracleFloor
            && tolerances.l3.logit_tolerance.is_none()
            && tolerances.l3.argmax_must_match
            && tolerances.l4.source == ToleranceSource::TodoDeriveFromOracleFloor
            && tolerances.l5.source == ToleranceSource::TodoLedgerBudget;
        log_result("tolerances_default_constructs", ok);
        assert!(ok, "{tolerances:#?}");
    }

    #[test]
    fn parity_levels_span_l0_to_l5() {
        let labels: Vec<_> = ParityLevel::ALL
            .iter()
            .map(|level| (level.index(), level.label()))
            .collect();
        let ok = labels
            == vec![
                (0, "L0_preprocess"),
                (1, "L1_per_op"),
                (2, "L2_per_layer"),
                (3, "L3_logits"),
                (4, "L4_tokens"),
                (5, "L5_end_to_end"),
            ];
        log_result("parity_levels_span_l0_to_l5", ok);
        assert!(ok, "{labels:?}");
    }

    #[test]
    fn invariant_trait_object_dispatches() {
        let invariants: Vec<Box<dyn Invariant>> = vec![
            Box::new(PlaceholderInvariant::new(
                "kv_cache_never_exceeds_reference_plus_window",
                InvariantKind::KvCacheBound,
            )),
            Box::new(PlaceholderInvariant::new(
                "same_input_twice_byte_identical",
                InvariantKind::Determinism,
            )),
        ];
        let results: Vec<_> = invariants
            .iter()
            .map(|invariant| invariant.validate())
            .collect();
        let ok = results.iter().all(|result| result.passed)
            && invariants[0].kind().label() == "kv_cache_bound"
            && invariants[1].kind().label() == "determinism";
        log_result("invariant_trait_object_dispatches", ok);
        assert!(ok, "{results:?}");
    }

    #[test]
    fn parity_gate_trait_object_dispatches() {
        let gate: Box<dyn ParityGate> = Box::new(PlaceholderParityGate::new(
            "l0_preprocess_placeholder",
            ParityLevel::L0Preprocess,
        ));
        let result = gate.validate(b"subject", b"oracle");
        let ok = result.passed && result.level == Some(ParityLevel::L0Preprocess);
        log_result("parity_gate_trait_object_dispatches", ok);
        assert!(ok, "{result:?}");
    }

    #[test]
    fn determinism_gate_checks_byte_identity() {
        let gate = DeterminismGate;
        let same = gate.validate_bytes(b"abc", b"abc");
        let different = gate.validate_bytes(b"abc", b"abd");
        let ok = same.passed && !different.passed;
        log_result("determinism_gate_checks_byte_identity", ok);
        assert!(ok, "same={same:?} different={different:?}");
    }
}
