//! Error type and the **stable** process exit-code mapping (plan §7.4).
//!
//! Exit codes are a public contract: `robot run_error` carries the same code in
//! its payload, and agents branch on them. Do not renumber existing codes.

use thiserror::Error;

/// Result alias used throughout the crate.
pub type FocrResult<T> = Result<T, FocrError>;

/// Top-level error. Each variant maps to a stable exit code via
/// [`FocrError::exit_code`].
#[derive(Debug, Error)]
pub enum FocrError {
    /// Usage / CLI argument error.
    #[error("usage error: {0}")]
    Usage(String),

    /// The model could not be found or resolved (plan §7.5).
    #[error("model not found / not resolvable: {0}")]
    ModelNotFound(String),

    /// An input image (or PDF page) could not be decoded.
    #[error("input decode error: {0}")]
    InputDecode(String),

    /// A per-stage / per-page budget or deadline was exceeded.
    #[error("budget/timeout exceeded: {0}")]
    Timeout(String),

    /// Cancelled (Ctrl+C / cooperative cancellation).
    #[error("cancelled")]
    Cancelled,

    /// A `.focrq` (or other) format/version mismatch.
    #[error("format/version mismatch: {0}")]
    FormatMismatch(String),

    /// A surface that is planned but not yet implemented in this phase.
    #[error("not yet implemented: {0}")]
    NotImplemented(String),

    /// Any other error.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl FocrError {
    /// Stable process exit code (plan §7.4). Keep in sync with the `robot
    /// run_error` payload. `0` is reserved for success.
    pub fn exit_code(&self) -> i32 {
        match self {
            FocrError::Usage(_) => 2,
            FocrError::ModelNotFound(_) => 3,
            FocrError::InputDecode(_) => 4,
            FocrError::Timeout(_) => 5,
            FocrError::Cancelled => 6,
            FocrError::FormatMismatch(_) => 7,
            FocrError::NotImplemented(_) | FocrError::Other(_) => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_codes_are_stable() {
        assert_eq!(FocrError::Usage("x".into()).exit_code(), 2);
        assert_eq!(FocrError::ModelNotFound("x".into()).exit_code(), 3);
        assert_eq!(FocrError::InputDecode("x".into()).exit_code(), 4);
        assert_eq!(FocrError::Timeout("x".into()).exit_code(), 5);
        assert_eq!(FocrError::Cancelled.exit_code(), 6);
        assert_eq!(FocrError::FormatMismatch("x".into()).exit_code(), 7);
        assert_eq!(FocrError::NotImplemented("x".into()).exit_code(), 1);
    }
}
