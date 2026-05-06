/// Common bunsen errors.
#[derive(Debug, thiserror::Error)]
pub enum BunsenError {
    /// Resource not found.
    #[error("{0}")]
    ResourceNotFound(String),

    /// Parse error.
    #[error("{0}")]
    ParseError(String),

    /// Invalid constraint.
    #[error("{0}")]
    Invalid(String),

    /// Error from an external component.
    #[error("{0}")]
    External(String),
}

/// Result type for bunsen operations.
pub type BunsenResult<T> = core::result::Result<T, BunsenError>;
