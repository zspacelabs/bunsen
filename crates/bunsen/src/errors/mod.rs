//! Common bunsen errors.

mod result_ext;
#[doc(inline)]
pub use result_ext::*;

/// Common bunsen error type.
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

impl BunsenError {
    /// Map an error to an External string error.
    pub fn external<E>(e: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        BunsenError::External(e.to_string())
    }
}

/// Result type for bunsen operations.
pub type BunsenResult<T> = core::result::Result<T, BunsenError>;
