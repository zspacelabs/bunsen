//! `--format`: YAML or JSON.

use bunsen::errors::{
    BunsenError,
    BunsenResult,
};
use serde::Serialize;

/// `--format`.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// YAML.
    #[value(alias = "yaml")]
    Yml,

    /// JSON, unindented.
    Json,
}

impl Format {
    /// `value` in this format: YAML, or JSON on one line, each ending in a
    /// newline. `what` names the value for the error.
    ///
    /// # Errors
    /// [`Internal`](bunsen::errors::BunsenErrorKind::Internal) if it does not
    /// serialize.
    pub fn render<T: Serialize>(
        self,
        value: &T,
        what: &str,
    ) -> BunsenResult<String> {
        match self {
            Format::Yml => serde_saphyr::to_string(value)
                .map_err(|e| BunsenError::internal(format!("render {what} as YAML: {e}"))),
            Format::Json => serde_json::to_string(value)
                .map(|json| json + "\n")
                .map_err(|e| BunsenError::internal(format!("render {what} as JSON: {e}"))),
        }
    }
}
