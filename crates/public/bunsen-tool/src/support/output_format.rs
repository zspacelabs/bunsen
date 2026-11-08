//! `--format`: YAML or JSON.

/// `--format`.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// YAML.
    #[value(alias = "yaml")]
    Yml,

    /// JSON, unindented.
    Json,
}
