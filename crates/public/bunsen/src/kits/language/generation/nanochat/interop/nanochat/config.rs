//! The config conversion.

/// Which of upstream's models a checkpoint is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoChatGeneration {
    /// Upstream's model as of late 2025 (`karpathy/nanochat` at
    /// `a088b7a6`): the policy's defaults.
    Upstream2025,

    /// Upstream's 2026 model (`a3ca42a6`): the 2025 one with
    /// [`with_upstream_2026`](crate::kits::language::generation::nanochat::NanoChatGptContractConfig::with_upstream_2026).
    Upstream2026,
}
