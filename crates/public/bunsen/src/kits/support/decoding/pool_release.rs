//! Releasing the device pool's free memory as a decode runs.
use core::num::NonZeroUsize;

use burn::tensor::Device;

/// How often a decode loop releases its device's free pool memory, counted
/// in steps of one position.
///
/// A decode's history reads grow by a position a step, so every step
/// allocates buffers of sizes the pool has not seen; the pool keeps their
/// pages, and a step slows as the decode grows (Qwen3-0.6B on CUDA: ~16.7 ms
/// a step near the prompt, ~47 ms by position 2000). Releasing the free
/// memory every 64 steps keeps a step flat (a 2000-step decode: 64.9 s ->
/// 33.9 s). Only between steps: a release before a prefill's large writes
/// makes them allocate afresh.
///
/// The decode loops ([`decode_cached`](super::decode_cached), Whisper's,
/// Qwen3-TTS's and ZONOS3's) release at the default, every
/// [`DEFAULT_EVERY`](Self::DEFAULT_EVERY) steps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolRelease {
    every: Option<NonZeroUsize>,
}

impl Default for PoolRelease {
    fn default() -> Self {
        Self::every(Self::DEFAULT_EVERY)
    }
}

impl PoolRelease {
    /// The default interval, in steps.
    pub const DEFAULT_EVERY: usize = 64;
    /// Never releases.
    pub const NEVER: Self = Self { every: None };

    /// Releases every `steps` steps; never, for zero.
    pub fn every(steps: usize) -> Self {
        Self {
            every: NonZeroUsize::new(steps),
        }
    }

    /// The interval, in steps; `None` for never.
    pub fn interval(&self) -> Option<usize> {
        self.every.map(NonZeroUsize::get)
    }

    /// Whether the release falls after step `step` (from zero).
    pub fn is_due(
        &self,
        step: usize,
    ) -> bool {
        self.every
            .is_some_and(|n| (step + 1).is_multiple_of(n.get()))
    }

    /// Releases `device`'s free pool memory if the release falls after step
    /// `step` (from zero).
    pub fn after_step(
        &self,
        step: usize,
        device: &Device,
    ) {
        if self.is_due(step) {
            device.memory_cleanup();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_due() {
        let every = PoolRelease::every(4);
        let due: alloc::vec::Vec<usize> = (0..12).filter(|&s| every.is_due(s)).collect();
        assert_eq!(due, [3, 7, 11]);
        assert!((0..100).all(|s| !PoolRelease::NEVER.is_due(s)));
        assert_eq!(PoolRelease::every(0), PoolRelease::NEVER);
        assert_eq!(PoolRelease::default().interval(), Some(64));
    }
}
