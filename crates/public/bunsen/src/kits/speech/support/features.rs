//! # Streaming features: audio in pieces of any size, to feature frames.

use core::fmt::Debug;

use burn::Tensor;

use crate::errors::BunsenResult;

/// A front end that turns audio, arriving in pieces of any size, into
/// feature frames.
pub trait StreamingFeatures: Clone + Debug + Send + Sync {
    /// Takes samples in, and returns the frames they complete.
    ///
    /// # Returns
    /// `[1, frames, features]`, or `None` when no frame completes.
    ///
    /// # Errors
    /// As the front end.
    fn push(
        &mut self,
        samples: &[f32],
    ) -> BunsenResult<Option<Tensor<3>>>;

    /// Ends the input, and returns the frames left.
    ///
    /// # Returns
    /// `[1, frames, features]`, or `None` when none are.
    ///
    /// # Errors
    /// As the front end.
    fn finish(&mut self) -> BunsenResult<Option<Tensor<3>>>;
}
