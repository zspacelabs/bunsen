use burn::module::Module;

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
        LookupError,
        WithOkOrPanic,
    },
    kits::speech::silero_vad::SileroVad,
};

/// Collection of sample-rate Silero VAD models.
///
/// What a Silero checkpoint holds, and what
/// [`SileroConstruct`](crate::kits::speech::silero_vad::pretrained::SileroConstruct)
/// builds: one [`SileroVad`] per sample rate (16 kHz and 8 kHz for the
/// upstream graph). A stream picks its branch by rate with
/// [`try_branch`](Self::try_branch) or [`expect_branch`](Self::expect_branch).
#[derive(Module, Debug)]
pub struct SileroVadCollection {
    /// Per-sample-rate models.
    pub branches: Vec<(usize, SileroVad)>,
}

impl SileroVadCollection {
    /// The model for the given sample rate.
    ///
    /// # Arguments
    /// * `sample_rate`: Sample rate in Hz.
    ///
    /// # Errors
    /// [`Lookup`](crate::errors::BunsenErrorKind::Lookup), with a
    /// [`LookupError`] cause whose candidates are the rates there are, when
    /// no branch runs at `sample_rate`.
    pub fn try_branch(
        &self,
        sample_rate: usize,
    ) -> BunsenResult<&SileroVad> {
        self.branches
            .iter()
            .find(|(rate, _)| *rate == sample_rate)
            .map(|(_, vad)| vad)
            .ok_or_else(|| {
                BunsenError::lookup(
                    LookupError::missing("sample-rate branch", sample_rate.to_string())
                        .with_candidates(self.branches.iter().map(|(rate, _)| rate.to_string())),
                )
            })
    }

    /// Select the branch for the given signal rate.
    ///
    /// Panics if the sample rate is not supported.
    pub fn expect_branch(
        &self,
        sample_rate: usize,
    ) -> &SileroVad {
        self.try_branch(sample_rate).ok_or_panic()
    }
}
