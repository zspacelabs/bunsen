use burn::{
    Tensor,
    config::Config,
    module::Module,
    prelude::Backend,
};

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
        ConstraintError,
        WithOkOrPanic,
    },
    kits::speech::silero_vad::SileroVad,
};

/// Common methods for [`SileroVadContextConfig`] and [`SileroVadContext`].
///
/// Implemented by:
/// * [`SileroVadContextConfig`]
/// * [`SileroVadContext`]
pub trait SileroVadContextMeta {
    /// The sample rate (in Hz) this context expects, e.g. `16000`.
    fn sample_rate(&self) -> usize;

    /// The batch size.
    fn batch_size(&self) -> usize;

    /// The size of the previous sequence window to preserve.
    fn context_size(&self) -> usize;
}

/// Config for [`SileroVadContext`]: one stream's continuation state.
///
/// [`init`](Self::init) opens a context against a [`SileroVad`] at this
/// config's rate; the model's
/// [`context_forward`](SileroVad::context_forward) consumes it and returns
/// the next one.
#[derive(Config, Debug)]
pub struct SileroVadContextConfig {
    /// The sample rate (in Hz) this context expects, e.g. `16000`.
    pub sample_rate: usize,

    /// The batch size.
    #[config(default = "1")]
    pub batch_size: usize,

    /// The size of the previous sequence window to preserve: the tail of
    /// each chunk that the next is prefixed with, in samples; above 0.
    ///
    /// Defaults to [`default_context_size`](Self::default_context_size) of
    /// the rate: upstream's 64 samples at 16 kHz and 32 at 8 kHz, and 0
    /// (refused) below 250 Hz.
    #[config(default = "Self::default_context_size(sample_rate)")]
    pub context_size: usize,
}

impl SileroVadContextMeta for SileroVadContextConfig {
    fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn context_size(&self) -> usize {
        self.context_size
    }
}

/// Context and state for sequential mode for [`SileroVad`].
///
/// What a stream carries between chunks, held by the caller rather than
/// the model: the tail of the last chunk, which the next is prefixed with,
/// and the recurrent state. So one loaded model serves any number of
/// streams, each with a context of its own; the Whisper stream driver
/// keeps one per stream it gates.
///
/// Built by [`SileroVadContextConfig`], or by
/// [`SileroVad::init_context`]; threaded through
/// [`SileroVad::context_forward`] and
/// [`SileroVad::context_forward_sequence`], each of which takes one and
/// returns the next. Implements [`SileroVadContextMeta`].
#[derive(Module, Debug)]
pub struct SileroVadContext<B: Backend> {
    /// The sample rate of the context.
    pub sample_rate: usize,

    /// The preceding input context, `[batch, context_size]`; at least one
    /// sample wide.
    pub context: Tensor<B, 2>,

    /// The current input state.
    pub state: Tensor<B, 3>,
}

impl<B: Backend> SileroVadContextMeta for SileroVadContext<B> {
    fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    fn batch_size(&self) -> usize {
        self.context.dims()[0]
    }

    fn context_size(&self) -> usize {
        self.context.dims()[1]
    }
}

impl SileroVadContextConfig {
    /// Upstream's context size for a rate: 4 ms of audio, which is 64
    /// samples at 16 kHz and 32 at 8 kHz, as silero-vad's `OnnxWrapper`
    /// prefixes each chunk (`context_size = 64 if sr == 16000 else 32`).
    ///
    /// Below 250 Hz this is 0, which [`try_init`](Self::try_init) refuses.
    pub fn default_context_size(sample_rate: usize) -> usize {
        sample_rate / 250
    }

    /// Opens a context against `vad`: a zero tail of `context_size` samples
    /// per row, and a zeroed recurrent state, at the model's rate.
    ///
    /// The fallible half of a `try_x` / `x` pair; the panicking half is
    /// [`init`](Self::init).
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
    /// [`ConstraintError`] cause, when `context_size` is 0, as
    /// [`default_context_size`](Self::default_context_size) gives below
    /// 250 Hz. Upstream always prefixes a tail; for no context at all, run
    /// the model's bare [`forward`](SileroVad::forward) instead.
    pub fn try_init<B: Backend>(
        &self,
        vad: &SileroVad<B>,
        device: &B::Device,
    ) -> BunsenResult<SileroVadContext<B>> {
        if self.context_size == 0 {
            return Err(BunsenError::from(ConstraintError::zero_or_empty(
                "SileroVadContextConfig",
                "context_size",
            ))
            .with_details(format!(
                "the default, sample_rate / 250, is 0 below 250 Hz; sample_rate = {}",
                self.sample_rate,
            )));
        }
        Ok(vad.init_context(self.batch_size, self.context_size, device))
    }

    /// Opens a context against `vad`; the panicking twin of
    /// [`try_init`](Self::try_init).
    ///
    /// # Panics
    ///
    /// When `context_size` is 0.
    pub fn init<B: Backend>(
        &self,
        vad: &SileroVad<B>,
        device: &B::Device,
    ) -> SileroVadContext<B> {
        self.try_init(vad, device).ok_or_panic()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        burner::module::ModuleInit,
        errors::{
            BunsenErrorKind,
            testing::ErrorMatcher,
        },
        kits::speech::silero_vad::SileroVadSignalConfig,
        support::testing::{
            PerformanceBackend,
            default_device,
        },
    };

    type B = PerformanceBackend;

    /// The tail each chunk is prefixed with is upstream's: 64 samples at
    /// 16 kHz and 32 at 8 kHz (`OnnxWrapper` in silero-vad's
    /// `utils_vad.py`: `context_size = 64 if sr == 16000 else 32`).
    #[test]
    fn test_default_context_size_is_upstreams() {
        assert_eq!(SileroVadContextConfig::new(16000).context_size(), 64);
        assert_eq!(SileroVadContextConfig::new(8000).context_size(), 32);
    }

    /// A zero `context_size`, set or defaulted below 250 Hz, is an error
    /// from `try_init`, not a context whose tail slice takes the whole
    /// chunk. At 250 Hz the default is 1 sample, which opens.
    #[test]
    #[serial_test::serial]
    fn test_try_init_refuses_a_zero_context() {
        let device = default_device();
        // A small model at each rate: the context takes the model's rate.
        let vad_at = |rate: usize| -> SileroVad<B> {
            SileroVadSignalConfig::new(rate, 33)
                .with_d_hidden(32)
                .with_d_bottleneck(16)
                .init(&device)
        };

        for cfg in [
            SileroVadContextConfig::new(16000).with_context_size(0),
            SileroVadContextConfig::new(249),
            SileroVadContextConfig::new(0),
        ] {
            assert_eq!(cfg.context_size(), 0);
            ErrorMatcher::kind(BunsenErrorKind::Illegal)
                .has_cause::<ConstraintError>()
                .assert_err(&cfg.try_init(&vad_at(cfg.sample_rate), &device));
        }

        let cfg = SileroVadContextConfig::new(250);
        assert_eq!(cfg.context_size(), 1);
        let context = cfg.try_init(&vad_at(250), &device).unwrap();
        assert_eq!(context.context.dims(), [1, 1]);
    }
}
