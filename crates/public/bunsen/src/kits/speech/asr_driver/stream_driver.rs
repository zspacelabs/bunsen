//! # The stream driver: the shared, immutable half of a transcription.

use alloc::{
    string::ToString,
    sync::Arc,
};

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
        ConstraintError,
        Rule,
    },
    kits::speech::asr_driver::{
        AsrEngine,
        AsrStreamContext,
        CommitRule,
        EmissionPolicy,
        LevelPolicy,
        StreamClock,
        VoiceActivityFilterConfig,
        VoiceActivityModel,
    },
};

/// The shared, immutable half of a transcription: an [`AsrEngine`], the
/// [`EmissionPolicy`], and the voice-activity model when one is attached.
///
/// Opens [`AsrStreamContext`]s, one per stream, each holding a clone of the
/// driver; the clone is cheap when the engine's model is shared. So one
/// driver serves any number of streams in one process, and
/// [`advance_ready`](crate::kits::speech::asr_driver::advance_ready) batches
/// their decodes.
#[derive(Clone, Debug)]
pub struct AsrStreamDriver<E: AsrEngine> {
    engine: E,
    emission: EmissionPolicy,
    vad: Option<Arc<dyn VoiceActivityModel>>,
    va_filter: Option<VoiceActivityFilterConfig>,
}

impl<E: AsrEngine> AsrStreamDriver<E> {
    /// A driver of `engine` under `emission`.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) when the policy
    /// would never decode (neither `window_full` nor `endpoint`), drafts on
    /// an interval without `endpoint` (whose gate is what says speech is in
    /// progress), or drafts on a zero interval (with a [`ConstraintError`]
    /// cause); [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported)
    /// for [`CommitRule::Agreement`].
    pub fn new(
        engine: E,
        emission: EmissionPolicy,
    ) -> BunsenResult<Self> {
        Self::check_emission(&emission)?;
        Ok(Self {
            engine,
            emission,
            vad: None,
            va_filter: None,
        })
    }

    fn check_emission(emission: &EmissionPolicy) -> BunsenResult<()> {
        let triggers = &emission.triggers;
        if !triggers.window_full && !triggers.endpoint {
            return Err(BunsenError::illegal(
                "EmissionPolicy.triggers: with neither the window_full nor the endpoint \
                 trigger nothing would ever decode",
            ));
        }
        if triggers.interval.is_some_and(|i| i.is_zero()) {
            return Err(BunsenError::from(ConstraintError::zero_or_empty(
                "EmissionPolicy",
                "triggers.interval",
            ))
            .with_details("an interval of zero would draft on every push"));
        }
        if triggers.interval.is_some() && !triggers.endpoint {
            return Err(BunsenError::illegal(
                "EmissionPolicy.triggers: the interval trigger drafts only while speech is in \
                 progress, which only the endpoint trigger's voice-activity gate tracks; turn \
                 endpoint on as well",
            ));
        }
        if let CommitRule::Agreement { .. } = emission.commit {
            return Err(BunsenError::unsupported(
                "EmissionPolicy.commit: the Agreement commit rule is not implemented; use \
                 Complete or LastTimestamp",
            ));
        }
        Ok(())
    }

    /// Attaches a voice-activity model and the filter that turns its
    /// probabilities into regions.
    ///
    /// Needed by any emission policy with the `endpoint` trigger; ignored by
    /// one without.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
    /// [`ConstraintError`] cause, if the model or the filter runs at a rate
    /// other than the engine's, or the filter's chunk is not the model's.
    pub fn with_vad(
        mut self,
        vad: Arc<dyn VoiceActivityModel>,
        filter: VoiceActivityFilterConfig,
    ) -> BunsenResult<Self> {
        self.check_vad(vad.as_ref(), &filter)?;
        self.vad = Some(vad);
        self.va_filter = Some(filter);
        Ok(self)
    }

    /// The model and the filter must agree with the engine on the rate, and
    /// with each other on the chunk.
    fn check_vad(
        &self,
        vad: &dyn VoiceActivityModel,
        filter: &VoiceActivityFilterConfig,
    ) -> BunsenResult<()> {
        let rate = self.sample_rate();
        if vad.sample_rate() != rate {
            return Err(must_equal(
                ("vad.sample_rate()", vad.sample_rate()),
                ("the driver's sample_rate()", rate),
            ));
        }
        if filter.sample_rate != rate {
            return Err(must_equal(
                ("filter.sample_rate", filter.sample_rate),
                ("the driver's sample_rate()", rate),
            ));
        }
        if filter.samples_per_chunk != vad.chunk_size() {
            return Err(must_equal(
                ("filter.samples_per_chunk", filter.samples_per_chunk),
                ("vad.chunk_size()", vad.chunk_size()),
            ));
        }
        Ok(())
    }

    /// The engine.
    pub fn engine(&self) -> &E {
        &self.engine
    }

    /// The driver with its engine replaced by `f` of it: for a builder of a
    /// concrete engine's settings.
    pub fn map_engine(
        mut self,
        f: impl FnOnce(E) -> E,
    ) -> Self {
        self.engine = f(self.engine);
        self
    }

    /// The emission policy.
    pub fn emission(&self) -> &EmissionPolicy {
        &self.emission
    }

    /// The voice-activity model, if one was attached.
    pub fn vad(&self) -> Option<&Arc<dyn VoiceActivityModel>> {
        self.vad.as_ref()
    }

    /// The voice-activity filter's config, if a model was attached.
    pub fn va_filter_config(&self) -> Option<&VoiceActivityFilterConfig> {
        self.va_filter.as_ref()
    }

    /// The engine's sample rate, in Hz. The stream's clock must run at it
    /// too.
    pub fn sample_rate(&self) -> usize {
        self.engine.sample_rate()
    }

    /// The draft interval in samples of media time, when the policy has
    /// one.
    pub fn interval_samples(&self) -> Option<usize> {
        self.emission
            .triggers
            .interval
            .map(|i| (i.as_secs_f64() * self.sample_rate() as f64).round() as usize)
    }

    /// The grid speech regions are snapped onto, in samples.
    pub fn grid_samples(&self) -> usize {
        self.engine.grid_frames() * self.engine.hop()
    }

    /// Opens a stream.
    ///
    /// # Arguments
    /// * `clock` - the stream's sample-to-time map. A bare stream gets
    ///   [`StreamClock::uniform`] at [`sample_rate`](Self::sample_rate).
    /// * `level` - which frames each window's level covers.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) if the clock
    /// does not run at the engine's [`sample_rate`](Self::sample_rate) (with
    /// a [`ConstraintError`] cause), or if the emission policy wants
    /// endpoints and no VAD was attached; as the engine's
    /// [`new_features`](AsrEngine::new_features).
    pub fn new_context(
        &self,
        clock: StreamClock,
        level: LevelPolicy,
    ) -> BunsenResult<AsrStreamContext<E>> {
        if clock.rate() != self.sample_rate() {
            return Err(ConstraintError::new(
                "AsrStreamDriver::new_context",
                "",
                Rule::Relation {
                    lhs: ("clock.rate()".into(), clock.rate().to_string()),
                    op: "==",
                    rhs: (
                        "the engine's sample_rate()".into(),
                        self.sample_rate().to_string(),
                    ),
                },
            )
            .into());
        }
        if self.emission.triggers.endpoint && self.vad.is_none() {
            return Err(BunsenError::illegal(
                "the endpoint trigger needs a voice-activity model; attach one with with_vad",
            ));
        }
        AsrStreamContext::init(self.clone(), clock, level)
    }
}

/// `lhs == rhs`, as [`AsrStreamDriver::with_vad`] refuses it.
fn must_equal(
    lhs: (&str, usize),
    rhs: (&str, usize),
) -> BunsenError {
    ConstraintError::new(
        "AsrStreamDriver::with_vad",
        "",
        Rule::Relation {
            lhs: (lhs.0.into(), lhs.1.to_string()),
            op: "==",
            rhs: (rhs.0.into(), rhs.1.to_string()),
        },
    )
    .into()
}
