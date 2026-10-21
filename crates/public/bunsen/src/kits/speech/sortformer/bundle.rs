//! A loaded Sortformer, with what it needs to diarize.
use std::sync::Arc;

use burn::tensor::Device;

use super::{
    SortformerFamily,
    SortformerLatency,
    SortformerStream,
    SortformerStreamingConfig,
    blocks::{
        Sortformer,
        SortformerMeta,
    },
};
use crate::{
    burner::module::ModuleInit,
    errors::{
        BunsenResult,
        ConstraintError,
        Rule,
    },
    kits::speech::{
        diarization::{
            SpeakerActivity,
            SpeakerActivityModel,
            SpeakerActivityStream,
            SpeakerTurn,
            TurnFilterConfig,
        },
        nemo::{
            NemoMelFrontEnd,
            NemoMelFrontEndConfig,
            NemoMelFrontEndMeta,
        },
    },
};

/// A Sortformer model and the front end it was trained behind: what a
/// pretrained loads to, and everything diarization needs.
///
/// [`diarizer`](Self::diarizer) makes it a [`SpeakerActivityModel`] under a
/// streaming config (its family's [`preset`](Self::preset), say), which a
/// stream driver attaches with
/// [`with_speakers`](crate::kits::speech::asr_driver::AsrStreamDriver::with_speakers);
/// [`diarize`](Self::diarize) runs a whole recording.
///
/// [`default_sortformer_factory`](super::pretrained::default_sortformer_factory)
/// builds one from a name.
#[derive(Debug)]
pub struct SortformerBundle {
    /// The model, shared by every stream.
    pub model: Arc<Sortformer>,

    /// The mel front end the model was trained behind.
    pub front_end: NemoMelFrontEnd,

    /// The family the model is of: whose streaming presets it runs under.
    pub family: SortformerFamily,
}

impl SortformerBundle {
    /// A bundle of `model`, of `family`, behind the front end `front_end`
    /// configures.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
    /// [`ConstraintError`] cause, when the front end's mel count is not the
    /// model's; as the front end's construction.
    pub fn new(
        model: Sortformer,
        family: SortformerFamily,
        front_end: &NemoMelFrontEndConfig,
        device: &Device,
    ) -> BunsenResult<Self> {
        if front_end.n_mels() != model.n_mels() {
            return Err(ConstraintError::new(
                "SortformerBundle",
                "front_end",
                Rule::Relation {
                    lhs: ("front_end.n_mels".into(), front_end.n_mels().to_string()),
                    op: "==",
                    rhs: ("model.n_mels()".into(), model.n_mels().to_string()),
                },
            )
            .into());
        }
        Ok(Self {
            model: Arc::new(model),
            front_end: front_end.try_init(device)?,
            family,
        })
    }

    /// The family's streaming preset at `latency`.
    ///
    /// # Errors
    /// As [`SortformerFamily::preset`].
    pub fn preset(
        &self,
        latency: SortformerLatency,
    ) -> BunsenResult<SortformerStreamingConfig> {
        self.family.preset(latency)
    }

    /// The sample rate, in Hz.
    pub fn sample_rate(&self) -> usize {
        self.front_end.sample_rate()
    }

    /// The model as a diarizer, streaming under `streaming`.
    ///
    /// # Errors
    /// As [`SortformerStreamingConfig::check`] against the model.
    pub fn diarizer(
        &self,
        streaming: SortformerStreamingConfig,
    ) -> BunsenResult<SortformerDiarizer> {
        streaming.check(self.model.n_speakers())?;
        Ok(SortformerDiarizer {
            model: self.model.clone(),
            front_end: self.front_end.clone(),
            streaming,
        })
    }

    /// The speaker turns of a whole recording: its samples streamed through
    /// [`diarizer`](Self::diarizer) under `streaming`, and the activity
    /// through `filter`.
    ///
    /// # Arguments
    /// * `samples` - mono, at [`sample_rate`](Self::sample_rate).
    ///
    /// # Errors
    /// As the model.
    pub fn diarize(
        &self,
        samples: &[f32],
        streaming: SortformerStreamingConfig,
        filter: &TurnFilterConfig,
    ) -> BunsenResult<Vec<SpeakerTurn>> {
        let diarizer = self.diarizer(streaming)?;
        let activity = diarizer.activity(samples)?;
        let frame_seconds = activity.frame_samples as f64 / self.sample_rate() as f64;
        Ok(filter.turns(&activity.probs, activity.n_speakers, frame_seconds))
    }
}

impl std::fmt::Display for SortformerBundle {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        let m = &self.model;
        write!(
            f,
            "{}: {} speakers, {} mels, {}-wide encoder, {} ms frames",
            self.family,
            m.n_speakers(),
            m.n_mels(),
            m.d_model(),
            1000 * self.front_end.hop() * m.subsampling_factor()
                / m.output_upsampling()
                / self.sample_rate()
        )
    }
}

/// A Sortformer under a streaming preset: a [`SpeakerActivityModel`], each
/// stream a [`SortformerStream`].
#[derive(Clone, Debug)]
pub struct SortformerDiarizer {
    model: Arc<Sortformer>,
    front_end: NemoMelFrontEnd,
    streaming: SortformerStreamingConfig,
}

impl SortformerDiarizer {
    /// The streaming preset.
    pub fn streaming(&self) -> &SortformerStreamingConfig {
        &self.streaming
    }

    /// Opens a stream.
    ///
    /// # Errors
    /// As [`SortformerStream::new`].
    pub fn stream(&self) -> BunsenResult<SortformerStream> {
        SortformerStream::new(
            self.model.clone(),
            self.front_end.clone(),
            self.streaming.clone(),
        )
    }

    /// The activity of a whole recording, streamed.
    ///
    /// # Errors
    /// As the model.
    pub fn activity(
        &self,
        samples: &[f32],
    ) -> BunsenResult<SpeakerActivity> {
        let mut stream = self.stream()?;
        let mut activity = stream.push(samples)?;
        activity.append(stream.finish()?)?;
        Ok(activity)
    }
}

impl SpeakerActivityModel for SortformerDiarizer {
    fn sample_rate(&self) -> usize {
        self.front_end.sample_rate()
    }

    fn frame_samples(&self) -> usize {
        self.front_end.hop() * self.model.subsampling_factor() / self.model.output_upsampling()
    }

    fn max_speakers(&self) -> usize {
        self.model.n_speakers()
    }

    fn new_stream(&self) -> BunsenResult<Box<dyn SpeakerActivityStream>> {
        Ok(Box::new(self.stream()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        kits::speech::sortformer::NemotronDiarizationConfig,
        support::testing::cpu_device,
    };

    fn bundle() -> SortformerBundle {
        let model: Sortformer = NemotronDiarizationConfig::new()
            .with_n_mels(24)
            .with_d_model(16)
            .with_n_layers(1)
            .with_n_heads(2)
            .with_d_hidden(8)
            .with_n_speakers(3)
            .init(&cpu_device());
        SortformerBundle::new(
            model,
            SortformerFamily::Nemotron,
            &NemoMelFrontEndConfig::nemotron_diarization().with_n_mels(24),
            &cpu_device(),
        )
        .unwrap()
    }

    #[test]
    fn test_diarizer_is_a_speaker_activity_model() {
        let bundle = bundle();
        let diarizer = bundle
            .diarizer(SortformerStreamingConfig::new(3, 1, 6, 12, 4))
            .unwrap();
        assert_eq!(diarizer.sample_rate(), 16_000);
        assert_eq!(diarizer.frame_samples(), 160);
        assert_eq!(diarizer.max_speakers(), 3);
        assert!(
            bundle
                .diarizer(SortformerStreamingConfig::new(3, 1, 6, 5, 4))
                .is_err()
        );
        assert!(bundle.to_string().contains("3 speakers"));
        assert!(bundle.to_string().contains("10 ms"));

        let samples = vec![0.0f32; 16_000];
        let mut stream = diarizer.new_stream().unwrap();
        let mut activity = stream.push(&samples).unwrap();
        activity.append(stream.finish().unwrap()).unwrap();
        assert_eq!(activity.frames(), 100);
        assert_eq!(
            activity.frames(),
            diarizer.activity(&samples).unwrap().frames()
        );

        let turns = bundle
            .diarize(
                &samples,
                SortformerStreamingConfig::new(3, 1, 6, 12, 4),
                &TurnFilterConfig::none(),
            )
            .unwrap();
        assert!(turns.iter().all(|t| t.end <= 1.0 + 1e-9));
    }

    /// The presets are the family's; the front end must read the model's
    /// mels.
    #[test]
    fn test_family_and_front_end() {
        let bundle = bundle();
        assert!(bundle.to_string().starts_with("Nemotron-3-Diarization: "));
        assert_eq!(
            bundle.preset(SortformerLatency::Low).unwrap(),
            SortformerStreamingConfig::nemotron_low_latency()
        );
        assert!(bundle.preset(SortformerLatency::High).is_err());

        let model = Arc::try_unwrap(bundle.model).unwrap();
        let err = SortformerBundle::new(
            model,
            SortformerFamily::Nemotron,
            &NemoMelFrontEndConfig::nemotron_diarization(),
            &cpu_device(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("front_end.n_mels"), "{err}");
    }
}
