//! # Voice-activity models: per-chunk speech probabilities for the driver.

use std::fmt::Debug;

use burn::{
    Tensor,
    module::Module,
    prelude::TensorData,
};
use dyn_clone::DynClone;

use crate::kits::speech::vad::silero_vad::{
    SileroVad,
    SileroVadContext,
    SileroVadContextConfig,
    SileroVadMeta,
};

/// A voice-activity model: the probability that each chunk of audio holds
/// speech, which a
/// [`VoiceActivityFilter`](crate::kits::speech::asr::driver::VoiceActivityFilter)
/// turns into speech regions.
///
/// Shared and immutable, like the driver it is attached to; each stream
/// opens its own [`VoiceActivityStream`] for its state.
pub trait VoiceActivityModel: Debug + Send + Sync {
    /// The sample rate it runs at, in Hz.
    fn sample_rate(&self) -> usize;

    /// The samples in one chunk: one probability per chunk.
    fn chunk_size(&self) -> usize;

    /// Opens a stream.
    fn new_stream(&self) -> Box<dyn VoiceActivityStream>;
}

/// One stream's voice-activity state.
///
/// `Clone`, through [`DynClone`], because the stream context that holds one
/// boxed is.
pub trait VoiceActivityStream: Debug + Send + Sync + DynClone {
    /// The speech probability of each chunk of `samples`, in order.
    ///
    /// # Arguments
    /// * `samples` - a whole number of chunks, following the samples of the
    ///   previous call.
    fn probabilities(
        &mut self,
        samples: Vec<f32>,
    ) -> Vec<f32>;
}

dyn_clone::clone_trait_object!(<> VoiceActivityStream);

impl VoiceActivityModel for SileroVad {
    fn sample_rate(&self) -> usize {
        SileroVadMeta::sample_rate(self)
    }

    fn chunk_size(&self) -> usize {
        SileroVadMeta::chunk_size(self)
    }

    fn new_stream(&self) -> Box<dyn VoiceActivityStream> {
        let device = self.devices()[0].clone();
        let context =
            SileroVadContextConfig::new(SileroVadMeta::sample_rate(self)).init(self, &device);
        Box::new(SileroVadStream {
            model: self.clone(),
            context: Some(context),
        })
    }
}

/// Silero's stream: the model, shared, and its recurrent state.
#[derive(Clone, Debug)]
struct SileroVadStream {
    model: SileroVad,

    /// `None` only while a step is in flight.
    context: Option<SileroVadContext>,
}

impl VoiceActivityStream for SileroVadStream {
    fn probabilities(
        &mut self,
        samples: Vec<f32>,
    ) -> Vec<f32> {
        let chunk = SileroVadMeta::chunk_size(&self.model);
        let steps = samples.len() / chunk;
        if steps == 0 {
            return Vec::new();
        }
        let device = self.model.devices()[0].clone();
        // The chunks in the model's own dtype, whatever the device's default
        // float.
        let dtype = crate::burner::module::InputDTypes::preferred_dtype(&self.model);
        let chunks: Tensor<3> = Tensor::from_data(
            TensorData::new(samples, [steps, 1, chunk]),
            (&device, burn::tensor::DType::from(dtype)),
        );

        let context = self.context.take().expect("present between steps");
        let (probs, context) = self.model.context_forward_sequence(chunks, context);
        self.context = Some(context);

        probs
            .to_data()
            .try_into_vec_as::<f32>()
            .expect("probabilities read back as f32")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        burner::module::ModuleInit,
        kits::speech::vad::silero_vad::SileroVadSignalConfig,
        support::{
            audio::load_audio_mono_sr,
            testing::cpu_device,
        },
    };

    /// Streamed in two calls, Silero's probabilities are those of one call
    /// over the same chunks. A property of the stream, not of the weights,
    /// so an untrained model shows it; `silero-model-validation` checks
    /// upstream's.
    #[test]
    fn test_silero_streams_as_one_call() {
        let device = cpu_device();
        device.seed(7);
        let model: SileroVad = SileroVadSignalConfig::standard_16khz().init(&device);
        let chunk = VoiceActivityModel::chunk_size(&model);
        let path = bunsen_test_audio::JFK_MOON_4S.path();
        let mut samples = load_audio_mono_sr(&path, 16_000).unwrap();
        samples.truncate(samples.len() / chunk * chunk);

        let whole = model.new_stream().probabilities(samples.clone());
        assert_eq!(whole.len(), samples.len() / chunk);

        let mut stream = model.new_stream();
        let cut = 37 * chunk;
        let mut parts = stream.probabilities(samples[..cut].to_vec());
        parts.extend(stream.probabilities(samples[cut..].to_vec()));

        assert_eq!(parts.len(), whole.len());
        for (a, b) in parts.iter().zip(&whole) {
            assert!((a - b).abs() < 1e-5, "{a} != {b}");
        }
    }

    /// A half model streams: the chunks go up in the model's dtype, not the
    /// device's default float (the CPU backend refuses a mixed op), and the
    /// probabilities read back as f32.
    #[test]
    fn test_silero_streams_in_its_own_dtype() {
        use burn::tensor::DType;

        use crate::burner::module::DTypeMapper;

        let device = cpu_device();
        device.seed(7);
        let model: SileroVad = SileroVadSignalConfig::standard_16khz().init(&device);
        let half = model.map(&mut DTypeMapper::new(DType::BF16));
        let chunk = VoiceActivityModel::chunk_size(&half);
        let path = bunsen_test_audio::JFK_MOON_4S.path();
        let mut samples = load_audio_mono_sr(&path, 16_000).unwrap();
        samples.truncate(8 * chunk);

        let probs = half.new_stream().probabilities(samples);
        assert_eq!(probs.len(), 8);
        assert!(probs.iter().all(|p| (0.0..=1.0).contains(p)), "{probs:?}");
    }
}
