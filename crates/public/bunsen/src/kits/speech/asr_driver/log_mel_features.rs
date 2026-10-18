//! # Streamed log-mels: the front end Whisper-family engines share.

use alloc::{
    vec,
    vec::Vec,
};

use burn::{
    Tensor,
    prelude::TensorData,
    tensor::Device,
};

use crate::{
    errors::{
        BunsenResult,
        ResultContext,
    },
    kits::speech::asr_driver::StreamingFeatures,
    ops::signal::perceptive_audio::{
        PerceptiveAudioConversionContext,
        PerceptiveAudioConverter,
        PerceptiveAudioConverterMeta,
    },
};

/// A streaming log-mel front end: whole hops of samples through a
/// [`PerceptiveAudioConverter`]'s context.
///
/// The first chunk must be long enough to produce a frame under the start
/// padding, so it waits for a window's worth of samples; after that a hop at
/// a time is enough. Finishing pads whatever remains with silence up to that
/// minimum, as a one-shot extractor pads a clip; a stream that never received
/// a sample finishes with nothing.
///
/// The frames are the converter's, `[1, frames, n_mels]`, before any
/// packaging: an engine floors and scales a window against its level when it
/// decodes it.
#[derive(Clone, Debug)]
pub struct LogMelFeatures {
    converter: PerceptiveAudioConverter,

    /// The converter's streaming state; `None` once finished.
    context: Option<PerceptiveAudioConversionContext>,

    /// Samples not yet a whole hop.
    staging: Vec<f32>,

    device: Device,
}

impl LogMelFeatures {
    /// A stream through `converter`, whose frames are built on `device`.
    pub fn new(
        converter: PerceptiveAudioConverter,
        device: Device,
    ) -> Self {
        Self {
            context: Some(converter.new_context(1)),
            converter,
            staging: Vec::new(),
            device,
        }
    }

    /// Moves whole hops from staging into the converter.
    fn drain(
        &mut self,
        flushing: bool,
    ) -> BunsenResult<Option<Tensor<3>>> {
        let (hop, n_fft) = (self.converter.hop(), self.converter.n_fft());
        let first = self
            .context
            .as_ref()
            .is_some_and(|ctx| ctx.carry().is_none());
        let minimum = if first {
            n_fft.div_ceil(hop) * hop
        } else {
            hop
        };

        let mut whole = self.staging.len() / hop * hop;
        if flushing {
            if self.staging.is_empty() {
                return Ok(None);
            }
            whole = self.staging.len().div_ceil(hop).max(minimum / hop) * hop;
            self.staging.resize(whole, 0.0);
        } else if whole < minimum {
            return Ok(None);
        }

        // `from_data` converts to the device's float dtype, so the samples
        // go up as they are.
        let chunk: Vec<f32> = self.staging.drain(..whole).collect();
        let waves: Tensor<2> = Tensor::from_data(TensorData::new(chunk, [1, whole]), &self.device);

        let ctx = self
            .context
            .take()
            .expect("the front end is open until the input ends");
        // The context built the chunk to the front end's grid, so a refusal
        // is the context's own fault.
        let (frames, ctx) = ctx.transform(waves).as_internal()?;
        self.context = Some(ctx);
        Ok(Some(frames))
    }
}

impl StreamingFeatures for LogMelFeatures {
    fn push(
        &mut self,
        samples: &[f32],
    ) -> BunsenResult<Option<Tensor<3>>> {
        self.staging.extend_from_slice(samples);
        self.drain(false)
    }

    fn finish(&mut self) -> BunsenResult<Option<Tensor<3>>> {
        if self.context.is_none() {
            return Ok(None);
        }
        let drained = self.drain(true)?;
        let tail = self.context.take().and_then(|mel| mel.finish());
        Ok(match (drained, tail) {
            (Some(a), Some(b)) => Some(Tensor::cat(vec![a, b], 1)),
            (a, b) => a.or(b),
        })
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Tolerance;

    use super::*;
    use crate::{
        kits::speech::whisper::blocks::WhisperFrontEndConfig,
        support::testing::cpu_device,
    };

    fn features() -> LogMelFeatures {
        let device = cpu_device();
        let converter = WhisperFrontEndConfig::new()
            .try_init_audio_converter(80, &device)
            .unwrap();
        LogMelFeatures::new(converter, device)
    }

    /// Every frame `push` and `finish` return, joined.
    fn run(
        mut features: LogMelFeatures,
        samples: &[f32],
        piece: usize,
    ) -> Tensor<3> {
        let mut out = Vec::new();
        for chunk in samples.chunks(piece) {
            out.extend(features.push(chunk).unwrap());
        }
        out.extend(features.finish().unwrap());
        Tensor::cat(out, 1)
    }

    /// Pushed in pieces of any size, a stream yields the frames of one push.
    #[test]
    fn test_pieces_yield_the_frames_of_one_push() {
        let samples: Vec<f32> = (0..9_000).map(|i| (i as f32 * 0.03).sin() * 0.4).collect();
        let whole = run(features(), &samples, samples.len());
        let [_, frames, n_mels] = whole.dims();
        assert_eq!(n_mels, 80);
        // Finishing pads to a whole hop; the centred STFT adds one frame.
        assert_eq!(frames, 9_000_usize.div_ceil(160) + 1);

        for piece in [1, 159, 160, 401, 3_000] {
            let parts = run(features(), &samples, piece);
            assert_eq!(parts.dims(), whole.dims(), "piece {piece}");
            parts
                .to_data()
                .assert_approx_eq::<f32>(&whole.to_data(), Tolerance::default());
        }
    }

    /// A stream that never received a sample finishes with nothing, and a
    /// finished one stays finished.
    #[test]
    fn test_empty_and_finished_streams() {
        let mut empty = features();
        assert!(empty.push(&[]).unwrap().is_none());
        assert!(empty.finish().unwrap().is_none());

        let mut short = features();
        assert!(short.push(&[0.1; 10]).unwrap().is_none());
        assert!(short.finish().unwrap().is_some());
        assert!(short.finish().unwrap().is_none());
    }
}
