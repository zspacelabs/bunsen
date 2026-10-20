//! A Sortformer, streamed: chunks, look-ahead, and the speaker cache.
use std::sync::Arc;

use burn::{
    Tensor,
    module::Module,
};

use super::{
    SortformerStreamingConfig,
    SpeakerCache,
    blocks::{
        Sortformer,
        SortformerMeta,
    },
};
use crate::{
    contracts::unpack_shape_contract,
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::{
        asr_driver::StreamingFeatures,
        diarization::{
            SpeakerActivity,
            SpeakerActivityStream,
        },
        nemo::{
            NemoMelFeatures,
            NemoMelFrontEnd,
            NemoMelFrontEndMeta,
        },
    },
};

/// A Sortformer over a stream of mel frames: the model run a chunk at a time
/// with its look-ahead, against a [`SpeakerCache`], `NeMo`'s streaming.
///
/// Mel frames go in through [`push`](Self::push) in pieces of any size; each
/// whole stack of `subsampling_factor` becomes an encoder frame, and once a
/// chunk and its look-ahead are pending a step runs and its chunk's frames
/// come out, final. [`finish`](Self::finish) runs what is left: the last
/// stack zero-padded, the last chunks with what look-ahead there is. However
/// the frames are split across calls, the output is the same.
///
/// The output is at the model's output rate: one frame per
/// `subsampling_factor / output_upsampling` mel frames.
#[derive(Clone, Debug)]
pub struct SortformerMelStream {
    model: Arc<Sortformer>,

    /// The silence slots' fill: the model's learned silence embedding.
    silence: Tensor<1>,

    cache: SpeakerCache,

    /// Mel frames not yet stacked, `[frames, n_mels]`.
    mels: Option<Tensor<2>>,

    /// Encoder frames not yet stepped past, `[frames, d_model]`.
    pending: Option<Tensor<2>>,

    /// Mel frames taken in.
    mel_frames: usize,

    /// Output frames emitted.
    emitted: usize,

    finished: bool,
}

impl SortformerMelStream {
    /// A stream through `model` under `config`.
    ///
    /// # Errors
    /// As [`SortformerStreamingConfig::check`];
    /// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for a model
    /// without a learned silence embedding.
    pub fn new(
        model: Arc<Sortformer>,
        config: SortformerStreamingConfig,
    ) -> BunsenResult<Self> {
        let silence = model
            .silence
            .as_ref()
            .ok_or_else(|| {
                BunsenError::unsupported(
                    "a streamed Sortformer needs a learned silence embedding to fill its cache's \
                     silence slots",
                )
            })?
            .val();
        let cache = SpeakerCache::new(config, model.n_speakers())?;
        Ok(Self {
            model,
            silence,
            cache,
            mels: None,
            pending: None,
            mel_frames: 0,
            emitted: 0,
            finished: false,
        })
    }

    /// The model.
    pub fn model(&self) -> &Arc<Sortformer> {
        &self.model
    }

    /// The speaker cache.
    pub fn cache(&self) -> &SpeakerCache {
        &self.cache
    }

    /// The mel frames per output frame.
    pub fn mels_per_output(&self) -> usize {
        self.model.subsampling_factor() / self.model.output_upsampling()
    }

    /// Takes mel frames in, and returns the output frames they finalize.
    ///
    /// # Arguments
    /// * `mels` - `[frames, n_mels]`, following the frames of the previous
    ///   call.
    ///
    /// # Returns
    /// `frames * n_speakers` probabilities, frame-major.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) after
    /// [`finish`](Self::finish).
    pub fn push(
        &mut self,
        mels: Tensor<2>,
    ) -> BunsenResult<Vec<f32>> {
        if self.finished {
            return Err(BunsenError::illegal(
                "the stream has ended; nothing more can be fed",
            ));
        }
        let [frames] = unpack_shape_contract!(
            ["frames", "n_mels"],
            &mels.dims(),
            &["frames"],
            &[("n_mels", self.model.n_mels())],
        );
        if frames == 0 {
            return Ok(Vec::new());
        }
        self.mel_frames += frames;
        self.mels = cat_present(self.mels.take(), Some(mels));
        self.stack(false);

        let c = self.cache.config().clone();
        let mut out = Vec::new();
        while self.pending_frames() >= c.chunk_len + c.right_context {
            out.extend(self.step(c.chunk_len, c.right_context)?);
        }
        Ok(out)
    }

    /// Ends the input, and returns the output frames left: the last stack
    /// zero-padded, the last chunks run with what look-ahead there is, and
    /// the output cut at the last mel frame.
    ///
    /// Idempotent.
    ///
    /// # Errors
    /// As the model.
    pub fn finish(&mut self) -> BunsenResult<Vec<f32>> {
        if self.finished {
            return Ok(Vec::new());
        }
        self.finished = true;
        self.stack(true);

        let c = self.cache.config().clone();
        let mut out = Vec::new();
        while self.pending_frames() > 0 {
            let chunk = c.chunk_len.min(self.pending_frames());
            let look_ahead = c.right_context.min(self.pending_frames() - chunk);
            out.extend(self.step(chunk, look_ahead)?);
        }

        // The output ends with the last mel frame, not with its stack.
        let total = self.mel_frames.div_ceil(self.mels_per_output());
        let s = self.model.n_speakers();
        let over = self.emitted.saturating_sub(total);
        out.truncate(out.len() - over.min(out.len() / s) * s);
        self.emitted -= over;
        Ok(out)
    }

    /// Output frames emitted so far.
    pub fn emitted(&self) -> usize {
        self.emitted
    }

    fn pending_frames(&self) -> usize {
        self.pending.as_ref().map_or(0, |p| p.dims()[0])
    }

    /// Stacks the whole groups of mel frames into encoder frames; with
    /// `last`, the remainder too, zero-padded.
    fn stack(
        &mut self,
        last: bool,
    ) {
        let Some(mels) = self.mels.take() else {
            return;
        };
        let factor = self.model.subsampling_factor();
        let n = mels.dims()[0];
        let whole = if last { n } else { n / factor * factor };
        if whole < n {
            self.mels = Some(mels.clone().slice_dim(0, whole..n));
        }
        if whole == 0 {
            return;
        }
        let (embs, _) = self
            .model
            .pre_encode(mels.slice_dim(0, 0..whole).unsqueeze_dim(0), None);
        let embs = embs.squeeze_dim(0);
        self.pending = cat_present(self.pending.take(), Some(embs));
    }

    /// One step: the cache, the FIFO, a chunk of `chunk` encoder frames and
    /// `look_ahead` more; the chunk's output out, the chunk into the FIFO.
    fn step(
        &mut self,
        chunk: usize,
        look_ahead: usize,
    ) -> BunsenResult<Vec<f32>> {
        let pending = self.pending.take().expect("frames are pending");
        let n = pending.dims()[0];
        let input = pending.clone().slice_dim(0, 0..chunk + look_ahead);
        let chunk_embs = pending.clone().slice_dim(0, 0..chunk);
        self.pending = (chunk < n).then(|| pending.slice_dim(0, chunk..n));

        let cached = self.cache.cached();
        let n_cached = cached.as_ref().map_or(0, |c| c.dims()[0]);
        let input = cat_present(cached, Some(input)).expect("the chunk is present");
        let probs = self.model.forward_pre_encoded(input.unsqueeze_dim(0), None);

        let up = self.model.output_upsampling();
        let s = self.model.n_speakers();
        let out: Vec<f32> = probs
            .output
            .slice_dim(1, n_cached * up..(n_cached + chunk) * up)
            .into_data()
            .try_into_vec_as::<f32>()
            .expect("read back f32 probabilities");
        let step: Vec<f32> = probs
            .encoder_rate
            .into_data()
            .try_into_vec_as::<f32>()
            .expect("read back f32 probabilities");
        self.cache.update(&step, chunk_embs, &self.silence)?;
        self.emitted += out.len() / s;
        Ok(out)
    }
}

/// A Sortformer over a stream of samples: `NeMo`'s mel front end, streamed,
/// into a [`SortformerMelStream`], as a diarizer stream
/// ([`SpeakerActivityStream`]).
///
/// Its mel frames are the one-shot front end's, however the samples are
/// split, so its activity is the one a whole recording would give.
#[derive(Clone, Debug)]
pub struct SortformerStream {
    features: NemoMelFeatures,
    hop: usize,
    mels: SortformerMelStream,

    /// Samples taken in.
    samples: usize,

    /// Mel frames handed on.
    handed: usize,

    /// Mel frames from the front end that the samples so far do not yet
    /// show are valid, `[frames, n_mels]`.
    held: Option<Tensor<2>>,
}

impl SortformerStream {
    /// A stream through `front_end` and `model`, under `config`.
    ///
    /// # Errors
    /// As [`SortformerMelStream::new`];
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) if the front end's
    /// mel count is not the model's.
    pub fn new(
        model: Arc<Sortformer>,
        front_end: NemoMelFrontEnd,
        config: SortformerStreamingConfig,
    ) -> BunsenResult<Self> {
        if front_end.n_mels() != model.n_mels() {
            return Err(BunsenError::illegal(format!(
                "SortformerStream: the front end makes {} mels, the model reads {}",
                front_end.n_mels(),
                model.n_mels()
            )));
        }
        let device = model.devices()[0].clone();
        let hop = front_end.hop();
        Ok(Self {
            features: NemoMelFeatures::new(front_end, device),
            hop,
            mels: SortformerMelStream::new(model, config)?,
            samples: 0,
            handed: 0,
            held: None,
        })
    }

    /// The mel-level stream underneath.
    pub fn mel_stream(&self) -> &SortformerMelStream {
        &self.mels
    }

    /// The samples in one output frame.
    pub fn frame_samples(&self) -> usize {
        self.hop * self.mels.mels_per_output()
    }

    /// The activity of the output frames in `probs`, after those emitted
    /// before them.
    fn activity(
        &self,
        probs: Vec<f32>,
    ) -> BunsenResult<SpeakerActivity> {
        let s = self.mels.model.n_speakers();
        let frames = probs.len() / s;
        let start = (self.mels.emitted() - frames) * self.frame_samples();
        SpeakerActivity::try_new(start, self.frame_samples(), s, probs)
    }

    /// Hands on the held mel frames below `limit` (a frame index).
    fn hand_on(
        &mut self,
        limit: usize,
    ) -> BunsenResult<Vec<f32>> {
        let Some(held) = self.held.take() else {
            return Ok(Vec::new());
        };
        let n = held.dims()[0];
        let take = limit.saturating_sub(self.handed).min(n);
        if take < n {
            self.held = Some(held.clone().slice_dim(0, take..n));
        }
        if take == 0 {
            return Ok(Vec::new());
        }
        self.handed += take;
        self.mels.push(held.slice_dim(0, 0..take))
    }

    fn hold(
        &mut self,
        frames: Option<Tensor<3>>,
    ) {
        if let Some(frames) = frames {
            self.held = cat_present(self.held.take(), Some(frames.squeeze_dim(0)));
        }
    }
}

impl SpeakerActivityStream for SortformerStream {
    fn push(
        &mut self,
        samples: &[f32],
    ) -> BunsenResult<SpeakerActivity> {
        self.samples += samples.len();
        let frames = self.features.push(samples)?;
        self.hold(frames);
        // `NeMo` keeps `samples / hop` mel frames: the samples so far vouch
        // for that many.
        let probs = self.hand_on(self.samples / self.hop)?;
        self.activity(probs)
    }

    fn finish(&mut self) -> BunsenResult<SpeakerActivity> {
        let frames = self.features.finish()?;
        self.hold(frames);
        let mut probs = self.hand_on(self.samples / self.hop)?;
        self.held = None;
        probs.extend(self.mels.finish()?);
        self.activity(probs)
    }
}

/// `a` then `b` along the frames, either possibly absent.
fn cat_present(
    a: Option<Tensor<2>>,
    b: Option<Tensor<2>>,
) -> Option<Tensor<2>> {
    match (a, b) {
        (Some(a), Some(b)) => Some(Tensor::cat(vec![a, b], 0)),
        (a, b) => a.or(b),
    }
}
