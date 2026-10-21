//! A Sortformer, streamed: chunks, look-ahead, and the speaker cache.
use std::sync::Arc;

use burn::{
    Tensor,
    module::Module,
};

use super::{
    SilenceFill,
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
/// with its context, against a [`SpeakerCache`], `NeMo`'s sync streaming.
///
/// Mel frames go in through [`push`](Self::push) in pieces of any size. Each
/// step covers `chunk_len` encoder frames of mels; once those and the
/// look-ahead's are in, the step pre-encodes its window (the left context's
/// mels, the chunk's, the look-ahead's), runs the model over the cache, the
/// FIFO and that window, and its chunk's frames come out, final.
/// [`finish`](Self::finish) runs what is left, the last chunks with what
/// look-ahead there is. However the frames are split across calls, the output
/// is the same.
///
/// The output is at the model's output rate: one frame per
/// `subsampling_factor / output_upsampling` mel frames.
#[derive(Clone, Debug)]
pub struct SortformerMelStream {
    model: Arc<Sortformer>,

    cache: SpeakerCache,

    /// Mel frames from `mels_start` on, `[frames, n_mels]`: the next step's
    /// left context and everything after.
    mels: Option<Tensor<2>>,

    /// The stream's mel frame `mels` starts at.
    mels_start: usize,

    /// The mel frame the next step's chunk starts at.
    next_chunk: usize,

    /// Mel frames taken in.
    mel_frames: usize,

    /// Output frames emitted.
    emitted: usize,

    finished: bool,
}

impl SortformerMelStream {
    /// A stream through `model` under `config`. The cache's silence slots
    /// hold the model's learned silence embedding when it has one
    /// (`Nemotron-3-Diarization`), and a running mean of the silent frames
    /// otherwise (Sortformer v2).
    ///
    /// # Errors
    /// As [`SortformerStreamingConfig::check`].
    pub fn new(
        model: Arc<Sortformer>,
        config: SortformerStreamingConfig,
    ) -> BunsenResult<Self> {
        let silence = match &model.silence {
            Some(silence) => SilenceFill::Learned(silence.val()),
            None => SilenceFill::running_mean(model.d_model(), &model.projection.weight.device()),
        };
        let cache = SpeakerCache::new(config, model.n_speakers(), silence)?;
        Ok(Self {
            model,
            cache,
            mels: None,
            mels_start: 0,
            next_chunk: 0,
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

        let c = self.cache.config().clone();
        let factor = self.model.subsampling_factor();
        let (chunk, look_ahead) = (c.chunk_len * factor, c.right_context * factor);
        let mut out = Vec::new();
        while self.next_chunk + chunk + look_ahead <= self.mel_frames {
            out.extend(self.step(self.next_chunk + chunk, look_ahead)?);
        }
        Ok(out)
    }

    /// Ends the input, and returns the output frames left: the last chunks
    /// run with what look-ahead there is, the last partial stack padded, and
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

        let c = self.cache.config().clone();
        let factor = self.model.subsampling_factor();
        let mut out = Vec::new();
        while self.next_chunk < self.mel_frames {
            let end = (self.next_chunk + c.chunk_len * factor).min(self.mel_frames);
            let look_ahead = (c.right_context * factor).min(self.mel_frames - end);
            out.extend(self.step(end, look_ahead)?);
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

    /// One step, `NeMo`'s `forward_streaming_step`: the chunk's mels
    /// `[next_chunk, end)` with up to the configured left context before them
    /// and `look_ahead` mels after, pre-encoded together; the model over the
    /// cache, the FIFO and that window; the chunk's output out, its
    /// embeddings into the FIFO.
    fn step(
        &mut self,
        end: usize,
        look_ahead: usize,
    ) -> BunsenResult<Vec<f32>> {
        let factor = self.model.subsampling_factor();
        let left_context = self.cache.config().left_context * factor;
        let start = self.next_chunk;
        let left = left_context.min(start);
        let mels = self.mels.clone().expect("mel frames are pending");
        let window = mels.slice_dim(
            0,
            start - left - self.mels_start..end + look_ahead - self.mels_start,
        );
        let (embs, _) = self.model.pre_encode(window.unsqueeze_dim(0), None);
        let embs = embs.squeeze_dim::<2>(0);
        let n_embs = embs.dims()[0];
        // `NeMo`'s `lc_enc` and `rc_enc`.
        let (left_enc, right_enc) = (
            (left as f64 / factor as f64).round() as usize,
            look_ahead.div_ceil(factor),
        );
        let chunk = n_embs - left_enc - right_enc;
        let chunk_embs = embs.clone().slice_dim(0, left_enc..left_enc + chunk);

        let cached = self.cache.cached();
        let n_cached = cached.as_ref().map_or(0, |c| c.dims()[0]);
        let input = cat_present(cached, Some(embs)).expect("the window is present");
        let probs = self.model.forward_pre_encoded(input.unsqueeze_dim(0), None);

        let up = self.model.output_upsampling();
        let s = self.model.n_speakers();
        let first = n_cached + left_enc;
        let out: Vec<f32> = probs
            .output
            .slice_dim(1, first * up..(first + chunk) * up)
            .into_data()
            .try_into_vec_as::<f32>()
            .expect("read back f32 probabilities");
        // The cache scores the cached frames and the chunk's, its context's
        // left out.
        let step: Vec<f32> = Tensor::cat(
            vec![
                probs.encoder_rate.clone().slice_dim(1, 0..n_cached),
                probs.encoder_rate.slice_dim(1, first..first + chunk),
            ],
            1,
        )
        .into_data()
        .try_into_vec_as::<f32>()
        .expect("read back f32 probabilities");
        self.cache.update(&step, chunk_embs)?;
        self.emitted += out.len() / s;

        // Keep what the next step's left context needs, and after.
        self.next_chunk = end;
        let keep_from = end.saturating_sub(left_context).max(self.mels_start);
        let n_mels = mels_len(&self.mels);
        self.mels = (keep_from - self.mels_start < n_mels).then(|| {
            self.mels
                .take()
                .expect("mel frames are pending")
                .slice_dim(0, keep_from - self.mels_start..n_mels)
        });
        self.mels_start = keep_from;
        Ok(out)
    }
}

/// The frames `mels` holds.
fn mels_len(mels: &Option<Tensor<2>>) -> usize {
    mels.as_ref().map_or(0, |m| m.dims()[0])
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
