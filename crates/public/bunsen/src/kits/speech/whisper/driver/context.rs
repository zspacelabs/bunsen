//! # The stream context: one transcription in progress.
//!
//! [`push`](WhisperStreamContext::push) is a fold: append to staging, drain
//! whole hops into the mel context, append the frames to the ring, offer them
//! to the clamp policy, offer the samples to the voice-activity gate, then
//! run whatever the emission policy says is due. The hop-alignment
//! requirement never reaches the caller &mdash; staging makes the API honest
//! against a sound card that hands over 480- or 1024-sample buffers.
//!
//! [`feed`](WhisperStreamContext::feed) is the first half of that fold and
//! [`advance`](WhisperStreamContext::advance) the second, so that a batch of
//! streams can be fed one at a time and advanced together by
//! [`advance_ready`](super::advance_ready).
//!
//! Nothing tensor-shaped crosses a window boundary. Continuity is three
//! host-side values: the seek pointer, the prompt carry, and the clock. The
//! ring holds every frame from `seek` onward and nothing before it.
//!
//! ## Regions
//!
//! With the `endpoint` trigger, a speech region closed by the gate is a
//! decode unit of its own: its frames are cut from the ring, decoded, and
//! committed with times off the parent stream's clock &mdash; region-as-
//! stream, without a second stream object. A full window is then decoded
//! only if speech is in progress inside it; a full window of silence is
//! skipped, not decoded, which is the gating half of voice activity.
//!
//! ## What a decode may touch
//!
//! Anything a provisional decode reaches takes `&self`: packaging asks the
//! clamp policy for a reference through `&self`, the model is pure, and the
//! prompt is read. That is what keeps drafts, when they arrive, from becoming
//! a second code path &mdash; and it is checkable today, through the
//! test-only probe.

use std::collections::VecDeque;

use burn::{
    Tensor,
    module::Module,
    prelude::{
        Backend,
        TensorData,
    },
};

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::{
        silero_vad::{
            SileroVadContext,
            SileroVadContextConfig,
            SileroVadMeta,
        },
        whisper::{
            blocks::WhisperMeta,
            decode::{
                DecodeConfig,
                Decoded,
                FallbackConfig,
                decode_with_fallback,
            },
            driver::{
                ClampPolicy,
                CommitRule,
                Emission,
                Segment,
                TimestampHistory,
                VoiceActivityFilter,
                driver_impl::WhisperDriver,
                support::{
                    SpeechRegion,
                    split_window,
                },
            },
        },
    },
    ops::signal::mels::{
        MelConversionContext,
        MelConverterMeta,
    },
};

/// One context's due unit, packaged and ready to join a batch.
struct Pending<B: Backend> {
    /// Index into the contexts being advanced.
    context: usize,
    unit: Due,
    /// A draft rather than a commit.
    draft: bool,
    /// The prompt it decodes under; what it is batched by.
    prompt: Vec<i64>,
    /// `[1, n_mels, width]`.
    window: Tensor<B, 3>,
}

/// Advances every context that has a decode due, batching the decodes.
///
/// Repeats until no context has anything due, so a context with several
/// windows waiting gets them all; a context with nothing due but a draft
/// due contributes the draft, batched the same way. Returns each context's
/// emissions, in the order of `contexts`.
///
/// # Arguments
/// * `driver` - the driver the contexts were opened from.
/// * `contexts` - the streams, fed through [`feed`](WhisperStreamContext::feed)
///   rather than [`push`](WhisperStreamContext::push), so that nothing has been
///   decoded yet.
///
/// # Errors
/// As [`WhisperStreamContext::advance`].
pub fn advance_ready<B: Backend>(
    driver: &WhisperDriver<B>,
    contexts: &mut [WhisperStreamContext<B>],
) -> BunsenResult<Vec<Vec<Emission>>> {
    let mut out: Vec<Vec<Emission>> = vec![Vec::new(); contexts.len()];

    loop {
        // One due unit per context, with the prompt it would decode under.
        let mut pending: Vec<Option<Pending<B>>> = Vec::new();
        for (i, ctx) in contexts.iter_mut().enumerate() {
            ctx.skip_silence();
            let (unit, draft) = match ctx.next_due() {
                Some(unit) => (unit, false),
                None => match ctx.draft_unit() {
                    Some(unit) => (unit, true),
                    None => continue,
                },
            };
            let frames = ctx.frames_at(&unit);
            ctx.ensure_language(&frames);
            pending.push(Some(Pending {
                context: i,
                unit,
                draft,
                prompt: ctx.prompt_now(),
                window: ctx.package_padded(frames),
            }));
        }
        if pending.is_empty() {
            return Ok(out);
        }

        // Group by prompt, in order of first appearance.
        let mut groups: Vec<(Vec<i64>, Vec<usize>)> = Vec::new();
        for (k, item) in pending.iter().enumerate() {
            let prompt = &item.as_ref().expect("not yet taken").prompt;
            match groups.iter_mut().find(|(p, _)| p == prompt) {
                Some((_, members)) => members.push(k),
                None => groups.push((prompt.clone(), vec![k])),
            }
        }

        for (prompt, members) in groups {
            let windows: Vec<Tensor<B, 3>> = members
                .iter()
                .map(|&k| pending[k].as_ref().expect("not yet taken").window.clone())
                .collect();
            let batch = Tensor::cat(windows, 0);

            // The first rung of the ladder, batched; any rung above it is
            // the context's own, which is rare.
            let config = driver.decode_config(prompt);
            let first = driver
                .model()
                .decode_windows_full(batch, &config, driver.filters());

            for (row, k) in members.into_iter().enumerate() {
                let item = pending[k].take().expect("taken once");
                let ctx = &mut contexts[item.context];
                let decoded = ctx.ladder(&config, item.window, Some(first[row].clone()));
                if item.draft {
                    out[item.context].push(ctx.draft_from(item.unit, decoded)?);
                } else {
                    out[item.context].extend(ctx.commit_due(item.unit, decoded)?);
                }
            }
        }
    }
}

/// One stream: the only stateful type in the driver.
///
/// Opened by [`WhisperDriver::new_context`]. Its tensor state &mdash; the
/// driver's handle, the mel carry, the frame ring, the VAD state &mdash; is
/// `Module` typed; everything else is host-side bookkeeping, small enough to
/// snapshot.
///
/// Not itself a `Module`, deliberately: `burn`'s derive treats every field
/// whose type mentions `B` as a module, `#[module(skip)]` or not, and
/// carries a skipped generic unchanged into the autodiff inner module, so
/// neither a boxed policy nor a policy type parameter survives it. The
/// policy stays behind its trait, boxed, and the context is `Clone + Debug`.
#[derive(Clone, Debug)]
pub struct WhisperStreamContext<B: Backend> {
    driver: WhisperDriver<B>,

    /// The mel front end's streaming state; `None` once flushed.
    mel: Option<MelConversionContext<B>>,

    /// `[1, frames, n_mels]` log-mels from `origin` onward; `None` when
    /// empty.
    frames: Option<Tensor<B, 3>>,

    /// Samples not yet a whole hop.
    staging: Vec<f32>,

    /// Samples pushed so far, staging included.
    samples_seen: usize,

    /// The stream frame index of the ring's first frame.
    origin: usize,

    /// The stream frame index the next window starts at.
    seek: usize,

    clock: TimestampHistory,

    /// Every committed id, in order.
    transcript: Vec<i64>,

    /// Where the prompt carry starts in the transcript: upstream's
    /// `prompt_reset_since`, moved past a decode that needed a high
    /// temperature so a failure does not feed the next window.
    prompt_reset: usize,

    /// Media time (samples seen) at the last draft or commit; drafts are
    /// paced from it.
    last_draft: usize,

    /// The language the prompt names: configured, or detected from the
    /// first window decoded; `None` until then on a detecting driver.
    language: Option<String>,

    clamp: Box<dyn ClampPolicy<B>>,

    /// Voice activity, when the driver has a VAD and the policy wants
    /// endpoints.
    vad: Option<VoiceActivity<B>>,

    finished: bool,

    /// Every window handed to a decode, in order. Test-only: how the
    /// chunking invariant is stated at frame level.
    #[cfg(test)]
    trace: Vec<Tensor<B, 3>>,
}

/// The voice-activity half of a stream: Silero's state, the filter, and the
/// regions it has closed but the decode has not yet consumed.
#[derive(Clone, Debug)]
struct VoiceActivity<B: Backend> {
    /// Silero's recurrent state; `None` only while a step is in flight.
    context: Option<SileroVadContext<B>>,

    filter: VoiceActivityFilter,

    /// Samples not yet a whole chunk.
    staging: Vec<f32>,

    /// Padding on each side of a region, in samples.
    pad: usize,

    /// The encoder grid closed regions are snapped onto, in samples.
    grid: usize,

    /// Closed regions, padded and snapped, oldest first.
    regions: VecDeque<SpeechRegion>,

    /// Where the last closed region ended, so the next one's padding cannot
    /// reach back into it.
    last_end: usize,
}

/// One decode unit: `count` frames from stream frame `start`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Due {
    pub(super) start: usize,
    pub(super) count: usize,
}

impl<B: Backend> WhisperStreamContext<B> {
    pub(super) fn open(
        driver: WhisperDriver<B>,
        clock: TimestampHistory,
        clamp: Box<dyn ClampPolicy<B>>,
    ) -> Self {
        let mel = driver.mel().new_context(1);

        let vad = match (driver.vad(), driver.filter_config()) {
            (Some(model), Some(gate)) if driver.emission().triggers.endpoint => {
                let device = model.devices()[0].clone();
                Some(VoiceActivity {
                    context: Some(
                        SileroVadContextConfig::new(model.sample_rate()).init(model, &device),
                    ),
                    filter: gate.init(),
                    staging: Vec::new(),
                    pad: gate.speech_pad_samples(),
                    grid: driver.encoder_grid(),
                    regions: VecDeque::new(),
                    last_end: 0,
                })
            }
            _ => None,
        };

        let language = driver.language().map(str::to_string);
        Self {
            driver,
            mel: Some(mel),
            frames: None,
            staging: Vec::new(),
            samples_seen: 0,
            origin: 0,
            seek: 0,
            clock,
            transcript: Vec::new(),
            prompt_reset: 0,
            last_draft: 0,
            language,
            clamp,
            vad,
            finished: false,
            #[cfg(test)]
            trace: Vec::new(),
        }
    }

    // ---- observation -------------------------------------------------

    /// The stream's clock.
    pub fn clock(&self) -> &TimestampHistory {
        &self.clock
    }

    /// Every committed id, in order.
    pub fn transcript(&self) -> &[i64] {
        &self.transcript
    }

    /// Samples pushed so far.
    pub fn samples_seen(&self) -> usize {
        self.samples_seen
    }

    /// The stream frame index the next window starts at.
    pub fn seek(&self) -> usize {
        self.seek
    }

    /// Frames produced so far.
    pub fn frames_seen(&self) -> usize {
        self.origin + self.ring_len()
    }

    /// Frames past the seek pointer, waiting for a decode.
    pub fn pending_frames(&self) -> usize {
        self.frames_seen() - self.seek
    }

    /// Whether the gate currently has a region open: speech in progress.
    /// Always `false` without voice activity.
    pub fn is_speaking(&self) -> bool {
        self.vad.as_ref().is_some_and(|v| v.filter.is_open())
    }

    /// Closed regions not yet decoded.
    pub fn regions_pending(&self) -> usize {
        self.vad.as_ref().map_or(0, |v| v.regions.len())
    }

    /// Whether [`flush`](Self::flush) or [`end_input`](Self::end_input) has
    /// run.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    fn ring_len(&self) -> usize {
        self.frames.as_ref().map_or(0, |f| f.dims()[1])
    }

    fn hop(&self) -> usize {
        self.driver.mel().hop()
    }

    // ---- input -------------------------------------------------------

    /// Pushes samples, of any length, and returns what became final.
    ///
    /// [`feed`](Self::feed) then [`advance`](Self::advance).
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] after the stream has ended.
    pub fn push(
        &mut self,
        samples: &[f32],
    ) -> BunsenResult<Vec<Emission>> {
        self.feed(samples)?;
        self.advance()
    }

    /// [`push`](Self::push), anchoring the clock: the first sample of
    /// `samples` was at media time `time`.
    ///
    /// # Errors
    /// As [`push`](Self::push) and [`TimestampHistory::anchor`].
    pub fn push_at(
        &mut self,
        samples: &[f32],
        time: f64,
    ) -> BunsenResult<Vec<Emission>> {
        self.clock.anchor(self.samples_seen, time)?;
        self.push(samples)
    }

    /// Takes samples in without decoding: the front end and the gate run,
    /// nothing else. Pair with [`advance`](Self::advance), or with
    /// [`advance_ready`](super::advance_ready) across many streams.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] after the stream has ended.
    pub fn feed(
        &mut self,
        samples: &[f32],
    ) -> BunsenResult<()> {
        if self.finished {
            return Err(BunsenError::Invalid(
                "the stream has ended; nothing more can be fed".to_string(),
            ));
        }

        self.staging.extend_from_slice(samples);
        self.samples_seen += samples.len();
        if let Some(vad) = &mut self.vad {
            vad.staging.extend_from_slice(samples);
        }

        self.drain_staging(false)?;
        self.run_vad(false);
        Ok(())
    }

    /// Runs every decode that is due, and returns what became final.
    pub fn advance(&mut self) -> BunsenResult<Vec<Emission>> {
        let mut out = Vec::new();
        loop {
            self.skip_silence();
            let Some(unit) = self.next_due() else {
                break;
            };
            let window = self.frames_at(&unit);
            self.ensure_language(&window);
            #[cfg(test)]
            self.trace.push(window.clone());
            let decoded = self.decode_frames(window);
            out.extend(self.commit_due(unit, decoded)?);
        }
        if let Some(unit) = self.draft_unit() {
            let window = self.frames_at(&unit);
            self.ensure_language(&window);
            let decoded = self.decode_frames(window);
            out.push(self.draft_from(unit, decoded)?);
        }
        Ok(out)
    }

    /// Ends the stream's input: flushes the front end, drops Whisper's
    /// trailing frame, and closes the gate. What that leaves due is decoded
    /// by the next [`advance`](Self::advance).
    ///
    /// Idempotent.
    pub fn end_input(&mut self) -> BunsenResult<()> {
        if self.finished {
            return Ok(());
        }
        self.finished = true;

        self.drain_staging(true)?;

        if let Some(mel) = self.mel.take()
            && let Some(tail) = mel.finish()
        {
            self.ingest(tail);
        }

        // Whisper's `stft[..., :-1]`: the stream's last frame is the end
        // padding's, and goes.
        if let Some(frames) = self.frames.take() {
            let n = frames.dims()[1];
            self.frames = (n > 1).then(|| frames.slice_dim(1, 0..n as isize - 1));
        }

        self.run_vad(true);
        Ok(())
    }

    /// Ends the stream and decodes whatever is left past the seek pointer:
    /// [`end_input`](Self::end_input) then [`advance`](Self::advance).
    ///
    /// Idempotent; a second flush returns nothing.
    pub fn flush(&mut self) -> BunsenResult<Vec<Emission>> {
        self.end_input()?;
        self.advance()
    }

    /// Moves whole hops from staging into the front end.
    ///
    /// The first chunk must be long enough to produce a frame under the
    /// start padding, so it waits for a window's worth of samples; after
    /// that a hop at a time is enough. Flushing pads whatever remains with
    /// silence up to that minimum, as upstream pads a clip; a stream that
    /// never received a sample flushes to nothing.
    fn drain_staging(
        &mut self,
        flushing: bool,
    ) -> BunsenResult<()> {
        let mel = self.driver.mel();
        let (hop, n_fft) = (mel.hop(), mel.n_fft());
        let first = self.mel.as_ref().is_some_and(|ctx| ctx.carry().is_none());
        let minimum = if first {
            n_fft.div_ceil(hop) * hop
        } else {
            hop
        };

        let mut whole = self.staging.len() / hop * hop;
        if flushing {
            if self.staging.is_empty() {
                return Ok(());
            }
            whole = self.staging.len().div_ceil(hop).max(minimum / hop) * hop;
            self.staging.resize(whole, 0.0);
        } else if whole < minimum {
            return Ok(());
        }

        let chunk: Vec<f64> = self.staging.drain(..whole).map(f64::from).collect();
        let device = self.driver.devices()[0].clone();
        let waves: Tensor<B, 2> = Tensor::from_data(TensorData::new(chunk, [1, whole]), &device);

        let ctx = self
            .mel
            .take()
            .expect("the front end is open until the input ends");
        let (frames, ctx) = ctx.transform(waves)?;
        self.mel = Some(ctx);
        self.ingest(frames);
        Ok(())
    }

    /// Appends frames to the ring, offering them to the clamp policy first.
    fn ingest(
        &mut self,
        new: Tensor<B, 3>,
    ) {
        self.clamp.observe(&new);
        self.frames = Some(match self.frames.take() {
            Some(ring) => Tensor::cat(vec![ring, new], 1),
            None => new,
        });
    }

    /// Runs Silero over every whole chunk in the VAD's staging, steps the
    /// gate, and queues the regions it closes. Flushing pads the last
    /// chunk with silence and closes an open region at the true length.
    fn run_vad(
        &mut self,
        flushing: bool,
    ) {
        let total = self.samples_seen;
        let Some(vad) = &mut self.vad else {
            return;
        };
        let model = self
            .driver
            .vad()
            .expect("a VAD is attached when voice activity is on");
        let chunk = model.chunk_size();

        if flushing && !vad.staging.is_empty() {
            let whole = vad.staging.len().div_ceil(chunk) * chunk;
            vad.staging.resize(whole, 0.0);
        }
        let whole = vad.staging.len() / chunk * chunk;
        if whole > 0 {
            let steps = whole / chunk;
            let samples: Vec<f32> = vad.staging.drain(..whole).collect();
            let device = model.devices()[0].clone();
            let chunks: Tensor<B, 3> =
                Tensor::from_data(TensorData::new(samples, [steps, 1, chunk]), &device);

            let context = vad.context.take().expect("present between steps");
            let (probs, context) = model.context_forward_sequence(chunks, context);
            vad.context = Some(context);

            let probs: Vec<f32> = probs.to_data().convert::<f32>().to_vec().unwrap();
            for p in probs {
                if let Some(raw) = vad.filter.step(p) {
                    vad.enqueue(raw, total);
                }
            }
        }

        if flushing && let Some(raw) = vad.filter.clone().finish(total) {
            vad.enqueue(raw, total);
        }
    }

    // ---- scheduling --------------------------------------------------

    /// With endpoints on, drops full windows that hold no speech: nothing
    /// closed inside them, and nothing open in them.
    pub(super) fn skip_silence(&mut self) {
        let width = self.driver.window_frames();
        let hop = self.hop();
        loop {
            let Some(vad) = &self.vad else {
                return;
            };
            if self.pending_frames() < width {
                return;
            }
            let (from, to) = (self.seek * hop, (self.seek + width) * hop);
            let closed_in = vad.regions.iter().any(|r| r.end > from && r.start < to);
            let open_in = vad.filter.open_since().is_some_and(|s| s < to);
            if closed_in || open_in {
                return;
            }
            self.seek += width;
            self.drop_before_seek();
        }
    }

    /// The next decode unit, if one is due.
    ///
    /// With endpoints on: the oldest closed region reaching past the seek
    /// pointer, a window at a time; else a full window, only while speech
    /// is in progress inside it. Without: a full window, and at the end of
    /// input whatever remains.
    pub(super) fn next_due(&self) -> Option<Due> {
        let width = self.driver.window_frames();
        let hop = self.hop();
        let available = self.frames_seen();

        if let Some(vad) = &self.vad {
            for region in &vad.regions {
                let (rs, re) = ((region.start / hop), (region.end / hop));
                if re <= self.seek {
                    continue;
                }
                let start = rs.max(self.seek);
                let count = (re - start).min(width);
                if start + count <= available {
                    return Some(Due { start, count });
                }
                // Its frames are still in the front end; at the end of
                // input, take what there is.
                return (self.finished && available > start).then(|| Due {
                    start,
                    count: available - start,
                });
            }

            if self.driver.emission().triggers.window_full && self.pending_frames() >= width {
                let to = (self.seek + width) * hop;
                if vad.filter.open_since().is_some_and(|s| s < to) {
                    return Some(Due {
                        start: self.seek,
                        count: width,
                    });
                }
            }
            return None;
        }

        if self.pending_frames() >= width {
            return Some(Due {
                start: self.seek,
                count: width,
            });
        }
        (self.finished && self.pending_frames() > 0).then(|| Due {
            start: self.seek,
            count: self.pending_frames(),
        })
    }

    /// The frames of a due unit: `[1, count, n_mels]`.
    pub(super) fn frames_at(
        &self,
        unit: &Due,
    ) -> Tensor<B, 3> {
        let ring = self.frames.as_ref().expect("due frames exist");
        let start = (unit.start - self.origin) as isize;
        ring.clone()
            .slice_dim(1, start..start + unit.count as isize)
    }

    // ---- decoding ----------------------------------------------------

    /// Packages a window against the clamp policy's reference and pads it
    /// out to the model's width: `[1, n_mels, width]`. Takes `&self`.
    pub(super) fn package_padded(
        &self,
        window: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let reference = self.clamp.reference(&window);
        let packaged = self.driver.front_end().package_window(window, reference);

        let width = self.driver.window_frames();
        let [_, n_mels, have] = packaged.dims();
        if have < width {
            let pad = Tensor::zeros([1, n_mels, width - have], &packaged.device());
            Tensor::cat(vec![packaged, pad], 2)
        } else {
            packaged
        }
    }

    /// Packages and decodes a window, up the temperature ladder. Takes
    /// `&self`: with [`frames_at`](Self::frames_at) this is the whole of a
    /// provisional decode.
    fn decode_frames(
        &self,
        window: Tensor<B, 3>,
    ) -> Decoded {
        let base = self.driver.decode_config(self.prompt_now());
        self.ladder(&base, self.package_padded(window), None)
    }

    /// Decodes a packaged window up the ladder, encoding it only if a rung
    /// past `first` is needed.
    ///
    /// # Arguments
    /// * `base` - the decode config at temperature zero.
    /// * `window` - `[1, n_mels, width]`, packaged.
    /// * `first` - the first rung's result when the caller has it.
    pub(super) fn ladder(
        &self,
        base: &DecodeConfig,
        window: Tensor<B, 3>,
        first: Option<Decoded>,
    ) -> Decoded {
        let model = self.driver.model();
        let mut xa: Option<Tensor<B, 3>> = None;
        decode_with_fallback(
            self.driver.fallback(),
            base,
            first,
            |config| {
                let xa = xa
                    .get_or_insert_with(|| model.forward_encoder(window.clone()))
                    .clone();
                model
                    .decode_features_full(xa, config, self.driver.filters())
                    .pop()
                    .expect("one row in, one row out")
            },
            |ids| self.text_of(ids),
        )
    }

    /// The text of some ids, when the driver can say.
    fn text_of(
        &self,
        ids: &[i64],
    ) -> Option<String> {
        let detokenizer = self.driver.detokenizer()?;
        detokenizer
            .detokenize(&self.driver.policy().text_ids(ids))
            .ok()
    }

    /// The prompt for the next window: the sot sequence, preceded by the
    /// transcript's tail after `<|startofprev|>` when carrying is on.
    pub(super) fn prompt_now(&self) -> Vec<i64> {
        let prompt = self.sot_now();
        let carried = &self.transcript[self.prompt_reset.min(self.transcript.len())..];
        if !self.driver.carries_prompt() || carried.is_empty() {
            return prompt;
        }

        // Upstream keeps `n_text_ctx / 2 - 1` tokens of context.
        let keep = self.driver.model().max_text_ctx() / 2 - 1;
        let tail = &carried[carried.len().saturating_sub(keep)..];

        let mut out = Vec::with_capacity(1 + tail.len() + prompt.len());
        out.push(self.driver.policy().ids().sot_prev);
        out.extend_from_slice(tail);
        out.extend_from_slice(&prompt);
        out
    }

    /// Detects the stream's language from `frames` when the driver leaves
    /// it to be detected and no window has been decoded yet: upstream's
    /// `detect_language`, on the same features the decode will use.
    pub(super) fn ensure_language(
        &mut self,
        frames: &Tensor<B, 3>,
    ) {
        if self.language.is_some() || !self.driver.detects_language() {
            return;
        }
        let model = self.driver.model();
        let xa = model.forward_encoder(self.package_padded(frames.clone()));
        let token = model.detect_language(xa, self.driver.policy().ids())[0];
        let code = self
            .driver
            .policy()
            .language_code(token)
            .expect("detection picks a language token");
        self.language = Some(code.to_string());
    }

    /// The language the stream's prompt names, once known.
    pub fn language(&self) -> Option<&str> {
        self.language.as_deref()
    }

    /// The sot sequence for this stream: the driver's, with the stream's
    /// language.
    fn sot_now(&self) -> Vec<i64> {
        self.driver
            .sot_sequence(self.language.as_deref())
            .expect("the language is known before a prompt is built")
    }

    /// Commits a decode of a due unit: records the ids, advances the seek
    /// pointer, drops the consumed frames and regions, and places the
    /// segments on the clock.
    ///
    /// A window the fallback policy calls silence (its no-speech
    /// probability over the threshold, its log probability poor) is
    /// skipped whole: nothing is emitted and the seek moves past it.
    ///
    /// Without timestamps the unit is one segment and the seek advances
    /// past it. With them the decode is split on its timestamps
    /// ([`split_window`]); the seek advances to the last closed timestamp,
    /// and the unfinished tail is either dropped, to be decoded again with
    /// more audio behind it (`CommitRule::Complete`, which is what
    /// upstream's seek loop does), or emitted as a draft first
    /// (`CommitRule::LastTimestamp`).
    ///
    /// A decode that needed a temperature above 0.5 resets the prompt
    /// carry, as upstream does, so a failure does not feed the next window.
    pub(super) fn commit_due(
        &mut self,
        unit: Due,
        decoded: Decoded,
    ) -> BunsenResult<Vec<Emission>> {
        let hop = self.hop();
        let mut out = Vec::new();

        let skip = self
            .driver
            .fallback()
            .should_skip(decoded.no_speech_prob.map(f64::from), decoded.avg_logprob());
        let tokens = decoded.tokens;

        if skip {
            self.seek = unit.start + unit.count;
        } else if self.driver.timestamps() {
            let split = split_window(
                &tokens,
                self.driver.policy().ids(),
                unit.count,
                self.driver.frames_per_timestamp(),
            );
            for segment in split.segments {
                let emission = self.segment_at(
                    unit.start + segment.start,
                    unit.start + segment.end,
                    segment.tokens,
                )?;
                self.transcript.extend_from_slice(&emission.tokens);
                out.push(Emission::Committed(emission));
            }
            if !split.tail.is_empty() && self.driver.emission().commit == CommitRule::LastTimestamp
            {
                let opens = (split.tail[0] - self.driver.policy().ids().timestamp_begin) as usize
                    * self.driver.frames_per_timestamp();
                // A tail opening past the unit's audio covers nothing yet;
                // it is decoded again with more audio behind it.
                if opens < unit.count {
                    let draft =
                        self.segment_at(unit.start + opens, unit.start + unit.count, split.tail)?;
                    out.push(Emission::Draft(draft));
                }
            }
            // Always forward, never past the unit.
            self.seek = unit.start + split.advance.clamp(1, unit.count);
        } else {
            let segment = self.segment_at(unit.start, unit.start + unit.count, tokens)?;
            self.transcript.extend_from_slice(&segment.tokens);
            out.push(Emission::Committed(segment));
            self.seek = unit.start + unit.count;
        }

        if !self.driver.carries_prompt() || FallbackConfig::resets_prompt(decoded.temperature) {
            self.prompt_reset = self.transcript.len();
        }

        self.last_draft = self.samples_seen;
        self.drop_before_seek();

        if let Some(vad) = &mut self.vad {
            let consumed = self.seek * hop;
            while vad.regions.front().is_some_and(|r| r.end <= consumed) {
                vad.regions.pop_front();
            }
        }

        Ok(out)
    }

    /// The draft due, if one is: under the `interval` trigger, while speech
    /// is in progress, once an interval of media time has passed since the
    /// last draft or commit, everything past the seek pointer.
    pub(super) fn draft_unit(&self) -> Option<Due> {
        let interval = self.driver.interval_samples()?;
        let count = self.pending_frames();
        let due = !self.finished
            && count > 0
            && self.is_speaking()
            && self.samples_seen.saturating_sub(self.last_draft) >= interval;
        due.then_some(Due {
            start: self.seek,
            count,
        })
    }

    /// Emits a decode of a draft unit as a draft: on the clock, detokenized
    /// when the driver can, and touching nothing but the draft pacing. It
    /// covers all audio since the last commit and supersedes the previous
    /// draft whole.
    pub(super) fn draft_from(
        &mut self,
        unit: Due,
        decoded: Decoded,
    ) -> BunsenResult<Emission> {
        self.last_draft = self.samples_seen;
        Ok(Emission::Draft(self.segment_at(
            unit.start,
            unit.start + unit.count,
            decoded.tokens,
        )?))
    }

    /// A segment over stream frames `[start, end)`, timed through the clock
    /// and detokenized when the driver can.
    fn segment_at(
        &self,
        start: usize,
        end: usize,
        tokens: Vec<i64>,
    ) -> BunsenResult<Segment> {
        let hop = self.hop();
        let text = match self.driver.detokenizer() {
            Some(detokenizer) => {
                Some(detokenizer.detokenize(&self.driver.policy().text_ids(&tokens))?)
            }
            None => None,
        };
        Ok(Segment {
            start: self.clock.time_at(start * hop),
            end: self.clock.time_at(end * hop),
            tokens,
            text,
        })
    }

    /// Retains the ring from the seek pointer onward, nothing before it.
    fn drop_before_seek(&mut self) {
        let drop = (self.seek - self.origin) as isize;
        self.frames = self.frames.take().and_then(|ring| {
            let n = ring.dims()[1] as isize;
            (drop < n).then(|| ring.slice_dim(1, drop..n))
        });
        self.origin = self.seek;
    }

    // ---- test probes -------------------------------------------------

    /// A provisional decode of everything past the seek pointer, without
    /// touching the context: what a draft says, on demand.
    #[cfg(test)]
    pub(crate) fn probe_decode(&self) -> Option<Vec<i64>> {
        let count = self.pending_frames();
        (count > 0).then(|| {
            self.decode_frames(self.frames_at(&Due {
                start: self.seek,
                count,
            }))
            .tokens
        })
    }

    /// Everything the context holds, as one comparable string.
    #[cfg(test)]
    pub(crate) fn fingerprint(&self) -> String {
        let data = |t: &Tensor<B, 3>| t.to_data().convert::<f32>().to_vec::<f32>().unwrap();
        let ring = self.frames.as_ref().map(data);
        let carry = self
            .mel
            .as_ref()
            .and_then(|m| m.carry())
            .map(|c| c.to_data().convert::<f32>().to_vec::<f32>().unwrap());
        let reference = self.frames.as_ref().map(|f| {
            self.clamp
                .reference(f)
                .to_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .unwrap()
        });
        let regions: Vec<SpeechRegion> = self
            .vad
            .as_ref()
            .map(|v| v.regions.iter().copied().collect())
            .unwrap_or_default();

        format!(
            "{:?}|{}|{}|{}|{:?}|{:?}|{}|{ring:?}|{carry:?}|{reference:?}|{regions:?}",
            self.staging,
            self.samples_seen,
            self.origin,
            self.seek,
            self.clock,
            self.transcript,
            self.finished,
        )
    }
}

impl<B: Backend> VoiceActivity<B> {
    /// Pads a raw region as far as the stream allows, snaps it outward onto
    /// the encoder grid, and queues it.
    ///
    /// The start pad cannot reach back into the previous region; the end
    /// pad cannot reach past the samples seen, which at close time are at
    /// least `min_silence` past the raw end &mdash; enough for the pad under
    /// both presets.
    fn enqueue(
        &mut self,
        raw: SpeechRegion,
        total: usize,
    ) {
        let start = raw.start.saturating_sub(self.pad).max(self.last_end);
        let end = (raw.end + self.pad).min(total).max(start);
        let mut region = SpeechRegion::new(start, end).snap_outward(self.grid);
        region.start = region.start.max(self.last_end);
        self.last_end = region.end;
        self.regions.push_back(region);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use burn::tensor::{
        Tolerance,
        backend::BackendTypes,
    };
    use serial_test::serial;

    use super::*;
    use crate::{
        burner::module::ModuleInit,
        errors::BunsenResult,
        kits::{
            speech::whisper::{
                Whisper,
                blocks::WhisperApiConfig,
                decode::{
                    GreedyDecodeConfig,
                    LogitFilter,
                },
                driver::{
                    EmissionPolicy,
                    MaxSeen,
                    PerWindow,
                    TokenPolicy,
                    Triggers,
                    WhisperSpecialIds,
                    context::advance_ready,
                    driver_impl::WhisperDriverConfig,
                    trim_stream_tail,
                },
            },
            tokens::Detokenizer,
        },
        support::testing::PerformanceBackend,
    };

    type B = PerformanceBackend;
    type F = <B as BackendTypes>::FloatElem;
    type Device = burn::prelude::Device<B>;

    /// A layout small enough for a tiny model: 5 base ranks, 1 language.
    fn tiny_layout() -> WhisperSpecialIds {
        WhisperSpecialIds::new(5, 1).unwrap()
    }

    /// A tiny model whose vocabulary fits the tiny layout and whose window
    /// is 16 frames (2560 samples), so a short clip has several windows.
    /// Seeded, so a run is the same run on the same backend.
    fn tiny_model_on<Bk: Backend>(device: &Bk::Device) -> Whisper<Bk> {
        Bk::seed(device, 7);
        WhisperApiConfig::new(
            /* n_mels */ 8,
            /* vocab_size */ tiny_layout().n_vocab(),
            /* d_model */ 64,
            /* max_audio_ctx */ 16,
            /* n_encoder_layers */ 1,
            /* max_text_ctx */ 16,
            /* n_decoder_layers */ 1,
        )
        .init(device)
    }

    fn tiny_model(device: &Device) -> Whisper<B> {
        tiny_model_on::<B>(device)
    }

    fn config(carry: bool) -> WhisperDriverConfig {
        WhisperDriverConfig::new()
            .with_language(Some("en".to_string()))
            .with_max_tokens(4)
            .with_condition_on_previous_text(carry)
    }

    fn driver(
        device: &Device,
        carry: bool,
    ) -> WhisperDriver<B> {
        config(carry)
            .init_with_policy(tiny_model(device), TokenPolicy::new(tiny_layout()), device)
            .unwrap()
    }

    /// The rate the tiny model, and every clip here, is at.
    const RATE: usize = 16_000;

    fn clock() -> TimestampHistory {
        TimestampHistory::uniform(RATE)
    }

    /// A deterministic 1.05 s clip: a tone under a bell-shaped envelope
    /// peaking mid-clip, a rising chirp, and a little noise. The loudest
    /// frames are in the middle on purpose, so the global maximum is never
    /// in the trailing frame that packaging drops.
    fn clip() -> Vec<f32> {
        clip_seeded(0.5, 440.0)
    }

    fn clip_seeded(
        peak_at: f32,
        tone_hz: f32,
    ) -> Vec<f32> {
        let n = 16_800;
        (0..n)
            .map(|i| {
                let t = i as f32 / RATE as f32;
                let envelope = (-(t - peak_at).powi(2) * 20.0).exp();
                let tone = 0.6 * envelope * (2.0 * std::f32::consts::PI * tone_hz * t).sin();
                let chirp =
                    0.2 * (2.0 * std::f32::consts::PI * (3_100.0 * t + 800.0 * t * t)).sin();
                let noise = 0.05 * (((i * 7919) % 1000) as f32 / 500.0 - 1.0);
                tone + chirp + noise
            })
            .collect()
    }

    /// Chunk sizes from a seeded generator: 1 to 1200 samples, so most are
    /// not hop multiples and staging is exercised.
    fn random_sizes(
        seed: usize,
        total: usize,
    ) -> Vec<usize> {
        let mut state = seed;
        let mut sizes = Vec::new();
        let mut left = total;
        while left > 0 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let size = ((state >> 33) as usize % 1200 + 1).min(left);
            sizes.push(size);
            left -= size;
        }
        sizes
    }

    fn push_in_pieces(
        ctx: &mut WhisperStreamContext<B>,
        audio: &[f32],
        sizes: &[usize],
    ) -> Vec<Emission> {
        let mut out = Vec::new();
        let mut at = 0;
        for &size in sizes {
            out.extend(ctx.push(&audio[at..at + size]).unwrap());
            at += size;
        }
        assert_eq!(at, audio.len());
        out.extend(ctx.flush().unwrap());
        out
    }

    fn tokens_of(emissions: &[Emission]) -> Vec<Vec<i64>> {
        emissions
            .iter()
            .map(|e| e.segment().tokens.clone())
            .collect()
    }

    /// The whole clip through the front end in one call, joined with its
    /// tail: what `package_mels` and `decode_chunked` start from.
    fn joined_mels(
        driver: &WhisperDriver<B>,
        audio: &[f32],
        device: &Device,
    ) -> Tensor<B, 3> {
        let n = audio.len();
        let waves: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(
                audio.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
                [1, n],
            ),
            device,
        );
        let (mels, ctx) = driver.mel().new_context(1).transform(waves).unwrap();
        match ctx.finish() {
            Some(tail) => Tensor::cat(vec![mels, tail], 1),
            None => mels,
        }
    }

    fn greedy_config(driver: &WhisperDriver<B>) -> GreedyDecodeConfig {
        GreedyDecodeConfig::new(driver.prompt().to_vec(), driver.policy().ids().eot)
            .with_max_tokens(driver.max_tokens())
    }

    /// **I5, against the one-shot path.** One push of the whole clip, with
    /// the running-maximum policy fed everything before the first decode,
    /// is `decode_chunked` over `package_mels` &mdash; window for window,
    /// id for id.
    #[test]
    #[serial]
    fn test_single_push_matches_decode_chunked() {
        let device = Device::default();
        let driver = driver(&device, false);
        let audio = clip();

        let mut ctx = driver.new_context(clock(), MaxSeen::new()).unwrap();
        let mut emissions = ctx.push(&audio).unwrap();
        emissions.extend(ctx.flush().unwrap());

        let expected = driver.model().decode_chunked(
            driver
                .front_end()
                .package_mels(joined_mels(&driver, &audio, &device)),
            &greedy_config(&driver),
        );

        assert_eq!(expected.len(), 7, "6 full windows and a remainder");
        assert_eq!(tokens_of(&emissions), expected);
        assert!(emissions.iter().all(Emission::is_committed));
        assert_eq!(
            ctx.transcript(),
            expected.concat(),
            "the transcript is the committed ids, in order",
        );
    }

    /// The chunking half of I5, stated as the equality caveat says it must
    /// be: the windows two streams decoded agree within tolerance always,
    /// their segments sit at the same times, and their ids agree whenever
    /// the windows are bit-identical.
    ///
    /// The front end's chunk-invariance is approximate on some backends (a
    /// matmul's accumulation order changes with the chunk), and an untrained
    /// model turns a last-digit difference into a flipped argmax. A trained
    /// model on speech does not; that is the validation crate's gate.
    fn assert_same_stream(
        a: (&WhisperStreamContext<B>, &[Emission]),
        b: (&WhisperStreamContext<B>, &[Emission]),
        label: &str,
    ) {
        let (ctx_a, got) = a;
        let (ctx_b, expected) = b;
        assert_eq!(
            ctx_a.trace.len(),
            ctx_b.trace.len(),
            "{label}: window count"
        );
        assert_eq!(got.len(), expected.len(), "{label}: emission count");

        let mut identical = true;
        for (w, (x, y)) in ctx_a.trace.iter().zip(&ctx_b.trace).enumerate() {
            let (x, y) = (x.to_data(), y.to_data());
            assert_eq!(x.shape, y.shape, "{label}: window {w} shape");
            x.assert_approx_eq::<F>(&y, Tolerance::rel_abs(1e-5, 1e-6));
            identical &= x == y;
        }

        for (g, e) in got.iter().zip(expected) {
            assert_eq!(g.segment().start, e.segment().start, "{label}: start");
            assert_eq!(g.segment().end, e.segment().end, "{label}: end");
        }

        if identical {
            assert_eq!(
                got, expected,
                "{label}: identical windows must decode identically"
            );
        } else {
            eprintln!("{label}: windows differ within tolerance; ids not compared");
        }
    }

    /// **I5, against chunking.** Random-sized pushes, most of them not hop
    /// multiples, decode the windows one push decodes &mdash; and one push
    /// decodes exactly what packaging each window by hand gives.
    #[test]
    #[serial]
    fn test_random_pushes_match_single_push() {
        let device = Device::default();
        let driver = driver(&device, false);
        let audio = clip();

        let mut whole = driver.new_context(clock(), PerWindow).unwrap();
        let mut expected = whole.push(&audio).unwrap();
        expected.extend(whole.flush().unwrap());

        for seed in [1, 2, 3] {
            let mut ctx = driver.new_context(clock(), PerWindow).unwrap();
            let got = push_in_pieces(&mut ctx, &audio, &random_sizes(seed, audio.len()));
            assert_same_stream((&ctx, &got), (&whole, &expected), &format!("seed {seed}"));
        }

        // A dynamic policy is the same policy, and the same chunking is the
        // same arithmetic, so this one is exact.
        let boxed: Box<dyn ClampPolicy<B>> = Box::new(PerWindow);
        let mut dynamic = driver.new_context(clock(), boxed).unwrap();
        let mut got = dynamic.push(&audio).unwrap();
        got.extend(dynamic.flush().unwrap());
        assert_eq!(got, expected);

        // The per-window one-shot path, by hand.
        let width = driver.window_frames();
        let trimmed = trim_stream_tail(joined_mels(&driver, &audio, &device));
        let frames = trimmed.dims()[1];
        let config = greedy_config(&driver);
        let mut by_hand = Vec::new();
        let mut at = 0;
        while at < frames {
            let count = width.min(frames - at);
            let window = trimmed
                .clone()
                .slice_dim(1, at as isize..(at + count) as isize);
            let mut packaged = driver
                .front_end()
                .package_window(window.clone(), PerWindow.reference(&window));
            if count < width {
                let pad = Tensor::zeros([1, packaged.dims()[1], width - count], &device);
                packaged = Tensor::cat(vec![packaged, pad], 2);
            }
            by_hand.push(driver.model().decode_window(packaged, &config));
            at += count;
        }
        assert_eq!(tokens_of(&expected), by_hand);
    }

    /// Segments sit on the clock: a window that starts at frame `f` starts
    /// at `f * hop / rate`, and the flushed remainder ends where the audio
    /// does. `push_at` moves all of it by anchoring.
    #[test]
    #[serial]
    fn test_segments_sit_on_the_clock() {
        let device = Device::default();
        let driver = driver(&device, false);
        let audio = clip();
        let hop = driver.mel().hop() as f64;
        let width = driver.window_frames();

        let mut ctx = driver.new_context(clock(), PerWindow).unwrap();
        let mut emissions = ctx.push_at(&audio, 100.0).unwrap();
        emissions.extend(ctx.flush().unwrap());

        let window_seconds = width as f64 * hop / driver.sample_rate() as f64;
        for (w, e) in emissions.iter().enumerate() {
            let segment = e.segment();
            assert!((segment.start - (100.0 + w as f64 * window_seconds)).abs() < 1e-9);
            if w + 1 < emissions.len() {
                assert!((segment.end - segment.start - window_seconds).abs() < 1e-9);
            }
        }

        // 16800 samples -> 104 frames streamed, +2 from finish, -1 trimmed:
        // 105, one per hop, and the remainder ends there.
        let last = emissions.last().unwrap().segment();
        assert_eq!(ctx.seek(), 105);
        assert!((last.end - (100.0 + 105.0 * hop / driver.sample_rate() as f64)).abs() < 1e-9);
        assert_eq!(ctx.pending_frames(), 0);
    }

    /// **I6.** A provisional decode leaves the context byte-identical, and
    /// says the same thing twice.
    #[test]
    #[serial]
    fn test_probe_decode_leaves_the_context_untouched() {
        let device = Device::default();
        let driver = driver(&device, false);
        let audio = clip();

        let mut ctx = driver.new_context(clock(), MaxSeen::new()).unwrap();
        // Enough for one committed window and a partial second one.
        let committed = ctx.push(&audio[..4_000]).unwrap();
        assert_eq!(committed.len(), 1);
        assert!(ctx.pending_frames() > 0);

        let before = ctx.fingerprint();
        let first = ctx.probe_decode().expect("frames are pending");
        let second = ctx.probe_decode().expect("frames are pending");
        assert_eq!(ctx.fingerprint(), before);
        assert_eq!(first, second);
        assert_eq!(ctx.transcript(), committed[0].segment().tokens);
    }

    /// With carrying on, a window is prompted with the transcript's tail
    /// after `<|startofprev|>`, exactly as decoding it by hand with that
    /// prompt would.
    #[test]
    #[serial]
    fn test_prompt_carry() {
        let device = Device::default();
        let driver = driver(&device, true);
        let audio = clip();
        let width = driver.window_frames();

        let mut ctx = driver.new_context(clock(), PerWindow).unwrap();
        let mut emissions = ctx.push(&audio).unwrap();
        emissions.extend(ctx.flush().unwrap());
        let got = tokens_of(&emissions);

        // By hand: the first window with the bare prompt, the second with
        // the carry built from the first's ids.
        let trimmed = trim_stream_tail(joined_mels(&driver, &audio, &device));
        let window_at = |w: usize| {
            let window = trimmed
                .clone()
                .slice_dim(1, (w * width) as isize..((w + 1) * width) as isize);
            driver
                .front_end()
                .package_window(window.clone(), PerWindow.reference(&window))
        };
        let ids = driver.policy().ids();
        let first = driver
            .model()
            .decode_window(window_at(0), &greedy_config(&driver));
        assert_eq!(got[0], first);

        let keep = driver.model().max_text_ctx() / 2 - 1;
        let mut prompt = vec![ids.sot_prev];
        prompt.extend_from_slice(&first[first.len().saturating_sub(keep)..]);
        prompt.extend_from_slice(driver.prompt());
        let carried = GreedyDecodeConfig::new(prompt, ids.eot).with_max_tokens(driver.max_tokens());
        let second = driver.model().decode_window(window_at(1), &carried);
        assert_eq!(got[1], second);

        // Without carrying, the same audio prompts every window the same
        // way, so this is only a real test if the two differ somewhere.
        let bare = driver
            .model()
            .decode_window(window_at(1), &greedy_config(&driver));
        assert!(
            !first.is_empty() || second != bare,
            "the tiny model emitted nothing to carry",
        );
    }

    /// A detokenizer on the driver puts text on every segment.
    #[test]
    #[serial]
    fn test_text_through_the_detokenizer() {
        #[derive(Debug)]
        struct Numbers;
        impl Detokenizer for Numbers {
            fn detokenize(
                &self,
                ids: &[i64],
            ) -> BunsenResult<String> {
                Ok(ids.iter().map(i64::to_string).collect::<Vec<_>>().join(" "))
            }
        }

        let device = Device::default();
        let driver = driver(&device, false).with_detokenizer(Arc::new(Numbers));
        let audio = clip();

        let mut ctx = driver.new_context(clock(), PerWindow).unwrap();
        let mut emissions = ctx.push(&audio).unwrap();
        emissions.extend(ctx.flush().unwrap());

        for e in &emissions {
            let segment = e.segment();
            let text_ids = driver.policy().text_ids(&segment.tokens);
            assert_eq!(
                segment.text.as_deref(),
                Some(Numbers.detokenize(&text_ids).unwrap().as_str())
            );
        }
    }

    /// The stream's lifecycle edges: nothing pushed, a stream shorter than a
    /// hop, flush twice, push after flush, feed and advance apart.
    #[test]
    #[serial]
    fn test_lifecycle_edges() {
        let device = Device::default();
        let driver = driver(&device, false);

        let mut empty = driver.new_context(clock(), PerWindow).unwrap();
        assert!(empty.flush().unwrap().is_empty());
        assert!(empty.flush().unwrap().is_empty(), "flush is idempotent");
        assert!(empty.push(&[0.0; 16]).is_err(), "no pushing after flush");
        assert!(empty.feed(&[0.0; 16]).is_err(), "nor feeding");
        assert!(empty.is_finished());
        assert!(!empty.is_speaking());
        assert_eq!(empty.regions_pending(), 0);

        // Shorter than a hop: padded with silence to something decodable.
        let mut short = driver.new_context(clock(), PerWindow).unwrap();
        assert!(short.push(&[0.1; 50]).unwrap().is_empty());
        let emissions = short.flush().unwrap();
        assert_eq!(emissions.len(), 1);
        assert!(short.pending_frames() == 0);

        // Feed then advance is push; end_input then advance is flush.
        let audio = clip();
        let mut split = driver.new_context(clock(), PerWindow).unwrap();
        split.feed(&audio).unwrap();
        assert_eq!(split.transcript().len(), 0, "feeding decodes nothing");
        let mut got = split.advance().unwrap();
        split.end_input().unwrap();
        got.extend(split.advance().unwrap());

        let mut pushed = driver.new_context(clock(), PerWindow).unwrap();
        let mut expected = pushed.push(&audio).unwrap();
        expected.extend(pushed.flush().unwrap());
        assert_eq!(got, expected);
    }

    /// **Batching is scheduling, nothing else.** Two streams advanced
    /// together commit exactly what each commits alone: same windows, same
    /// times, same ids &mdash; and a third stream with no audio is left
    /// alone.
    #[test]
    #[serial]
    fn test_advance_ready_matches_solo() {
        let device = Device::default();
        let driver = driver(&device, false).with_logit_filters(decisive());
        let clips = [clip_seeded(0.5, 440.0), clip_seeded(0.3, 660.0)];

        let mut solo = Vec::new();
        for audio in &clips {
            let mut ctx = driver.new_context(clock(), PerWindow).unwrap();
            let mut emissions = ctx.push(audio).unwrap();
            emissions.extend(ctx.flush().unwrap());
            solo.push(emissions);
        }
        if tokens_of(&solo[0]) == tokens_of(&solo[1]) {
            eprintln!(
                "the seeded tiny model says the same of both clips; scheduling is still pinned"
            );
        }

        let mut contexts: Vec<WhisperStreamContext<B>> = (0..3)
            .map(|_| driver.new_context(clock(), PerWindow).unwrap())
            .collect();
        // The two clips go in whole, so the front end's arithmetic is the
        // solo run's; what differs is only that six windows per stream are
        // decoded twelve at a time. The third stream never gets audio.
        for (ctx, audio) in contexts.iter_mut().zip(&clips) {
            ctx.feed(audio).unwrap();
        }
        let mut batched = advance_ready(&driver, &mut contexts).unwrap();
        assert_eq!(batched[0].len(), 6, "six full windows before the end");
        for ctx in contexts.iter_mut().take(2) {
            ctx.end_input().unwrap();
        }
        let tail = advance_ready(&driver, &mut contexts).unwrap();
        for (all, more) in batched.iter_mut().zip(tail) {
            all.extend(more);
        }

        assert_eq!(batched[0], solo[0]);
        assert_eq!(batched[1], solo[1]);
        assert!(batched[2].is_empty());
        assert_eq!(contexts[0].seek(), 105);
    }

    /// Batching is by prompt: a stream that has already committed a window
    /// carries a different prompt from one that has not, so the two form
    /// two groups in one pass, and each decodes as it would alone. The
    /// decode is dictated from the prompt, so a window batched under the
    /// wrong prompt would show.
    #[test]
    #[serial]
    fn test_advance_ready_with_prompt_carry() {
        let device = Device::default();
        let driver = driver(&device, true).with_logit_filters(decisive());
        let audio = clip();
        // 4000 samples: one 16-frame window and change.
        let first = RATE / 4;

        // Solo: A decodes its first window early, then the rest; B all at
        // once.
        let mut a = driver.new_context(clock(), PerWindow).unwrap();
        let mut solo_a = a.push(&audio[..first]).unwrap();
        assert_eq!(solo_a.len(), 1, "one window committed early");
        solo_a.extend(a.push(&audio[first..]).unwrap());
        solo_a.extend(a.flush().unwrap());
        let mut b = driver.new_context(clock(), PerWindow).unwrap();
        let mut solo_b = b.push(&audio).unwrap();
        solo_b.extend(b.flush().unwrap());
        assert_ne!(
            solo_a[0].segment().tokens,
            solo_a[1].segment().tokens,
            "the carried prompt changes what is decoded"
        );

        // Batched: A has committed its first window, B nothing; fed the
        // rest, they advance together under two prompts.
        let mut ctx_a = driver.new_context(clock(), PerWindow).unwrap();
        let mut batched_a = ctx_a.push(&audio[..first]).unwrap();
        ctx_a.feed(&audio[first..]).unwrap();
        ctx_a.end_input().unwrap();
        let mut ctx_b = driver.new_context(clock(), PerWindow).unwrap();
        ctx_b.feed(&audio).unwrap();
        ctx_b.end_input().unwrap();
        let mut contexts = vec![ctx_a, ctx_b];
        let out = advance_ready(&driver, &mut contexts).unwrap();
        batched_a.extend(out[0].clone());
        assert_eq!(batched_a, solo_a);
        assert_eq!(out[1], solo_b);
    }

    /// Dictates the decode: whatever the untrained model thinks (its
    /// near-ties flip between runs and backends), the next token is the
    /// scripted one for its position in the window. The script obeys the
    /// timestamp grammar, so the rules, applied after it, pass it through.
    #[derive(Debug)]
    struct Script(Vec<i64>);

    impl<Bk: Backend> LogitFilter<Bk> for Script {
        fn apply(
            &self,
            logits: Tensor<Bk, 2>,
            tokens: &[Vec<i64>],
            prompt_len: usize,
        ) -> Tensor<Bk, 2> {
            let [rows, vocab] = logits.dims();
            let mut data = vec![f32::NEG_INFINITY; rows * vocab];
            for (row, t) in tokens.iter().enumerate() {
                let at = (t.len() - prompt_len).min(self.0.len() - 1);
                data[row * vocab + self.0[at] as usize] = 0.0;
            }
            Tensor::from_data(
                burn::tensor::TensorData::new(data, [rows, vocab]),
                &logits.device(),
            )
        }
    }

    /// Dictates the decode from the prompt: at sampled position `at` the
    /// token is `(prompt_len + at) % 4 + 1`, so streams under different
    /// prompts decode differently, and a window batched under the wrong
    /// prompt would show. Independent of the model, whose forward on this
    /// backend varies from call to call by more than a near-tie.
    #[derive(Debug)]
    struct Echo;

    impl LogitFilter<B> for Echo {
        fn apply(
            &self,
            logits: Tensor<B, 2>,
            tokens: &[Vec<i64>],
            prompt_len: usize,
        ) -> Tensor<B, 2> {
            let [rows, vocab] = logits.dims();
            let mut data = vec![f32::NEG_INFINITY; rows * vocab];
            for (row, t) in tokens.iter().enumerate() {
                let at = t.len() - prompt_len;
                data[row * vocab + (prompt_len + at) % 4 + 1] = 0.0;
            }
            Tensor::from_data(
                burn::tensor::TensorData::new(data, [rows, vocab]),
                &logits.device(),
            )
        }
    }

    /// The filters that make a plumbing test's decode a function of its
    /// prompt alone.
    fn decisive() -> Vec<Arc<dyn LogitFilter<B>>> {
        vec![Arc::new(Echo)]
    }

    /// Each window's script: two closed segments and a reopened tail,
    /// `<|1|> a <|3|><|3|> b <|5|><|5|> c`, at the cap of 8 tokens. The
    /// first timestamp is within the cap of index 2, so the cap is a real
    /// constraint that the script satisfies; the seek advances 10 frames
    /// of 16 per decode, so windows overlap.
    fn script() -> Vec<i64> {
        let tb = tiny_layout().timestamp_begin;
        vec![tb + 1, 1, tb + 3, tb + 3, 2, tb + 5, tb + 5, 3]
    }

    /// A driver that emits timestamps under the script. No prompt carry:
    /// the tiny model's text context is 16, and a carried prompt would
    /// leave the script no room.
    fn timestamped(
        device: &Device,
        commit: CommitRule,
    ) -> WhisperDriver<B> {
        let scripted: Arc<dyn LogitFilter<B>> = Arc::new(Script(script()));
        config(false)
            .with_max_tokens(8)
            .with_timestamps(true)
            .with_max_initial_timestamp(Some(0.04))
            .with_emission(EmissionPolicy::new(Triggers::new(), commit))
            .init_with_policy(tiny_model(device), TokenPolicy::new(tiny_layout()), device)
            .unwrap()
            .with_logit_filters(vec![scripted])
    }

    /// Pushes a clip through a fresh context and flushes it.
    fn run_clip(
        driver: &WhisperDriver<B>,
        audio: &[f32],
        at: Option<f64>,
    ) -> Vec<Emission> {
        let mut ctx = driver.new_context(clock(), PerWindow).unwrap();
        let mut emissions = match at {
            Some(time) => ctx.push_at(audio, time).unwrap(),
            None => ctx.push(audio).unwrap(),
        };
        emissions.extend(ctx.flush().unwrap());
        assert_eq!(ctx.pending_frames(), 0, "a flush consumes everything");
        assert_eq!(ctx.seek(), 105);
        let committed: Vec<i64> = emissions
            .iter()
            .filter(|e| e.is_committed())
            .flat_map(|e| e.segment().tokens.clone())
            .collect();
        assert_eq!(
            ctx.transcript(),
            &committed[..],
            "the transcript is what was committed"
        );
        emissions
    }

    /// With timestamps on, a window's decode is split on its consecutive
    /// timestamps into segments on the clock, and the seek advances to the
    /// last closed timestamp rather than a whole window: 105 frames at 10
    /// per decode is 11 decodes, 22 segments, each 4 frames long.
    #[test]
    #[serial]
    fn test_timestamps_split_windows_into_segments() {
        let device = Device::default();
        let driver = timestamped(&device, CommitRule::Complete);
        let tb = tiny_layout().timestamp_begin;
        let hop = driver.mel().hop() as f64;
        let frame = |f: f64| f * hop / driver.sample_rate() as f64;
        assert_eq!(driver.frames_per_timestamp(), 2);
        assert_eq!(
            driver.filters().len(),
            2,
            "the test's filter, then the rules appended by the driver"
        );

        let emissions = run_clip(&driver, &clip(), None);
        assert_eq!(emissions.len(), 22);

        for (k, e) in emissions.iter().enumerate() {
            assert!(e.is_committed(), "Complete never drafts");
            let s = e.segment();
            let (window, which) = (k / 2, k % 2);
            let seek = (window * 10) as f64;
            let expected = if which == 0 {
                (vec![tb + 1, 1, tb + 3], seek + 2.0, seek + 6.0)
            } else {
                (vec![tb + 3, 2, tb + 5], seek + 6.0, seek + 10.0)
            };
            assert_eq!(s.tokens, expected.0, "segment {k}");
            assert!(
                (s.start - frame(expected.1)).abs() < 1e-9,
                "segment {k}: {s:?}"
            );
            assert!(
                (s.end - frame(expected.2)).abs() < 1e-9,
                "segment {k}: {s:?}"
            );
        }
    }

    /// **I8.** The same clip on a clock anchored at 100 s yields the same
    /// segments, shifted by 100 s.
    #[test]
    #[serial]
    fn test_segment_times_are_invariant_to_clock_origin() {
        let device = Device::default();
        let driver = timestamped(&device, CommitRule::Complete);
        let audio = clip();

        let base = run_clip(&driver, &audio, None);
        let shifted = run_clip(&driver, &audio, Some(100.0));
        assert_eq!(base.len(), shifted.len());
        for (a, b) in base.iter().zip(&shifted) {
            assert_eq!(a.segment().tokens, b.segment().tokens);
            assert!((b.segment().start - a.segment().start - 100.0).abs() < 1e-9);
            assert!((b.segment().end - a.segment().end - 100.0).abs() < 1e-9);
        }
    }

    /// Under `LastTimestamp` the unfinished tail of a window is emitted as
    /// a draft opening on its timestamp, and what is committed is exactly
    /// what `Complete` commits.
    #[test]
    #[serial]
    fn test_last_timestamp_drafts_the_tail() {
        let device = Device::default();
        let audio = clip();
        let complete = run_clip(&timestamped(&device, CommitRule::Complete), &audio, None);
        let last = run_clip(
            &timestamped(&device, CommitRule::LastTimestamp),
            &audio,
            None,
        );

        let committed: Vec<&Emission> = last.iter().filter(|e| e.is_committed()).collect();
        assert_eq!(committed.len(), complete.len());
        for (a, b) in committed.iter().zip(&complete) {
            assert_eq!(a.segment(), b.segment());
        }

        let tb = tiny_layout().timestamp_begin;
        let drafts: Vec<&Emission> = last.iter().filter(|e| !e.is_committed()).collect();
        // One per decode, except the last: its 5 frames end before the
        // tail's timestamp at frame 10, so there is nothing to draft yet.
        assert_eq!(drafts.len(), 10);
        for draft in drafts {
            let s = draft.segment();
            assert_eq!(s.tokens, vec![tb + 5, 3], "the reopened tail");
            assert!(s.start <= s.end);
        }
    }

    /// Degenerate at temperature zero, sound once sampling has broken the
    /// loop: while the sampled history is all `1`s (or empty) the logits
    /// are nearly flat with `1` a hair ahead, so the argmax loops on `1` at
    /// a log probability near `-log(vocab)`; once any other token appears,
    /// which only sampling can bring, the logits peak on `2`.
    #[derive(Debug)]
    struct Degenerate;

    impl LogitFilter<B> for Degenerate {
        fn apply(
            &self,
            logits: Tensor<B, 2>,
            tokens: &[Vec<i64>],
            prompt_len: usize,
        ) -> Tensor<B, 2> {
            let [rows, vocab] = logits.dims();
            let mut data = vec![-0.01f32; rows * vocab];
            for (row, t) in tokens.iter().enumerate() {
                let at = row * vocab;
                if t[prompt_len..].iter().all(|&x| x == 1) {
                    data[at + 1] = 0.0;
                } else {
                    data[at..at + vocab].fill(0.0);
                    data[at + 2] = 30.0;
                }
            }
            Tensor::from_data(
                burn::tensor::TensorData::new(data, [rows, vocab]),
                &logits.device(),
            )
        }
    }

    /// A carrying driver under the degenerate filter and a fallback
    /// policy.
    fn degenerate_driver(
        device: &Device,
        fallback: FallbackConfig,
    ) -> WhisperDriver<B> {
        let filter: Arc<dyn LogitFilter<B>> = Arc::new(Degenerate);
        config(true)
            .with_max_tokens(8)
            .with_fallback(fallback)
            .init_with_policy(tiny_model(device), TokenPolicy::new(tiny_layout()), device)
            .unwrap()
            .with_logit_filters(vec![filter])
    }

    /// The ladder: at temperature zero the decode loops and fails the log
    /// probability threshold; without a ladder that is what is emitted.
    /// With upstream's ladder, sampling at the first rung above zero breaks
    /// the loop and the decode recovers, below the prompt-reset line, so
    /// the carry stands. A ladder whose only rung above zero is past that
    /// line recovers too, and resets the carry.
    #[test]
    #[serial]
    fn test_fallback_ladder_recovers() {
        let device = Device::default();
        let audio = clip();
        let flat = degenerate_driver(&device, FallbackConfig::new());
        let mut ctx = flat.new_context(clock(), PerWindow).unwrap();
        let mut looped = ctx.push(&audio).unwrap();
        looped.extend(ctx.flush().unwrap());
        assert!(!looped.is_empty());
        // Eight 1s in the first window; fewer after, once the carried
        // prompt has taken its share of the tiny model's 16-token context.
        assert!(
            looped
                .iter()
                .all(|e| !e.segment().tokens.is_empty()
                    && e.segment().tokens.iter().all(|&t| t == 1)),
            "no ladder: the loop is what there is: {looped:?}"
        );
        let carry_len = ctx.prompt_now().len();
        assert!(carry_len > flat.prompt().len(), "the carry stands");

        let climbing = degenerate_driver(&device, FallbackConfig::upstream());
        let mut ctx = climbing.new_context(clock(), PerWindow).unwrap();
        let mut recovered = ctx.push(&audio).unwrap();
        recovered.extend(ctx.flush().unwrap());
        assert_eq!(recovered.len(), looped.len());
        for e in &recovered {
            let t = &e.segment().tokens;
            assert!(t.contains(&2), "sampling broke the loop: {t:?}");
        }
        assert!(
            ctx.prompt_now().len() > climbing.prompt().len(),
            "recovered below 0.5: the carry stands"
        );

        let hot = degenerate_driver(
            &device,
            FallbackConfig::new().with_temperatures(vec![0.0, 0.8]),
        );
        let mut ctx = hot.new_context(clock(), PerWindow).unwrap();
        let mut reset = ctx.push(&audio).unwrap();
        reset.extend(ctx.flush().unwrap());
        assert!(reset.iter().all(|e| e.segment().tokens.contains(&2)));
        assert_eq!(
            ctx.prompt_now(),
            hot.prompt(),
            "past 0.5: the carry was reset"
        );
    }

    /// The silence rule: a window whose no-speech probability is over the
    /// threshold and whose log probability is poor is skipped whole, the
    /// seek moving past it with nothing emitted; a good enough log
    /// probability keeps it.
    #[test]
    #[serial]
    fn test_no_speech_skips_a_window() {
        let device = Device::default();
        let audio = clip();

        let skipping = degenerate_driver(
            &device,
            FallbackConfig::new().with_no_speech_threshold(Some(0.0)),
        );
        let mut ctx = skipping.new_context(clock(), PerWindow).unwrap();
        let mut emissions = ctx.push(&audio).unwrap();
        emissions.extend(ctx.flush().unwrap());
        assert!(
            emissions.is_empty(),
            "every window is silence: {emissions:?}"
        );
        assert_eq!(ctx.seek(), 105, "the seek still moved past them");
        assert!(ctx.transcript().is_empty());

        let kept = degenerate_driver(
            &device,
            FallbackConfig::new()
                .with_no_speech_threshold(Some(0.0))
                .with_logprob_threshold(Some(-1e9)),
        );
        let mut ctx = kept.new_context(clock(), PerWindow).unwrap();
        let mut emissions = ctx.push(&audio).unwrap();
        emissions.extend(ctx.flush().unwrap());
        assert!(
            !emissions.is_empty(),
            "a good enough log probability keeps a window"
        );
    }

    /// A multilingual driver given no language detects it from the first
    /// window: on a layout with one language, that one, and the stream then
    /// decodes exactly as one configured with it. Solo and batched agree.
    #[test]
    #[serial]
    fn test_language_is_detected_per_stream() {
        let device = Device::default();
        let detecting = WhisperDriverConfig::new()
            .with_max_tokens(4)
            .init_with_policy(
                tiny_model(&device),
                TokenPolicy::new(tiny_layout()),
                &device,
            )
            .unwrap()
            .with_logit_filters(decisive());
        assert!(detecting.detects_language());
        assert!(detecting.prompt().is_empty());
        let only = crate::kits::speech::whisper::blocks::LANGUAGES[0];
        assert_eq!(only, "en");

        let audio = clip();
        let mut solo = detecting.new_context(clock(), PerWindow).unwrap();
        assert_eq!(solo.language(), None);
        let mut expected = solo.push(&audio).unwrap();
        expected.extend(solo.flush().unwrap());
        assert_eq!(solo.language(), Some(only));

        // The same as configuring it.
        let configured = driver(&device, true).with_logit_filters(decisive());
        assert_eq!(run_clip(&configured, &audio, None), expected);

        // Batched: fed, then advanced together.
        let mut batch = vec![
            detecting.new_context(clock(), PerWindow).unwrap(),
            detecting.new_context(clock(), PerWindow).unwrap(),
        ];
        for ctx in &mut batch {
            ctx.feed(&audio).unwrap();
            ctx.end_input().unwrap();
        }
        let out = advance_ready(&detecting, &mut batch).unwrap();
        assert_eq!(out[0], expected);
        assert_eq!(out[1], expected);
        assert_eq!(batch[1].language(), Some(only));
    }

    /// The driver reports the model's rate, and the grid derived from it.
    #[test]
    fn test_driver_reports_the_models_rate() {
        let device = Device::default();
        let driver = WhisperDriverConfig::new()
            .init_with_policy(
                tiny_model(&device),
                TokenPolicy::new(tiny_layout()),
                &device,
            )
            .unwrap();
        assert_eq!(driver.sample_rate(), 16_000);
        assert_eq!(driver.mel().hop(), 160);
        assert_eq!(driver.frames_per_timestamp(), 2);
        assert_eq!(driver.encoder_grid(), 320);
        assert_eq!(driver.interval_samples(), None);
    }

    /// The configuration refuses what this slice cannot do, with a reason,
    /// and refuses a mismatched language.
    #[test]
    fn test_init_refuses_the_unsupported() {
        let device = Device::default();
        let policy = TokenPolicy::new(tiny_layout());
        let base = WhisperDriverConfig::new().with_language(Some("en".to_string()));

        assert!(
            base.init_with_policy(tiny_model(&device), policy.clone(), &device)
                .is_ok()
        );
        // Multilingual without a language detects it per stream.
        assert!(
            WhisperDriverConfig::new()
                .init_with_policy(tiny_model(&device), policy.clone(), &device)
                .unwrap()
                .detects_language()
        );
        assert!(
            base.clone()
                .with_timestamps(true)
                .init_with_policy(tiny_model(&device), policy.clone(), &device)
                .is_ok()
        );
        assert!(
            base.clone()
                .with_emission(EmissionPolicy::responsive())
                .init_with_policy(tiny_model(&device), policy.clone(), &device)
                .is_ok(),
            "responsive is the third deployment target",
        );
        assert!(
            base.clone()
                .with_emission(EmissionPolicy::new(
                    Triggers::new().with_interval(Some(std::time::Duration::ZERO)),
                    CommitRule::Complete,
                ))
                .init_with_policy(tiny_model(&device), policy.clone(), &device)
                .is_err(),
            "an interval of zero",
        );
        assert!(
            base.clone()
                .with_language(Some("xx".to_string()))
                .init_with_policy(tiny_model(&device), policy.clone(), &device)
                .is_err()
        );

        // Conservative is constructible, but a stream under it needs a VAD.
        let conservative = base
            .clone()
            .with_emission(EmissionPolicy::conservative())
            .init_with_policy(tiny_model(&device), policy, &device)
            .unwrap();
        assert!(conservative.new_context(clock(), PerWindow).is_err());

        // A clock at the wrong rate is refused at the stream, not later.
        let driver = driver(&device, false);
        assert!(
            driver
                .new_context(TimestampHistory::uniform(8_000), PerWindow)
                .is_err()
        );
    }

    /// Against the real voice-activity model: regions become segments in
    /// the parent stream's frame of reference, and silence is never
    /// decoded.
    #[cfg(feature = "silero-weights")]
    mod voice_activity {
        use super::*;
        use crate::{
            kits::speech::{
                silero_vad::SileroVad,
                whisper::driver::{
                    CommitRule,
                    Triggers,
                    VoiceActivityFilterConfig,
                },
            },
            support::{
                audio::load_audio_mono_sr,
                testing::CpuBackend,
            },
        };

        /// Silero and the tiny model on the CPU backend, so the golden
        /// regions from the gate's own test hold here too.
        type C = CpuBackend;
        type CDevice = burn::prelude::Device<C>;

        fn speech() -> Vec<f32> {
            let path = concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/testdata/audio/jfk_moon_4s.mp3"
            );
            load_audio_mono_sr(path, RATE).unwrap()
        }

        fn conservative_driver(device: &CDevice) -> WhisperDriver<C> {
            let vad = SileroVad::<C>::load_16khz_pretrained(device).unwrap();
            config(false)
                .with_emission(EmissionPolicy::conservative())
                .init_with_policy(
                    tiny_model_on::<C>(device),
                    TokenPolicy::new(tiny_layout()),
                    device,
                )
                .unwrap()
                .with_vad(vad, VoiceActivityFilterConfig::fast_whisper_burn())
                .unwrap()
        }

        /// A model or a filter at another rate, or a filter with the wrong
        /// chunk, is refused at attach time; the matching pair is not.
        #[test]
        #[serial]
        fn test_with_vad_refuses_a_mismatch() {
            let device = CDevice::default();
            let driver = || {
                config(false)
                    .with_emission(EmissionPolicy::conservative())
                    .init_with_policy(
                        tiny_model_on::<C>(&device),
                        TokenPolicy::new(tiny_layout()),
                        &device,
                    )
                    .unwrap()
            };
            let filter = VoiceActivityFilterConfig::fast_whisper_burn;

            let eight = SileroVad::<C>::load_8khz_pretrained(&device).unwrap();
            assert!(driver().with_vad(eight, filter()).is_err());

            let vad = SileroVad::<C>::load_16khz_pretrained(&device).unwrap();
            assert!(
                driver()
                    .with_vad(vad.clone(), filter().with_sample_rate(8_000))
                    .is_err()
            );
            assert!(
                driver()
                    .with_vad(vad.clone(), filter().with_samples_per_chunk(256))
                    .is_err()
            );
            assert!(driver().configure_vad_filter(|f| f).is_err(), "no VAD");

            let attached = driver().with_vad(vad, filter()).unwrap();
            assert_eq!(
                attached.filter_config().map(|f| f.samples_per_chunk),
                Some(512)
            );
            assert!(
                attached
                    .configure_vad_filter(|f| f.with_samples_per_chunk(256))
                    .is_err()
            );
        }

        /// The gate's golden regions, padded and snapped outward, are
        /// exactly what gets decoded: the segments tile each region, a
        /// window at a time, nothing lands in the pause between them, and
        /// their times are the parent clock's, anchor and all. Pushed in
        /// random pieces, so the VAD's own staging is exercised.
        #[test]
        #[serial]
        fn test_regions_become_segments_on_the_parent_clock() {
            let device = CDevice::default();
            let vad = SileroVad::<C>::load_16khz_pretrained(&device).unwrap();
            let regions_only = EmissionPolicy::new(
                Triggers::new().with_window_full(false).with_endpoint(true),
                CommitRule::LastTimestamp,
            );
            let driver = config(false)
                .with_emission(regions_only)
                .init_with_policy(
                    tiny_model_on::<C>(&device),
                    TokenPolicy::new(tiny_layout()),
                    &device,
                )
                .unwrap()
                .with_vad(vad, VoiceActivityFilterConfig::fast_whisper_burn())
                .unwrap();
            let audio = speech();
            assert_eq!(audio.len(), 64_000);

            let mut ctx = driver.new_context(clock(), PerWindow).unwrap();
            let sizes = random_sizes(11, audio.len());
            let mut emissions = Vec::new();
            let mut at = 0;
            for (k, &size) in sizes.iter().enumerate() {
                let piece = &audio[at..at + size];
                emissions.extend(if k == 0 {
                    ctx.push_at(piece, 100.0).unwrap()
                } else {
                    ctx.push(piece).unwrap()
                });
                at += size;
            }
            emissions.extend(ctx.flush().unwrap());

            // (0, 16352) and (29728, 56800) from the gate's golden test,
            // snapped outward onto the 320-sample grid.
            let expected = [(0usize, 16_640usize), (29_440, 56_960)];
            let window = driver.window_frames() as f64 * driver.mel().hop() as f64
                / driver.sample_rate() as f64;
            let close = |a: f64, b: f64| (a - b).abs() < 1e-9;

            let mut segments = emissions.iter().map(Emission::segment).peekable();
            for (start, end) in expected {
                let (start, end) = (
                    100.0 + start as f64 / RATE as f64,
                    100.0 + end as f64 / RATE as f64,
                );
                let mut at = start;
                while let Some(segment) = segments.next_if(|s| s.start < end) {
                    assert!(close(segment.start, at), "gap before {segment:?}");
                    assert!(segment.end - segment.start <= window + 1e-9);
                    assert!(segment.end <= end + 1e-9, "past the region: {segment:?}");
                    at = segment.end;
                }
                assert!(
                    close(at, end),
                    "region not covered to {end}: stopped at {at}\n{emissions:#?}"
                );
            }
            assert!(segments.next().is_none(), "segments outside every region");
            assert!(emissions.iter().all(Emission::is_committed));
            assert_eq!(ctx.regions_pending(), 0);
            assert!(!ctx.is_speaking());
            assert_eq!(
                emissions.len(),
                (104_usize).div_ceil(16) + (172_usize).div_ceil(16),
                "a window at a time over 104 and 172 frames",
            );
        }

        /// Under `conservative()` the full-window trigger runs too: a
        /// window with speech in progress decodes as it fills, so the window
        /// that opens a region may begin a little before it; but nothing is
        /// ever decoded from the pause, and both regions are covered.
        #[test]
        #[serial]
        fn test_conservative_skips_silence() {
            let device = CDevice::default();
            let driver = conservative_driver(&device);
            let audio = speech();

            let mut ctx = driver.new_context(clock(), PerWindow).unwrap();
            let mut emissions = ctx.push(&audio).unwrap();
            emissions.extend(ctx.flush().unwrap());

            let window = driver.window_frames() as f64 * driver.mel().hop() as f64
                / driver.sample_rate() as f64;
            let (gap_start, gap_end) = (16_640.0 / RATE as f64, 29_440.0 / RATE as f64);
            let mut covered_to = 0.0_f64;
            for e in &emissions {
                let s = e.segment();
                assert!(
                    !(s.start >= gap_start && s.end <= gap_end),
                    "a window of silence was decoded: {s:?}",
                );
                assert!(s.start >= covered_to - 1e-9, "out of order: {s:?}");
                if s.start > gap_start + 1e-9 {
                    assert!(
                        s.start >= gap_end - window - 1e-9,
                        "started deep in the pause: {s:?}"
                    );
                }
                covered_to = s.end;
            }
            assert!(
                (covered_to - 56_960.0 / RATE as f64).abs() < 1e-9,
                "covered to {covered_to}\n{emissions:#?}"
            );
            assert_eq!(ctx.regions_pending(), 0);
        }

        /// A responsive driver: conservative plus a draft every 50 ms of
        /// media time while speech is in progress &mdash; shorter than
        /// `responsive()`'s 600 ms because the tiny model's window is 0.16 s
        /// of audio, and a full window commits before a longer interval
        /// could draft. Its decode is scripted so the pin is about
        /// scheduling, not the untrained model.
        fn responsive_driver(device: &CDevice) -> WhisperDriver<C> {
            let vad = SileroVad::<C>::load_16khz_pretrained(device).unwrap();
            let scripted: Arc<dyn LogitFilter<C>> = Arc::new(Script(vec![3, 1, 4, 1]));
            let policy = EmissionPolicy::new(
                Triggers::new()
                    .with_endpoint(true)
                    .with_interval(Some(std::time::Duration::from_millis(50))),
                CommitRule::LastTimestamp,
            );
            config(false)
                .with_emission(policy)
                .init_with_policy(
                    tiny_model_on::<C>(device),
                    TokenPolicy::new(tiny_layout()),
                    device,
                )
                .unwrap()
                .with_vad(vad, VoiceActivityFilterConfig::fast_whisper_burn())
                .unwrap()
                .with_logit_filters(vec![scripted])
        }

        /// The clip pushed 100 ms at a time, then flushed.
        fn in_pieces(ctx: &mut WhisperStreamContext<C>) -> Vec<Emission> {
            let audio = speech();
            let mut out = Vec::new();
            for piece in audio.chunks(RATE / 10) {
                out.extend(ctx.push(piece).unwrap());
            }
            out.extend(ctx.flush().unwrap());
            out
        }

        /// **I9.** Adding the interval trigger cannot change the transcript:
        /// `responsive()` commits exactly what `conservative()` commits, and
        /// drafts besides: each covering the audio since the last commit,
        /// paced at least an interval apart, none after the last commit.
        #[test]
        #[serial]
        fn test_responsive_commits_what_conservative_commits() {
            let device = CDevice::default();
            let scripted: Arc<dyn LogitFilter<C>> = Arc::new(Script(vec![3, 1, 4, 1]));
            let conservative = conservative_driver(&device).with_logit_filters(vec![scripted]);
            let responsive = responsive_driver(&device);

            let mut ctx = conservative.new_context(clock(), PerWindow).unwrap();
            let quiet = in_pieces(&mut ctx);
            assert!(
                quiet.iter().all(|e| e.is_committed()),
                "conservative never drafts"
            );

            let mut ctx = responsive.new_context(clock(), PerWindow).unwrap();
            let chatty = in_pieces(&mut ctx);
            let committed: Vec<&Emission> = chatty.iter().filter(|e| e.is_committed()).collect();
            assert_eq!(committed.len(), quiet.len());
            for (a, b) in committed.iter().zip(&quiet) {
                assert_eq!(a.segment(), b.segment());
            }

            let drafts: Vec<&Segment> = chatty
                .iter()
                .filter(|e| !e.is_committed())
                .map(|e| e.segment())
                .collect();
            assert!(drafts.len() >= 3, "{} drafts", drafts.len());
            let interval = 0.05;
            let mut last_end = f64::NEG_INFINITY;
            let mut last_commit_end = 0.0;
            let mut seen_commits = 0;
            for e in &chatty {
                let s = e.segment();
                if e.is_committed() {
                    last_commit_end = s.end;
                    seen_commits += 1;
                    continue;
                }
                assert!(
                    s.start >= last_commit_end - 1e-9,
                    "a draft starts at the seek: {s:?}"
                );
                assert!(s.end > s.start, "{s:?}");
                assert!(
                    s.end - last_end >= interval - 1e-9,
                    "paced: {s:?} after {last_end}"
                );
                assert_eq!(s.tokens, vec![3, 1, 4, 1]);
                last_end = s.end;
            }
            assert!(seen_commits > 0);
            assert!(
                chatty.last().unwrap().is_committed(),
                "the flush commits everything; nothing is left to draft"
            );
        }

        /// Batched, a responsive stream drafts and commits exactly as it
        /// does alone: fed a piece at a time and advanced after each.
        #[test]
        #[serial]
        fn test_advance_ready_drafts_like_solo() {
            let device = CDevice::default();
            let driver = responsive_driver(&device);
            let audio = speech();

            let mut solo = driver.new_context(clock(), PerWindow).unwrap();
            let mut batch = vec![driver.new_context(clock(), PerWindow).unwrap()];
            let (mut expected, mut got) = (Vec::new(), Vec::new());
            for piece in audio.chunks(RATE / 10) {
                expected.extend(solo.push(piece).unwrap());
                batch[0].feed(piece).unwrap();
                got.extend(advance_ready(&driver, &mut batch).unwrap().remove(0));
            }
            expected.extend(solo.flush().unwrap());
            batch[0].end_input().unwrap();
            got.extend(advance_ready(&driver, &mut batch).unwrap().remove(0));
            assert_eq!(got, expected);
            assert!(
                expected.iter().any(|e| !e.is_committed()),
                "there were drafts"
            );
        }

        /// Under the offline policy an attached VAD is simply unused: the
        /// clip decodes as one window at the end, silence and all.
        #[test]
        #[serial]
        fn test_offline_ignores_the_vad() {
            let device = CDevice::default();
            let vad = SileroVad::<C>::load_16khz_pretrained(&device).unwrap();
            let driver = config(false)
                .init_with_policy(
                    tiny_model_on::<C>(&device),
                    TokenPolicy::new(tiny_layout()),
                    &device,
                )
                .unwrap()
                .with_vad(vad, VoiceActivityFilterConfig::fast_whisper_burn())
                .unwrap();
            let audio = speech();

            let mut ctx = driver.new_context(clock(), PerWindow).unwrap();
            let mut emissions = ctx.push(&audio).unwrap();
            emissions.extend(ctx.flush().unwrap());

            // 400 frames of 16 per window: 25 windows, no remainder.
            assert_eq!(emissions.len(), 25);
            assert_eq!(ctx.regions_pending(), 0);
        }
    }
}
