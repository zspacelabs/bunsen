//! # The stream context: one stream in progress.

use std::{
    collections::VecDeque,
    sync::Arc,
};

use burn::Tensor;

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::{
        asr_driver::{
            AsrEngine,
            AsrStreamDriver,
            CommitRule,
            DecodeRequest,
            LevelPolicy,
            LevelTracker,
            SpeakerAttribution,
            SpeechRegion,
            StreamClock,
            StreamingFeatures,
            TranscriptEvent,
            TranscriptSegment,
            VoiceActivityFilter,
            VoiceActivityStream,
            WindowDecode,
            support::frame_ring::FrameRing,
        },
        diarization::{
            SpeakerActivity,
            SpeakerActivityStream,
        },
    },
};

/// Advances every context that has a decode due, batching the decodes.
///
/// Repeats until no context has anything due, so a context with several
/// units due (a burst of closed regions, a long write) advances through all
/// of them. Each round takes one unit from every context that has one and
/// hands them to the engine together; each context's events are what its
/// own [`read`](AsrStreamContext::read) would have returned.
///
/// # Arguments
/// * `driver` - the driver the contexts were opened from.
/// * `contexts` - the streams, fed through [`write`](AsrStreamContext::write)
///   rather than [`write_read`](AsrStreamContext::write_read), so that nothing
///   has been decoded yet.
///
/// # Errors
/// As [`AsrStreamContext::read`].
pub fn advance_ready<E: AsrEngine>(
    driver: &AsrStreamDriver<E>,
    contexts: &mut [AsrStreamContext<E>],
) -> BunsenResult<Vec<Vec<TranscriptEvent>>> {
    let mut out: Vec<Vec<TranscriptEvent>> = vec![Vec::new(); contexts.len()];

    loop {
        // One due unit per context: a commit if one is due, else a draft.
        let mut due: Vec<(usize, Due, bool)> = Vec::new();
        for (i, ctx) in contexts.iter_mut().enumerate() {
            ctx.skip_silence();
            if let Some(unit) = ctx.next_due() {
                due.push((i, unit, false));
            } else if let Some(unit) = ctx.draft_unit() {
                due.push((i, unit, true));
            }
        }
        if due.is_empty() {
            for (i, ctx) in contexts.iter_mut().enumerate() {
                out[i].extend(ctx.release());
            }
            return Ok(out);
        }

        let requests = due
            .iter()
            .map(|&(i, unit, draft)| contexts[i].request(unit, draft))
            .collect();
        let decodes = driver.engine().decode(requests)?;

        for ((i, unit, draft), decode) in due.into_iter().zip(decodes) {
            let ctx = &mut contexts[i];
            if draft {
                let event = ctx.draft_from(unit, decode)?;
                out[i].extend(event);
            } else {
                out[i].extend(ctx.commit_due(unit, decode)?);
            }
        }
    }
}

/// One stream: the only stateful type of the driver.
///
/// Opened by [`AsrStreamDriver::new_context`], with the stream's
/// [`StreamClock`] and its [`LevelPolicy`]; it holds a clone of the driver,
/// so a driver opens as many as it has streams. Samples go in through
/// [`write_read`](Self::write_read), or [`write`](Self::write) and
/// [`read`](Self::read) apart, and come out as [`TranscriptEvent`]s;
/// [`end_read`](Self::end_read) ends the stream and decodes what is left.
///
/// A decode reads the context through `&self`, and only a commit changes
/// it, so drafting never changes what the stream commits.
#[derive(Clone, Debug)]
pub struct AsrStreamContext<E: AsrEngine> {
    driver: AsrStreamDriver<E>,

    /// The feature front end's streaming state.
    features: E::Features,

    /// The features from `origin` onward.
    frames: FrameRing,

    /// Samples pushed so far.
    samples_seen: usize,

    /// The stream frame index of the ring's first frame.
    origin: usize,

    /// The stream frame index the next window starts at.
    seek: usize,

    clock: StreamClock,

    level: LevelTracker<E::Level>,

    /// The engine's carry for this stream.
    state: E::StreamState,

    /// Every committed id, in order.
    transcript: Vec<i64>,

    /// Media time (samples seen) at the last draft or commit; drafts are
    /// paced from it.
    last_draft: usize,

    /// Voice activity, when the driver has a VAD and the policy wants
    /// endpoints.
    vad: Option<VoiceActivity>,

    /// Speaker activity, when the driver has a diarizer.
    speakers: Option<SpeakerTimeline>,

    finished: bool,
}

/// The speaker half of a stream: the diarizer's state, the activity it has
/// finalized, and the events waiting for it.
#[derive(Clone, Debug)]
struct SpeakerTimeline {
    stream: Box<dyn SpeakerActivityStream>,

    attribution: Arc<dyn SpeakerAttribution>,

    /// Finalized frames, from a margin before the oldest stretch that may
    /// still be attributed.
    activity: SpeakerActivity,

    /// The diarizer's input has ended: every frame is final.
    finished: bool,

    /// Committed segments, oldest first, each with its stream samples,
    /// waiting for the diarizer to finalize the frames they cover.
    held: VecDeque<(usize, usize, TranscriptEvent)>,

    /// The latest draft produced while commits were held: it follows them
    /// out.
    held_draft: Option<(usize, usize, TranscriptEvent)>,
}

/// Frames kept before the oldest stretch still to be attributed, for a rule
/// that reads a little before a segment's start.
const SPEAKER_MARGIN_FRAMES: usize = 8;

/// The voice-activity half of a stream: the model's state, the filter, and
/// the regions it has closed but the decode has not yet consumed.
#[derive(Clone, Debug)]
struct VoiceActivity {
    stream: Box<dyn VoiceActivityStream>,

    filter: VoiceActivityFilter,

    /// The samples in one chunk.
    chunk: usize,

    /// Samples not yet a whole chunk.
    staging: Vec<f32>,

    /// Padding on each side of a region, in samples.
    pad: usize,

    /// The grid closed regions are snapped onto, in samples.
    grid: usize,

    /// Closed regions, padded and snapped, oldest first.
    regions: VecDeque<SpeechRegion>,

    /// Where the last closed region ended, so the next one's padding cannot
    /// reach back into it.
    last_end: usize,
}

/// One decode unit: `count` frames from stream frame `start`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Due {
    start: usize,
    count: usize,
}

impl<E: AsrEngine> AsrStreamContext<E> {
    /// Initialize a new context.
    ///
    /// [`AsrStreamDriver::new_context`] is the way in: it checks that the
    /// clock runs at the engine's rate and that a policy wanting endpoints
    /// has a voice-activity model, which this does not.
    ///
    /// # Errors
    /// As the engine's [`new_features`](AsrEngine::new_features).
    pub fn init(
        driver: AsrStreamDriver<E>,
        clock: StreamClock,
        level: LevelPolicy,
    ) -> BunsenResult<Self> {
        let engine = driver.engine();
        let features = engine.new_features()?;
        let state = engine.new_stream_state();

        let vad = match (driver.vad(), driver.va_filter_config()) {
            (Some(model), Some(gate)) if driver.emission().triggers.endpoint => {
                Some(VoiceActivity {
                    stream: model.new_stream(),
                    filter: gate.init(),
                    chunk: model.chunk_size(),
                    staging: Vec::new(),
                    pad: gate.speech_pad_samples(),
                    grid: driver.grid_samples(),
                    regions: VecDeque::new(),
                    last_end: 0,
                })
            }
            _ => None,
        };

        let speakers = match driver.speakers() {
            Some(attachment) => Some(SpeakerTimeline {
                stream: attachment.model.new_stream()?,
                attribution: attachment.attribution.clone(),
                activity: SpeakerActivity::empty(
                    0,
                    attachment.model.frame_samples(),
                    attachment.model.max_speakers(),
                ),
                finished: false,
                held: VecDeque::new(),
                held_draft: None,
            }),
            None => None,
        };

        // Two windows: a full window can wait for its decode while the next
        // one fills, before the ring compacts.
        let frames = FrameRing::new(2 * engine.window_frames());
        Ok(Self {
            features,
            frames,
            samples_seen: 0,
            origin: 0,
            seek: 0,
            clock,
            level: LevelTracker::new(level),
            state,
            transcript: Vec::new(),
            last_draft: 0,
            vad,
            speakers,
            finished: false,
            driver,
        })
    }

    // ---- observation -------------------------------------------------

    /// The driver the stream was opened from.
    pub fn driver(&self) -> &AsrStreamDriver<E> {
        &self.driver
    }

    /// The stream's clock.
    pub fn clock(&self) -> &StreamClock {
        &self.clock
    }

    /// The stream's level.
    pub fn level(&self) -> &LevelTracker<E::Level> {
        &self.level
    }

    /// The engine's carry for this stream.
    pub fn engine_state(&self) -> &E::StreamState {
        &self.state
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
        self.origin + self.frames.len()
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

    /// The diarizer's finalized activity, when the driver has one: from a
    /// few frames before the oldest stretch still to be attributed.
    pub fn speaker_activity(&self) -> Option<&SpeakerActivity> {
        self.speakers.as_ref().map(|t| &t.activity)
    }

    /// Committed segments waiting for the diarizer to cover them.
    pub fn held_segments(&self) -> usize {
        self.speakers.as_ref().map_or(0, |t| t.held.len())
    }

    /// Whether [`end_read`](Self::end_read) or [`end_input`](Self::end_input)
    /// has run.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    fn hop(&self) -> usize {
        self.driver.engine().hop()
    }

    fn window_frames(&self) -> usize {
        self.driver.engine().window_frames()
    }

    // ---- input -------------------------------------------------------

    /// Anchors the clock, then [`write_read`](Self::write_read): the first
    /// of `samples` is placed at media time `time`, and later times follow
    /// from there.
    ///
    /// # Errors
    /// As [`write_read`](Self::write_read) and [`StreamClock::anchor`].
    pub fn anchor_write_read(
        &mut self,
        time: f64,
        samples: &[f32],
    ) -> BunsenResult<Vec<TranscriptEvent>> {
        self.clock.anchor(self.samples_seen, time)?;
        self.write_read(samples)
    }

    /// Writes samples and returns the [`TranscriptEvent`]s they made due:
    /// [`write`](Self::write) then [`read`](Self::read).
    ///
    /// # Errors
    /// As [`write`](Self::write) and [`read`](Self::read).
    pub fn write_read(
        &mut self,
        samples: &[f32],
    ) -> BunsenResult<Vec<TranscriptEvent>> {
        self.write(samples)?;
        self.read()
    }

    /// Takes samples in without decoding: the front end and the gate run,
    /// nothing else. Pair with [`read`](Self::read), or with
    /// [`advance_ready`] across many streams.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) after the stream
    /// has ended; as the engine's front end.
    pub fn write(
        &mut self,
        samples: &[f32],
    ) -> BunsenResult<()> {
        if self.finished {
            return Err(BunsenError::illegal(
                "the stream has ended; nothing more can be fed",
            ));
        }

        self.samples_seen += samples.len();
        if let Some(vad) = &mut self.vad {
            vad.staging.extend_from_slice(samples);
        }

        if let Some(frames) = self.features.push(samples)? {
            self.ingest(frames);
        }
        self.run_vad(false);
        if let Some(timeline) = &mut self.speakers {
            let frames = timeline.stream.push(samples)?;
            timeline.activity.append(frames)?;
        }
        Ok(())
    }

    /// Runs every decode that is due, and returns what it emitted: each
    /// decode's events in order, then an `interval` draft if one is due.
    ///
    /// With a diarizer, a committed segment comes out once the diarizer has
    /// finalized the frames it covers, which may be a later `read`; events
    /// keep their order, and a draft produced behind held commits follows
    /// them out, replacing any earlier one.
    ///
    /// # Errors
    /// As the engine's decode, and its text.
    pub fn read(&mut self) -> BunsenResult<Vec<TranscriptEvent>> {
        let mut out = Vec::new();
        loop {
            self.skip_silence();
            let Some(unit) = self.next_due() else {
                break;
            };
            let decode = self.decode_one(unit, false)?;
            out.extend(self.commit_due(unit, decode)?);
        }
        if let Some(unit) = self.draft_unit() {
            let decode = self.decode_one(unit, true)?;
            out.extend(self.draft_from(unit, decode)?);
        }
        out.extend(self.release());
        Ok(out)
    }

    /// Ends the stream and decodes whatever is left past the seek pointer:
    /// [`end_input`](Self::end_input) then [`read`](Self::read).
    ///
    /// Idempotent; a second flush returns nothing.
    ///
    /// # Errors
    /// As [`end_input`](Self::end_input) and [`read`](Self::read).
    pub fn end_read(&mut self) -> BunsenResult<Vec<TranscriptEvent>> {
        self.end_input()?;
        self.read()
    }

    /// Ends the stream's input: flushes the front end, drops the engine's
    /// trailing frames, and closes the gate. What that leaves due is decoded
    /// by the next [`read`](Self::read).
    ///
    /// Idempotent.
    ///
    /// # Errors
    /// As the engine's front end.
    pub fn end_input(&mut self) -> BunsenResult<()> {
        if self.finished {
            return Ok(());
        }
        self.finished = true;

        if let Some(tail) = self.features.finish()? {
            self.ingest(tail);
        }
        self.frames
            .drop_back(self.driver.engine().end_frames_dropped());

        self.run_vad(true);
        if let Some(timeline) = &mut self.speakers {
            let frames = timeline.stream.finish()?;
            timeline.activity.append(frames)?;
            timeline.finished = true;
        }
        Ok(())
    }

    /// Appends frames to the ring, offering them to the level first.
    fn ingest(
        &mut self,
        new: Tensor<3>,
    ) {
        self.level.observe(new.clone());
        self.frames.push(new);
    }

    /// Runs the voice-activity model over every whole chunk in the VAD's
    /// staging, steps the gate, and queues the regions it closes. Flushing
    /// pads the last chunk with silence and closes an open region at the
    /// true length.
    fn run_vad(
        &mut self,
        flushing: bool,
    ) {
        let total = self.samples_seen;
        let Some(vad) = &mut self.vad else {
            return;
        };
        let chunk = vad.chunk;

        if flushing && !vad.staging.is_empty() {
            let whole = vad.staging.len().div_ceil(chunk) * chunk;
            vad.staging.resize(whole, 0.0);
        }
        let whole = vad.staging.len() / chunk * chunk;
        if whole > 0 {
            let samples: Vec<f32> = vad.staging.drain(..whole).collect();
            for p in vad.stream.probabilities(samples) {
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
    /// closed inside them, and nothing open in them. An open region counts
    /// from where it will start once closed: its onset less the start pad,
    /// snapped down onto the grid, so the window holding the pad is kept. A
    /// window is judged only once the VAD has ruled a start pad past its
    /// end: an onset there would pad back into it.
    fn skip_silence(&mut self) {
        let width = self.window_frames();
        let hop = self.hop();

        loop {
            let Some(vad) = &self.vad else {
                return;
            };
            if self.pending_frames() < width {
                return;
            }
            let (from, to) = (self.seek * hop, (self.seek + width) * hop);
            if vad.filter.sample_count() < to + vad.pad {
                return;
            }
            let closed_in = vad.regions.iter().any(|r| r.end > from && r.start < to);
            let open_in = vad
                .filter
                .open_since()
                .is_some_and(|s| vad.padded_start(s) < to);
            if closed_in || open_in {
                return;
            }
            self.seek += width;
            self.drop_before_seek();
        }
    }

    /// The next decode unit, if one is due: [`unit_due`](Self::unit_due),
    /// held while a diarizer has not yet finalized the unit's audio (the
    /// input's end excepted), so an engine that reads the activity sees all
    /// of it, however the audio was chunked.
    fn next_due(&self) -> Option<Due> {
        let unit = self.unit_due()?;
        if let Some(timeline) = &self.speakers
            && !timeline.finished
            && timeline.activity.end() < (unit.start + unit.count) * self.hop()
        {
            return None;
        }
        Some(unit)
    }

    /// The next decode unit by the emission policy alone.
    ///
    /// With endpoints on: the oldest closed region reaching past the seek
    /// pointer, a window at a time; else a full window, only while speech
    /// is in progress inside it. Without: a full window, and at the end of
    /// input whatever remains.
    fn unit_due(&self) -> Option<Due> {
        let width = self.window_frames();
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
                if vad
                    .filter
                    .open_since()
                    .is_some_and(|s| vad.padded_start(s) < to)
                {
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

    /// The draft due, if one is: under the `interval` trigger, while speech
    /// is in progress, once an interval of media time has passed since the
    /// last draft or commit, everything past the seek pointer.
    fn draft_unit(&self) -> Option<Due> {
        let interval = self.driver.interval_samples()?;
        let count = self.pending_frames().min(self.window_frames());
        let due = !self.finished
            && count > 0
            && self.is_speaking()
            && self.samples_seen.saturating_sub(self.last_draft) >= interval;
        due.then_some(Due {
            start: self.seek,
            count,
        })
    }

    // ---- decoding ----------------------------------------------------

    /// The request that decodes a unit: its frames, and the level they are
    /// packaged against. Takes `&self`.
    fn request(
        &self,
        unit: Due,
        draft: bool,
    ) -> DecodeRequest<'_, E> {
        let window = self.frames.window(unit.start - self.origin, unit.count);
        let hop = self.hop();
        DecodeRequest {
            state: &self.state,
            transcript: &self.transcript,
            start: unit.start * hop,
            level: self.level.reference(window.clone()),
            window,
            draft,
            speakers: self.speakers.as_ref().map(|t| {
                t.activity
                    .window(unit.start * hop, (unit.start + unit.count) * hop)
            }),
        }
    }

    /// Decodes one unit on its own.
    fn decode_one(
        &self,
        unit: Due,
        draft: bool,
    ) -> BunsenResult<WindowDecode<E::Carry>> {
        Ok(self
            .driver
            .engine()
            .decode(vec![self.request(unit, draft)])?
            .pop()
            .expect("one request in, one decode out"))
    }

    /// Commits a decode of a due unit: records the ids, advances the seek
    /// pointer, drops the consumed frames and regions, places the segments
    /// on the clock, and hands the engine its carry.
    ///
    /// A skipped decode commits nothing, and the seek moves past the unit.
    /// Otherwise each complete segment is committed and the seek advances by
    /// the decode's `advance`, at least one frame and at most the unit; the
    /// unfinished tail is decoded again with more audio behind it, and under
    /// [`CommitRule::LastTimestamp`] emitted first as a draft.
    fn commit_due(
        &mut self,
        unit: Due,
        decode: WindowDecode<E::Carry>,
    ) -> BunsenResult<Vec<TranscriptEvent>> {
        let hop = self.hop();
        let mut out = Vec::new();

        if decode.skip {
            self.seek = unit.start + unit.count;
        } else {
            for segment in decode.segments {
                let (start, end) = (unit.start + segment.start, unit.start + segment.end);
                let emission = self.segment_at(start, end, segment.tokens)?;
                self.transcript.extend_from_slice(&emission.tokens);
                out.push((start * hop, end * hop, TranscriptEvent::Committed(emission)));
            }
            if let Some(tail) = decode.tail
                && self.driver.emission().commit == CommitRule::LastTimestamp
                // A tail opening past the unit's audio covers nothing yet.
                && tail.start < unit.count
            {
                let (start, end) = (unit.start + tail.start, unit.start + unit.count);
                let draft = self.segment_at(start, end, tail.tokens)?;
                out.push((start * hop, end * hop, TranscriptEvent::Draft(draft)));
            }
            // Always forward, never past the unit.
            self.seek = unit.start + decode.advance.clamp(1, unit.count);
        }

        self.driver
            .engine()
            .commit(&mut self.state, &self.transcript, decode.carry);

        self.last_draft = self.samples_seen;
        self.drop_before_seek();

        if let Some(vad) = &mut self.vad {
            let consumed = self.seek * hop;
            while vad.regions.front().is_some_and(|r| r.end <= consumed) {
                vad.regions.pop_front();
            }
        }

        Ok(self.route(out))
    }

    /// Emits a decode of a draft unit as a draft: on the clock, with text
    /// when the engine can say, and touching nothing but the draft pacing.
    /// It covers all audio since the last commit and supersedes the previous
    /// draft whole.
    ///
    /// With a diarizer and commits held, the draft is held behind them.
    fn draft_from(
        &mut self,
        unit: Due,
        decode: WindowDecode<E::Carry>,
    ) -> BunsenResult<Vec<TranscriptEvent>> {
        self.last_draft = self.samples_seen;
        let hop = self.hop();
        let (start, end) = (unit.start, unit.start + unit.count);
        let draft = TranscriptEvent::Draft(self.segment_at(start, end, decode.tokens)?);
        Ok(self.route(vec![(start * hop, end * hop, draft)]))
    }

    /// Events on their way out, each with the stream samples it covers.
    ///
    /// Without a diarizer they go out as they are. With one, every commit is
    /// held until [`release`](Self::release) finds the diarizer has covered
    /// it, and a draft goes out at once, attributed from what the diarizer
    /// has finalized, unless commits are held: then it waits behind them.
    fn route(
        &mut self,
        events: Vec<(usize, usize, TranscriptEvent)>,
    ) -> Vec<TranscriptEvent> {
        let Some(timeline) = &mut self.speakers else {
            return events.into_iter().map(|(_, _, e)| e).collect();
        };
        let mut out = Vec::new();
        for (start, end, event) in events {
            match event {
                TranscriptEvent::Committed(_) => timeline.held.push_back((start, end, event)),
                TranscriptEvent::Draft(_) if timeline.held.is_empty() => {
                    out.push(timeline.attribute(start, end, event))
                }
                TranscriptEvent::Draft(_) => timeline.held_draft = Some((start, end, event)),
            }
        }
        out.extend(self.release());
        out
    }

    /// The held events the diarizer now covers, attributed, in order; the
    /// held draft once no commit is ahead of it. Trims the activity behind
    /// what is still to be attributed.
    fn release(&mut self) -> Vec<TranscriptEvent> {
        let seek = self.seek * self.hop();
        let Some(timeline) = &mut self.speakers else {
            return Vec::new();
        };
        let mut out = Vec::new();
        while let Some(&(_, end, _)) = timeline.held.front() {
            if !timeline.finished && timeline.activity.end() < end {
                break;
            }
            let (start, end, event) = timeline.held.pop_front().expect("present");
            out.push(timeline.attribute(start, end, event));
        }
        if timeline.held.is_empty()
            && let Some((start, end, event)) = timeline.held_draft.take()
        {
            out.push(timeline.attribute(start, end, event));
        }

        let oldest = timeline
            .held
            .front()
            .map_or(seek, |&(start, _, _)| start.min(seek));
        let margin = SPEAKER_MARGIN_FRAMES * timeline.activity.frame_samples;
        timeline.activity.drop_before(oldest.saturating_sub(margin));
        out
    }

    /// A segment over stream frames `[start, end)`, timed through the clock,
    /// with text when the engine can say.
    fn segment_at(
        &self,
        start: usize,
        end: usize,
        tokens: Vec<i64>,
    ) -> BunsenResult<TranscriptSegment> {
        let hop = self.hop();
        Ok(TranscriptSegment {
            start: self.clock.time_at(start * hop),
            end: self.clock.time_at(end * hop),
            text: self.driver.engine().text(&tokens)?,
            tokens,
            speaker: None,
        })
    }

    /// Retains the ring from the seek pointer onward, nothing before it.
    fn drop_before_seek(&mut self) {
        self.frames.drop_front(self.seek - self.origin);
        self.origin = self.seek;
    }
}

impl SpeakerTimeline {
    /// `event` with its segment attributed over stream samples
    /// `[start, end)`.
    fn attribute(
        &self,
        start: usize,
        end: usize,
        mut event: TranscriptEvent,
    ) -> TranscriptEvent {
        let speaker = self.attribution.attribute(&self.activity, start, end);
        match &mut event {
            TranscriptEvent::Committed(s) | TranscriptEvent::Draft(s) => s.speaker = speaker,
        }
        event
    }
}

impl VoiceActivity {
    /// Where a region with onset `onset` starts once closed: the start pad
    /// back from the onset, no earlier than the previous region's end,
    /// snapped down onto the grid ([`enqueue`](Self::enqueue)'s start).
    fn padded_start(
        &self,
        onset: usize,
    ) -> usize {
        let start = onset.saturating_sub(self.pad).max(self.last_end);
        (start / self.grid * self.grid).max(self.last_end)
    }

    /// Pads a raw region as far as the stream allows, snaps it outward onto
    /// the grid, and queues it.
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
    use std::{
        sync::{
            Arc,
            atomic::{
                AtomicUsize,
                Ordering,
            },
        },
        time::Duration,
    };

    use burn::{
        prelude::TensorData,
        tensor::Device,
    };

    use super::*;
    use crate::{
        errors::{
            BunsenErrorKind,
            testing::ErrorMatcher,
        },
        kits::speech::{
            asr_driver::{
                DecodeTriggers,
                EmissionPolicy,
                MaxLevel,
                StreamingFeatures,
                TimedTokens,
                VoiceActivityFilterConfig,
                VoiceActivityModel,
            },
            diarization::SpeakerActivityModel,
            testing::ScriptedDiarizer,
        },
        support::testing::cpu_device,
    };

    /// Samples per second.
    const RATE: usize = 100;
    /// Samples per frame.
    const HOP: usize = 2;
    /// Frames per window.
    const WIDTH: usize = 8;
    /// The VAD's chunk, in samples.
    const CHUNK: usize = 4;

    /// One frame per hop: the hop's mean, as the frame's one feature.
    #[derive(Clone, Debug)]
    struct ToyFeatures {
        staging: Vec<f32>,
        device: Device,
    }

    impl ToyFeatures {
        fn frames(
            &self,
            samples: Vec<f32>,
        ) -> Option<Tensor<3>> {
            let means: Vec<f32> = samples
                .chunks(HOP)
                .map(|hop| hop.iter().sum::<f32>() / HOP as f32)
                .collect();
            let n = means.len();
            (n > 0).then(|| Tensor::from_data(TensorData::new(means, [1, n, 1]), &self.device))
        }
    }

    impl StreamingFeatures for ToyFeatures {
        fn push(
            &mut self,
            samples: &[f32],
        ) -> BunsenResult<Option<Tensor<3>>> {
            self.staging.extend_from_slice(samples);
            let whole = self.staging.len() / HOP * HOP;
            let samples = self.staging.drain(..whole).collect();
            Ok(self.frames(samples))
        }

        fn finish(&mut self) -> BunsenResult<Option<Tensor<3>>> {
            let mut samples = std::mem::take(&mut self.staging);
            if !samples.is_empty() {
                samples.resize(HOP, 0.0);
            }
            Ok(self.frames(samples))
        }
    }

    /// A stream's carry: the commits seen, and the level of the last.
    #[derive(Clone, Debug, Default, PartialEq)]
    struct ToyState {
        commits: usize,
        last_level: Option<f32>,
    }

    /// Decodes each frame to `round(10 * value)`. With `split`, a segment
    /// every `split` frames and the remainder a tail, as timestamps would;
    /// without, the window is one segment.
    #[derive(Clone, Debug)]
    struct ToyEngine {
        device: Device,
        split: Option<usize>,
        decodes: Arc<AtomicUsize>,
    }

    impl ToyEngine {
        fn new(split: Option<usize>) -> Self {
            Self {
                device: cpu_device(),
                split,
                decodes: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn decodes(&self) -> usize {
            self.decodes.load(Ordering::SeqCst)
        }
    }

    impl AsrEngine for ToyEngine {
        type Carry = f32;
        type Features = ToyFeatures;
        type Level = MaxLevel;
        type StreamState = ToyState;

        fn sample_rate(&self) -> usize {
            RATE
        }

        fn hop(&self) -> usize {
            HOP
        }

        fn window_frames(&self) -> usize {
            WIDTH
        }

        fn grid_frames(&self) -> usize {
            2
        }

        fn new_features(&self) -> BunsenResult<ToyFeatures> {
            Ok(ToyFeatures {
                staging: Vec::new(),
                device: self.device.clone(),
            })
        }

        fn new_stream_state(&self) -> ToyState {
            ToyState::default()
        }

        fn decode(
            &self,
            requests: Vec<DecodeRequest<'_, Self>>,
        ) -> BunsenResult<Vec<WindowDecode<f32>>> {
            Ok(requests
                .into_iter()
                .map(|request| {
                    self.decodes.fetch_add(1, Ordering::SeqCst);
                    let level = request.level.max.into_data().try_to_vec::<f32>().unwrap()[0];
                    let values = request.window.into_data().try_to_vec::<f32>().unwrap();
                    let tokens: Vec<i64> =
                        values.iter().map(|v| (v * 10.0).round() as i64).collect();
                    let count = tokens.len();
                    let Some(k) = self.split else {
                        return WindowDecode::whole(count, tokens, level);
                    };
                    let complete = count / k * k;
                    WindowDecode {
                        segments: (0..complete)
                            .step_by(k)
                            .map(|s| TimedTokens {
                                start: s,
                                end: s + k,
                                tokens: tokens[s..s + k].to_vec(),
                            })
                            .collect(),
                        tail: (complete < count).then(|| TimedTokens {
                            start: complete,
                            end: count,
                            tokens: tokens[complete..].to_vec(),
                        }),
                        advance: complete,
                        skip: false,
                        tokens,
                        carry: level,
                    }
                })
                .collect())
        }

        fn commit(
            &self,
            state: &mut ToyState,
            _transcript: &[i64],
            carry: f32,
        ) {
            state.commits += 1;
            state.last_level = Some(carry);
        }

        fn text(
            &self,
            tokens: &[i64],
        ) -> BunsenResult<Option<String>> {
            Ok(Some(
                tokens
                    .iter()
                    .map(i64::to_string)
                    .collect::<Vec<_>>()
                    .join(" "),
            ))
        }
    }

    /// Speech where a chunk's mean magnitude passes a quarter.
    #[derive(Clone, Debug)]
    struct EnergyVad;

    impl VoiceActivityModel for EnergyVad {
        fn sample_rate(&self) -> usize {
            RATE
        }

        fn chunk_size(&self) -> usize {
            CHUNK
        }

        fn new_stream(&self) -> Box<dyn VoiceActivityStream> {
            Box::new(EnergyVad)
        }
    }

    impl VoiceActivityStream for EnergyVad {
        fn probabilities(
            &mut self,
            samples: Vec<f32>,
        ) -> Vec<f32> {
            samples
                .chunks(CHUNK)
                .map(|c| {
                    let energy = c.iter().map(|v| v.abs()).sum::<f32>() / CHUNK as f32;
                    if energy > 0.25 { 1.0 } else { 0.0 }
                })
                .collect()
        }
    }

    /// 80 ms of silence closes a region; 40 ms of padding on each side.
    fn gate() -> VoiceActivityFilterConfig {
        VoiceActivityFilterConfig::new()
            .with_sample_rate(RATE)
            .with_samples_per_chunk(CHUNK)
            .with_min_silence_ms(80)
            .with_speech_pad_ms(40)
    }

    fn driver(
        engine: ToyEngine,
        emission: EmissionPolicy,
    ) -> AsrStreamDriver<ToyEngine> {
        let driver = AsrStreamDriver::new(engine, emission).unwrap();
        if driver.emission().triggers.endpoint {
            driver.with_vad(Arc::new(EnergyVad), gate()).unwrap()
        } else {
            driver
        }
    }

    fn open(driver: &AsrStreamDriver<ToyEngine>) -> AsrStreamContext<ToyEngine> {
        driver
            .new_context(StreamClock::uniform(RATE), LevelPolicy::PerWindow)
            .unwrap()
    }

    /// A ramp of distinct values, no silence.
    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i * 7) % 11) as f32 / 10.0).collect()
    }

    /// Silence, speech, a long silence, speech, silence.
    fn speech() -> Vec<f32> {
        let mut out = vec![0.0; 40];
        out.extend((0..32).map(|i| 0.5 + (i % 5) as f32 / 10.0));
        out.extend(vec![0.0; 48]);
        out.extend((0..20).map(|i| 0.6 + (i % 3) as f32 / 10.0));
        out.extend(vec![0.0; 40]);
        out
    }

    /// Writes `samples` in pieces of `sizes` (cycled), then ends the stream.
    fn run(
        ctx: &mut AsrStreamContext<ToyEngine>,
        samples: &[f32],
        sizes: &[usize],
    ) -> Vec<TranscriptEvent> {
        let mut out = Vec::new();
        let mut at = 0;
        for &size in sizes.iter().cycle() {
            if at >= samples.len() {
                break;
            }
            let end = (at + size).min(samples.len());
            out.extend(ctx.write_read(&samples[at..end]).unwrap());
            at = end;
        }
        out.extend(ctx.end_read().unwrap());
        out
    }

    fn committed(events: &[TranscriptEvent]) -> Vec<TranscriptSegment> {
        events
            .iter()
            .filter(|e| e.is_committed())
            .map(|e| e.segment().clone())
            .collect()
    }

    /// Offline, every frame is committed once, a window at a time, whatever
    /// the pieces the audio arrives in.
    #[test]
    fn test_offline_commits_every_frame_however_chunked() {
        let samples = ramp(100);
        let driver = driver(ToyEngine::new(None), EmissionPolicy::offline());

        let whole = run(&mut open(&driver), &samples, &[samples.len()]);
        for sizes in [&[1][..], &[3, 7], &[16], &[5, 1, 11]] {
            assert_eq!(run(&mut open(&driver), &samples, sizes), whole, "{sizes:?}");
        }

        // 50 frames: six full windows and a remainder of two.
        let segments = committed(&whole);
        assert_eq!(segments.len(), 7);
        let expected: Vec<i64> = samples
            .chunks(HOP)
            .map(|hop| ((hop[0] + hop[1]) / 2.0 * 10.0).round() as i64)
            .collect();
        let tokens: Vec<i64> = segments.iter().flat_map(|s| s.tokens.clone()).collect();
        assert_eq!(tokens, expected);
        let text: Vec<String> = expected[..WIDTH].iter().map(i64::to_string).collect();
        assert_eq!(segments[0].text, Some(text.join(" ")));
    }

    /// Segments are placed on the stream's clock, origin and all.
    #[test]
    fn test_segments_sit_on_the_clock() {
        let driver = driver(ToyEngine::new(None), EmissionPolicy::offline());
        let mut ctx = open(&driver);
        let events = ctx.anchor_write_read(10.0, &ramp(32)).unwrap();
        let segments = committed(&events);
        assert_eq!(segments.len(), 2);
        // A window is 8 frames of 2 samples at 100 Hz: 0.16 s.
        assert!((segments[0].start - 10.0).abs() < 1e-9);
        assert!((segments[0].end - 10.16).abs() < 1e-9);
        assert!((segments[1].end - 10.32).abs() < 1e-9);
    }

    /// With timed segments, the seek stops at the last complete one; the
    /// tail is decoded again, and under `LastTimestamp` drafted first.
    /// Drafting it changes nothing committed.
    #[test]
    fn test_split_windows_and_their_tails() {
        // 48 frames: windows advance by 6, the last unit is 6 whole frames.
        let samples = ramp(96);
        let complete = driver(ToyEngine::new(Some(3)), EmissionPolicy::offline());
        let last_timestamp = driver(
            ToyEngine::new(Some(3)),
            EmissionPolicy {
                commit: CommitRule::LastTimestamp,
                ..EmissionPolicy::offline()
            },
        );

        let a = run(&mut open(&complete), &samples, &[samples.len()]);
        let b = run(&mut open(&last_timestamp), &samples, &[samples.len()]);

        assert!(a.iter().all(TranscriptEvent::is_committed));
        assert!(b.iter().any(|e| !e.is_committed()));
        assert_eq!(committed(&a), committed(&b));

        let segments = committed(&a);
        assert_eq!(segments.len(), 16);
        assert!(segments.windows(2).all(|w| w[0].end == w[1].start));
        assert!((segments[15].end - 0.96).abs() < 1e-9);
    }

    /// Under `endpoint`, windows of silence are never decoded, and what is
    /// committed covers the speech.
    #[test]
    fn test_silence_is_skipped() {
        let samples = speech();
        let offline_engine = ToyEngine::new(None);
        let gated_engine = ToyEngine::new(None);
        let offline = driver(offline_engine.clone(), EmissionPolicy::offline());
        let gated = driver(gated_engine.clone(), EmissionPolicy::conservative());

        run(&mut open(&offline), &samples, &[samples.len()]);
        let events = run(&mut open(&gated), &samples, &[samples.len()]);

        assert!(gated_engine.decodes() < offline_engine.decodes());
        let segments = committed(&events);
        assert!(!segments.is_empty());
        // Speech spans 0.40 s to 0.72 s and 1.20 s to 1.40 s; regions are
        // padded by 0.04 s and snapped to 0.04 s.
        assert!(segments.first().unwrap().start >= 0.36 - 1e-9);
        assert!(segments.last().unwrap().end <= 1.44 + 1e-9);
        let covered = |t: f64| segments.iter().any(|s| s.start <= t && t < s.end);
        assert!(covered(0.5) && covered(1.3));
        assert!(!covered(0.9));
    }

    /// Drafts on an interval add events, and never change what is
    /// committed: under any chunking, the same as without them.
    ///
    /// The chunking itself can change what is committed: a window decoded
    /// while speech is still open starts at the seek pointer, not at the
    /// region's padded start.
    #[test]
    fn test_drafts_never_change_commits() {
        let samples = speech();
        let conservative = driver(ToyEngine::new(None), EmissionPolicy::conservative());
        let responsive = driver(
            ToyEngine::new(None),
            EmissionPolicy {
                triggers: DecodeTriggers::new()
                    .with_endpoint(true)
                    .with_interval(Some(Duration::from_millis(60))),
                ..EmissionPolicy::conservative()
            },
        );

        for sizes in [&[4][..], &[6, 2], &[samples.len()]] {
            let mut quiet_ctx = open(&conservative);
            let quiet = run(&mut quiet_ctx, &samples, sizes);
            let mut ctx = open(&responsive);
            let loud = run(&mut ctx, &samples, sizes);
            assert_eq!(committed(&loud), committed(&quiet), "{sizes:?}");
            assert_eq!(ctx.engine_state(), quiet_ctx.engine_state(), "{sizes:?}");
            assert_eq!(ctx.transcript(), quiet_ctx.transcript(), "{sizes:?}");
            if sizes[0] < samples.len() {
                assert!(loud.iter().any(|e| !e.is_committed()), "{sizes:?}");
            }
        }
    }

    /// Batched across streams, each stream gets what its own reads give.
    #[test]
    fn test_advance_ready_matches_solo() {
        let streams = [speech(), ramp(70), speech()[20..].to_vec()];
        let driver = driver(ToyEngine::new(Some(3)), EmissionPolicy::conservative());

        let solo: Vec<Vec<TranscriptEvent>> = streams
            .iter()
            .map(|s| run(&mut open(&driver), s, &[10]))
            .collect();

        let mut contexts: Vec<_> = streams.iter().map(|_| open(&driver)).collect();
        let mut batched: Vec<Vec<TranscriptEvent>> = vec![Vec::new(); streams.len()];
        let longest = streams.iter().map(Vec::len).max().unwrap();
        for at in (0..longest).step_by(10) {
            for (ctx, s) in contexts.iter_mut().zip(&streams) {
                if at < s.len() {
                    ctx.write(&s[at..(at + 10).min(s.len())]).unwrap();
                }
            }
            for (i, events) in advance_ready(&driver, &mut contexts)
                .unwrap()
                .into_iter()
                .enumerate()
            {
                batched[i].extend(events);
            }
        }
        for ctx in contexts.iter_mut() {
            ctx.end_input().unwrap();
        }
        for (i, events) in advance_ready(&driver, &mut contexts)
            .unwrap()
            .into_iter()
            .enumerate()
        {
            batched[i].extend(events);
        }
        assert_eq!(batched, solo);
    }

    /// The level reaches the engine: a window's own, or the loudest seen.
    #[test]
    fn test_level_policy_reaches_the_engine() {
        // A loud first window, then quiet ones.
        let mut samples = vec![0.9; 16];
        samples.extend(vec![0.2; 32]);
        let driver = driver(ToyEngine::new(None), EmissionPolicy::offline());

        let level = |policy| {
            let mut ctx = driver
                .new_context(StreamClock::uniform(RATE), policy)
                .unwrap();
            run(&mut ctx, &samples, &[samples.len()]);
            ctx.engine_state().last_level.unwrap()
        };
        assert!((level(LevelPolicy::PerWindow) - 0.2).abs() < 1e-6);
        assert!((level(LevelPolicy::Running) - 0.9).abs() < 1e-6);
    }

    const FRAME: usize = 4;
    const LATENCY: usize = 30;

    fn with_speakers(
        driver: AsrStreamDriver<ToyEngine>,
        switch: usize,
    ) -> AsrStreamDriver<ToyEngine> {
        driver
            .with_speakers(
                Arc::new(ScriptedDiarizer {
                    sample_rate: RATE,
                    frame_samples: FRAME,
                    n_speakers: 2,
                    latency: LATENCY,
                    script: vec![(0, Some(0)), (switch, Some(1))],
                }),
                Arc::new(crate::kits::speech::asr_driver::MeanActivity::default()),
            )
            .unwrap()
    }

    /// The segments without their speakers.
    fn unlabelled(events: &[TranscriptEvent]) -> Vec<TranscriptEvent> {
        events
            .iter()
            .cloned()
            .map(|mut e| {
                match &mut e {
                    TranscriptEvent::Committed(s) | TranscriptEvent::Draft(s) => s.speaker = None,
                }
                e
            })
            .collect()
    }

    /// Every segment carries the speaker of the audio it covers, and the
    /// transcript is the one the driver makes without a diarizer.
    #[test]
    fn test_speakers_label_every_segment() {
        let samples = ramp(96);
        // Segments of 3 frames, 6 samples: the switch falls on a boundary.
        let plain = driver(ToyEngine::new(Some(3)), EmissionPolicy::offline());
        let labelled = with_speakers(plain.clone(), 48);

        let want = run(&mut open(&plain), &samples, &[7]);
        let got = run(&mut open(&labelled), &samples, &[7]);
        assert_eq!(unlabelled(&got), want);

        let segments = committed(&got);
        assert!(!segments.is_empty());
        for s in &segments {
            let speaker = s.speaker.expect("labelled").index;
            let mid = 0.5 * (s.start + s.end) * RATE as f64;
            assert_eq!(speaker, usize::from(mid >= 48.0), "{s:?}");
        }
    }

    /// A unit waits to be decoded until the diarizer has finalized its audio,
    /// so every commit comes out attributed from final activity; however the
    /// audio is chunked, the events are the same.
    #[test]
    fn test_decodes_wait_for_the_diarizer() {
        let samples = ramp(120);
        let labelled = with_speakers(
            driver(ToyEngine::new(Some(3)), EmissionPolicy::offline()),
            40,
        );

        let whole = run(&mut open(&labelled), &samples, &[samples.len()]);
        for sizes in [&[1][..], &[5, 3], &[16]] {
            let mut ctx = open(&labelled);
            let mut events = Vec::new();
            let mut waited = false;
            let mut at = 0;
            for &size in sizes.iter().cycle() {
                if at >= samples.len() {
                    break;
                }
                let end = (at + size).min(samples.len());
                for e in ctx.write_read(&samples[at..end]).unwrap() {
                    if e.is_committed() {
                        let covered = ctx.speaker_activity().unwrap().end() as f64 / RATE as f64;
                        assert!(e.segment().end <= covered + 1e-9);
                    }
                    events.push(e);
                }
                // A full window is pending, and not yet decoded.
                waited |= ctx.pending_frames() >= WIDTH;
                at = end;
            }
            events.extend(ctx.end_read().unwrap());
            assert_eq!(ctx.held_segments(), 0);
            assert!(waited, "{sizes:?}");
            assert_eq!(events, whole, "{sizes:?}");
        }
    }

    /// With a diarizer, drafts still never change commits.
    #[test]
    fn test_drafts_never_change_labelled_commits() {
        let samples = speech();
        let conservative = with_speakers(
            driver(ToyEngine::new(None), EmissionPolicy::conservative()),
            100,
        );
        let responsive = with_speakers(
            driver(
                ToyEngine::new(None),
                EmissionPolicy {
                    triggers: DecodeTriggers::new()
                        .with_endpoint(true)
                        .with_interval(Some(Duration::from_millis(60))),
                    ..EmissionPolicy::conservative()
                },
            ),
            100,
        );
        for sizes in [&[4][..], &[6, 2]] {
            let quiet = run(&mut open(&conservative), &samples, sizes);
            let loud = run(&mut open(&responsive), &samples, sizes);
            assert_eq!(committed(&loud), committed(&quiet), "{sizes:?}");
            assert!(loud.iter().any(|e| !e.is_committed()), "{sizes:?}");
            assert!(committed(&loud).iter().all(|s| s.speaker.is_some()));
        }
    }

    /// Batched across streams with a diarizer, each stream gets what its own
    /// reads give.
    #[test]
    fn test_advance_ready_matches_solo_with_speakers() {
        let streams = [speech(), ramp(70)];
        let driver = with_speakers(
            driver(ToyEngine::new(Some(3)), EmissionPolicy::conservative()),
            60,
        );
        let solo: Vec<Vec<TranscriptEvent>> = streams
            .iter()
            .map(|s| run(&mut open(&driver), s, &[10]))
            .collect();

        let mut contexts: Vec<_> = streams.iter().map(|_| open(&driver)).collect();
        let mut batched: Vec<Vec<TranscriptEvent>> = vec![Vec::new(); streams.len()];
        let longest = streams.iter().map(Vec::len).max().unwrap();
        for at in (0..longest).step_by(10) {
            for (ctx, s) in contexts.iter_mut().zip(&streams) {
                if at < s.len() {
                    ctx.write(&s[at..(at + 10).min(s.len())]).unwrap();
                }
            }
            for (i, events) in advance_ready(&driver, &mut contexts)
                .unwrap()
                .into_iter()
                .enumerate()
            {
                batched[i].extend(events);
            }
        }
        for ctx in contexts.iter_mut() {
            ctx.end_input().unwrap();
        }
        for (i, events) in advance_ready(&driver, &mut contexts)
            .unwrap()
            .into_iter()
            .enumerate()
        {
            batched[i].extend(events);
        }
        assert_eq!(batched, solo);
    }

    /// The engine sees the window's activity; the activity is trimmed behind
    /// what is still to be attributed.
    #[test]
    fn test_requests_carry_the_window_activity() {
        let labelled = with_speakers(driver(ToyEngine::new(None), EmissionPolicy::offline()), 40);
        let mut ctx = open(&labelled);
        ctx.write(&ramp(64)).unwrap();
        let unit = ctx.next_due().expect("a full window");
        let request = ctx.request(unit, false);
        let activity = request.speakers.expect("a diarizer is attached");
        assert_eq!(activity.start, unit.start * HOP / FRAME * FRAME);
        assert!(activity.end() <= (unit.start + unit.count) * HOP + FRAME);

        let plain = driver(ToyEngine::new(None), EmissionPolicy::offline());
        let mut ctx = open(&plain);
        ctx.write(&ramp(64)).unwrap();
        let unit = ctx.next_due().unwrap();
        assert!(ctx.request(unit, false).speakers.is_none());

        // After a long stream, only a margin before the seek is kept.
        let mut ctx = open(&labelled);
        run(&mut ctx, &ramp(400), &[16]);
        let activity = ctx.speaker_activity().unwrap();
        assert!(
            activity.frames() <= SPEAKER_MARGIN_FRAMES + 1,
            "{}",
            activity.frames()
        );
    }

    #[test]
    fn test_speakers_must_share_the_rate() {
        #[derive(Debug)]
        struct Elsewhere;
        impl SpeakerActivityModel for Elsewhere {
            fn sample_rate(&self) -> usize {
                RATE * 2
            }

            fn frame_samples(&self) -> usize {
                FRAME
            }

            fn max_speakers(&self) -> usize {
                2
            }

            fn new_stream(&self) -> BunsenResult<Box<dyn SpeakerActivityStream>> {
                unreachable!()
            }
        }
        let driver = driver(ToyEngine::new(None), EmissionPolicy::offline());
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("sample_rate")
            .assert_err(&driver.with_speakers(
                Arc::new(Elsewhere),
                Arc::new(crate::kits::speech::asr_driver::MeanActivity::default()),
            ));
    }

    #[test]
    fn test_refuses_bad_settings() {
        let engine = ToyEngine::new(None);
        let never = EmissionPolicy {
            triggers: DecodeTriggers::new().with_window_full(false),
            ..EmissionPolicy::offline()
        };
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("nothing would ever decode")
            .assert_err(&AsrStreamDriver::new(engine.clone(), never));

        let zero = EmissionPolicy {
            triggers: DecodeTriggers::new()
                .with_endpoint(true)
                .with_interval(Some(Duration::ZERO)),
            ..EmissionPolicy::conservative()
        };
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .assert_err(&AsrStreamDriver::new(engine.clone(), zero));

        let ungated = EmissionPolicy {
            triggers: DecodeTriggers::new().with_interval(Some(Duration::from_millis(60))),
            ..EmissionPolicy::offline()
        };
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("endpoint")
            .assert_err(&AsrStreamDriver::new(engine.clone(), ungated));

        let agreement = EmissionPolicy {
            commit: CommitRule::Agreement { runs: 2 },
            ..EmissionPolicy::offline()
        };
        ErrorMatcher::kind(BunsenErrorKind::Unsupported)
            .message_contains("not implemented")
            .assert_err(&AsrStreamDriver::new(engine.clone(), agreement));

        // Endpoints without a VAD; a clock at another rate.
        let bare = AsrStreamDriver::new(engine.clone(), EmissionPolicy::conservative()).unwrap();
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("with_vad")
            .assert_err(&bare.new_context(StreamClock::uniform(RATE), LevelPolicy::PerWindow));
        let offline = AsrStreamDriver::new(engine.clone(), EmissionPolicy::offline()).unwrap();
        ErrorMatcher::kind(BunsenErrorKind::Illegal).assert_err(
            &offline.new_context(StreamClock::uniform(RATE * 2), LevelPolicy::PerWindow),
        );

        // A filter at another rate, or another chunk.
        ErrorMatcher::kind(BunsenErrorKind::Illegal).assert_err(
            &offline
                .clone()
                .with_vad(Arc::new(EnergyVad), gate().with_sample_rate(RATE * 2)),
        );
        ErrorMatcher::kind(BunsenErrorKind::Illegal).assert_err(&offline.clone().with_vad(
            Arc::new(EnergyVad),
            gate().with_samples_per_chunk(CHUNK * 2),
        ));

        // Nothing is fed after the end.
        let mut ctx = open(&offline);
        ctx.end_read().unwrap();
        assert!(ctx.end_read().unwrap().is_empty());
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("the stream has ended")
            .assert_err(&ctx.write(&[0.0; 4]));
    }
}
