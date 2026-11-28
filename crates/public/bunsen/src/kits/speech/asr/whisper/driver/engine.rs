//! # The Whisper engine: its config, and the model as an ASR engine.

use alloc::sync::Arc;

use burn::{
    Tensor,
    config::Config,
    module::Module,
    tensor::Device,
};

use crate::{
    errors::{
        BunsenResult,
        ConstraintError,
        ResultContext,
        WithOkOrPanic,
    },
    kits::{
        speech::{
            asr::{
                driver::{
                    AsrEngine,
                    DecodeRequest,
                    LevelPolicy,
                    SpeakerCutConfig,
                    TimedTokens,
                    WindowDecode,
                },
                whisper::{
                    DecodeConfig,
                    Whisper,
                    WhisperConstruct,
                    WhisperFallbackConfig,
                    WhisperMeta,
                    WhisperPolicy,
                    blocks::AUDIO_ENCODER_STRIDE,
                    decode::DecodedTokens,
                    driver::{
                        WhisperTask,
                        WhisperTokenLayout,
                        support::segments::split_window,
                    },
                    logit_filters::ApplyTimestampRules,
                },
            },
            support::{
                MaxLevel,
                log_mel::{
                    LogMelFeatures,
                    LogMelFrontEndConfig,
                },
            },
        },
        support::{
            decoding::{
                LogitFilter,
                SearchConfig,
            },
            tokens::Detokenizer,
        },
    },
    ops::signal::perceptive_audio::{
        PerceptiveAudioConverter,
        PerceptiveAudioConverterMeta,
    },
};

/// Config for a [`WhisperEngine`]: what each window is decoded as.
///
/// Set the language, task, timestamps, search and fallback ladder here, then
/// build the engine over a loaded [`WhisperConstruct`](crate::kits::speech::asr::whisper::WhisperConstruct) with
/// [`try_init`](Self::try_init), and the driver over the engine and an
/// [`EmissionPolicy`](crate::kits::speech::asr::driver::EmissionPolicy) with [`AsrStreamDriver::new`](crate::kits::speech::asr::driver::AsrStreamDriver::new), as for every ASR engine.
/// A bare model is a construct through [`WhisperConstruct::from_model`](crate::kits::speech::asr::whisper::WhisperConstruct::from_model), or
/// [`WhisperConstruct::new`](crate::kits::speech::asr::whisper::WhisperConstruct::new) with an explicit token layout. The engine keeps
/// a copy of this config; the driver opens a
/// [`WhisperStreamContext`](super::WhisperStreamContext) per stream with
/// [`new_context`](super::WhisperStreamDriver::new_context).
#[derive(Config, Debug)]
pub struct WhisperEngineConfig {
    /// The language of the speech, as a
    /// [`LANGUAGES`](crate::kits::speech::asr::whisper::blocks::LANGUAGES)
    /// code.
    ///
    /// `None` on a multilingual checkpoint detects the language per stream
    /// from its first committed window, as upstream's `transcribe()` does
    /// from its first; a draft before that detects one for its own decode
    /// and keeps nothing. Must be `None` for an English-only checkpoint,
    /// which takes no language token.
    #[config(default = "None")]
    pub language: Option<String>,

    /// Transcribe, or translate to English. Ignored by an English-only
    /// checkpoint, which takes no task token.
    #[config(default = "WhisperTask::Transcribe")]
    pub task: WhisperTask,

    /// Let the model emit timestamp tokens, under upstream's timestamp
    /// rules. Emissions are then split on them, and the seek pointer
    /// advances to the last closed timestamp rather than a whole window.
    #[config(default = false)]
    pub timestamps: bool,

    /// Under `timestamps`, the latest time the first timestamp of a window
    /// may name, in seconds; upstream's default is one second.
    #[config(default = "Some(1.0)")]
    pub max_initial_timestamp: Option<f64>,

    /// Cap on tokens generated per window.
    #[config(default = 224)]
    pub max_tokens: usize,

    /// The search of each window: its beams, patience and ranking. Its
    /// temperature and trajectories are each rung's of the
    /// [`fallback`](Self::fallback) ladder.
    #[config(default = "SearchConfig::new()")]
    pub search: SearchConfig,

    /// Prompt each window with the tail of the transcript so far, after
    /// `<|startofprev|>`, as upstream's `condition_on_previous_text` does.
    #[config(default = true)]
    pub condition_on_previous_text: bool,

    /// With a diarizer attached, a window is cut at a change of speaker
    /// before it is decoded
    /// ([`SpeakerCutConfig::cut`](crate::kits::speech::asr::driver::SpeakerCutConfig::cut)), so a
    /// segment is one speaker's as far as the turns go; `None` decodes
    /// windows as the driver's emission policy cuts them.
    #[config(default = "Some(SpeakerCutConfig::new())")]
    pub speaker_cut: Option<SpeakerCutConfig>,

    /// The temperature ladder and its thresholds. The default ladder is
    /// temperature zero alone; [`WhisperFallbackConfig::upstream`] is
    /// `transcribe()`'s.
    #[config(default = "WhisperFallbackConfig::new()")]
    pub fallback: WhisperFallbackConfig,
}

impl WhisperEngineConfig {
    /// Sets the decode fields from `policy`: the task, the language, the
    /// timestamps, the beam size (the rest of the search as it is), the token
    /// cap, the prompt carry, and the fallback (upstream's ladder, or
    /// temperature zero alone). A construct's
    /// [`policy`](WhisperConstruct::policy) is what its presets resolved to.
    pub fn with_policy(
        self,
        policy: &WhisperPolicy,
    ) -> Self {
        let search = self.search.clone().with_beam_size(policy.beam_size);
        self.with_task(policy.task)
            .with_language(policy.language.clone())
            .with_timestamps(policy.timestamps)
            .with_search(search)
            .with_max_tokens(policy.max_tokens)
            .with_condition_on_previous_text(policy.condition_on_previous_text)
            .with_fallback(if policy.fallback {
                WhisperFallbackConfig::upstream()
            } else {
                WhisperFallbackConfig::new()
            })
    }

    /// Builds the engine over a loaded construct, sharing it.
    ///
    /// See the [errors convention](crate::errors#convention-try_x-and-x).
    ///
    /// The model and its layout come from the construct. So do, when the
    /// construct has a vocabulary, upstream's default logit filters and, under
    /// the `tokenizer` feature, a detokenizer, so emissions carry text;
    /// the driver's
    /// [`with_logit_filters`](super::WhisperStreamDriver::with_logit_filters) and
    /// [`with_detokenizer`](super::WhisperStreamDriver::with_detokenizer) replace
    /// either. Several engines over one construct are `Arc::clone`s of it.
    ///
    /// # Errors
    /// - [`Illegal`](crate::errors::BunsenErrorKind::Illegal) if the layout
    ///   does not fit the model or its vocabulary (see
    ///   [`WhisperConstruct::try_validate`](crate::kits::speech::asr::whisper::WhisperConstruct::try_validate)), if a language or task is given
    ///   for an English-only layout, or if the configuration breaks a rule: a
    ///   zero `search.beam_size` or `fallback.ladder.best_of`, no fallback
    ///   temperature, or a `search.patience` that collects no candidates. A
    ///   broken field rule carries a [`ConstraintError`].
    /// - [`Lookup`](crate::errors::BunsenErrorKind::Lookup) if the language is
    ///   not one the layout has (see [`WhisperTokenLayout::sot_sequence`]).
    /// - As [`LogMelFrontEndConfig::try_init_audio_converter`] for the model's
    ///   front end.
    pub fn try_init(
        &self,
        construct: Arc<WhisperConstruct>,
        device: &Device,
    ) -> BunsenResult<WhisperEngine> {
        construct.try_validate()?;
        let special_ids = *construct.layout.ids();

        if !special_ids.is_multilingual() && self.language.is_some() {
            return Err(crate::errors::ConstraintError::custom(
                "WhisperEngineConfig",
                "language",
                "an English-only checkpoint takes no language",
            )
            .into());
        }
        let task = special_ids.is_multilingual().then_some(self.task);
        // A multilingual checkpoint with no language detects it per
        // stream; its prompt is built when the language is known.
        let prompt = match (special_ids.is_multilingual(), self.language.as_deref()) {
            (true, None) => Vec::new(),
            (_, language) => construct
                .layout
                .sot_sequence(language, task, self.timestamps)?,
        };
        let max_initial_timestamp_index = self.max_initial_timestamp.map(|seconds| {
            (seconds / construct.layout.layout().timestamp_step_seconds).round() as usize
        });
        let mut filters: Vec<Arc<dyn LogitFilter>> = construct.default_filters();
        if self.timestamps {
            filters.push(Arc::new(ApplyTimestampRules::new(
                &special_ids,
                max_initial_timestamp_index,
            )));
        }

        const OWNER: &str = "WhisperEngineConfig";
        self.search
            .try_validate()
            .context("WhisperEngineConfig.search")?;
        if self.fallback.ladder.temperatures.is_empty() {
            return Err(
                ConstraintError::zero_or_empty(OWNER, "fallback.ladder.temperatures").into(),
            );
        }
        if self.fallback.ladder.best_of == Some(0) {
            return Err(ConstraintError::zero_or_empty(OWNER, "fallback.ladder.best_of").into());
        }

        let audio_converter = construct
            .model
            .front_end()
            .try_init_audio_converter(construct.model.n_mels(), device)?;

        #[cfg(feature = "tokenizer")]
        let detokenizer: Option<Arc<dyn Detokenizer>> = construct
            .detokenizer()?
            .map(|detokenizer| Arc::new(detokenizer) as Arc<dyn Detokenizer>);
        #[cfg(not(feature = "tokenizer"))]
        let detokenizer: Option<Arc<dyn Detokenizer>> = None;

        let engine = WhisperEngine {
            config: self.clone(),
            construct,
            audio_converter,
            prompt,
            task,
            max_initial_timestamp_index,
            filters,
            detokenizer,
        };
        Ok(engine)
    }

    /// [`try_init`](Self::try_init), panicking.
    ///
    /// # Panics
    /// As [`try_init`](Self::try_init) errs.
    pub fn init(
        &self,
        construct: Arc<WhisperConstruct>,
        device: &Device,
    ) -> WhisperEngine {
        self.try_init(construct, device).ok_or_panic()
    }
}

/// Whisper as an [`AsrEngine`](crate::kits::speech::asr::driver::AsrEngine):
/// the model and its token layout (a [`WhisperConstruct`](crate::kits::speech::asr::whisper::WhisperConstruct) behind an `Arc`),
/// the mel front end, the logit filters, the decode settings and the
/// detokenizer.
///
/// The engine streams log-mels a hop at a time, floors each window against
/// a [`MaxLevel`](crate::kits::speech::support::MaxLevel), decodes up
/// the fallback ladder, splits a timestamped decode into segments, carries
/// the transcript's tail into the next prompt, and detects a stream's
/// language from its first committed window.
#[derive(Clone, Debug)]
pub struct WhisperEngine {
    pub(super) config: WhisperEngineConfig,
    pub(super) audio_converter: PerceptiveAudioConverter,
    /// The model, its layout and its vocabulary, shared.
    pub(super) construct: Arc<WhisperConstruct>,
    /// The sot sequence every window's decode opens with; empty when the
    /// language is detected per stream.
    pub(super) prompt: Vec<i64>,
    /// The task token's meaning; `None` for an English-only layout.
    pub(super) task: Option<WhisperTask>,
    pub(super) max_initial_timestamp_index: Option<usize>,
    /// Applied to the logits every step, in order: the caller's, then the
    /// timestamp rules when timestamps are on.
    pub(super) filters: Vec<Arc<dyn LogitFilter>>,
    pub(super) detokenizer: Option<Arc<dyn Detokenizer>>,
}

impl WhisperEngine {
    /// The driver configuration.
    pub fn config(&self) -> &WhisperEngineConfig {
        &self.config
    }

    /// The construct: the model, its layout and its vocabulary.
    pub fn construct(&self) -> &Arc<WhisperConstruct> {
        &self.construct
    }

    /// The model.
    pub fn whisper_model(&self) -> &Whisper {
        &self.construct.model
    }

    /// The mel front end.
    pub fn audio_converter(&self) -> &PerceptiveAudioConverter {
        &self.audio_converter
    }

    /// The token layout, derived from the model.
    pub fn token_layout(&self) -> &WhisperTokenLayout {
        &self.construct.layout
    }

    /// The sot sequence every window's decode opens with.
    pub fn prompt(&self) -> &[i64] {
        &self.prompt
    }

    /// The logit filters every decode applies.
    pub fn filters(&self) -> &[Arc<dyn LogitFilter>] {
        &self.filters
    }

    /// The task; `None` for an English-only layout.
    pub fn task(&self) -> Option<WhisperTask> {
        self.task
    }

    /// Whether streams detect their language, from their first committed
    /// window.
    pub fn detects_language(&self) -> bool {
        self.prompt.is_empty()
    }

    /// The sot sequence for `language`, under this driver's task and
    /// timestamp setting.
    ///
    /// # Errors
    /// As [`WhisperTokenLayout::sot_sequence`].
    pub fn sot_sequence(
        &self,
        language: Option<&str>,
    ) -> BunsenResult<Vec<i64>> {
        self.construct
            .layout
            .sot_sequence(language, self.task, self.config.timestamps)
    }

    /// Audio frames per timestamp index: two.
    pub fn frames_per_timestamp(&self) -> usize {
        AUDIO_ENCODER_STRIDE
    }

    /// The decode of one window under this driver, given its prompt.
    pub fn decode_config(
        &self,
        prompt: Vec<i64>,
    ) -> DecodeConfig {
        let ids = self.construct.layout.ids();
        DecodeConfig::new(prompt, ids.eot)
            .with_max_tokens(self.config.max_tokens)
            .with_search(self.config.search.clone())
            .with_sot_token(Some(ids.sot))
            .with_no_speech_token(Some(ids.no_speech))
    }

    /// The detokenizer, if one was attached.
    pub fn detokenizer(&self) -> Option<&Arc<dyn Detokenizer>> {
        self.detokenizer.as_ref()
    }

    /// Frames per decode window: the model's audio context.
    pub fn window_frames(&self) -> usize {
        self.construct.model.max_audio_ctx()
    }

    /// The audio front end the model's log-mels are computed with.
    pub fn front_end(&self) -> &LogMelFrontEndConfig {
        self.construct.model.front_end()
    }

    /// The encoder grid in samples: one timestamp step, which is
    /// [`frames_per_timestamp`](Self::frames_per_timestamp) mel hops. 320 at
    /// 16 kHz.
    pub fn encoder_grid(&self) -> usize {
        self.frames_per_timestamp() * self.audio_converter.hop()
    }

    /// The devices the model lives on.
    pub fn devices(&self) -> Vec<Device> {
        self.construct.model.devices()
    }
}

/// What a Whisper stream carries from one window to the next.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WhisperStreamState {
    /// The language the prompt names: configured, or detected from the
    /// first window committed; `None` until then on a detecting driver.
    pub language: Option<String>,

    /// Where the prompt carry starts in the transcript: upstream's
    /// `prompt_reset_since`, moved past a decode that needed a high
    /// temperature so a failure does not feed the next window.
    pub prompt_reset: usize,
}

/// What a Whisper decode hands back to its commit.
#[derive(Clone, Debug, PartialEq)]
pub struct WhisperCarry {
    /// The temperature the decode was accepted at.
    pub temperature: f64,

    /// The language the decode detected, when the stream had none.
    pub detected: Option<String>,
}

/// One request, packaged: what it decodes under, and its window.
struct Packaged {
    /// The prompt it decodes under; what it is batched by.
    prompt: Vec<i64>,
    /// `[1, n_mels, width]`.
    window: Tensor<3>,
    /// The language detected for it, when its stream had none.
    detected: Option<String>,
    /// The window's frames.
    count: usize,
}

impl WhisperEngine {
    /// Floors a window at the range below its level and pads it out to the
    /// model's width: `[1, n_mels, width]`.
    fn package_padded(
        &self,
        window: Tensor<3>,
        level: &MaxLevel,
    ) -> Tensor<3> {
        let packaged = self.front_end().package_window(window, level.max.clone());

        let width = self.window_frames();
        let [_, n_mels, have] = packaged.dims();
        if have < width {
            let pad = Tensor::zeros(
                [1, n_mels, width - have],
                (&packaged.device(), packaged.dtype()),
            );
            Tensor::cat(vec![packaged, pad], 2)
        } else {
            packaged
        }
    }

    /// The prompt a window decodes under: the sot sequence naming
    /// `language`, preceded by the transcript's tail after
    /// `<|startofprev|>` when carrying is on.
    pub(super) fn prompt_under(
        &self,
        state: &WhisperStreamState,
        transcript: &[i64],
        language: Option<&str>,
    ) -> Vec<i64> {
        let prompt = self
            .sot_sequence(language)
            .expect("the language is known before a prompt is built");
        let carried = &transcript[state.prompt_reset.min(transcript.len())..];
        if !self.config.condition_on_previous_text || carried.is_empty() {
            return prompt;
        }

        // Upstream keeps `n_text_ctx / 2 - 1` tokens of context.
        let keep = self.whisper_model().max_text_ctx() / 2 - 1;
        let tail = &carried[carried.len().saturating_sub(keep)..];

        let mut out = Vec::with_capacity(1 + tail.len() + prompt.len());
        out.push(self.token_layout().ids().sot_prev);
        out.extend_from_slice(tail);
        out.extend_from_slice(&prompt);
        out
    }

    /// The language a packaged window detects, when the stream has none and
    /// the driver leaves it to be detected: upstream's `detect_language`, on
    /// the features the decode will use.
    fn undetected_language(
        &self,
        state: &WhisperStreamState,
        packaged: &Tensor<3>,
    ) -> Option<String> {
        if state.language.is_some() || !self.detects_language() {
            return None;
        }
        let model = self.whisper_model();
        let xa = model.forward_encoder(packaged.clone());
        let token = model.detect_language(xa, self.token_layout().ids())[0];
        let code = self
            .token_layout()
            .language_code(token)
            .expect("detection picks a language token");
        Some(code.to_string())
    }

    /// Packages a request, and finds the prompt it decodes under.
    ///
    /// A request on a stream with no language yet detects one from its own
    /// window and decodes under it. A commit's detection reaches the stream
    /// through its carry; a draft's is dropped, so drafting cannot change
    /// the language the stream commits under.
    fn package(
        &self,
        request: &DecodeRequest<'_, Self>,
    ) -> Packaged {
        let count = request
            .speaker_cut(self.config.speaker_cut.as_ref(), self)
            .unwrap_or(request.window.dims()[1]);
        let window = self.package_padded(
            request.window.clone().slice_dim(1, 0..count),
            &request.level,
        );
        let detected = self.undetected_language(request.state, &window);
        let language = detected.as_deref().or(request.state.language.as_deref());
        Packaged {
            prompt: self.prompt_under(request.state, request.transcript, language),
            window,
            detected,
            count,
        }
    }

    /// Decodes a packaged window up the ladder, encoding it only if a rung
    /// past `first` is needed.
    ///
    /// # Arguments
    /// * `base` - the decode config at temperature zero.
    /// * `window` - `[1, n_mels, width]`, packaged.
    /// * `first` - the first rung's result when the caller has it.
    fn ladder(
        &self,
        base: &DecodeConfig,
        window: Tensor<3>,
        first: Option<DecodedTokens>,
    ) -> DecodedTokens {
        let model = self.whisper_model();
        let mut xa: Option<Tensor<3>> = None;
        self.config.fallback.decode(
            base,
            first,
            |config| {
                let xa = xa
                    .get_or_insert_with(|| model.forward_encoder(window.clone()))
                    .clone();
                model
                    .decode_features_full(xa, config, &self.filters)
                    .pop()
                    .expect("one row in, one row out")
            },
            |ids| self.text(ids).ok().flatten(),
        )
    }

    /// Lays a decode out for the core.
    ///
    /// A window the fallback policy calls silence (its no-speech probability
    /// over the threshold, its log probability poor) is skipped whole.
    /// Without timestamps the window is one segment. With them the decode is
    /// split on its timestamps ([`split_window`]): the stream advances to the
    /// last closed timestamp, and the unfinished tail opens at its first
    /// timestamp.
    fn lay_out(
        &self,
        decoded: DecodedTokens,
        count: usize,
        detected: Option<String>,
    ) -> WindowDecode<WhisperCarry> {
        let carry = WhisperCarry {
            temperature: decoded.temperature,
            detected,
        };
        let skip = self
            .config
            .fallback
            .should_skip(decoded.no_speech_prob.map(f64::from), decoded.avg_logprob());
        let tokens = decoded.tokens;

        if skip {
            return WindowDecode {
                tokens,
                skip: true,
                segments: Vec::new(),
                advance: count,
                tail: None,
                carry,
            };
        }
        if !self.config.timestamps {
            return WindowDecode::whole(count, tokens, carry);
        }

        let ids = self.token_layout().ids();
        let fpt = self.frames_per_timestamp();
        let split = split_window(&tokens, ids, count, fpt);
        let tail = (!split.tail.is_empty()).then(|| TimedTokens {
            start: (split.tail[0] - ids.timestamp_begin) as usize * fpt,
            end: count,
            tokens: split.tail,
        });
        WindowDecode {
            segments: split
                .segments
                .into_iter()
                .map(|segment| TimedTokens {
                    start: segment.start,
                    end: segment.end,
                    tokens: segment.tokens,
                })
                .collect(),
            tokens,
            skip: false,
            advance: split.advance,
            tail,
            carry,
        }
    }
}

impl AsrEngine for WhisperEngine {
    type Carry = WhisperCarry;
    type Features = LogMelFeatures;
    type Level = MaxLevel;
    type StreamState = WhisperStreamState;

    fn sample_rate(&self) -> usize {
        self.construct.model.sample_rate()
    }

    fn hop(&self) -> usize {
        self.audio_converter.hop()
    }

    fn window_frames(&self) -> usize {
        WhisperEngine::window_frames(self)
    }

    fn grid_frames(&self) -> usize {
        self.frames_per_timestamp()
    }

    /// [`Running`](LevelPolicy::Running): each window floored against the
    /// loudest the stream has been.
    fn level_policy(&self) -> LevelPolicy {
        LevelPolicy::Running
    }

    /// The language the stream's prompt names, once known: configured, or
    /// detected by the stream's first commit.
    fn language<'s>(
        &self,
        state: &'s WhisperStreamState,
    ) -> Option<&'s str> {
        state.language.as_deref()
    }

    /// Whisper's `stft[..., :-1]`: the stream's last frame is the end
    /// padding's, and goes.
    fn end_frames_dropped(&self) -> usize {
        1
    }

    fn new_features(&self) -> BunsenResult<LogMelFeatures> {
        Ok(LogMelFeatures::new(
            self.audio_converter.clone(),
            self.devices()[0].clone(),
        ))
    }

    fn new_stream_state(&self) -> WhisperStreamState {
        WhisperStreamState {
            language: self.config.language.clone(),
            prompt_reset: 0,
        }
    }

    /// Windows that decode under the same prompt share one batched first
    /// rung; a window that needs a rung above it climbs the rest of the
    /// ladder alone. A lone window decodes on its own, encoding only when
    /// its first rung needs it.
    fn decode(
        &self,
        requests: Vec<DecodeRequest<'_, Self>>,
    ) -> BunsenResult<Vec<WindowDecode<WhisperCarry>>> {
        let packaged: Vec<Packaged> = requests.iter().map(|r| self.package(r)).collect();

        if let [one] = packaged.as_slice() {
            let config = self.decode_config(one.prompt.clone());
            let decoded = self.ladder(&config, one.window.clone(), None);
            return Ok(vec![self.lay_out(decoded, one.count, one.detected.clone())]);
        }

        // Group by prompt, in order of first appearance.
        let mut groups: Vec<(Vec<i64>, Vec<usize>)> = Vec::new();
        for (k, item) in packaged.iter().enumerate() {
            match groups.iter_mut().find(|(p, _)| *p == item.prompt) {
                Some((_, members)) => members.push(k),
                None => groups.push((item.prompt.clone(), vec![k])),
            }
        }

        let mut out: Vec<Option<WindowDecode<WhisperCarry>>> = vec![None; packaged.len()];
        for (prompt, members) in groups {
            let batch = Tensor::cat(
                members
                    .iter()
                    .map(|&k| packaged[k].window.clone())
                    .collect(),
                0,
            );
            let config = self.decode_config(prompt);
            let first = self
                .whisper_model()
                .decode_windows_full(batch, &config, &self.filters);
            for (row, k) in members.into_iter().enumerate() {
                let item = &packaged[k];
                let decoded = self.ladder(&config, item.window.clone(), Some(first[row].clone()));
                out[k] = Some(self.lay_out(decoded, item.count, item.detected.clone()));
            }
        }
        Ok(out
            .into_iter()
            .map(|d| d.expect("every request decoded"))
            .collect())
    }

    /// Fixes a detected language, and resets the prompt carry after a decode
    /// that needed a temperature above 0.5 (as upstream does, so a failure
    /// does not feed the next window), or always when carrying is off.
    fn commit(
        &self,
        state: &mut WhisperStreamState,
        transcript: &[i64],
        carry: WhisperCarry,
    ) {
        if state.language.is_none() {
            state.language = carry.detected;
        }
        if !self.config.condition_on_previous_text
            || WhisperFallbackConfig::resets_prompt(carry.temperature)
        {
            state.prompt_reset = transcript.len();
        }
    }

    fn text(
        &self,
        tokens: &[i64],
    ) -> BunsenResult<Option<String>> {
        match &self.detokenizer {
            Some(detokenizer) => Ok(Some(
                detokenizer.detokenize(&self.token_layout().text_ids(tokens))?,
            )),
            None => Ok(None),
        }
    }
}
