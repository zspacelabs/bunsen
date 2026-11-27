//! [`Zonos3Engine`]: text and a speaker to codes, and codes to speech.

use alloc::{
    collections::BTreeMap,
    format,
    string::String,
    sync::Arc,
    vec,
    vec::Vec,
};

use burn::{
    Tensor,
    config::Config,
    module::Module,
    prelude::{
        Int,
        TensorData,
        s,
    },
    tensor::{
        DType,
        Device,
    },
};

use super::{
    Zonos3Stop,
    Zonos3TakeRows,
    zonos3_output_stage,
};
use crate::{
    blocks::transformers::decoders::stack::DecoderStackCache,
    errors::{
        BunsenError,
        BunsenResult,
        LookupError,
    },
    kits::{
        speech::tts::{
            driver::{
                TtsCapabilities,
                TtsCloneMode,
                TtsDriver,
                TtsDriverConfig,
                TtsLanguage,
                TtsSamplingOverrides,
            },
            zonos3::{
                Zonos3Construct,
                blocks::{
                    Zonos3LmMeta,
                    Zonos3RowConditioning,
                },
                prompt::{
                    Zonos3PromptRequest,
                    Zonos3Speaker,
                    condition_bucket,
                    zonos3_text_ids,
                },
                sampler::{
                    Zonos3Sampler,
                    Zonos3SamplerConfig,
                },
                voices::Zonos3Voice,
            },
        },
        support::{
            decoding::PoolRelease,
            text::{
                BunsenLanguageTag,
                TextNormalizationConfig,
                ZyphraNormalizationConfig,
            },
            tokens::Tokenizer,
        },
    },
    ops::transformers::attention::KVCacheStorage,
};

/// The release's text normalization, which a [`Zonos3Driver`] runs a
/// request's text through: line breaks collapsed, then Zyphra's normalizer
/// (feature `zyphra-textnorm`), with no markup tags and no pre-pass, as the
/// release's reference normalizes.
pub fn zonos3_release_normalization() -> TextNormalizationConfig {
    TextNormalizationConfig::Chain(vec![
        TextNormalizationConfig::CollapseLineBreaks,
        TextNormalizationConfig::Zyphra(ZyphraNormalizationConfig::new()),
    ])
}

/// ZONOS3-Mini's TTS driver: [`TtsDriver`] over [`Zonos3Engine`].
pub type Zonos3Driver<T> = TtsDriver<Zonos3Engine<T>>;

/// The driver of `engine`, its text normalized as the release normalizes
/// it ([`zonos3_release_normalization`]).
///
/// # Errors
/// The normalizer's, building it.
pub fn zonos3_driver<T: Tokenizer + Send + Sync>(
    engine: Zonos3Engine<T>
) -> BunsenResult<Zonos3Driver<T>> {
    TtsDriver::new(
        engine,
        TtsDriverConfig::new().with_normalization(zonos3_release_normalization()),
    )
}

/// The family a ZONOS3 voice file names.
pub const ZONOS3_VOICE_FAMILY: &str = "zonos3";

/// The clone mode a ZONOS3 voice file is made in: the speaker encoder's
/// embedding, spoken through the prompt's speaker row.
pub const ZONOS3_EMBEDDING_MODE: &str = "embedding";

/// The tensor a ZONOS3 voice file holds: the raw embedding, `[S]` f32.
pub const ZONOS3_EMBEDDING_TENSOR: &str = "embedding";

/// How [`Zonos3Engine`] speaks in a stock voice: the release ships each as
/// both its reference codes and its speaker embedding.
#[derive(Config, Copy, Debug, PartialEq, Eq)]
pub enum Zonos3SpeakerMode {
    /// Its codes: its reference recording's codec codes, a prefix of audio
    /// rows (the release's default).
    Codes,

    /// Its speaker embedding, through the prompt's speaker row (the
    /// studio's default, and the mode a cloned voice speaks in).
    Embedding,
}

impl Zonos3SpeakerMode {
    /// `voice` as a speaker in this mode.
    pub fn speaker(
        self,
        voice: &Zonos3Voice,
    ) -> Zonos3Speaker {
        match self {
            Self::Codes => Zonos3Speaker::Codes(voice.codes.clone()),
            Self::Embedding => Zonos3Speaker::Embedding(voice.embedding.clone()),
        }
    }
}

/// Config for [`Zonos3Engine`]; its defaults are the release's
/// (`config.json`'s `generation`,
/// [`Zonos3GenerationMirror::engine_config`](crate::kits::speech::tts::zonos3::interop::zonos::Zonos3GenerationMirror::engine_config)).
#[derive(Config, Debug)]
pub struct Zonos3EngineConfig {
    /// The language a request that names none is in: a request code of the
    /// release's `languages` table (`en_us`, `fr_fr`), or a language tag
    /// ([`Zonos3Construct::language_tag`]).
    #[config(default = "String::from(\"en_us\")")]
    pub default_language: String,

    /// The stock voice a request that names none speaks in.
    #[config(default = "String::from(\"Savannah\")")]
    pub default_voice: String,

    /// How a stock voice is spoken in.
    #[config(default = "Zonos3SpeakerMode::Codes")]
    pub speaker_mode: Zonos3SpeakerMode,

    /// A value for each condition the take is conditioned on, by its name
    /// (the construct's [`conditions`](Zonos3Construct::conditions):
    /// `loudness`, `fmax`, `speaking_rate`), bucketed by its edges; a
    /// condition without one is the unknown bucket.
    #[config(default = "BTreeMap::new()")]
    pub conditions: BTreeMap<String, f64>,

    /// How each step's codes are sampled.
    #[config(default = "Zonos3SamplerConfig::new()")]
    pub sampler: Zonos3SamplerConfig,

    /// The silence prefill's frames, at most
    /// [`silence_prefill_max_frames`](Self::silence_prefill_max_frames).
    #[config(default = 3)]
    pub silence_prefill_frames: usize,

    /// The longest silence prefill.
    #[config(default = 30)]
    pub silence_prefill_max_frames: usize,

    /// The most rows a take generates; `None`, as many as the model's
    /// context leaves after the prompt (fewer is always the bound).
    #[config(default = "None")]
    pub max_new_frames: Option<usize>,

    /// The silence prefill's frames a take's decode starts with, the last
    /// ones.
    #[config(default = 4)]
    pub overlap_frames: usize,

    /// The output gain's headroom, in dB.
    #[config(default = 3.0)]
    pub headroom_db: f64,

    /// The output limiter's knee, in dBFS.
    #[config(default = -1.0)]
    pub output_limit_dbfs: f64,

    /// The decode cache's storage.
    #[config(default = "KVCacheStorage::Auto")]
    pub storage: KVCacheStorage,
}

/// What [`Zonos3Engine`] speaks: one take.
#[derive(Clone, Debug, PartialEq)]
pub struct Zonos3Request {
    /// The text, as the model reads it: normalized already (a
    /// [`Zonos3Driver`] normalizes a request's).
    pub text: String,

    /// The speaker.
    pub speaker: Zonos3Speaker,

    /// Each condition's bucket; `None` (or a short list) is the unknown
    /// bucket.
    pub condition_buckets: Vec<Option<usize>>,

    /// Seeds the device's random numbers before the take; `None` leaves
    /// them as they are.
    pub seed: Option<u64>,

    /// The most rows the take generates, under the config's
    /// [`max_new_frames`](Zonos3EngineConfig::max_new_frames); `None`, the
    /// config's.
    pub max_frames: Option<usize>,
}

impl Zonos3Request {
    /// `text` in `speaker`'s voice, every condition unknown, unseeded.
    pub fn new(
        text: impl Into<String>,
        speaker: Zonos3Speaker,
    ) -> Self {
        Self {
            text: text.into(),
            speaker,
            condition_buckets: Vec::new(),
            seed: None,
            max_frames: None,
        }
    }

    /// Seeds the take.
    pub fn with_seed(
        mut self,
        seed: u64,
    ) -> Self {
        self.seed = Some(seed);
        self
    }

    /// At most `frames` rows long.
    pub fn with_max_frames(
        mut self,
        frames: usize,
    ) -> Self {
        self.max_frames = Some(frames);
        self
    }
}

/// A take's codes: the frames its decode covers.
#[derive(Clone, Debug, PartialEq)]
pub struct Zonos3Codes {
    /// The frames, `[F][n]`, undelayed.
    pub frames: Vec<Vec<u32>>,

    /// The first frame's index in the stream (the silence prefill's frames
    /// counted from 0).
    pub start: usize,

    /// Why generation stopped.
    pub stop: Zonos3Stop,

    /// The rows generated.
    pub rows: usize,
}

/// A take in progress ([`Zonos3Engine::start_take`]): the model's cache and
/// its last logits, the rows sampled so far, and the sampler.
///
/// A `Module` over bare tensors: moved with `to_device`, never recorded or
/// visited.
#[derive(Module, Debug)]
pub struct Zonos3TakeRun {
    /// The model's decode cache.
    pub cache: DecoderStackCache,

    /// `[n, V]`: the last row's logits, each codebook's.
    pub logits: Tensor<2>,

    /// The rows sampled so far, and the stop rule.
    #[module(skip)]
    pub take: Zonos3TakeRows,

    /// Each codebook's codes so far, the sampler's history.
    pub history: Vec<Vec<i64>>,

    /// The prompt's silence prefill rows, `[P][n]`.
    pub silence: Vec<Vec<u32>>,

    /// The text pad, each generated row's text column.
    pub text_pad: u32,

    /// The sampler.
    #[module(skip)]
    pub sampler: Zonos3Sampler,

    /// When the device pool's free memory is released.
    #[module(skip)]
    pub pool: PoolRelease,

    /// The rows run through the model after the prompt.
    pub step: usize,

    /// Why the take stopped; `None` while it runs.
    #[module(skip)]
    pub stop: Option<Zonos3Stop>,
}

impl Zonos3TakeRun {
    /// The take's codes so far: the frames a whole decode covers from
    /// `overlap` frames into the silence prefill
    /// ([`Zonos3TakeRows::frames`]).
    ///
    /// # Panics
    /// For a take still running.
    pub fn codes(
        &self,
        overlap: usize,
    ) -> Zonos3Codes {
        let stop = self.stop.expect("a take that stopped");
        let (frames, start) = self.take.frames(&self.silence, overlap);
        Zonos3Codes {
            frames,
            start,
            stop,
            rows: self.take.rows().len(),
        }
    }
}

/// ZONOS3-Mini as a TTS engine
/// ([`TtsEngine`](crate::kits::speech::tts::driver::TtsEngine)): text and a
/// speaker to codes, one take decoded whole, as the release generates it.
///
/// A take is the text (normalized by its driver, [`Zonos3Driver`]); its prompt
/// ([`Zonos3PromptConfig::build`](crate::kits::speech::tts::zonos3::prompt::Zonos3PromptConfig::build))
/// run through the language model, then one row a step through its decode
/// cache, each row's codes sampled by [`Zonos3Sampler`] and fed back with the
/// text pad, until [`Zonos3TakeRows`]'s stop rule; then the frames it covers,
/// decoded by the codec, through the output stage ([`zonos3_output_stage`]).
#[derive(Debug)]
pub struct Zonos3Engine<T: Tokenizer> {
    construct: Arc<Zonos3Construct>,
    language: BunsenLanguageTag,
    sampler: Zonos3Sampler,
    tokenizer: T,
    pub(super) capabilities: TtsCapabilities,
    pub(super) condition_buckets: Vec<Option<usize>>,
    config: Zonos3EngineConfig,
}

impl<T: Tokenizer> Zonos3Engine<T> {
    /// The driver of a loaded model, its text read by `tokenizer` (the
    /// construct's [`tokenizer`](Zonos3Construct::tokenizer) file).
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a silence
    /// prefill past its limit; the default language's
    /// ([`Zonos3Construct::language_tag`]);
    /// [`Lookup`](crate::errors::BunsenErrorKind::Lookup) for a default voice
    /// that is not a stock voice, or a condition the release does not have.
    pub fn new(
        construct: Arc<Zonos3Construct>,
        tokenizer: T,
        config: Zonos3EngineConfig,
    ) -> BunsenResult<Self> {
        if config.silence_prefill_frames > config.silence_prefill_max_frames {
            return Err(crate::errors::ConstraintError::new(
                "Zonos3EngineConfig",
                "silence_prefill_frames",
                crate::errors::Rule::Relation {
                    lhs: (
                        "silence_prefill_frames".into(),
                        config.silence_prefill_frames.to_string(),
                    ),
                    op: "<=",
                    rhs: (
                        "silence_prefill_max_frames".into(),
                        config.silence_prefill_max_frames.to_string(),
                    ),
                },
            )
            .into());
        }
        let language = construct.language_tag(&config.default_language)?;
        if !construct.voices.contains_key(&config.default_voice) {
            return Err(BunsenError::lookup(
                LookupError::missing("stock voice", &config.default_voice)
                    .with_candidates(construct.voices.keys()),
            ));
        }
        if let Some(name) = config
            .conditions
            .keys()
            .find(|name| !construct.conditions.iter().any(|c| &c.name == *name))
        {
            return Err(BunsenError::lookup(
                LookupError::missing("condition", name)
                    .with_candidates(construct.conditions.iter().map(|c| c.name.clone())),
            ));
        }
        let condition_buckets = construct
            .conditions
            .iter()
            .map(|c| {
                config
                    .conditions
                    .get(&c.name)
                    .map(|&value| condition_bucket(&c.edges, value))
            })
            .collect();
        let sampler = Self::sampler(&construct, &config.sampler)?;
        let capabilities = TtsCapabilities {
            languages: construct
                .language_tags
                .iter()
                .map(|(key, tag)| TtsLanguage {
                    key: key.clone(),
                    tag: tag.clone(),
                })
                .collect(),
            default_voice: Some(config.default_voice.clone()),
            stock_voices: true,
            described_voices: false,
            instruction: false,
            clone_modes: if construct.speaker.is_some() {
                vec![TtsCloneMode::clip(ZONOS3_EMBEDDING_MODE)]
            } else {
                Vec::new()
            },
        };
        Ok(Self {
            construct,
            language,
            sampler,
            tokenizer,
            capabilities,
            condition_buckets,
            config,
        })
    }

    fn sampler(
        construct: &Zonos3Construct,
        config: &Zonos3SamplerConfig,
    ) -> BunsenResult<Zonos3Sampler> {
        let lm = &construct.lm_config;
        config.try_init(lm.codebook_size, lm.eoa_id(), lm.audio_vocab())
    }

    /// The model.
    pub fn construct(&self) -> &Zonos3Construct {
        &self.construct
    }

    /// The default language, as a tag.
    pub fn language(&self) -> &BunsenLanguageTag {
        &self.language
    }

    /// The config.
    pub fn config(&self) -> &Zonos3EngineConfig {
        &self.config
    }

    /// The output's sample rate, in Hz.
    pub fn sample_rate(&self) -> u32 {
        self.construct.sample_rate
    }

    /// The stock voices, by name.
    pub fn voices(&self) -> &BTreeMap<String, Zonos3Voice> {
        &self.construct.voices
    }

    /// The stock voice `name`.
    ///
    /// # Errors
    /// [`Lookup`](crate::errors::BunsenErrorKind::Lookup) for a name that is
    /// not a stock voice.
    pub fn stock_voice(
        &self,
        name: &str,
    ) -> BunsenResult<&Zonos3Voice> {
        let voices = self.voices();
        voices.get(name).ok_or_else(|| {
            BunsenError::lookup(
                LookupError::missing("stock voice", name).with_candidates(voices.keys()),
            )
        })
    }

    /// A stock voice as a speaker in the config's
    /// [`speaker_mode`](Zonos3EngineConfig::speaker_mode).
    ///
    /// # Errors
    /// As [`stock_voice`](Self::stock_voice).
    pub fn stock_speaker(
        &self,
        name: &str,
    ) -> BunsenResult<Zonos3Speaker> {
        Ok(self.config.speaker_mode.speaker(self.stock_voice(name)?))
    }

    /// A stock voice as a reference prefix, the release's default speaker
    /// mode, whatever the config's.
    ///
    /// # Errors
    /// As [`stock_voice`](Self::stock_voice).
    pub fn voice_reference(
        &self,
        name: &str,
    ) -> BunsenResult<Zonos3Speaker> {
        Ok(Zonos3SpeakerMode::Codes.speaker(self.stock_voice(name)?))
    }

    pub(crate) fn device(&self) -> Device {
        self.construct.lm.embedders[0].weight.device()
    }

    /// The last row's logits of `[1, T, K, V]`, `[K, V]`.
    fn last_logits(logits: Tensor<4>) -> Tensor<2> {
        let [_, t, k, v] = logits.dims();
        logits.slice(s![0..1, t - 1..t, .., ..]).reshape([k, v])
    }

    /// The take's codes: the prompt, the generated rows, and the frames they
    /// cover, sampled by the config's
    /// [`sampler`](Zonos3EngineConfig::sampler).
    ///
    /// # Errors
    /// The tokenizer's, the prompt's
    /// ([`build`](crate::kits::speech::tts::zonos3::prompt::Zonos3PromptConfig::build)); and
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a prompt
    /// that leaves the model no room to generate, or a budget of none.
    pub fn generate_codes(
        &self,
        request: &Zonos3Request,
    ) -> BunsenResult<Zonos3Codes> {
        self.generate_codes_sampled(request, &self.sampler)
    }

    /// [`generate_codes`](Self::generate_codes), its sampler the config's
    /// with `overrides` over it.
    ///
    /// # Errors
    /// As [`generate_codes`](Self::generate_codes).
    pub fn generate_codes_with(
        &self,
        request: &Zonos3Request,
        overrides: &TtsSamplingOverrides,
    ) -> BunsenResult<Zonos3Codes> {
        self.generate_codes_sampled(request, &self.try_sampler_with(overrides)?)
    }

    /// The sampler a take with `overrides` draws by: the config's, the
    /// overrides over it (none, the engine's own).
    ///
    /// # Errors
    /// As [`Zonos3SamplerConfig::try_init`]: an override it refuses, such as
    /// a repetition penalty that is not positive.
    pub fn try_sampler_with(
        &self,
        overrides: &TtsSamplingOverrides,
    ) -> BunsenResult<Zonos3Sampler> {
        if *overrides == TtsSamplingOverrides::default() {
            return Ok(self.sampler.clone());
        }
        let mut config = self.config.sampler.clone();
        if let Some(t) = overrides.temperature {
            config.temperature = t;
        }
        if let Some(k) = overrides.top_k {
            config.top_k = k;
        }
        if let Some(p) = overrides.top_p {
            config.top_p = p;
        }
        if let Some(p) = overrides.min_p {
            config.min_p = p;
        }
        if let Some(r) = overrides.repetition_penalty {
            config.repetition_penalty = r;
        }
        Self::sampler(&self.construct, &config)
    }

    /// [`try_sampler_with`](Self::try_sampler_with), panicking with its
    /// error's report.
    ///
    /// # Panics
    /// As `try_sampler_with`'s errors.
    pub fn sampler_with(
        &self,
        overrides: &TtsSamplingOverrides,
    ) -> Zonos3Sampler {
        use crate::errors::WithOkOrPanic;
        self.try_sampler_with(overrides).ok_or_panic()
    }

    /// [`generate_codes`](Self::generate_codes), sampled by `sampler`.
    ///
    /// # Errors
    /// As [`generate_codes`](Self::generate_codes).
    fn generate_codes_sampled(
        &self,
        request: &Zonos3Request,
        sampler: &Zonos3Sampler,
    ) -> BunsenResult<Zonos3Codes> {
        let mut run = self.start_take(request, sampler.clone())?;
        while self.step_take(&mut run).is_none() {}
        Ok(run.codes(self.config.overlap_frames))
    }

    /// A take started, sampled by `sampler`: its prompt built and run
    /// through the model, the device seeded, nothing sampled yet.
    /// [`step_take`](Self::step_take) then samples it a row at a time;
    /// [`generate_codes`](Self::generate_codes) is that loop.
    ///
    /// # Errors
    /// As [`generate_codes`](Self::generate_codes).
    pub fn start_take(
        &self,
        request: &Zonos3Request,
        sampler: Zonos3Sampler,
    ) -> BunsenResult<Zonos3TakeRun> {
        let lm = &self.construct.lm;
        let layout = &self.construct.prompt;
        let max_rows = self.construct.lm_config.max_rows;
        let prompt = layout.build(&Zonos3PromptRequest {
            text_ids: zonos3_text_ids(&self.tokenizer, &request.text)?,
            speaker: request.speaker.clone(),
            condition_buckets: request.condition_buckets.clone(),
            silence_prefill_frames: self.config.silence_prefill_frames,
        })?;
        let r = prompt.rows.len();
        if r >= max_rows {
            return Err(BunsenError::illegal(format!(
                "a prompt of {r} rows leaves no room in the model's {max_rows}"
            )));
        }
        let budget = self
            .config
            .max_new_frames
            .unwrap_or(max_rows)
            .min(request.max_frames.unwrap_or(max_rows))
            .min(max_rows - r);
        if budget == 0 {
            return Err(BunsenError::illegal("a take of no frames"));
        }

        let device = self.device();
        if let Some(seed) = request.seed {
            device.seed(seed);
        }
        let n = layout.n_codebooks;
        let pad = layout.audio_pad;
        let eoa = self.construct.lm_config.eoa_id() as u32;
        let cold_start = prompt
            .rows
            .last()
            .is_none_or(|row| row[..n].iter().all(|&c| c == pad));
        let mut cache = lm.decode_cache(1, r + budget, self.config.storage, &device);
        let logits = Self::last_logits(lm.forward(
            prompt.ids(&device),
            &prompt.conditioning(&device),
            Some(&mut cache),
        ));
        Ok(Zonos3TakeRun {
            cache,
            logits,
            take: Zonos3TakeRows::new(n, eoa, pad, cold_start, budget),
            history: vec![Vec::new(); n],
            silence: prompt.rows[r - prompt.silence_rows..].to_vec(),
            text_pad: layout.text_pad,
            sampler,
            pool: PoolRelease::default(),
            step: 0,
            stop: None,
        })
    }

    /// The take's next row: sampled, recorded, and run through the model;
    /// or why the take stopped.
    ///
    /// # Panics
    /// For a take that stopped already.
    pub fn step_take(
        &self,
        run: &mut Zonos3TakeRun,
    ) -> Option<Zonos3Stop> {
        assert!(run.stop.is_none(), "a step of a take that stopped");
        let device = self.device();
        let logits = run.logits.clone();
        let mut codes: Vec<u32> = run
            .sampler
            .sample(logits, &run.history)
            .into_iter()
            .map(|c| c as u32)
            .collect();
        let stop = run.take.push(&mut codes);
        for (j, &c) in codes.iter().enumerate() {
            run.history[j].push(i64::from(c));
        }
        if stop.is_some() {
            run.stop = stop;
            return stop;
        }
        let n = codes.len();
        let row: Vec<i64> = codes
            .iter()
            .map(|&c| i64::from(c))
            .chain([i64::from(run.text_pad)])
            .collect();
        let ids: Tensor<3, Int> = Tensor::from_data(TensorData::new(row, [1, 1, n + 1]), &device);
        run.logits = Self::last_logits(self.construct.lm.forward(
            ids,
            &Zonos3RowConditioning::default(),
            Some(&mut run.cache),
        ));
        run.pool.after_step(run.step, &device);
        run.step += 1;
        None
    }

    /// The samples of a take's codes: the codec's audio, through the output
    /// stage; none for no frames.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for frames not as
    /// wide as the codec's codebooks, or a code past them;
    /// [`Internal`](crate::errors::BunsenErrorKind::Internal) for audio that
    /// is not finite.
    pub fn render(
        &self,
        codes: &Zonos3Codes,
    ) -> BunsenResult<Vec<f32>> {
        let frames = &codes.frames;
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        let codec = &self.construct.codec;
        let size = self.construct.codec_config.codebook_size;
        let q = codec.quantizers.len();
        if frames
            .iter()
            .any(|f| f.len() != q || f.iter().any(|&c| c as usize >= size))
        {
            return Err(BunsenError::illegal(format!(
                "decoded frames must be {q} codes below {size}"
            )));
        }
        let ids = self.frame_ids(frames);
        self.output_samples(codec.forward(ids))
    }

    /// The codec's audio, `[B, N]`, through the output stage, as samples;
    /// refused when not finite. What `render` and a take's stream share.
    pub(crate) fn output_samples(
        &self,
        audio: Tensor<2>,
    ) -> BunsenResult<Vec<f32>> {
        let mut samples: Vec<f32> = audio
            .cast(DType::F32)
            .into_data()
            .try_into_vec_as::<f32>()
            .map_err(|e| BunsenError::internal(format!("the codec's audio: {e:?}")))?;
        if samples.iter().any(|x| !x.is_finite()) {
            return Err(BunsenError::internal("the codec's audio is not finite"));
        }
        zonos3_output_stage(
            &mut samples,
            self.config.headroom_db,
            self.config.output_limit_dbfs,
        );
        Ok(samples)
    }

    /// A take's frames, `[F][n]`, as the codec reads them: `[1, n, F]`.
    pub(crate) fn frame_ids(
        &self,
        frames: &[Vec<u32>],
    ) -> Tensor<3, Int> {
        let (q, f) = (frames[0].len(), frames.len());
        let flat: Vec<i64> = (0..q)
            .flat_map(|j| frames.iter().map(move |frame| i64::from(frame[j])))
            .collect();
        Tensor::from_data(TensorData::new(flat, [1, q, f]), &self.device())
    }
}
