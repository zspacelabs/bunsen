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
    errors::{
        BunsenError,
        BunsenResult,
        LookupError,
    },
    kits::{
        speech::tts::{
            driver::{
                TtsCapabilities,
                TtsDriver,
                TtsDriverConfig,
                TtsLanguage,
                TtsSamplingOverrides,
            },
            zonos3::{
                Zonos3Construct,
                Zonos3GenerationJson,
                lm::{
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

/// Config for [`Zonos3Engine`]; its defaults are the release's
/// (`config.json`'s `generation`, [`from_generation`](Self::from_generation)).
#[derive(Config, Debug)]
pub struct Zonos3EngineConfig {
    /// The language a request that names none is in: a request code of the
    /// release's `languages` table (`en_us`, `fr_fr`), or a language tag
    /// ([`Zonos3GenerationJson::language_tag`]).
    #[config(default = "String::from(\"en_us\")")]
    pub default_language: String,

    /// The stock voice a request that names none speaks in.
    #[config(default = "String::from(\"Savannah\")")]
    pub default_voice: String,

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

impl Zonos3EngineConfig {
    /// The settings of a release's `generation` section (its markup tags
    /// aside: the normalizer keeps none, as the release's reference).
    pub fn from_generation(generation: &Zonos3GenerationJson) -> Self {
        Self::new()
            .with_sampler(Zonos3SamplerConfig::from_generation(generation))
            .with_silence_prefill_frames(generation.silence_prefill_frames)
            .with_silence_prefill_max_frames(generation.silence_prefill_max_frames)
            .with_overlap_frames(generation.overlap_frames)
            .with_headroom_db(generation.headroom_db)
            .with_output_limit_dbfs(generation.output_limit_dbfs)
    }
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
    /// ([`Zonos3GenerationJson::language_tag`]);
    /// [`Lookup`](crate::errors::BunsenErrorKind::Lookup) for a default voice
    /// that is not a stock voice, or a condition the release does not have.
    pub fn new(
        construct: Arc<Zonos3Construct>,
        tokenizer: T,
        config: Zonos3EngineConfig,
    ) -> BunsenResult<Self> {
        if config.silence_prefill_frames > config.silence_prefill_max_frames {
            return Err(BunsenError::illegal(format!(
                "a silence prefill of {} frames is past the limit of {}",
                config.silence_prefill_frames, config.silence_prefill_max_frames
            )));
        }
        let language = construct
            .generation
            .language_tag(&config.default_language)?;
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
        let sampler = Self::sampler(&construct, &config.sampler);
        let capabilities = TtsCapabilities {
            languages: construct
                .generation
                .languages
                .iter()
                .map(|(key, grammar)| {
                    Ok(TtsLanguage {
                        key: key.clone(),
                        tag: BunsenLanguageTag::parse(grammar)?,
                    })
                })
                .collect::<BunsenResult<_>>()?,
            default_voice: Some(config.default_voice.clone()),
            stock_voices: true,
            described_voices: false,
            instruction: false,
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
    ) -> Zonos3Sampler {
        let lm = &construct.lm_config;
        config.init(lm.codebook_size, lm.eoa_id(), lm.audio_vocab())
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

    /// A stock voice as a reference prefix, the release's default speaker
    /// mode.
    ///
    /// # Errors
    /// [`Lookup`](crate::errors::BunsenErrorKind::Lookup) for a name that is
    /// not a stock voice.
    pub fn voice_reference(
        &self,
        name: &str,
    ) -> BunsenResult<Zonos3Speaker> {
        let voices = self.voices();
        voices
            .get(name)
            .map(|voice| Zonos3Speaker::Reference(voice.codes.clone()))
            .ok_or_else(|| {
                BunsenError::lookup(
                    LookupError::missing("stock voice", name).with_candidates(voices.keys()),
                )
            })
    }

    fn device(&self) -> Device {
        self.construct.lm.head.weight.device()
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
        self.generate_codes_sampled(request, &Self::sampler(&self.construct, &config))
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
        let mut logits = Self::last_logits(lm.forward(
            prompt.ids(&device),
            &prompt.conditioning(&device),
            Some(&mut cache),
        ));

        let mut take = Zonos3TakeRows::new(n, eoa, pad, cold_start, budget);
        let mut history: Vec<Vec<i64>> = vec![Vec::new(); n];
        let stop = loop {
            let mut codes: Vec<u32> = sampler
                .sample(logits, &history)
                .into_iter()
                .map(|c| c as u32)
                .collect();
            let stop = take.push(&mut codes);
            for (j, &c) in codes.iter().enumerate() {
                history[j].push(i64::from(c));
            }
            if let Some(stop) = stop {
                break stop;
            }
            let row: Vec<i64> = codes
                .iter()
                .map(|&c| i64::from(c))
                .chain([i64::from(layout.text_pad)])
                .collect();
            let ids: Tensor<3, Int> =
                Tensor::from_data(TensorData::new(row, [1, 1, n + 1]), &device);
            logits = Self::last_logits(lm.forward(
                ids,
                &Zonos3RowConditioning::default(),
                Some(&mut cache),
            ));
        };

        let silence = &prompt.rows[r - prompt.silence_rows..];
        let (frames, start) = take.frames(silence, self.config.overlap_frames);
        Ok(Zonos3Codes {
            frames,
            start,
            stop,
            rows: take.rows().len(),
        })
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
        let f = frames.len();
        let flat: Vec<i64> = (0..q)
            .flat_map(|j| frames.iter().map(move |frame| i64::from(frame[j])))
            .collect();
        let ids: Tensor<3, Int> =
            Tensor::from_data(TensorData::new(flat, [1, q, f]), &self.device());
        let mut samples: Vec<f32> = codec
            .forward(ids)
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
}
