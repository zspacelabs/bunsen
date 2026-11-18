//! [`Zonos3Driver`]: text and a speaker to speech.

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
    Zonos3Normalizer,
    Zonos3Stop,
    Zonos3TakeRows,
    Zonos3TextNormalization,
    Zonos3ZyphraNormalization,
    zonos3_output_stage,
};
use crate::{
    errors::{
        BunsenError,
        BunsenResult,
        LookupError,
    },
    kits::{
        speech::tts::zonos3::{
            Zonos3Construct,
            Zonos3GenerationJson,
            lm::Zonos3RowConditioning,
            prompt::{
                Zonos3PromptRequest,
                Zonos3Speaker,
                zonos3_text_ids,
            },
            sampler::{
                Zonos3Sampler,
                Zonos3SamplerConfig,
            },
            voices::Zonos3Voice,
        },
        support::tokens::Tokenizer,
    },
    ops::transformers::attention::KVCacheStorage,
};

/// Config for [`Zonos3Driver`]; its defaults are the release's
/// (`config.json`'s `generation`, [`from_generation`](Self::from_generation)).
#[derive(Config, Debug)]
pub struct Zonos3DriverConfig {
    /// How a request's text is normalized; Zyphra's normalizer in English
    /// by default, the release's (feature `zyphra-textnorm`).
    #[config(default = "Zonos3TextNormalization::Zyphra(Zonos3ZyphraNormalization::new())")]
    pub normalization: Zonos3TextNormalization,

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

impl Zonos3DriverConfig {
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

/// What [`Zonos3Driver`] speaks.
#[derive(Clone, Debug, PartialEq)]
pub struct Zonos3Request {
    /// The text, as typed: the driver normalizes it
    /// ([`normalization`](Zonos3DriverConfig::normalization)).
    pub text: String,

    /// The speaker.
    pub speaker: Zonos3Speaker,

    /// Each condition's bucket; `None` (or a short list) is the unknown
    /// bucket.
    pub condition_buckets: Vec<Option<usize>>,

    /// Seeds the device's random numbers before the take; `None` leaves
    /// them as they are.
    pub seed: Option<u64>,
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

/// A take: its samples, and the codes they decode.
#[derive(Clone, Debug, PartialEq)]
pub struct Zonos3Speech {
    /// The samples, mono, in `[-1, 1]`.
    pub samples: Vec<f32>,

    /// Their rate, in Hz.
    pub sample_rate: u32,

    /// The codes.
    pub codes: Zonos3Codes,
}

/// ZONOS3-Mini's driver: text and a speaker to speech, one take decoded
/// whole, as the release generates it.
///
/// A take is the text, normalized ([`Zonos3Normalizer`]); its prompt
/// ([`Zonos3PromptConfig::build`](crate::kits::speech::tts::zonos3::prompt::Zonos3PromptConfig::build))
/// run through the language model, then one row a step through its decode
/// cache, each row's codes sampled by [`Zonos3Sampler`] and fed back with the
/// text pad, until [`Zonos3TakeRows`]'s stop rule; then the frames it covers,
/// decoded by the codec, through the output stage ([`zonos3_output_stage`]).
#[derive(Debug)]
pub struct Zonos3Driver<T: Tokenizer> {
    construct: Arc<Zonos3Construct>,
    normalizer: Zonos3Normalizer,
    sampler: Zonos3Sampler,
    tokenizer: T,
    config: Zonos3DriverConfig,
}

impl<T: Tokenizer> Zonos3Driver<T> {
    /// The driver of a loaded model, its text read by `tokenizer` (the
    /// construct's [`tokenizer`](Zonos3Construct::tokenizer) file).
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a silence
    /// prefill past its limit; the normalizer's
    /// ([`Zonos3Normalizer::new`]).
    pub fn new(
        construct: Arc<Zonos3Construct>,
        tokenizer: T,
        config: Zonos3DriverConfig,
    ) -> BunsenResult<Self> {
        if config.silence_prefill_frames > config.silence_prefill_max_frames {
            return Err(BunsenError::illegal(format!(
                "a silence prefill of {} frames is past the limit of {}",
                config.silence_prefill_frames, config.silence_prefill_max_frames
            )));
        }
        let normalizer =
            Zonos3Normalizer::new(&config.normalization, &construct.generation.languages)?;
        let lm = &construct.lm_config;
        let sampler = config
            .sampler
            .init(lm.codebook_size, lm.eoa_id(), lm.audio_vocab());
        Ok(Self {
            construct,
            normalizer,
            sampler,
            tokenizer,
            config,
        })
    }

    /// The model.
    pub fn construct(&self) -> &Zonos3Construct {
        &self.construct
    }

    /// The config.
    pub fn config(&self) -> &Zonos3DriverConfig {
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

    /// The take's codes: the text normalized, the prompt, the generated
    /// rows, and the frames they cover.
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
        let lm = &self.construct.lm;
        let layout = &self.construct.prompt;
        let max_rows = self.construct.lm_config.max_rows;
        let prompt = layout.build(&Zonos3PromptRequest {
            text_ids: zonos3_text_ids(&self.tokenizer, &self.normalizer.normalize(&request.text))?,
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
            let mut codes: Vec<u32> = self
                .sampler
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

    /// A take: [`generate_codes`](Self::generate_codes), then
    /// [`render`](Self::render).
    ///
    /// # Errors
    /// As theirs.
    pub fn synthesize(
        &self,
        request: &Zonos3Request,
    ) -> BunsenResult<Zonos3Speech> {
        let codes = self.generate_codes(request)?;
        Ok(Zonos3Speech {
            samples: self.render(&codes)?,
            sample_rate: self.sample_rate(),
            codes,
        })
    }
}
