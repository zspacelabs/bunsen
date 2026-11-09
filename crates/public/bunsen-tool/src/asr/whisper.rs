//! Transcribing with Whisper: the construct through a stream driver.
use std::{
    path::{
        Path,
        PathBuf,
    },
    sync::Arc,
};

use bunsen::{
    data::pretrained::{
        PretrainedCache,
        PretrainedCacheOptions,
        ResourceBundle,
    },
    errors::{
        BunsenResult,
        ResultContext,
    },
    kits::{
        speech::asr::{
            driver::TranscriptSegment,
            whisper::{
                driver::{
                    WhisperConstruct,
                    WhisperStreamDriver,
                    WhisperStreamDriverConfig,
                },
                pretrained::{
                    CHECKPOINT,
                    OPENAI_LOCAL_DIR,
                    VOCABULARY,
                    WhisperConstructConfig,
                    default_whisper_factory,
                },
            },
        },
        support::construct::Deferred,
    },
};
use burn::tensor::Device;

use crate::asr::{
    Emission,
    Segment,
    Transcript,
    segment_speaker,
    speaker_turns,
    stream,
};

/// Whisper's own transcription flags: paths, not policy. The policy is the
/// model's, its presets and patch resolved
/// (`--model 'openai/small[translate]{beam_size: 1}'`).
#[derive(clap::Args, Debug, Default)]
pub struct WhisperFlags {
    /// A vocabulary file (`multilingual.tiktoken`, `gpt2.tiktoken`) in place
    /// of the one the checkpoint's layout selects; trusted as given.
    #[arg(long)]
    pub vocab: Option<PathBuf>,

    /// `openai-whisper`'s download cache, whose files are used in place;
    /// `~/.cache/whisper` when omitted.
    #[arg(long)]
    pub upstream_cache_dir: Option<PathBuf>,
}

impl WhisperFlags {
    /// `options`, with `--upstream-cache-dir` as `openai-whisper`'s cache.
    pub fn cache_options(
        &self,
        options: PretrainedCacheOptions,
    ) -> PretrainedCacheOptions {
        match &self.upstream_cache_dir {
            Some(dir) => options.with_local_dir(OPENAI_LOCAL_DIR, dir.clone()),
            None => options,
        }
    }
}

/// What `--model` names: a checkpoint on disk, as a one-resource bundle under
/// [`CHECKPOINT`] (its vocabulary the one its layout selects), or a row of
/// the Whisper factory; `--vocab` over either.
///
/// # Errors
/// [`Policy`](bunsen::errors::BunsenErrorKind::Policy) when the factory
/// does not have the name, or the map cannot take the vocabulary: `spec`
/// and `--vocab` come from the command line.
pub fn resolve(
    spec: &str,
    flags: Option<&WhisperFlags>,
    cache: &PretrainedCache,
) -> BunsenResult<Deferred<WhisperConstructConfig>> {
    let path = Path::new(spec);
    let mut model = if path.is_file() {
        Deferred::from_resource_bundle(ResourceBundle::given(spec, CHECKPOINT, path)).as_policy()?
    } else {
        default_whisper_factory()?
            .resolve(spec, cache)
            .as_policy()?
    };
    if let Some(vocab) = flags.and_then(|f| f.vocab.as_ref()) {
        model = model
            .with_overlay(ResourceBundle::given("--vocab", VOCABULARY, vocab))
            .as_policy()?;
    }
    Ok(model)
}

/// The driver for `bundle` under its policy and `emission`.
pub fn driver(
    bundle: Arc<WhisperConstruct>,
    emission: Emission,
    device: &Device,
) -> BunsenResult<WhisperStreamDriver> {
    // The model's policy, its presets and patch resolved; an English-only
    // checkpoint is told no language.
    let mut policy = bundle.policy.clone();
    if !bundle.layout.ids().is_multilingual()
        && let Some(language) = policy.language.take()
    {
        log::warn!("English-only checkpoint: language {language} does not apply");
    }
    let driver: WhisperStreamDriver = WhisperStreamDriverConfig::new()
        .with_policy(&policy)
        .with_emission(emission.into())
        .init_from_construct(bundle, device)
        // The settings come from the command line.
        .as_policy()?;
    Ok(driver)
}

/// Transcribes one recording through `driver`.
pub fn transcribe(
    driver: &WhisperStreamDriver,
    samples: &[f32],
    chunk: usize,
) -> BunsenResult<(Transcript, Vec<TranscriptSegment>)> {
    let (committed, ctx) = stream::stream(driver, samples, chunk)?;
    let segments = committed
        .iter()
        .map(|s| Segment {
            start: s.start,
            end: s.end,
            text: s.text.as_deref().unwrap_or("").trim().to_string(),
            tokens: s.tokens.clone(),
            speaker: segment_speaker(s),
        })
        .collect();
    let transcript = Transcript {
        text: stream::join_text(&committed),
        turns: driver.speakers().map(|_| speaker_turns(&committed)),
        language: ctx.language().map(String::from),
        words: None,
        segments: Some(segments),
    };
    Ok((transcript, committed))
}
