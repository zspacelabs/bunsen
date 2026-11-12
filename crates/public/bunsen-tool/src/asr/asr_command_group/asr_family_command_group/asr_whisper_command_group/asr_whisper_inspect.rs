//! `asr family whisper inspect`.

use bunsen::{
    data::{
        builds::GEOMETRY,
        resources::ResourceCache,
    },
    errors::{
        BunsenError,
        BunsenResult,
        LookupError,
    },
    kits::speech::asr::whisper::{
        WhisperApiConfig,
        WhisperGeometry,
        driver::WhisperSpecialIds,
        pretrained::{
            CHECKPOINT,
            GeometryMismatch,
            VOCABULARY,
            WhisperVocabulary,
            default_whisper_factory,
        },
    },
};
use bunsen_app::{
    cache::WeightsCacheArgs,
    logging::{
        LevelFilter,
        LogArgs,
        LogPolicy,
    },
};

use crate::asr::whisper;

/// Scans a Whisper checkpoint and checks it against its prefab: its
/// resources and where each stands, the geometry it promised, and the one
/// its tensors' shapes say. No backend is involved; nothing is built.
#[derive(clap::Args, Debug)]
pub struct AsrWhisperInspectCommand {
    #[clap(flatten)]
    pub logging: LogArgs,

    #[clap(flatten)]
    pub cache: WeightsCacheArgs,

    /// A model name, as `--model` takes it, or a checkpoint's path.
    pub name: String,
}

/// The token layout a config's vocabulary size implies, which selects the
/// vocabulary.
fn layout_of(cfg: &WhisperApiConfig) -> BunsenResult<WhisperSpecialIds> {
    Ok(*cfg.token_layout.policy_for_vocab(cfg.vocab_size)?.ids())
}

impl AsrWhisperInspectCommand {
    pub fn run(&self) -> BunsenResult<()> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Warn, LevelFilter::Warn))?;
        let cache: ResourceCache = self.cache.init()?;
        let model = whisper::resolve(&self.name, None, &cache)?;
        println!("model: {}", model.id());
        if let Some((provider, row)) = model.model.build() {
            let description = default_whisper_factory()?
                .provider(provider)
                .map(|p| p.description().to_string())
                .unwrap_or_default();
            println!("  provider: {provider} ({description})");
            println!("  description: {}", row.description);
        }

        let map = model.to_resource_set();
        println!("resources:");
        for (key, r) in &map.resources {
            let pin = match &r.sha256 {
                Some(sha256) => format!("sha256 {sha256}"),
                None => "unpinned".to_string(),
            };
            println!(
                "  {key}: {} ({}, {pin}) [{}]",
                r.file,
                r.format.as_deref().unwrap_or("unlabeled"),
                cache.status(r),
            );
            for source in &r.sources {
                println!("    {source}");
            }
        }

        let promised: Option<WhisperGeometry> = model.model.stated(GEOMETRY)?;
        if let Some(geometry) = &promised {
            match geometry.prefab() {
                Some(prefab) => println!("geometry: prefab {prefab}"),
                None => println!("geometry: stated"),
            }
            println!("  {geometry:?}");
        }

        let loaded = cache.load(&map)?;
        let checkpoint = loaded.family(CHECKPOINT);
        if checkpoint.is_empty() {
            return Err(BunsenError::lookup(
                LookupError::missing("resource", CHECKPOINT).with_candidates(loaded.keys()),
            )
            .context(model.id()));
        }
        for (key, part) in checkpoint {
            println!("{key}: {} ({})", part.path.display(), part.provenance);
        }

        // A named model whose checkpoint does not scan as its prefab is a
        // `GeometryMismatch` from `scan`; report it as the finding it is
        // rather than a failure. Any other error is a failure.
        let cfg = match model.scan(&loaded) {
            Ok(cfg) => cfg,
            Err(e) if promised.is_some() && e.find::<GeometryMismatch>().is_some() => {
                println!("MISMATCH: {e}");
                return Ok(());
            }
            Err(e) => return Err(e),
        };
        let geometry = cfg.geometry();
        println!("scanned: {geometry:?}");
        println!("  heads: {}", geometry.n_heads());
        println!("  front end: {:?}", cfg.front_end);

        match (&promised, geometry.prefab()) {
            (Some(_), _) => println!("  matches the stated geometry"),
            (None, Some(prefab)) => println!("  geometry is prefab {}", prefab),
            (None, None) => println!("  geometry matches no built-in prefab"),
        }

        // A row declares its vocabulary, listed above; a path has only the
        // rule, applied to what the scan found.
        if map.get(VOCABULARY).is_none() {
            let vocab = WhisperVocabulary::for_layout(&layout_of(&cfg)?).resource_set();
            let r = vocab.try_get(VOCABULARY)?;
            println!(
                "vocabulary (by the checkpoint's layout): {} ({})",
                vocab.name,
                cache.status(r)
            );
        }
        Ok(())
    }
}
