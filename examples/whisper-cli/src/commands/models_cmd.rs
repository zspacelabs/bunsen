use std::collections::BTreeMap;

use bunsen::{
    data::{
        cache::verify_sha256,
        pretrained::{
            CacheStatus,
            Construct,
            PretrainedCache,
            PretrainedFactory,
            PretrainedProvider,
            PretrainedRef,
            StaticResourceMap,
        },
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::whisper::{
        WhisperApiConfig,
        driver::WhisperSpecialIds,
        pretrained::{
            CHECKPOINT,
            OPENAI_LOCAL_DIR,
            OPENAI_VOCABULARIES_MAPS,
            PytorchWhisperScanner,
            VOCABULARY,
            WHISPER_KIT,
            WHISPER_PREFABS,
            WhisperConstruct,
            WhisperVocabulary,
            default_whisper_factory,
            openai_download_root,
        },
    },
};
use clap_common::logging::{
    LogArgs,
    LogLevelNum,
};

use crate::whisper_clap::{
    ScannerArgs,
    WeightsCacheArgs,
};

/// Lists, fetches and inspects the models `--model` can name.
///
/// No backend is involved: nothing here builds a module. `inspect` scans a
/// checkpoint's geometry through bunsen's scanner, which reads shapes, not
/// tensors.
#[derive(clap::Args, Debug)]
pub struct ModelsCmd {
    #[clap(flatten)]
    pub logging: LogArgs,

    #[clap(flatten)]
    pub cache: WeightsCacheArgs,

    #[command(subcommand)]
    action: ModelsAction,
}

#[derive(clap::Subcommand, Debug)]
enum ModelsAction {
    /// List the pretrained models and their resources, and where each one
    /// stands.
    List,

    /// List the prefabs: the geometries, without weights.
    Prefabs,

    /// Bring every resource of the named models into the cache, fetching
    /// what is not local: a path's vocabulary is the one its checkpoint's
    /// layout selects, and a name's checkpoint must scan as its prefab.
    Fetch {
        /// Re-hash each pinned file after it is resolved, the cached ones
        /// included.
        #[arg(long)]
        verify: bool,

        #[clap(flatten)]
        scanner: ScannerArgs,

        /// Model names, as `--model` takes them.
        #[arg(required = true)]
        names: Vec<String>,
    },

    /// Scan a model's checkpoint and check it against its prefab.
    Inspect {
        /// A model name, as `--model` takes it, or a checkpoint path.
        name: String,

        #[clap(flatten)]
        scanner: ScannerArgs,
    },
}

impl ModelsCmd {
    pub fn run(&self) -> BunsenResult<()> {
        self.logging.init(Some(LogLevelNum::Info))?;
        let cache = self.cache.init()?;
        let factory = default_whisper_factory()?;

        match &self.action {
            ModelsAction::List => list(&factory, &cache),
            ModelsAction::Prefabs => prefabs(),
            ModelsAction::Fetch {
                verify,
                scanner,
                names,
            } => fetch(&factory, &cache, names, *verify, &scanner.scanner()),
            ModelsAction::Inspect { name, scanner } => {
                inspect(&factory, &cache, name, &scanner.scanner())
            }
        }
    }
}

fn resolve(
    factory: &PretrainedFactory,
    name: &str,
) -> BunsenResult<PretrainedRef> {
    factory.resolve_for::<WhisperConstruct>(name)
}

fn list(
    factory: &PretrainedFactory,
    cache: &PretrainedCache,
) -> BunsenResult<()> {
    println!("cache: {}", cache.cache_dir().display());
    let upstream = cache
        .local_dirs()
        .get(OPENAI_LOCAL_DIR)
        .cloned()
        .or_else(openai_download_root);
    match upstream {
        Some(dir) => println!("upstream cache: {}", dir.display()),
        None => println!("upstream cache: (no home directory)"),
    }

    for provider in factory.providers() {
        println!();
        list_provider(cache, provider.as_ref());
    }
    println!();
    list_maps(cache, "vocabularies", OPENAI_VOCABULARIES_MAPS);
    Ok(())
}

/// One provider's rows: each with its prefab, where its resources stand
/// taken together, and their keys.
fn list_provider(
    cache: &PretrainedCache,
    provider: &dyn PretrainedProvider,
) {
    println!(
        "{}: {} ({}; {})",
        provider.name(),
        provider.description(),
        provider.license().unwrap_or("license per row"),
        provider.origin().unwrap_or("origin per row"),
    );
    println!(
        "  NAME                             PREFAB           STATUS     RESOURCES              DESCRIPTION"
    );
    for row in provider.list() {
        let aliases = if row.aliases.is_empty() {
            String::new()
        } else {
            format!(" (also: {})", row.aliases.join(", "))
        };
        println!(
            "  {:<32} {:<16} {:<10} {:<22} {}{aliases}",
            provider.id(&row),
            row.prefab.as_deref().unwrap_or("-"),
            summarize(&cache.map_status(WHISPER_KIT, &row.resources)).to_string(),
            row.resources.keys().join("+"),
            row.description,
        );
    }
}

/// Where a whole map stands: the worst of its resources', since one remote
/// resource means a fetch.
fn summarize(status: &BTreeMap<String, CacheStatus>) -> CacheStatus {
    let rank = |s: &CacheStatus| match s {
        CacheStatus::Cached => 0,
        CacheStatus::Bundled => 1,
        CacheStatus::LocalDir => 2,
        CacheStatus::Remote => 3,
    };
    status
        .values()
        .copied()
        .max_by_key(rank)
        .unwrap_or(CacheStatus::Cached)
}

/// Standalone maps, one line per resource.
fn list_maps(
    cache: &PretrainedCache,
    what: &str,
    maps: &[&StaticResourceMap<'_>],
) {
    println!("{what}:");
    println!("  NAME                          KEY          KIND           STATUS     DESCRIPTION");
    for map in maps {
        let owned = map.to_map();
        for (key, r) in &owned.resources {
            println!(
                "  {:<29} {:<12} {:<14} {:<10} {}",
                map.name,
                key,
                r.kind.as_deref().unwrap_or("-"),
                cache.status(WHISPER_KIT, r).to_string(),
                map.description,
            );
        }
    }
}

/// The token layout a config's vocabulary size implies, which selects the
/// vocabulary.
fn layout_of(cfg: &WhisperApiConfig) -> BunsenResult<WhisperSpecialIds> {
    Ok(*cfg.token_layout.policy_for_vocab(cfg.vocab_size)?.ids())
}

fn prefabs() -> BunsenResult<()> {
    println!("{}: {}", WHISPER_PREFABS.name, WHISPER_PREFABS.description);
    println!(
        "  NAME             MELS  VOCAB D_MODEL HEADS  ENC  DEC AUDIO_CTX TEXT_CTX  DESCRIPTION"
    );
    for prefab in WHISPER_PREFABS.iter() {
        let g = prefab.to_config().geometry();
        println!(
            "  {:<16} {:>4} {:>6} {:>7} {:>5} {:>4} {:>4} {:>9} {:>8}  {}",
            prefab.name,
            g.n_mels,
            g.vocab_size,
            g.d_model,
            g.n_heads(),
            g.n_encoder_layers,
            g.n_decoder_layers,
            g.max_audio_ctx,
            g.max_text_ctx,
            prefab.description,
        );
    }
    Ok(())
}

fn fetch(
    factory: &PretrainedFactory,
    cache: &PretrainedCache,
    names: &[String],
    verify: bool,
    scanner: &PytorchWhisperScanner,
) -> BunsenResult<()> {
    for name in names {
        let model = resolve(factory, name)?;
        // The hook's plan, before anything but the checkpoint is fetched: a
        // path gets the vocabulary its checkpoint's layout selects, and a
        // name is checked against the geometry it promised.
        let hook = WhisperConstruct::new().with_scanner(scanner.clone());
        let map = hook.plan(&model, cache)?;
        let loaded = cache.load(WHISPER_KIT, &map)?;
        println!("{}:", model.id());
        for (key, part) in loaded.iter() {
            println!("  {key}: {} ({})", part.path.display(), part.provenance);
            if verify {
                match map.get(key).and_then(|r| r.sha256.as_deref()) {
                    Some(sha256) => {
                        verify_sha256(&part.path, sha256)?;
                        println!("    sha256 {sha256} ok");
                    }
                    None => println!("    (unpinned; no digest to verify against)"),
                }
            }
        }
    }
    Ok(())
}

fn inspect(
    factory: &PretrainedFactory,
    cache: &PretrainedCache,
    name: &str,
    scanner: &PytorchWhisperScanner,
) -> BunsenResult<()> {
    let model = resolve(factory, name)?;
    println!("model: {}", model.id());
    if let Some((provider, row)) = model.named() {
        let description = factory
            .provider(provider)
            .map(|p| p.description().to_string())
            .unwrap_or_default();
        println!("  provider: {provider} ({description})");
        println!("  description: {}", row.description);
    }

    let map = model.to_map();
    println!("resources:");
    for (key, r) in &map.resources {
        let pin = match &r.sha256 {
            Some(sha256) => format!("sha256 {sha256}"),
            None => "unpinned".to_string(),
        };
        println!(
            "  {key}: {} ({}, {pin}) [{}]",
            r.file,
            r.kind.as_deref().unwrap_or("unlabeled"),
            cache.status(WHISPER_KIT, r),
        );
        for source in &r.sources {
            println!("    {source}");
        }
    }

    let promised = model.prefab(&WHISPER_PREFABS);
    if let Some(prefab) = &promised {
        println!("prefab: {} ({})", prefab.name, prefab.description);
        println!("  {:?}", prefab.to_config().geometry());
    }

    let loaded = cache.load(WHISPER_KIT, &map)?;
    let checkpoint = loaded.get(CHECKPOINT).ok_or_else(|| {
        BunsenError::ResourceNotFound(format!("{}: no {CHECKPOINT} resource", model.id()))
    })?;
    println!(
        "checkpoint: {} ({})",
        checkpoint.path.display(),
        checkpoint.provenance
    );

    // A named model that does not scan as its prefab is an error from
    // `scan`; report it as the finding it is rather than a failure.
    let hook = WhisperConstruct::new().with_scanner(scanner.clone());
    let cfg = match hook.scan(&model, &checkpoint.path) {
        Ok(cfg) => cfg,
        Err(BunsenError::Invalid(msg)) if promised.is_some() => {
            println!("MISMATCH: {msg}");
            return Ok(());
        }
        Err(e) => return Err(e),
    };
    let geometry = cfg.geometry();
    println!("scanned: {geometry:?}");
    println!("  heads: {}", geometry.n_heads());
    println!("  front end: {:?}", cfg.front_end);

    match (&promised, geometry.prefab()) {
        (Some(prefab), _) => println!("  matches prefab {}", prefab.name),
        (None, Some(prefab)) => println!("  geometry is prefab {}", prefab.name),
        (None, None) => println!("  geometry matches no built-in prefab"),
    }

    // A row declares its vocabulary, listed above; a path has only the
    // rule, applied to what the scan found.
    if map.get(VOCABULARY).is_none() {
        let vocab = WhisperVocabulary::for_layout(&layout_of(&cfg)?)
            .map()
            .to_map();
        let r = vocab.try_get(VOCABULARY)?;
        println!(
            "vocabulary (by the checkpoint's layout): {} ({})",
            vocab.name,
            cache.status(WHISPER_KIT, r)
        );
    }
    Ok(())
}
