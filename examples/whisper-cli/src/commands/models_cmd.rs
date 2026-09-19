use bunsen::{
    data::{
        cache::verify_sha256,
        pretrained::WeightsCache,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::whisper::pretrained::{
        OPENAI_LOCAL_DIR,
        WHISPER_KIT,
        WHISPER_PREFABS,
        WHISPER_PROVIDERS,
        openai_download_root,
        prefab_for_geometry,
    },
};
use clap_common::logging::{
    LogArgs,
    LogLevelNum,
};

use crate::{
    models::loader::{
        ModelRef,
        scan_model,
    },
    whisper_clap::WeightsCacheArgs,
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
    /// List the pretrained models, and where each one stands.
    List,

    /// List the prefabs: the geometries, without weights.
    Prefabs,

    /// Bring models into the cache, fetching what is not local.
    Fetch {
        /// Re-hash each file after it is resolved, the bundled and cached
        /// ones included.
        #[arg(long)]
        verify: bool,

        /// Model names, as `--model` takes them.
        #[arg(required = true)]
        names: Vec<String>,
    },

    /// Scan a model's checkpoint and check it against its prefab.
    Inspect {
        /// A model name, as `--model` takes it, or a checkpoint path.
        name: String,
    },
}

impl ModelsCmd {
    pub fn run(&self) -> BunsenResult<()> {
        self.logging.init(Some(LogLevelNum::Info))?;
        let cache = self.cache.init()?;

        match &self.action {
            ModelsAction::List => list(&cache),
            ModelsAction::Prefabs => prefabs(),
            ModelsAction::Fetch { verify, names } => fetch(&cache, names, *verify),
            ModelsAction::Inspect { name } => inspect(&cache, name),
        }
    }
}

fn list(cache: &WeightsCache) -> BunsenResult<()> {
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

    for provider in WHISPER_PROVIDERS {
        println!();
        println!(
            "{}: {} ({}; {})",
            provider.name,
            provider.description,
            provider.license.unwrap_or("license unknown"),
            provider.origin.unwrap_or("origin unknown"),
        );
        println!(
            "  NAME                   PREFAB           FORMAT         STATUS          DESCRIPTION"
        );
        for pretrained in provider.items {
            let aliases = if pretrained.aliases.is_empty() {
                String::new()
            } else {
                format!(" (also: {})", pretrained.aliases.join(", "))
            };
            println!(
                "  {:<22} {:<16} {:<14} {:<15} {}{aliases}",
                provider.id(pretrained),
                pretrained.prefab,
                pretrained.format.to_string(),
                cache
                    .status(WHISPER_KIT, provider.name, &pretrained.to_descriptor())
                    .to_string(),
                pretrained.description,
            );
        }
    }
    Ok(())
}

fn prefabs() -> BunsenResult<()> {
    println!("{}: {}", WHISPER_PREFABS.name, WHISPER_PREFABS.description);
    println!(
        "  NAME             MELS  VOCAB D_MODEL HEADS  ENC  DEC AUDIO_CTX TEXT_CTX  DESCRIPTION"
    );
    for prefab in WHISPER_PREFABS.items {
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
    cache: &WeightsCache,
    names: &[String],
    verify: bool,
) -> BunsenResult<()> {
    for name in names {
        let model = ModelRef::resolve(name)?;
        let located = model.locate(cache)?;
        println!(
            "{}: {} ({})",
            model.id(),
            located.path.display(),
            located.provenance
        );

        if verify {
            match &model {
                ModelRef::Pretrained { pretrained, .. } => match pretrained.sha256 {
                    Some(sha256) => {
                        verify_sha256(&located.path, sha256)?;
                        println!("  sha256 {sha256} ok");
                    }
                    None => println!("  (unpinned; no digest to verify against)"),
                },
                ModelRef::Path(_) => {
                    println!("  (a path has no digest to verify against)");
                }
            }
        }
    }
    Ok(())
}

fn inspect(
    cache: &WeightsCache,
    name: &str,
) -> BunsenResult<()> {
    let model = ModelRef::resolve(name)?;
    println!("model: {}", model.id());

    if let ModelRef::Pretrained {
        provider,
        pretrained,
    } = &model
    {
        println!("  provider: {} ({})", provider.name, provider.description);
        println!("  format: {}", pretrained.format);
        println!("  sha256: {}", pretrained.sha256.unwrap_or("unpinned"));
        println!(
            "  status: {}",
            cache.status(WHISPER_KIT, provider.name, &pretrained.to_descriptor())
        );
        println!("  sources:");
        for source in pretrained.sources {
            println!("    {source}");
        }
    }

    let promised = model.prefab();
    if let Some(prefab) = &promised {
        println!("prefab: {} ({})", prefab.name, prefab.description);
        println!("  {:?}", prefab.to_config().geometry());
    }

    let located = model.locate(cache)?;
    println!(
        "checkpoint: {} ({})",
        located.path.display(),
        located.provenance
    );

    // A named model that does not scan as its prefab is an error from
    // `scan_model`; report it as the finding it is rather than a failure.
    let cfg = match scan_model(&model, &located.path) {
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

    match (&promised, prefab_for_geometry(&geometry)) {
        (Some(prefab), _) => println!("  matches prefab {}", prefab.name),
        (None, Some(prefab)) => println!("  geometry is prefab {}", prefab.name),
        (None, None) => println!("  geometry matches no built-in prefab"),
    }
    Ok(())
}
