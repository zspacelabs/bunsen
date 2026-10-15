//! `phonon models`.
use bunsen::errors::BunsenResult;
use bunsen_app::logging::{
    LogArgs,
    LogLevelNum,
};

use crate::{
    args::WeightsCacheArgs,
    transcriber::Kit,
};

/// Lists and fetches the models `--model` can name.
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
    /// List the models, their files, and whether each is cached.
    List,

    /// Bring every file of the named models into the cache.
    Fetch {
        /// Model names, as `--model` takes them.
        #[arg(required = true)]
        names: Vec<String>,
    },
}

impl ModelsCmd {
    pub fn run(&self) -> BunsenResult<()> {
        self.logging.init(Some(LogLevelNum::Info))?;
        let cache = self.cache.init()?;
        match &self.action {
            ModelsAction::List => {
                println!("cache: {}", cache.cache_dir().display());
                for kit in Kit::ALL {
                    for (provider, row) in kit.list()? {
                        println!();
                        println!("{provider}:{}", row.name);
                        println!("  {}", row.description);
                        if !row.aliases.is_empty() {
                            println!("  aliases: {}", row.aliases.join(", "));
                        }
                        for (key, r) in &row.resources.resources {
                            println!(
                                "  {key}: {} ({}) [{}]",
                                r.file,
                                r.kind.as_deref().unwrap_or("unlabeled"),
                                cache.status(kit.cache_kit(), r)
                            );
                        }
                    }
                }
                Ok(())
            }
            ModelsAction::Fetch { names } => {
                for name in names {
                    let (kit, _) = Kit::of(name)?;
                    let map = match kit {
                        Kit::Parakeet => {
                            bunsen::kits::speech::parakeet::pretrained::default_parakeet_factory()?
                                .resolve(name, &cache)?
                                .to_map()
                        }
                        Kit::Qwen3Asr => {
                            bunsen::kits::speech::qwen3_asr::pretrained::default_qwen3_asr_factory(
                            )?
                            .resolve(name, &cache)?
                            .to_map()
                        }
                    };
                    let loaded = cache.load(kit.cache_kit(), &map)?;
                    for (key, part) in &loaded.parts {
                        println!(
                            "{name} {key}: {} ({})",
                            part.path.display(),
                            part.provenance
                        );
                    }
                }
                Ok(())
            }
        }
    }
}
