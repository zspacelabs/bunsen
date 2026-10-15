//! `phonon models`.
use bunsen::{
    errors::BunsenResult,
    kits::speech::parakeet::pretrained::{
        PARAKEET_KIT,
        default_parakeet_factory,
    },
};
use bunsen_app::logging::{
    LogArgs,
    LogLevelNum,
};

use crate::args::WeightsCacheArgs;

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
        let factory = default_parakeet_factory()?;
        match &self.action {
            ModelsAction::List => {
                println!("cache: {}", cache.cache_dir().display());
                for (provider, row) in factory.list() {
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
                            cache.status(PARAKEET_KIT, r)
                        );
                    }
                }
                Ok(())
            }
            ModelsAction::Fetch { names } => {
                for name in names {
                    let model = factory.resolve(name, &cache)?;
                    let loaded = cache.load(PARAKEET_KIT, &model.to_map())?;
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
