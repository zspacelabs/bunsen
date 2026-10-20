//! `bunsen-asr <family> list`.
use bunsen::errors::BunsenResult;
use bunsen_app::{
    cache::WeightsCacheArgs,
    logging::{
        LevelFilter,
        LogArgs,
        LogPolicy,
    },
};

use crate::family::Family;

/// Lists the models `--model` can name, their licences and files, and
/// whether each file is cached.
#[derive(clap::Args, Debug)]
pub struct ListCmd {
    #[clap(flatten)]
    pub logging: LogArgs,

    #[clap(flatten)]
    pub cache: WeightsCacheArgs,
}

impl ListCmd {
    pub fn run(
        &self,
        family: Family,
    ) -> BunsenResult<()> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Info, LevelFilter::Info))?;
        let cache = self.cache.init()?;
        let listing = family.listing()?;
        println!(
            "{} models; cache: {}",
            family.name(),
            cache.cache_dir().display()
        );
        for (provider, row) in listing.rows {
            let default = if row.name == family.default_model() {
                " (default)"
            } else {
                ""
            };
            println!();
            println!("{provider}:{}{default}", row.name);
            println!("  {}", row.description);
            println!("  license: {}", row.license.as_deref().unwrap_or("unknown"));
            if !row.aliases.is_empty() {
                println!("  aliases: {}", row.aliases.join(", "));
            }
            for (key, r) in &row.resources.resources {
                println!(
                    "  {key}: {} ({}) [{}]",
                    r.file,
                    r.kind.as_deref().unwrap_or("unlabeled"),
                    cache.status(family.cache_kit(), r)
                );
            }
        }
        for provider in listing.unlisted {
            println!();
            println!(
                "{provider}:org/repo  any Hugging Face repo in transformers' layout, resolved on use"
            );
        }
        Ok(())
    }
}
