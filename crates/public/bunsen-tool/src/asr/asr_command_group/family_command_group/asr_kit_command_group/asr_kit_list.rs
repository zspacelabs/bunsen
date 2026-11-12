//! `asr family <family> list` and `asr diarize list`.

use bunsen::errors::BunsenResult;
use bunsen_app::{
    cache::{
        WeightsCacheArgs,
        human_bytes,
    },
    logging::{
        LevelFilter,
        LogArgs,
        LogPolicy,
    },
};

use crate::{
    asr::AsrKit,
    support::{
        preset_lines,
        print_providers,
    },
};

/// Lists the models `--model` can name, their licences, their presets, and
/// their files, and whether each file is cached.
#[derive(clap::Args, Debug)]
pub struct AsrKitListCommand {
    #[clap(flatten)]
    pub logging: LogArgs,

    #[clap(flatten)]
    pub cache: WeightsCacheArgs,
}

impl AsrKitListCommand {
    pub fn run(
        &self,
        kit: AsrKit,
    ) -> BunsenResult<()> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Info, LevelFilter::Info))?;
        let cache = self.cache.init()?;
        let listing = kit.listing()?;
        println!(
            "{} models; cache: {}",
            kit.name(),
            cache.cache_dir().display()
        );
        print_providers(&listing.providers);
        for (provider, row) in listing.rows {
            let default = if row.name == kit.default_model() {
                " (default)"
            } else {
                ""
            };
            println!();
            println!("{provider}:{}{default}", row.name);
            println!("  {}", row.description);
            let resources = row.resource_set();
            if let Some(size) = resources.size() {
                println!("  size: {}", human_bytes(size));
            }
            let dtypes = resources.dtypes();
            if !dtypes.is_empty() {
                println!("  dtype: {}", dtypes.join(", "));
            }
            println!("  license: {}", row.license.as_deref().unwrap_or("unknown"));
            if !row.aliases.is_empty() {
                let aliases: Vec<&str> = row.aliases.iter().map(|a| a.leaf()).collect();
                println!("  aliases: {}", aliases.join(", "));
            }
            for line in preset_lines(&row.presets) {
                println!("{line}");
            }
            for (key, r) in &resources.resources {
                println!(
                    "  {key}: {} ({}) [{}]",
                    r.file,
                    r.format.as_deref().unwrap_or("unlabeled"),
                    cache.status(r)
                );
            }
        }
        for provider in &listing.providers {
            if listing.unlisted.iter().any(|p| p == provider.name()) {
                println!();
                println!("{}:org/repo  {}", provider.name(), provider.description());
            }
        }
        Ok(())
    }
}
