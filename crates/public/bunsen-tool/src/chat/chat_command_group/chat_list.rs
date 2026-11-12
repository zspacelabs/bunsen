//! `chat list`.

use bunsen::errors::BunsenResult;
use bunsen_app::{
    cache::WeightsCacheArgs,
    logging::{
        LevelFilter,
        LogArgs,
        LogPolicy,
    },
};

use crate::{
    chat::ChatModelFamily,
    support::{
        print_model_row,
        print_providers,
    },
};

/// Lists checkpoints: every family's under `chat list`, one family's under
/// `chat family <family> list`.
#[derive(clap::Args, Debug)]
pub struct ChatListCommand {
    #[clap(flatten)]
    logging: LogArgs,

    #[clap(flatten)]
    cache: WeightsCacheArgs,

    /// Also print each file's first source, where it is fetched from (as
    /// `url: ...`).
    #[arg(long)]
    urls: bool,
}

impl ChatListCommand {
    /// Prints each of `families`' checkpoints, as [`print_model_row`] does.
    pub fn run(
        &self,
        families: &[ChatModelFamily],
    ) -> BunsenResult<()> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Info, LevelFilter::Info))?;
        let cache = self.cache.init()?;
        println!("cache: {}", cache.cache_dir().display());
        for &family in families {
            let default = family.default_model();
            println!();
            println!("# {} checkpoints (the default: {default})", family.name());
            print_providers(&family.providers()?);
            for (id, row) in family.rows()? {
                print_model_row(&cache, &id, &row, default, self.urls);
            }
            if let Some(note) = family.hf_note() {
                println!();
                println!("{note}");
            }
        }
        Ok(())
    }
}
