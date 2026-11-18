//! `tts voices`.

use std::io::{
    BufRead,
    IsTerminal,
    Write,
};

use bunsen::{
    data::{
        formats::SafetensorsCheckpoint,
        resources::{
            Resource,
            ResourceSet,
        },
    },
    errors::{
        BunsenError,
        BunsenResult,
        sys_at,
    },
    kits::speech::tts::zonos3::{
        pretrained::{
            VOICES,
            default_zonos3_factory,
        },
        voices::zonos3_voices,
    },
};
use bunsen_app::{
    cache::{
        WeightsCacheArgs,
        human_bytes,
    },
    logging::{
        LogArgs,
        LogPolicy,
    },
};
use clap::ArgMatches;
use log::LevelFilter;

use crate::{
    support::ChooseArgs,
    tts::choose_tts_model,
};

/// Lists a checkpoint's stock voices.
#[derive(clap::Args, Debug)]
pub struct TtsVoicesCommand {
    #[clap(flatten)]
    logging: LogArgs,

    #[clap(flatten)]
    cache: WeightsCacheArgs,

    /// The checkpoint, chosen as `tts run` chooses it.
    #[clap(flatten)]
    choose: ChooseArgs,

    /// Fetch the voices file without asking, when it is not cached.
    #[arg(long)]
    yes: bool,
}

impl TtsVoicesCommand {
    /// Prints each voice, by name, with its reference length; `matches` are
    /// the command's own, for the stage flags' order.
    pub fn run(
        &self,
        matches: &ArgMatches,
    ) -> BunsenResult<()> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Warn, LevelFilter::Warn))?;
        let cache = self.cache.init()?;
        let flags = self.choose.flags(matches)?;
        let model = choose_tts_model(&self.choose.model, &flags)?;
        let plan = default_zonos3_factory()?
            .resolve(&model, &cache)?
            .plan(&cache)?;
        let voices = plan.try_get(VOICES)?;
        if cache.cached_path(voices)?.is_none() {
            let stdin = std::io::stdin();
            let interactive = stdin.is_terminal();
            confirm_fetch(voices, self.yes, interactive, &mut stdin.lock())?;
        }
        let mut set = ResourceSet::new(plan.name.clone());
        set.insert(voices.clone());
        let loaded = cache.load(&set)?;
        let voices =
            zonos3_voices(SafetensorsCheckpoint::single(loaded.expect(VOICES)?).tensors()?)?;
        println!(
            "# {model}: {} voices, each its reference frames",
            voices.len()
        );
        for (name, voice) in &voices {
            println!("{name}  {} frames", voice.codes.len());
        }
        Ok(())
    }
}

/// Whether to fetch `voices`, which the cache does not hold: yes with
/// `yes`; else asked on `input`, when `interactive`, the default no.
///
/// # Errors
/// [`Policy`](bunsen::errors::BunsenErrorKind::Policy) for a no, or no
/// terminal to ask on; a read's failure.
fn confirm_fetch(
    voices: &Resource,
    yes: bool,
    interactive: bool,
    input: &mut impl BufRead,
) -> BunsenResult<()> {
    if yes {
        return Ok(());
    }
    let size = voices
        .size
        .map_or_else(String::new, |n| format!(" ({})", human_bytes(n)));
    let from = voices
        .urls()
        .first()
        .map_or_else(String::new, |url| format!(" from {url}"));
    let what = format!("the voices file{size} is not cached");
    if !interactive {
        return Err(BunsenError::policy(format!(
            "{what}; pass --yes to fetch it{from}"
        )));
    }
    eprint!("{what}; fetch it{from}? [y/N] ");
    std::io::stderr()
        .flush()
        .map_err(sys_at("write to", "stderr"))?;
    let mut answer = String::new();
    input
        .read_line(&mut answer)
        .map_err(sys_at("read", "stdin"))?;
    match answer.trim().to_lowercase().as_str() {
        "y" | "yes" => Ok(()),
        _ => Err(BunsenError::policy(
            "the voices file was not fetched; nothing to list",
        )),
    }
}

#[cfg(test)]
mod tests {
    use bunsen::errors::BunsenErrorKind;

    use super::*;

    fn voices() -> Resource {
        default_zonos3_factory()
            .unwrap()
            .lookup(bunsen::kits::speech::tts::zonos3::pretrained::DEFAULT_ZONOS3_MODEL)
            .unwrap()
            .1
            .resource_set()
            .get(VOICES)
            .unwrap()
            .clone()
    }

    /// `--yes` fetches; a yes on the terminal fetches; anything else, or no
    /// terminal, does not, and says how to.
    #[test]
    fn test_confirm_fetch() {
        let voices = voices();
        let ask = |yes: bool, interactive: bool, typed: &str| {
            confirm_fetch(&voices, yes, interactive, &mut typed.as_bytes())
        };
        assert!(ask(true, false, "").is_ok());
        assert!(ask(false, true, "y\n").is_ok());
        assert!(ask(false, true, "Yes\n").is_ok());
        for typed in ["\n", "n\n", ""] {
            assert_eq!(
                ask(false, true, typed).unwrap_err().kind(),
                BunsenErrorKind::Policy,
                "{typed:?}"
            );
        }
        let error = ask(false, false, "y\n").unwrap_err();
        assert_eq!(error.kind(), BunsenErrorKind::Policy);
        assert!(error.to_string().contains("--yes"), "{error}");
    }
}
