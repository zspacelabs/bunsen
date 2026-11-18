//! `tts voices`.

use std::{
    collections::BTreeMap,
    io::{
        BufRead,
        IsTerminal,
        Write,
    },
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
            VOICE_INFO,
            VOICES,
            default_zonos3_factory,
        },
        voices::{
            Zonos3Voice,
            with_zonos3_voice_info,
            zonos3_voice_info,
            zonos3_voices,
        },
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

    /// Fetch the voices' files without asking, when they are not cached.
    #[arg(long)]
    yes: bool,
}

impl TtsVoicesCommand {
    /// Prints each voice: its name, what the release says of it, and its
    /// reference length; `matches` are the command's own, for the stage
    /// flags' order.
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
        let mut set = ResourceSet::new(plan.name.clone());
        set.insert(plan.try_get(VOICES)?.clone());
        if let Some(info) = plan.get(VOICE_INFO) {
            set.insert(info.clone());
        }
        let mut missing = Vec::new();
        for resource in set.resources.values() {
            if cache.cached_path(resource)?.is_none() {
                missing.push(resource);
            }
        }
        if !missing.is_empty() {
            let stdin = std::io::stdin();
            let interactive = stdin.is_terminal();
            confirm_fetch(&missing, self.yes, interactive, &mut stdin.lock())?;
        }
        let loaded = cache.load(&set)?;
        let mut voices =
            zonos3_voices(SafetensorsCheckpoint::single(loaded.expect(VOICES)?).tensors()?)?;
        if let Some(info) = loaded.get(VOICE_INFO) {
            let path = &info.path;
            let json = std::fs::read_to_string(path).map_err(sys_at("read", path))?;
            with_zonos3_voice_info(&mut voices, zonos3_voice_info(&json)?)?;
        }
        println!("# {model}: {} stock voices", voices.len());
        print!("{}", voice_table(&voices));
        Ok(())
    }
}

/// The voices as a table: name, language, style, gender, and the length of
/// their reference codes in codec frames; `-` where the release says
/// nothing.
fn voice_table(voices: &BTreeMap<String, Zonos3Voice>) -> String {
    let mut rows = vec![[
        "name".to_string(),
        "language".to_string(),
        "style".to_string(),
        "gender".to_string(),
        "reference".to_string(),
    ]];
    for (name, voice) in voices {
        let info = voice.info.as_ref();
        let dash = || "-".to_string();
        rows.push([
            name.clone(),
            info.map_or_else(dash, |i| i.language.clone()),
            info.and_then(|i| i.style.clone()).unwrap_or_else(dash),
            info.map_or_else(dash, |i| i.gender.clone()),
            format!("{} frames", voice.codes.len()),
        ]);
    }
    let widths: Vec<usize> = (0..5)
        .map(|c| rows.iter().map(|r| r[c].chars().count()).max().unwrap_or(0))
        .collect();
    let mut out = String::new();
    for row in rows {
        let line: Vec<String> = row
            .iter()
            .zip(&widths)
            .map(|(cell, &w)| format!("{cell:<w$}"))
            .collect();
        out.push_str(line.join("  ").trim_end());
        out.push('\n');
    }
    out
}

/// Whether to fetch the `missing` files, which the cache does not hold: yes
/// with `yes`; else asked on `input`, when `interactive`, the default no.
///
/// # Errors
/// [`Policy`](bunsen::errors::BunsenErrorKind::Policy) for a no, or no
/// terminal to ask on; a read's failure.
fn confirm_fetch(
    missing: &[&Resource],
    yes: bool,
    interactive: bool,
    input: &mut impl BufRead,
) -> BunsenResult<()> {
    if yes {
        return Ok(());
    }
    let names: Vec<&str> = missing.iter().map(|r| r.file.as_str()).collect();
    let size: u64 = missing.iter().filter_map(|r| r.size).sum();
    let from = missing
        .first()
        .and_then(|r| r.urls().first().copied())
        .and_then(|url| url.split('/').nth(2))
        .map_or_else(String::new, |host| format!(" from {host}"));
    let what = format!(
        "{} ({}) {} not cached",
        names.join(" and "),
        human_bytes(size),
        if missing.len() == 1 { "is" } else { "are" }
    );
    if !interactive {
        return Err(BunsenError::policy(format!(
            "{what}; pass --yes to fetch {}{from}",
            if missing.len() == 1 { "it" } else { "them" }
        )));
    }
    eprint!(
        "{what}; fetch {}{from}? [y/N] ",
        if missing.len() == 1 { "it" } else { "them" }
    );
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
            "the voices were not fetched; nothing to list",
        )),
    }
}

#[cfg(test)]
mod tests {
    use bunsen::{
        errors::BunsenErrorKind,
        kits::speech::tts::zonos3::voices::Zonos3VoiceInfo,
    };

    use super::*;

    fn resources() -> Vec<Resource> {
        let set = default_zonos3_factory()
            .unwrap()
            .lookup(bunsen::kits::speech::tts::zonos3::pretrained::DEFAULT_ZONOS3_MODEL)
            .unwrap()
            .1
            .resource_set();
        [VOICES, VOICE_INFO]
            .iter()
            .map(|k| set.get(k).unwrap().clone())
            .collect()
    }

    /// `--yes` fetches; a yes on the terminal fetches; anything else, or no
    /// terminal, does not, and says how to.
    #[test]
    fn test_confirm_fetch() {
        let resources = resources();
        let missing: Vec<&Resource> = resources.iter().collect();
        let ask = |yes: bool, interactive: bool, typed: &str| {
            confirm_fetch(&missing, yes, interactive, &mut typed.as_bytes())
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
        let message = error.to_string();
        assert!(message.contains("--yes"), "{message}");
        assert!(
            message.contains("voices.safetensors and voices.json"),
            "{message}"
        );
        assert!(message.contains("from huggingface.co"), "{message}");
    }

    /// One row a voice, aligned; `-` where nothing is said of it.
    #[test]
    fn test_voice_table() {
        let voice = |info: Option<Zonos3VoiceInfo>| Zonos3Voice {
            codes: vec![vec![0; 10]; 3],
            embedding: vec![0.0; 4],
            info,
        };
        let voices = BTreeMap::from([
            (
                "Cillian".to_string(),
                voice(Some(Zonos3VoiceInfo {
                    language: "English".into(),
                    style: Some("Irish".into()),
                    gender: "male".into(),
                    file: "Cillian_en-Irish_M_1".into(),
                })),
            ),
            ("Zed".to_string(), voice(None)),
        ]);
        assert_eq!(
            voice_table(&voices),
            "name     language  style  gender  reference\n\
             Cillian  English   Irish  male    3 frames\n\
             Zed      -         -      -       3 frames\n"
        );
    }
}
