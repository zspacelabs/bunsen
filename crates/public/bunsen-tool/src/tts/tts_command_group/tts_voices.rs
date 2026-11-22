//! `tts voices`.

use std::io::{
    BufRead,
    IsTerminal,
    Write,
};

use bunsen::{
    data::resources::{
        LoadedResourceSet,
        Resource,
        ResourceCache,
        ResourceSet,
    },
    errors::{
        BunsenError,
        BunsenResult,
        sys_at,
    },
    kits::speech::tts::{
        driver::{
            TtsVoiceInfo,
            TtsVoicePrint,
        },
        qwen3_tts::{
            interop::qwen_tts::Qwen3TtsPromptMirror,
            pretrained::{
                CONFIG,
                default_qwen3_tts_factory,
            },
        },
        zonos3::{
            driver::zonos3_voice_list,
            interop::zonos::load_zonos3_voices,
            pretrained::{
                VOICE_INFO,
                VOICES,
                default_zonos3_factory,
            },
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
    tts::{
        TtsFamily,
        choose_tts_model,
        voice_files,
    },
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

    /// The family the checkpoint is chosen among, as `tts run` takes it.
    #[arg(long, value_enum)]
    family: Option<TtsFamily>,

    /// Fetch the voices' files without asking, when they are not cached.
    #[arg(long)]
    yes: bool,

    /// A voice directory, whose voice files are listed too.
    ///
    /// Each with the family, mode and build it was made for.
    #[arg(long)]
    voice_dir: Option<std::path::PathBuf>,
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
        let (model, family) = choose_tts_model(&self.choose.model, &flags, self.family)?;
        let voices = match family {
            TtsFamily::Zonos3 => self.zonos3_voices(&model, &cache)?,
            TtsFamily::Qwen3Tts => self.qwen3_tts_voices(&model, &cache)?,
        };
        println!("# {model}: {} stock voices", voices.len());
        print!("{}", voice_table(&voices));
        if let Some(dir) = &self.voice_dir {
            let files = voice_files(dir)?;
            println!("# {}: {} voice files", dir.display(), files.len());
            let mut rows = vec![["name", "family", "mode", "build"].map(String::from)];
            for (name, path) in files {
                let voice = TtsVoicePrint::read(&path)?;
                rows.push([
                    name,
                    voice.identity.family,
                    voice.identity.mode,
                    voice.identity.encoder,
                ]);
            }
            print!("{}", table(&rows));
        }
        Ok(())
    }

    /// `set` loaded, its files fetched first if they are not cached and the
    /// user agrees ([`confirm_fetch`]).
    fn load(
        &self,
        cache: &ResourceCache,
        set: &ResourceSet,
    ) -> BunsenResult<LoadedResourceSet> {
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
        cache.load(set)
    }

    /// ZONOS3's voices, from its voices' files alone.
    fn zonos3_voices(
        &self,
        model: &str,
        cache: &ResourceCache,
    ) -> BunsenResult<Vec<TtsVoiceInfo>> {
        let plan = default_zonos3_factory()?
            .resolve(model, cache)?
            .plan(cache)?;
        let mut set = ResourceSet::new(plan.name.clone());
        set.insert(plan.try_get(VOICES)?.clone());
        if let Some(info) = plan.get(VOICE_INFO) {
            set.insert(info.clone());
        }
        let loaded = self.load(cache, &set)?;
        Ok(zonos3_voice_list(&load_zonos3_voices(&loaded)?))
    }

    /// Qwen3-TTS's speakers, from its `config.json` alone.
    fn qwen3_tts_voices(
        &self,
        model: &str,
        cache: &ResourceCache,
    ) -> BunsenResult<Vec<TtsVoiceInfo>> {
        let plan = default_qwen3_tts_factory()?
            .resolve(model, cache)?
            .plan(cache)?;
        let mut set = ResourceSet::new(plan.name.clone());
        set.insert(plan.try_get(CONFIG)?.clone());
        let loaded = self.load(cache, &set)?;
        let path = loaded.expect(CONFIG)?;
        let text = std::fs::read_to_string(path).map_err(sys_at("read", path))?;
        let json: Qwen3TtsPromptMirror = serde_json::from_str(&text)
            .map_err(|e| BunsenError::invalid_resource(format!("{}: {e}", path.display())))?;
        Ok(Qwen3TtsPromptMirror::prompt_config(&json)?.voices())
    }
}

/// The voices as a table: name, language (its default, a model key, or what
/// the release says), style and gender; `-` where nothing is said.
fn voice_table(voices: &[TtsVoiceInfo]) -> String {
    let mut rows = vec![[
        "name".to_string(),
        "language".to_string(),
        "style".to_string(),
        "gender".to_string(),
    ]];
    for voice in voices {
        let dash = || "-".to_string();
        rows.push([
            voice.name.clone(),
            voice
                .default_language
                .clone()
                .or_else(|| voice.description.clone())
                .unwrap_or_else(dash),
            voice.style.clone().unwrap_or_else(dash),
            voice.gender.clone().unwrap_or_else(dash),
        ]);
    }
    table(&rows)
}

/// `rows` as a table, each column as wide as its widest cell.
fn table(rows: &[[String; 4]]) -> String {
    let widths: Vec<usize> = (0..4)
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
    use bunsen::errors::BunsenErrorKind;

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
        let mut cillian = TtsVoiceInfo::stock("Cillian");
        cillian.description = Some("English".into());
        cillian.style = Some("Irish".into());
        cillian.gender = Some("male".into());
        let mut eric = TtsVoiceInfo::stock("eric");
        eric.default_language = Some("sichuan_dialect".into());
        let voices = [cillian, eric, TtsVoiceInfo::stock("Zed")];
        assert_eq!(
            voice_table(&voices),
            "name     language         style  gender\n\
             Cillian  English          Irish  male\n\
             eric     sichuan_dialect  -      -\n\
             Zed      -                -      -\n"
        );
    }
}
