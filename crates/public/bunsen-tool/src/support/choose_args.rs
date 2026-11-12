//! The model-choosing flags: `--model`, the stage flags and `--patch`.
use std::rc::Rc;

use bunsen::{
    data::catalog::{
        CatalogErrorExt,
        PretrainedCatalogStack,
        builtin_stack,
        select::state::SelectRun,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
};
use bunsen_app::catalog::{
    SelectionFlags,
    SelectionStageArgs,
};
use clap::ArgMatches;
use serde_json::{
    Map,
    Value,
};

/// The built-in catalog's stack, imported once per thread: what the
/// commands choose among. (A stack is not `Sync`, so it is not shared across
/// threads.)
///
/// # Errors
/// [`Policy`](bunsen::errors::BunsenErrorKind::Policy) if the built-in
/// catalog doesn't load, which would be a bug in the built catalog.
pub fn builtin_catalog() -> BunsenResult<Rc<PretrainedCatalogStack>> {
    thread_local! {
        static STACK: Result<Rc<PretrainedCatalogStack>, String> =
            builtin_stack().map(Rc::new).map_err(|e| e.problem_message());
    }
    STACK.with(|s| s.clone().map_err(BunsenError::policy))
}

/// Logs, at `INFO`, the build `run` chose and why: each stage that removed
/// builds, with the flag or default it came from.
pub fn log_choice(
    chosen: &str,
    run: &SelectRun,
) {
    let why: Vec<String> = run
        .trace
        .stages
        .iter()
        .filter(|s| s.dropped > 0)
        .map(|s| format!("`{}` ({}) dropped {}", s.op, s.origin, s.dropped))
        .collect();
    if why.is_empty() {
        log::info!("chose {chosen}");
    } else {
        log::info!("chose {chosen}: {}", why.join("; "));
    }
}

/// How a command chooses its model and patches its config, for
/// `#[command(flatten)]`.
///
/// The stage flags are read back in command-line order from the command's
/// [`ArgMatches`] ([`flags`](Self::flags)), so a command taking these needs
/// its own matches.
#[derive(clap::Args, Debug, Clone, Default)]
pub struct ChooseArgs {
    /// The model: a build's name, a group of builds, or `hf:org/repo`.
    ///
    /// A build by its full name, its bare name or an alias
    /// (`qwen/qwen3/qwen3-0.6b`, `qwen3-0.6b`); a group of them (`qwen`,
    /// `gemma/gemma`), which the defaults and the stage flags choose within;
    /// a named selection (`default`); or `hf:org/repo`, a Hugging Face repo
    /// no build describes. Repeatable: several names are one `from`.
    #[arg(long = "model", value_name = "NAME")]
    pub model: Vec<String>,

    #[command(flatten)]
    pub stages: SelectionStageArgs,

    /// A merge patch over the chosen model's config (`{beam_size: 1}`).
    ///
    /// A YAML mapping; repeatable, merged in order.
    #[arg(long = "patch", value_name = "{..}")]
    pub patch: Vec<String>,
}

impl ChooseArgs {
    /// The stage and preset flags, in command-line order, read from
    /// `matches`: the matches of the command these flags are flattened into.
    ///
    /// # Errors
    /// As [`SelectionFlags::read`] and [`builtin_catalog`].
    pub fn flags(
        &self,
        matches: &ArgMatches,
    ) -> BunsenResult<SelectionFlags> {
        SelectionFlags::read(&*builtin_catalog()?, matches)
    }

    /// The `--patch`es merged in order; `None` without any.
    ///
    /// # Errors
    /// [`Policy`](bunsen::errors::BunsenErrorKind::Policy) for a patch that is
    /// not a YAML mapping.
    pub fn patch(&self) -> BunsenResult<Option<Value>> {
        if self.patch.is_empty() {
            return Ok(None);
        }
        let mut merged = Value::Object(Map::new());
        for text in &self.patch {
            let value: Value = serde_saphyr::from_str(text)
                .map_err(|e| BunsenError::policy(format!("`--patch {text}` is not YAML: {e}")))?;
            if !value.is_object() {
                return Err(BunsenError::policy(format!(
                    "`--patch {text}` is not a mapping"
                )));
            }
            bunsen::support::merge_patch::apply(&mut merged, &value);
        }
        Ok(Some(merged))
    }
}

#[cfg(test)]
mod tests {
    use clap::{
        CommandFactory,
        FromArgMatches,
    };
    use serde_json::json;

    use super::*;

    #[derive(clap::Parser, Debug)]
    struct Args {
        #[command(flatten)]
        choose: ChooseArgs,
    }

    #[test]
    fn test_flags() {
        let matches = Args::command()
            .try_get_matches_from([
                "x",
                "--model",
                "qwen",
                "--limit",
                "2",
                "--patch",
                "{beam_size: 1, vad: {keep: true}}",
                "--patch",
                "{vad: {keep: null}}",
                "--off",
                "thinking",
            ])
            .unwrap();
        let args = Args::from_arg_matches(&matches).unwrap();
        assert_eq!(args.choose.model, ["qwen"]);
        assert_eq!(
            args.choose.patch().unwrap(),
            Some(json!({"beam_size": 1, "vad": {}}))
        );
        let flags = args.choose.flags(&matches).unwrap();
        assert_eq!(flags.stages.len(), 1);
        assert_eq!(flags.presets.presets, ["no-thinking"]);
        assert_eq!(ChooseArgs::default().patch().unwrap(), None);
        let bad = ChooseArgs {
            patch: vec!["[1]".to_string()],
            ..ChooseArgs::default()
        };
        assert!(bad.patch().is_err());
    }
}
