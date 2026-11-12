//! How the chat commands choose a checkpoint, on the catalog pipeline.
use bunsen::{
    data::catalog::{
        CatalogErrorExt,
        CatalogName,
        select::{
            choose::{
                NamedSelection,
                SelectRequest,
                SelectionContext,
            },
            query::{
                LabeledStage,
                NameList,
                Stage,
            },
            state::SelectEnv,
        },
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
};
use bunsen_app::catalog::SelectionFlags;

use crate::{
    chat::ChatModelFamily,
    support::{
        builtin_catalog,
        log_choice,
    },
};

/// The name every chat context has for its default checkpoint, which a
/// command without `--model` stands on.
pub const DEFAULT_SELECTION: &str = "default";

impl ChatModelFamily {
    /// The family whose command name is `name`.
    pub fn from_name(name: &str) -> Option<ChatModelFamily> {
        ChatModelFamily::ALL.into_iter().find(|f| f.name() == name)
    }

    /// The family's listed checkpoints, by their full names in the
    /// built-in catalog: what the chat commands choose among.
    ///
    /// # Errors
    /// As [`rows`](Self::rows), and
    /// [`Illegal`](bunsen::errors::BunsenErrorKind::Illegal) for a row whose
    /// name is not a catalog name, which the tests pin none is.
    pub fn builds(self) -> BunsenResult<Vec<CatalogName>> {
        Ok(self.rows()?.into_iter().map(|(_, row)| row.name).collect())
    }
}

/// The chat commands' selection context, within `family` or across every
/// family:
///
/// - the defaults, applied to every choice but a checkpoint named exactly: a
///   tuned model (`keep: instruct`, labeled `instruct`), and each release
///   group's newest weighted first (`release: latest`, labeled `latest`);
/// - [`DEFAULT_SELECTION`]: the family's default checkpoint
///   ([`ChatModelFamily::default_model`]), or across families Qwen3's.
pub fn chat_selection(family: Option<ChatModelFamily>) -> SelectionContext {
    let default = family.unwrap_or(ChatModelFamily::Qwen3).default_model();
    let defaults: Vec<LabeledStage> = serde_saphyr::from_str(
        "[{keep: instruct, as: instruct}, {weigh: {by: [{release: latest}]}, as: latest}]",
    )
    .expect("the chat defaults are stages");
    SelectionContext {
        defaults,
        named: vec![NamedSelection {
            name: DEFAULT_SELECTION.to_string(),
            description: format!("the default checkpoint, {default}"),
            run: vec![Stage::From(NameList(vec![default.to_string()])).into()],
        }],
    }
}

/// The checkpoint `names` (each `--model`) and `flags` (the stage and
/// preset flags) choose, within `family` or across every family, and its
/// family:
///
/// - no name is [`DEFAULT_SELECTION`];
/// - one `provider:ref` name of a provider that lists nothing (`hf:org/repo`)
///   is that ref, of `family`, or across families Qwen3's, as no build
///   describes it; it takes no stage flags. `catalog:` is dropped, its rows
///   being the builds chosen among;
/// - anything else is chosen among the builds: the presets `flags` select keep
///   only the builds that offer them, whatever their order on the command line.
///
/// # Errors
/// [`Policy`](bunsen::errors::BunsenErrorKind::Policy) for stage flags with
/// an `hf:` ref; as [`SelectionContext::choose`], whose errors name the
/// stages that left nothing; as [`ChatModelFamily::builds`] and
/// [`builtin_catalog`].
pub fn choose_checkpoint(
    family: Option<ChatModelFamily>,
    names: &[String],
    flags: &SelectionFlags,
) -> BunsenResult<(ChatModelFamily, String)> {
    let mut names: Vec<String> = names.to_vec();
    if let [only] = names.as_mut_slice()
        && let Some((provider, rest)) = only.split_once(':')
    {
        if provider != "catalog" {
            if !flags.stages.is_empty() {
                return Err(BunsenError::policy(format!(
                    "`{only}` is a ref, not a choice: it takes no selection flags"
                )));
            }
            return Ok((family.unwrap_or(ChatModelFamily::Qwen3), only.clone()));
        }
        *only = rest.to_string();
    }
    if names.is_empty() {
        names.push(DEFAULT_SELECTION.to_string());
    }
    let families: Vec<ChatModelFamily> = match family {
        Some(family) => vec![family],
        None => ChatModelFamily::ALL.to_vec(),
    };
    let mut universe = Vec::new();
    let mut of_family = Vec::new();
    for family in families {
        for name in family.builds()? {
            of_family.push((name.clone(), family));
            universe.push(name);
        }
    }
    let stack = builtin_catalog()?;
    let env = SelectEnv::new(&stack);
    let request = SelectRequest {
        names,
        stages: flags.stages.clone(),
        ..SelectRequest::default()
    };
    let (chosen, run) = chat_selection(family)
        .choose(&env, Some(&universe), flags.prefix(), &request)
        .map_err(|e| BunsenError::policy(e.problem_message()))?;
    let name = &chosen.build.name;
    log_choice(&name.to_string(), &run);
    let family = of_family
        .iter()
        .find(|(n, _)| n == name)
        .map(|(_, f)| *f)
        .expect("the choice is one of the universe's builds");
    Ok((family, name.to_string()))
}

#[cfg(test)]
mod tests {
    use bunsen_app::catalog::SelectionStageArgs;
    use clap::CommandFactory;

    use super::*;

    /// A stand-in for a chat command's selection flags.
    #[derive(clap::Parser, Debug)]
    struct Flags {
        #[arg(long)]
        model: Vec<String>,
        #[command(flatten)]
        stages: SelectionStageArgs,
    }

    fn choose_with(
        family: Option<ChatModelFamily>,
        argv: &[&str],
    ) -> BunsenResult<(ChatModelFamily, String)> {
        let matches = Flags::command()
            .try_get_matches_from([&["flags"], argv].concat())
            .unwrap();
        let names: Vec<String> = matches
            .get_many::<String>("model")
            .map(|v| v.cloned().collect())
            .unwrap_or_default();
        let flags = SelectionFlags::read(&builtin_catalog().unwrap(), &matches)?;
        choose_checkpoint(family, &names, &flags)
    }

    fn choose(
        family: Option<ChatModelFamily>,
        argv: &[&str],
    ) -> (ChatModelFamily, String) {
        choose_with(family, argv).unwrap()
    }

    /// Every family's default is its own default checkpoint; across
    /// families, Qwen3's.
    #[test]
    fn test_defaults() {
        for family in ChatModelFamily::ALL {
            assert_eq!(
                choose(Some(family), &[]),
                (family, family.default_model().to_string())
            );
        }
        assert_eq!(
            choose(None, &[]),
            (
                ChatModelFamily::Qwen3,
                ChatModelFamily::Qwen3.default_model().to_string()
            )
        );
    }

    /// Every listed checkpoint, named by its full name or its bare name, is
    /// itself, of its family, across families and within its own.
    #[test]
    fn test_every_row_is_its_own() {
        for family in ChatModelFamily::ALL {
            for (_, row) in family.rows().unwrap() {
                let full = row.name.to_string();
                for name in [full.as_str(), row.name.leaf()] {
                    assert_eq!(
                        choose(None, &["--model", name]),
                        (family, full.clone()),
                        "{name}"
                    );
                    assert_eq!(
                        choose(Some(family), &["--model", name]),
                        (family, full.clone()),
                        "{name}"
                    );
                }
                for alias in &row.aliases {
                    let alias = alias.leaf();
                    assert_eq!(choose(None, &["--model", alias]).0, family, "{alias}");
                }
            }
        }
        assert_eq!(
            choose(None, &["--model", "catalog:gemma/gemma/gemma-2-2b-it"]).0,
            ChatModelFamily::Gemma
        );
    }

    /// An `hf:` repo is its ref, of the family the command is in, else
    /// Qwen3's, and takes no stage flags.
    #[test]
    fn test_hub_repos_pass_through() {
        assert_eq!(
            choose(None, &["--model", "hf:Qwen/Qwen3-0.6B"]),
            (ChatModelFamily::Qwen3, "hf:Qwen/Qwen3-0.6B".to_string())
        );
        assert_eq!(
            choose(
                Some(ChatModelFamily::Gemma),
                &["--model", "hf:unsloth/gemma-3-1b-it"]
            ),
            (
                ChatModelFamily::Gemma,
                "hf:unsloth/gemma-3-1b-it".to_string()
            )
        );
        assert!(choose_with(None, &["--model", "hf:Qwen/Qwen3-0.6B", "--limit", "1"]).is_err());
    }

    /// A group is a set the defaults choose within; stage flags narrow it
    /// further, in order.
    #[test]
    fn test_selections() {
        let (family, name) = choose(None, &["--model", "gemma"]);
        assert_eq!(family, ChatModelFamily::Gemma);
        assert!(name.ends_with("-it"), "{name}: a tuned model");
        let (family, _) = choose(None, &["--model", "nanochat-students"]);
        assert_eq!(family, ChatModelFamily::NanoChat);
        let (_, name) = choose(None, &["--model", "qwen3", "--eq", "thinking", "always"]);
        assert_eq!(name, "qwen/qwen3/qwen3-4b-thinking-2507");
        let (_, name) = choose(
            None,
            &[
                "--model",
                "qwen/qwen3",
                "--unset",
                "instruct",
                "--keep",
                "{instruct: false}",
                "--asc",
                "parameters",
            ],
        );
        assert_eq!(name, "qwen/qwen3/qwen3-0.6b-base");
        assert!(choose_with(None, &["--model", "no-such-model"]).is_err());
    }

    /// The presets keep only the builds that offer them, wherever the flag
    /// is written: `--on thinking` across families leaves the families
    /// that think, and within one that doesn't, nothing.
    #[test]
    fn test_presets_narrow() {
        for argv in [
            &["--on", "thinking", "--model", "qwen"][..],
            &["--model", "qwen", "--on", "thinking"][..],
        ] {
            let (family, _) = choose(None, argv);
            assert!(
                matches!(family, ChatModelFamily::Qwen3 | ChatModelFamily::Qwen3_5),
                "{argv:?}"
            );
        }
        let (family, name) = choose(Some(ChatModelFamily::Qwen3_5), &["--off", "thinking"]);
        assert_eq!(family, ChatModelFamily::Qwen3_5);
        assert_eq!(name, ChatModelFamily::Qwen3_5.default_model());
        assert!(choose_with(Some(ChatModelFamily::NanoChat), &["--on", "thinking"]).is_err());
    }
}
