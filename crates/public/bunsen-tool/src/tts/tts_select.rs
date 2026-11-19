//! How `tts` chooses a checkpoint, on the catalog pipeline.
use bunsen::{
    data::catalog::{
        CatalogErrorExt,
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
    kits::speech::tts::zonos3::pretrained::DEFAULT_ZONOS3_MODEL,
};
use bunsen_app::catalog::SelectionFlags;

use crate::{
    support::{
        builtin_catalog,
        log_choice,
    },
    tts::{
        TtsFamily,
        tts_builds,
    },
};

/// `tts`'s selection context: the default, applied to every choice but a
/// checkpoint named exactly (each release group's newest weighted first,
/// labeled `latest`), and `default`, which a command without `--model`
/// takes: `family`'s default checkpoint, or, without one,
/// [`DEFAULT_ZONOS3_MODEL`].
pub fn tts_selection(family: Option<TtsFamily>) -> SelectionContext {
    let default = family.map_or(DEFAULT_ZONOS3_MODEL, TtsFamily::default_model);
    let defaults: Vec<LabeledStage> =
        serde_saphyr::from_str("[{weigh: {by: [{release: latest}]}, as: latest}]")
            .expect("the tts defaults are stages");
    SelectionContext {
        defaults,
        named: vec![NamedSelection {
            name: "default".to_string(),
            description: format!("the default checkpoint, {default}"),
            run: vec![Stage::From(NameList(vec![default.to_string()])).into()],
        }],
    }
}

/// The TTS checkpoint `names` (each `--model`) and `flags` (the stage and
/// preset flags) choose among every family's listed builds, or `family`'s
/// alone, and its family: no name is `default`; one `provider:ref` of a
/// provider that lists nothing (`hf:org/repo`) is that ref, of `family`, and
/// takes no stage flags.
///
/// # Errors
/// [`Policy`](bunsen::errors::BunsenErrorKind::Policy) for stage flags with
/// an `hf:` ref, or a ref without a family; as [`SelectionContext::choose`]
/// and [`builtin_catalog`].
pub fn choose_tts_model(
    names: &[String],
    flags: &SelectionFlags,
    family: Option<TtsFamily>,
) -> BunsenResult<(String, TtsFamily)> {
    let mut names = names.to_vec();
    if let [only] = names.as_mut_slice()
        && let Some((provider, rest)) = only.split_once(':')
    {
        if provider != "catalog" {
            if !flags.stages.is_empty() {
                return Err(BunsenError::policy(format!(
                    "`{only}` is a ref, not a choice: it takes no selection flags"
                )));
            }
            let family = family.ok_or_else(|| {
                BunsenError::policy(format!(
                    "`{only}` names no family: pass --family (zonos3 or qwen3-tts)"
                ))
            })?;
            return Ok((only.clone(), family));
        }
        *only = rest.to_string();
    }
    if names.is_empty() {
        names.push("default".to_string());
    }
    let builds: Vec<_> = tts_builds()?
        .into_iter()
        .filter(|(_, f)| family.is_none_or(|family| family == *f))
        .collect();
    let universe: Vec<_> = builds.iter().map(|(name, _)| name.clone()).collect();
    let stack = builtin_catalog()?;
    let env = SelectEnv::new(&stack);
    let request = SelectRequest {
        names,
        stages: flags.stages.clone(),
        ..SelectRequest::default()
    };
    let (chosen, run) = tts_selection(family)
        .choose(&env, Some(&universe), flags.prefix(), &request)
        .map_err(|e| BunsenError::policy(e.problem_message()))?;
    let chosen_family = builds
        .iter()
        .find(|(name, _)| *name == chosen.build.name)
        .map(|(_, f)| *f)
        .expect("the choice is one of the universe's builds");
    let name = chosen.build.name.to_string();
    log_choice(&name, &run);
    Ok((name, chosen_family))
}

#[cfg(test)]
mod tests {
    use bunsen_app::catalog::SelectionStageArgs;
    use clap::CommandFactory;

    use super::*;

    /// A stand-in for `tts run`'s selection flags.
    #[derive(clap::Parser, Debug)]
    struct Flags {
        #[arg(long)]
        model: Vec<String>,
        #[command(flatten)]
        stages: SelectionStageArgs,
    }

    fn choose(argv: &[&str]) -> (String, TtsFamily) {
        choose_in(argv, None)
    }

    fn choose_in(
        argv: &[&str],
        family: Option<TtsFamily>,
    ) -> (String, TtsFamily) {
        let matches = Flags::command()
            .try_get_matches_from([&["flags"], argv].concat())
            .unwrap();
        let names: Vec<String> = matches
            .get_many::<String>("model")
            .map(|v| v.cloned().collect())
            .unwrap_or_default();
        let flags = SelectionFlags::read(&builtin_catalog().unwrap(), &matches).unwrap();
        choose_tts_model(&names, &flags, family).unwrap()
    }

    /// The default is ZONOS3's, or the family's; every listed checkpoint of
    /// every family, by its name, is itself, of its family; a ref takes the
    /// family it is given, and needs one.
    #[test]
    fn test_choose() {
        assert_eq!(
            choose(&[]),
            (DEFAULT_ZONOS3_MODEL.to_string(), TtsFamily::Zonos3)
        );
        for family in TtsFamily::ALL {
            assert_eq!(
                choose_in(&[], Some(family)),
                (family.default_model().to_string(), family)
            );
            for (_, row) in family.rows().unwrap() {
                let name = row.name.to_string();
                assert_eq!(choose(&["--model", &name]), (name.clone(), family));
            }
        }
        assert_eq!(choose(&["--model", "zonos3-mini"]).0, DEFAULT_ZONOS3_MODEL);
        assert_eq!(
            choose(&["--model", "qwen3-tts"]).1,
            TtsFamily::Qwen3Tts,
            "a family's name chooses among its builds"
        );
        assert_eq!(
            choose_in(&["--model", "hf:org/repo"], Some(TtsFamily::Qwen3Tts)),
            ("hf:org/repo".to_string(), TtsFamily::Qwen3Tts)
        );
        let flags = SelectionFlags::default();
        let error = choose_tts_model(&["hf:org/repo".to_string()], &flags, None).unwrap_err();
        assert_eq!(error.kind(), bunsen::errors::BunsenErrorKind::Policy);
    }
}
