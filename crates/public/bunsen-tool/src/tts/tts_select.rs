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
    kits::speech::tts::zonos3::pretrained::{
        DEFAULT_ZONOS3_MODEL,
        default_zonos3_factory,
    },
};
use bunsen_app::catalog::SelectionFlags;

use crate::support::{
    builtin_catalog,
    log_choice,
};

/// `tts`'s selection context: the default, applied to every choice but a
/// checkpoint named exactly (each release group's newest weighted first,
/// labeled `latest`), and `default`, [`DEFAULT_ZONOS3_MODEL`], which a
/// command without `--model` takes.
pub fn tts_selection() -> SelectionContext {
    let defaults: Vec<LabeledStage> =
        serde_saphyr::from_str("[{weigh: {by: [{release: latest}]}, as: latest}]")
            .expect("the tts defaults are stages");
    SelectionContext {
        defaults,
        named: vec![NamedSelection {
            name: "default".to_string(),
            description: format!("the default checkpoint, {DEFAULT_ZONOS3_MODEL}"),
            run: vec![Stage::From(NameList(vec![DEFAULT_ZONOS3_MODEL.to_string()])).into()],
        }],
    }
}

/// The ZONOS3 checkpoint `names` (each `--model`) and `flags` (the stage and
/// preset flags) choose among the listed ones: no name is `default`; one
/// `provider:ref` of a provider that lists nothing (`hf:org/repo`) is that
/// ref, and takes no stage flags.
///
/// # Errors
/// [`Policy`](bunsen::errors::BunsenErrorKind::Policy) for stage flags with
/// an `hf:` ref; as [`SelectionContext::choose`] and [`builtin_catalog`].
pub fn choose_tts_model(
    names: &[String],
    flags: &SelectionFlags,
) -> BunsenResult<String> {
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
            return Ok(only.clone());
        }
        *only = rest.to_string();
    }
    if names.is_empty() {
        names.push("default".to_string());
    }
    let universe: Vec<_> = default_zonos3_factory()?
        .list()
        .into_iter()
        .map(|(_, row)| row.name)
        .collect();
    let stack = builtin_catalog()?;
    let env = SelectEnv::new(&stack);
    let request = SelectRequest {
        names,
        stages: flags.stages.clone(),
        ..SelectRequest::default()
    };
    let (chosen, run) = tts_selection()
        .choose(&env, Some(&universe), flags.prefix(), &request)
        .map_err(|e| BunsenError::policy(e.problem_message()))?;
    let name = chosen.build.name.to_string();
    log_choice(&name, &run);
    Ok(name)
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

    fn choose(argv: &[&str]) -> String {
        let matches = Flags::command()
            .try_get_matches_from([&["flags"], argv].concat())
            .unwrap();
        let names: Vec<String> = matches
            .get_many::<String>("model")
            .map(|v| v.cloned().collect())
            .unwrap_or_default();
        let flags = SelectionFlags::read(&builtin_catalog().unwrap(), &matches).unwrap();
        choose_tts_model(&names, &flags).unwrap()
    }

    /// The default is ZONOS3's default; every listed checkpoint, by its
    /// name, is itself.
    #[test]
    fn test_choose() {
        assert_eq!(choose(&[]), DEFAULT_ZONOS3_MODEL);
        for (_, row) in default_zonos3_factory().unwrap().list() {
            assert_eq!(
                choose(&["--model", &row.name.to_string()]),
                row.name.to_string()
            );
        }
        assert_eq!(choose(&["--model", "zonos3-mini"]), DEFAULT_ZONOS3_MODEL);
        assert_eq!(choose(&["--model", "hf:org/repo"]), "hf:org/repo");
    }
}
