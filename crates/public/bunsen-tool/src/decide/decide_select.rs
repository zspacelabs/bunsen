//! How `decide` chooses a checkpoint, on the catalog pipeline.
use bunsen::{
    data::pretrained::build_catalog::{
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
    kits::decision::laya::pretrained::{
        DEFAULT_LAYA_MODEL,
        default_laya_factory,
    },
};
use bunsen_app::catalog::SelectionFlags;

use crate::support::{
    builtin_catalog,
    log_choice,
};

/// `decide`'s selection context: the default, applied to every choice but
/// a checkpoint named exactly (each release group's newest weighted first,
/// labeled `latest`), and `default`, [`DEFAULT_LAYA_MODEL`], which a command
/// without `--model` takes.
pub fn decide_selection() -> SelectionContext {
    let defaults: Vec<LabeledStage> =
        serde_saphyr::from_str("[{weigh: {by: [{release: latest}]}, as: latest}]")
            .expect("the decide defaults are stages");
    SelectionContext {
        defaults,
        named: vec![NamedSelection {
            name: "default".to_string(),
            description: format!("the default checkpoint, {DEFAULT_LAYA_MODEL}"),
            run: vec![Stage::From(NameList(vec![DEFAULT_LAYA_MODEL.to_string()])).into()],
        }],
    }
}

/// The Laya checkpoint `names` (each `--model`) and `flags` (the stage and
/// preset flags) choose among the listed ones: no name is `default`; one
/// `provider:ref` of a provider that lists nothing (`hf:org/repo`) is that
/// ref, and takes no stage flags.
///
/// # Errors
/// [`Policy`](bunsen::errors::BunsenErrorKind::Policy) for stage flags with
/// an `hf:` ref; as [`SelectionContext::choose`] and [`builtin_catalog`].
pub fn choose_decide_model(
    names: &[String],
    flags: &SelectionFlags,
) -> BunsenResult<String> {
    let mut names = names.to_vec();
    if let [only] = names.as_mut_slice()
        && let Some((provider, rest)) = only.split_once(':')
    {
        if provider != "well-known" {
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
    let universe = default_laya_factory()?
        .list()
        .into_iter()
        .map(|(_, row)| {
            CatalogName::parse(&row.name).map_err(|e| {
                BunsenError::illegal(format!("`{}`: {}", row.name, e.problem_message()))
            })
        })
        .collect::<BunsenResult<Vec<_>>>()?;
    let stack = builtin_catalog()?;
    let env = SelectEnv::new(&stack);
    let request = SelectRequest {
        names,
        stages: flags.stages.clone(),
        ..SelectRequest::default()
    };
    let (chosen, run) = decide_selection()
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

    /// A stand-in for `decide run`'s selection flags.
    #[derive(clap::Parser, Debug)]
    struct Flags {
        #[arg(long)]
        model: Vec<String>,
        #[command(flatten)]
        stages: SelectionStageArgs,
    }

    fn choose_with(argv: &[&str]) -> BunsenResult<String> {
        let matches = Flags::command()
            .try_get_matches_from([&["flags"], argv].concat())
            .unwrap();
        let names: Vec<String> = matches
            .get_many::<String>("model")
            .map(|v| v.cloned().collect())
            .unwrap_or_default();
        let flags = SelectionFlags::read(&builtin_catalog().unwrap(), &matches)?;
        choose_decide_model(&names, &flags)
    }

    fn choose(argv: &[&str]) -> String {
        choose_with(argv).unwrap()
    }

    /// The default is Laya's default; every listed checkpoint, by its name,
    /// is itself; the stage flags choose by tags.
    #[test]
    fn test_choose() {
        assert_eq!(choose(&[]), DEFAULT_LAYA_MODEL);
        for (_, row) in default_laya_factory().unwrap().list() {
            assert_eq!(choose(&["--model", &row.name]), row.name);
        }
        assert_eq!(
            choose(&["--model", "laya", "--asc", "parameters"]),
            "laya/laya/laya-multilingual",
            "the smaller model first"
        );
        assert_eq!(choose(&["--model", "hf:org/repo"]), "hf:org/repo");
        assert!(choose_with(&["--model", "laya", "--has", "language", "zz-Zzzz"]).is_err());
    }
}
