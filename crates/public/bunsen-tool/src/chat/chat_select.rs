//! How the chat commands choose a checkpoint.
use bunsen::{
    data::pretrained::selection::SelectionCandidate,
    errors::BunsenResult,
};
use serde_json::json;

use crate::{
    chat::ChatModelFamily,
    select::{
        NamedSelection,
        SelectionContext,
    },
};

/// The tag each chat candidate carries, set when it is built: the name of its
/// family ([`ChatModelFamily::name`]).
pub const CHAT_FAMILY_TAG: &str = "chat.family";

/// The name every chat context has for its default checkpoint, which a
/// `--select` without a name stands on.
pub const DEFAULT_SELECTION: &str = "default";

impl ChatModelFamily {
    /// The family whose command name is `name`.
    pub fn from_name(name: &str) -> Option<ChatModelFamily> {
        ChatModelFamily::ALL.into_iter().find(|f| f.name() == name)
    }

    /// The family's listed checkpoints, as selection candidates tagged with
    /// the family ([`CHAT_FAMILY_TAG`]).
    ///
    /// # Errors
    /// As [`rows`](Self::rows).
    pub fn candidates(self) -> BunsenResult<Vec<SelectionCandidate>> {
        Ok(self
            .rows()?
            .into_iter()
            .map(|(_, row)| SelectionCandidate::new(row).with_tag(CHAT_FAMILY_TAG, self.name()))
            .collect())
    }
}

/// The chat commands' selection context, within `family` or across every
/// family:
///
/// - the defaults, applied to every choice but a checkpoint named exactly: a
///   tuned model (`instruct: true`), each release group's newest (`latest`),
///   the heaviest first (`order: [-weight]`);
/// - [`DEFAULT_SELECTION`]: the family's default checkpoint
///   ([`ChatModelFamily::default_model`]), or across families Qwen3's.
pub fn chat_selection(family: Option<ChatModelFamily>) -> SelectionContext {
    let default = family.unwrap_or(ChatModelFamily::Qwen3).default_model();
    SelectionContext {
        defaults: json!({
            "require": {"instruct": true},
            "latest": true,
            "order": ["-weight"],
        }),
        named: vec![NamedSelection {
            name: DEFAULT_SELECTION.to_string(),
            description: format!("the default checkpoint, {default}"),
            selection: json!({"name": default}),
        }],
    }
}

/// The checkpoint `selects` (each `--select`) and `presets` (each
/// `--preset`) choose, within `family` or across every family, and its
/// family:
///
/// - no `--select` is [`DEFAULT_SELECTION`];
/// - one `provider:ref` name of a provider that lists nothing (`hf:org/repo`)
///   is that ref, of `family`, or across families Qwen3's, as no row describes
///   it; `well-known:` is dropped, its rows being the candidates;
/// - anything else is chosen among the candidates
///   ([`SelectionContext::choose`]).
///
/// # Errors
/// As [`SelectionContext::choose`] and [`ChatModelFamily::candidates`].
pub fn choose_checkpoint(
    family: Option<ChatModelFamily>,
    selects: &[String],
    presets: &[String],
) -> BunsenResult<(ChatModelFamily, String)> {
    let mut selects: Vec<String> = selects.to_vec();
    if let [only] = selects.as_mut_slice()
        && !only.trim_start().starts_with('{')
        && let Some((provider, rest)) = only.split_once(':')
    {
        if provider != "well-known" {
            return Ok((family.unwrap_or(ChatModelFamily::Qwen3), only.clone()));
        }
        *only = rest.to_string();
    }
    if selects.is_empty() {
        selects.push(DEFAULT_SELECTION.to_string());
    }
    let mut universe = Vec::new();
    match family {
        Some(family) => universe.extend(family.candidates()?),
        None => {
            for family in ChatModelFamily::ALL {
                universe.extend(family.candidates()?);
            }
        }
    }
    let choice = chat_selection(family).choose(universe, &selects, presets)?;
    let chosen = choice
        .candidate
        .text(CHAT_FAMILY_TAG)
        .and_then(ChatModelFamily::from_name)
        .expect("every chat candidate is tagged with its family");
    Ok((chosen, choice.candidate.pretrained.name))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn choose(
        family: Option<ChatModelFamily>,
        selects: &[&str],
    ) -> (ChatModelFamily, String) {
        let selects: Vec<String> = selects.iter().map(|s| s.to_string()).collect();
        choose_checkpoint(family, &selects, &[]).unwrap()
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

    /// Every listed checkpoint, named by its full id or its bare name, is
    /// itself, of its family, across families and within its own.
    #[test]
    fn test_every_row_is_its_own() {
        for family in ChatModelFamily::ALL {
            for (_, row) in family.rows().unwrap() {
                let bare = row
                    .name
                    .rsplit_once('/')
                    .map_or(row.name.as_str(), |(_, n)| n);
                for name in [row.name.as_str(), bare] {
                    assert_eq!(choose(None, &[name]), (family, row.name.clone()), "{name}");
                    assert_eq!(
                        choose(Some(family), &[name]),
                        (family, row.name.clone()),
                        "{name}"
                    );
                }
                for alias in &row.aliases {
                    assert_eq!(choose(None, &[alias]).0, family, "{alias}");
                }
            }
        }
        assert_eq!(
            choose(None, &["nanochat-d20-rl"]),
            (
                ChatModelFamily::NanoChat,
                "nanochat-students/rl-d20".to_string()
            )
        );
        assert_eq!(
            choose(None, &["well-known:gemma/gemma/gemma-2-2b-it"]).0,
            ChatModelFamily::Gemma
        );
    }

    /// An `hf:` repo is its ref, of the family the command is in, else
    /// Qwen3's.
    #[test]
    fn test_hub_repos_pass_through() {
        assert_eq!(
            choose(None, &["hf:Qwen/Qwen3-0.6B"]),
            (ChatModelFamily::Qwen3, "hf:Qwen/Qwen3-0.6B".to_string())
        );
        assert_eq!(
            choose(Some(ChatModelFamily::Gemma), &["hf:unsloth/gemma-3-1b-it"]),
            (
                ChatModelFamily::Gemma,
                "hf:unsloth/gemma-3-1b-it".to_string()
            )
        );
    }

    /// A fuzzy name is a set the defaults choose within; an object narrows
    /// it further.
    #[test]
    fn test_selections() {
        let (family, name) = choose(None, &["gemma-3"]);
        assert_eq!(family, ChatModelFamily::Gemma);
        assert!(name.ends_with("-it"), "{name}: a tuned model");
        let (family, _) = choose(None, &["nanochat"]);
        assert_eq!(
            family,
            ChatModelFamily::NanoChat,
            "a prefix of its rows' names"
        );
        let (_, name) = choose(None, &["{require: {thinking: always}}"]);
        assert_eq!(name, "qwen/qwen3/qwen3-4b-thinking-2507");
        let (_, name) = choose(
            None,
            &[
                "qwen3",
                "{require: {instruct: null, thinking: never}, order: [parameters]}",
            ],
        );
        assert_eq!(name, "qwen/qwen3/qwen3-0.6b-base");
        assert!(choose_checkpoint(None, &["no-such-model".to_string()], &[]).is_err());
    }

    /// A `--preset` narrows the choice to the rows that have it.
    #[test]
    fn test_presets_narrow() {
        let (family, _) = choose_checkpoint(
            None,
            &["{require: {thinking: switch}}".to_string()],
            &["thinking".to_string()],
        )
        .unwrap();
        assert!(matches!(
            family,
            ChatModelFamily::Qwen3 | ChatModelFamily::Qwen3_5
        ));
    }
}
