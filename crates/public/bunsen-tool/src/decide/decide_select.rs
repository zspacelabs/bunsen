//! How `decide` chooses a checkpoint.
use bunsen::{
    data::pretrained::selection::SelectionCandidate,
    errors::BunsenResult,
    kits::decision::laya::pretrained::{
        DEFAULT_LAYA_MODEL,
        default_laya_factory,
    },
};
use serde_json::json;

use crate::select::{
    NamedSelection,
    SelectionContext,
};

/// `decide`'s selection context: the defaults, applied to every choice but
/// a checkpoint named exactly (each release sequence's newest, the heaviest
/// first), and `default`, [`DEFAULT_LAYA_MODEL`], which a command without
/// `--select` takes.
pub fn decide_selection() -> SelectionContext {
    SelectionContext {
        defaults: json!({"latest": true, "order": ["-weight"]}),
        named: vec![NamedSelection {
            name: "default".to_string(),
            description: format!("the default checkpoint, {DEFAULT_LAYA_MODEL}"),
            selection: json!({"name": DEFAULT_LAYA_MODEL}),
        }],
    }
}

/// The Laya checkpoint `selects` (each `--select`) choose among the listed
/// ones: no `--select` is `default`; one `provider:ref` of a provider that
/// lists nothing (`hf:org/repo`) is that ref.
///
/// # Errors
/// As [`SelectionContext::choose`].
pub fn choose_decide_model(selects: &[String]) -> BunsenResult<String> {
    let mut selects = selects.to_vec();
    if let [only] = selects.as_mut_slice()
        && !only.trim_start().starts_with('{')
        && let Some((provider, rest)) = only.split_once(':')
    {
        if provider != "well-known" {
            return Ok(only.clone());
        }
        *only = rest.to_string();
    }
    if selects.is_empty() {
        selects.push("default".to_string());
    }
    let universe: Vec<SelectionCandidate> = default_laya_factory()?
        .list()
        .into_iter()
        .map(|(_, row)| SelectionCandidate::new(row))
        .collect();
    Ok(decide_selection()
        .choose(universe, &selects, &[])?
        .candidate
        .pretrained
        .name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn choose(selects: &[&str]) -> String {
        let selects: Vec<String> = selects.iter().map(|s| s.to_string()).collect();
        choose_decide_model(&selects).unwrap()
    }

    /// The default is Laya's default; every listed checkpoint, by its name,
    /// is itself; a selection chooses by tags.
    #[test]
    fn test_choose() {
        assert_eq!(choose(&[]), DEFAULT_LAYA_MODEL);
        for (_, row) in default_laya_factory().unwrap().list() {
            assert_eq!(choose(&[&row.name]), row.name);
        }
        assert_eq!(
            choose(&["{prefer: [{language: de, otherwise: 0.5}]}"]),
            "laya/laya-multilingual",
            "only the multilingual model's `*` might hear German"
        );
        assert_eq!(choose(&["hf:org/repo"]), "hf:org/repo");
        assert!(choose_decide_model(&["{require: {language: zz-Zzzz}}".to_string()]).is_err());
    }
}
