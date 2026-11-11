//! The built-in catalog: the layers `bunsen-catalog` carries, imported.
//!
//! `bunsen-catalog` holds the catalog the catalog builder writes from its
//! probe-contract sources (`validation/catalog_builder`): the prelude, whose
//! `$contracts` are the kits' own (`kits::support::catalog_contracts`), the
//! commons, and a layer per kit, each with the bunsen feature it needs.
//! [`builtin_stack`] imports them in order, skipping a layer whose feature
//! this build of bunsen doesn't have.

use super::{
    PretrainedCatalog,
    PretrainedCatalogStack,
    error::{
        CatalogErrorExt,
        CatalogResult,
        CatalogRule,
        catalog_error,
    },
};

/// Whether this build of bunsen has `feature`, as the built-in catalog's
/// manifest names it.
///
/// # Errors
/// [`CatalogRule::Parse`] for a feature bunsen doesn't know.
fn compiled(feature: &str) -> CatalogResult<bool> {
    match feature {
        "kits" => Ok(cfg!(feature = "kits")),
        "speech" => Ok(cfg!(feature = "speech")),
        other => Err(catalog_error(
            CatalogRule::Parse,
            format!("the built-in catalog names the feature `{other}`, which bunsen doesn't have"),
        )),
    }
}

/// The built-in catalog's layers, in import order, with whether this build
/// imports each.
///
/// # Errors
/// [`CatalogRule::Parse`] for a layer whose feature bunsen doesn't know.
pub fn builtin_layers() -> CatalogResult<Vec<(bunsen_catalog::CatalogFile, bool)>> {
    bunsen_catalog::files()
        .into_iter()
        .map(|file| {
            let imported = match file.feature {
                None => true,
                Some(feature) => compiled(feature).map_err(|e| e.in_layer(file.label()))?,
            };
            Ok((file, imported))
        })
        .collect()
}

/// The built-in catalog, as a stack: each of its layers this build of
/// bunsen has the feature for, in order.
///
/// # Errors
/// Whatever a push finds, which would be a bug in the built catalog: the
/// builder imports every layer before it writes them.
pub fn builtin_stack() -> CatalogResult<PretrainedCatalogStack> {
    let mut stack = PretrainedCatalogStack::new();
    for (file, imported) in builtin_layers()? {
        if !imported {
            continue;
        }
        let label = file.label();
        let document = if file.name.ends_with(".json") {
            PretrainedCatalog::from_json(file.text)
        } else {
            PretrainedCatalog::from_yaml(file.text)
        }
        .map_err(|e| e.in_layer(label))?;
        stack.push(label, None, document)?;
    }
    Ok(stack)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The built-in catalog imports whole, and its base layer is first.
    #[test]
    fn test_the_builtin_stack_imports() {
        let stack = builtin_stack().unwrap();
        let layers = builtin_layers().unwrap();
        assert_eq!(
            stack.layers().len(),
            layers.iter().filter(|(_, imported)| *imported).count()
        );
        assert_eq!(stack.layers()[0].label(), "prelude");
    }

    /// Its contracts are the compiled kits' own, as their fragments write
    /// them: a kit's changed `contract.yml` needs a `cargo catalog update`.
    #[cfg(all(feature = "kits", feature = "store"))]
    #[test]
    fn test_the_builtin_contracts_are_the_kits() {
        use crate::{
            data::pretrained::build_catalog::document::OrderedNode,
            kits::support::catalog_contracts::{
                contracts_section,
                kit_catalog_contracts,
            },
        };

        let stack = builtin_stack().unwrap();
        let built = stack.layers()[0].document().contracts.clone().unwrap();
        let fragments = PretrainedCatalog::from_node(OrderedNode::Map(vec![(
            "$contracts".to_string(),
            contracts_section(&kit_catalog_contracts()).unwrap(),
        )]))
        .unwrap()
        .contracts
        .unwrap();
        #[cfg(feature = "speech")]
        assert_eq!(built, fragments);
        #[cfg(not(feature = "speech"))]
        for contract in &fragments {
            assert!(built.contains(contract), "{}", contract.name());
        }
    }

    /// A build of each kit resolves, by its short name.
    #[cfg(feature = "speech")]
    #[test]
    fn test_builtin_builds_resolve() {
        let stack = builtin_stack().unwrap();
        for (name, contract) in [
            ("qwen3-0.6b", "qwen3-lm"),
            ("qwen3.5-0.8b", "qwen3.5-lm"),
            ("gemma-3-1b-it", "gemma-lm"),
            ("rl-d20", "nanochat-lm"),
            ("laya", "laya-decision"),
            ("torchvision/resnet50", "resnet-classifier"),
            ("large", "whisper-asr"),
            ("phonon-2", "parakeet-asr"),
            ("phonon-1", "qwen3-asr/stated"),
            ("voxtral-mini-4b-realtime", "voxtral-asr/stated"),
            ("sortformer-v2", "sortformer-diarizer"),
            ("vad-16k", "silero-vad"),
        ] {
            let found = stack.find_build(name).unwrap();
            let build = stack.resolve_build(&found).unwrap();
            assert_eq!(
                build.tag("contract"),
                Some(&serde_json::json!(contract)),
                "{name}"
            );
        }
    }
}
