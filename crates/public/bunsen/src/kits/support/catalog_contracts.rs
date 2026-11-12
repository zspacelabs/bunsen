//! # The kits' build-catalog contracts
//!
//! Each kit with a pretrained index writes the contract its loader
//! implements, a signature of the resources and arguments it reads, in a
//! `contract.yml` beside the loader (`{KIT}_CONTRACT`). A fragment is a map
//! of `$contracts` entries: the kit's contract, its variants and its types
//! ([`catalog_format`](crate::data::catalog::catalog_format)).
//! The catalog builder gathers every kit's fragment into the base catalog's
//! `$contracts` section, so the built catalog's contracts are the ones the
//! loaders are written against, and each kit's tests hold its fragment to the
//! resource keys and formats the kit reads.

use alloc::{
    string::String,
    vec::Vec,
};

use crate::{
    data::catalog::{
        document::OrderedNode,
        error::{
            CatalogErrorExt,
            CatalogRule,
            catalog_error,
        },
    },
    errors::BunsenResult,
};

/// One kit's contract fragment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KitCatalogContract {
    /// The kit, by its name (`{KIT}_KIT`).
    pub kit: &'static str,

    /// The fragment: `$contracts` entries, as YAML.
    pub yaml: &'static str,
}

/// The contract fragment of every compiled kit with a pretrained index, in
/// a fixed order.
pub fn kit_catalog_contracts() -> Vec<KitCatalogContract> {
    use crate::kits::{
        decision::laya::pretrained::{
            LAYA_CONTRACT,
            LAYA_KIT,
        },
        images::resnet::pretrained::{
            RESNET_CONTRACT,
            RESNET_KIT,
        },
        language::{
            gemma::pretrained::{
                GEMMA_CONTRACT,
                GEMMA_KIT,
            },
            nanochat::pretrained::{
                NANOCHAT_CONTRACT,
                NANOCHAT_KIT,
            },
            qwen3::pretrained::{
                QWEN3_CONTRACT,
                QWEN3_KIT,
            },
            qwen3_5::pretrained::{
                QWEN3_5_CONTRACT,
                QWEN3_5_KIT,
            },
        },
    };

    let contract = |kit, yaml| KitCatalogContract { kit, yaml };
    #[allow(unused_mut)]
    let mut contracts = alloc::vec![
        contract(
            crate::data::resources::hub::HUB_KIT,
            crate::data::resources::hub::HF_MIRROR_CONTRACT,
        ),
        contract(
            crate::data::datasets::DATASETS_KIT,
            crate::data::datasets::PARQUET_SHARDS_CONTRACT,
        ),
        contract(LAYA_KIT, LAYA_CONTRACT),
        contract(RESNET_KIT, RESNET_CONTRACT),
        contract(GEMMA_KIT, GEMMA_CONTRACT),
        contract(NANOCHAT_KIT, NANOCHAT_CONTRACT),
        contract(QWEN3_KIT, QWEN3_CONTRACT),
        contract(QWEN3_5_KIT, QWEN3_5_CONTRACT),
    ];
    #[cfg(feature = "speech")]
    {
        use crate::kits::speech::{
            asr::{
                parakeet::pretrained::{
                    PARAKEET_CONTRACT,
                    PARAKEET_KIT,
                },
                qwen3_asr::pretrained::{
                    QWEN3_ASR_CONTRACT,
                    QWEN3_ASR_KIT,
                },
                voxtral::pretrained::{
                    VOXTRAL_CONTRACT,
                    VOXTRAL_KIT,
                },
                whisper::pretrained::{
                    WHISPER_CONTRACT,
                    WHISPER_KIT,
                },
            },
            diarization::sortformer::pretrained::{
                SORTFORMER_CONTRACT,
                SORTFORMER_KIT,
            },
            vad::silero_vad::pretrained::{
                SILERO_CONTRACT,
                SILERO_KIT,
            },
        };
        contracts.extend([
            contract(PARAKEET_KIT, PARAKEET_CONTRACT),
            contract(QWEN3_ASR_KIT, QWEN3_ASR_CONTRACT),
            contract(VOXTRAL_KIT, VOXTRAL_CONTRACT),
            contract(WHISPER_KIT, WHISPER_CONTRACT),
            contract(SORTFORMER_KIT, SORTFORMER_CONTRACT),
            contract(SILERO_KIT, SILERO_CONTRACT),
        ]);
    }
    contracts
}

/// The `$contracts` section `contracts` make: their entries, in order.
///
/// # Errors
/// [`CatalogRule::Parse`] when a fragment isn't a YAML map, and
/// [`CatalogRule::Structure`] when two fragments declare one name.
pub fn contracts_section(contracts: &[KitCatalogContract]) -> BunsenResult<OrderedNode> {
    let mut entries: Vec<(String, OrderedNode)> = Vec::new();
    for contract in contracts {
        let at = alloc::format!("the {} kit's contract", contract.kit);
        let OrderedNode::Map(fragment) =
            OrderedNode::from_yaml(contract.yaml).map_err(|e| e.at(&at))?
        else {
            return Err(catalog_error(
                CatalogRule::Parse,
                "a contract fragment is a map of `$contracts` entries",
            )
            .at(&at));
        };
        for (name, entry) in fragment {
            if entries.iter().any(|(n, _)| n.eq_ignore_ascii_case(&name)) {
                return Err(catalog_error(
                    CatalogRule::Structure,
                    alloc::format!("`{name}` is declared by another kit's contract too"),
                )
                .at(&at));
            }
            entries.push((name, entry));
        }
    }
    Ok(OrderedNode::Map(entries))
}

/// What the kits' tests hold their fragments to.
#[cfg(test)]
pub(crate) mod testing {
    use alloc::{
        format,
        string::{
            String,
            ToString,
        },
        vec::Vec,
    };

    use super::*;
    use crate::data::catalog::{
        BunsenCatalog,
        BunsenCatalogStack,
        document::CatalogEntry,
    };

    /// `yaml`'s contracts as lines, one per slot, after it loads as a base
    /// catalog: `{contract}: {slot} [{formats}]` for a resource slot,
    /// `{contract}: {slot}: {type}` for an argument slot, a `?` after an
    /// optional slot's name, and `{contract}: abstract` for an abstract
    /// contract; a variant's contract is its path (`qwen3-asr/stated`).
    pub(crate) fn contract_lines(yaml: &'static str) -> Vec<String> {
        let section = contracts_section(&[KitCatalogContract { kit: "test", yaml }]).unwrap();
        let catalog = BunsenCatalog::from_node(OrderedNode::Map(alloc::vec![(
            "$contracts".to_string(),
            section,
        )]))
        .unwrap();
        let mut stack = BunsenCatalogStack::new();
        stack.push("contracts", None, catalog.clone()).unwrap();

        fn walk(
            prefix: &str,
            entries: &[CatalogEntry],
            lines: &mut Vec<String>,
        ) {
            for entry in entries {
                let CatalogEntry::Contract(contract) = entry else {
                    continue;
                };
                let path = if prefix.is_empty() {
                    contract.name.clone()
                } else {
                    format!("{prefix}/{}", contract.name)
                };
                if contract.abstract_ {
                    lines.push(format!("{path}: abstract"));
                }
                for (slot, spec) in &contract.resources {
                    let optional = if spec.optional { "?" } else { "" };
                    lines.push(format!(
                        "{path}: {slot}{optional} [{}]",
                        spec.formats.join(", ")
                    ));
                }
                for (slot, spec) in &contract.args {
                    let optional = if spec.optional { "?" } else { "" };
                    lines.push(format!("{path}: {slot}{optional}: {}", spec.type_expr));
                }
                walk(&path, &contract.children, lines);
            }
        }
        let mut lines = Vec::new();
        walk("", catalog.contracts.as_deref().unwrap_or(&[]), &mut lines);
        lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every compiled kit's fragment loads, and together they make one
    /// `$contracts` section with no name twice.
    #[test]
    fn test_the_fragments_make_one_section() {
        let contracts = kit_catalog_contracts();
        let OrderedNode::Map(section) = contracts_section(&contracts).unwrap() else {
            unreachable!()
        };
        assert!(section.len() >= contracts.len());
        for contract in &contracts {
            assert!(!testing::contract_lines(contract.yaml).is_empty());
        }
    }

    /// A name two fragments declare is refused.
    #[test]
    fn test_a_repeated_name_is_refused() {
        let a = KitCatalogContract {
            kit: "a",
            yaml: "m: { $kind: contract, $description: A. }",
        };
        let b = KitCatalogContract {
            kit: "b",
            yaml: "M: { $kind: contract, $description: B. }",
        };
        let err = contracts_section(&[a, b]).unwrap_err();
        assert!(err.has(CatalogRule::Structure), "{err}");
    }
}
