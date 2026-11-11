//! The built-in catalog's rows held to the per-kit catalogs' rows.

use alloc::{
    format,
    string::{
        String,
        ToString,
    },
    vec::Vec,
};

use serde_json::{
    Value,
    json,
};

use crate::data::pretrained::{
    BuildCatalogProvider,
    EmbeddedCatalog,
    Pretrained,
    PretrainedProvider,
    build_catalog::presets::PresetSelection,
};

/// One kit: its old catalog, its contract, and its stated geometry by
/// prefab name, when its contract states one.
struct Kit {
    kit: &'static str,
    catalog: &'static EmbeddedCatalog,
    contract: &'static str,
    geometry: Option<fn(&str) -> Value>,
}

fn kits() -> Vec<Kit> {
    use crate::kits::{
        decision::laya::pretrained::LAYA_CATALOG,
        images::resnet::pretrained::{
            RESNET_CATALOG,
            RESNET_PREFABS,
        },
        language::{
            gemma::pretrained::GEMMA_CATALOG,
            nanochat::pretrained::NANOCHAT_CATALOG,
            qwen3::pretrained::QWEN3_CATALOG,
            qwen3_5::pretrained::QWEN3_5_CATALOG,
        },
        speech::{
            asr::{
                parakeet::pretrained::PARAKEET_CATALOG,
                qwen3_asr::pretrained::{
                    QWEN3_ASR_CATALOG,
                    QWEN3_ASR_PREFABS,
                },
                voxtral::pretrained::{
                    VOXTRAL_CATALOG,
                    VOXTRAL_PREFABS,
                },
                whisper::pretrained::WHISPER_CATALOG,
            },
            diarization::sortformer::pretrained::SORTFORMER_CATALOG,
            vad::silero_vad::pretrained::SILERO_CATALOG,
        },
    };
    let kit = |kit, catalog, contract, geometry| Kit {
        kit,
        catalog,
        contract,
        geometry,
    };
    fn resnet(prefab: &str) -> Value {
        serde_json::to_value(RESNET_PREFABS.expect_lookup_prefab(prefab).to_config()).unwrap()
    }
    fn qwen3_asr(prefab: &str) -> Value {
        serde_json::to_value(QWEN3_ASR_PREFABS.expect_lookup_prefab(prefab).to_config()).unwrap()
    }
    fn voxtral(prefab: &str) -> Value {
        serde_json::to_value(VOXTRAL_PREFABS.expect_lookup_prefab(prefab).to_config()).unwrap()
    }
    alloc::vec![
        kit("laya", &LAYA_CATALOG, "laya-decision", None),
        kit(
            "resnet",
            &RESNET_CATALOG,
            "resnet-classifier",
            Some(resnet as fn(&str) -> Value),
        ),
        kit("gemma", &GEMMA_CATALOG, "gemma-lm", None),
        kit("nanochat", &NANOCHAT_CATALOG, "nanochat-lm", None),
        kit("qwen3", &QWEN3_CATALOG, "qwen3-lm", None),
        kit("qwen3.5", &QWEN3_5_CATALOG, "qwen3.5-lm", None),
        kit("parakeet", &PARAKEET_CATALOG, "parakeet-asr", None),
        kit(
            "qwen3-asr",
            &QWEN3_ASR_CATALOG,
            "qwen3-asr",
            Some(qwen3_asr)
        ),
        kit("voxtral", &VOXTRAL_CATALOG, "voxtral-asr", Some(voxtral)),
        kit("whisper", &WHISPER_CATALOG, "whisper-asr", None),
        kit(
            "sortformer",
            &SORTFORMER_CATALOG,
            "sortformer-diarizer",
            None
        ),
        kit("silero", &SILERO_CATALOG, "silero-vad", None),
    ]
}

/// How much a run compared.
#[derive(Debug, Default)]
struct Compared {
    resources: usize,
    tags: usize,
    selections: usize,
    geometries: usize,
}

/// What differs between `old` and `new`, as lines.
fn differences(
    kit: &Kit,
    old: &Pretrained,
    new: &Pretrained,
    compared: &mut Compared,
) -> Vec<String> {
    let mut out = Vec::new();
    let mut differ = |what: String| out.push(format!("{}: {what}", old.name));

    let sentence = |s: &str| s.trim_end_matches('.').to_string();
    if sentence(&old.description) != sentence(&new.description) {
        differ(format!(
            "description {:?} vs {:?}",
            old.description, new.description
        ));
    }
    if (&old.license, &old.origin) != (&new.license, &new.origin) {
        differ(format!(
            "license/origin {:?} vs {:?}",
            (&old.license, &old.origin),
            (&new.license, &new.origin)
        ));
    }
    for alias in &old.aliases {
        if !new.aliases.contains(alias) {
            differ(format!("no alias {alias:?}"));
        }
    }

    // The files: the same keys, each the same file, pin, size, kind,
    // namespace and sources; a dtype the old row left out may be given.
    let keys = |row: &Pretrained| row.resources.resources.keys().cloned().collect::<Vec<_>>();
    if keys(old) != keys(new) {
        differ(format!("resources {:?} vs {:?}", keys(old), keys(new)));
    }
    for (key, o) in &old.resources.resources {
        let Some(n) = new.resources.resources.get(key) else {
            continue;
        };
        compared.resources += 1;
        let mut o = o.clone();
        if o.dtype.is_none() {
            o.dtype = n.dtype.clone();
        }
        if &o != n {
            differ(format!("resource {key}: {o:?} vs {n:?}"));
        }
    }

    // The tags: each old one, as the new form names it.
    for (key, value) in old.tags.iter() {
        let (new_key, expected) = match key {
            "base" => ("instruct", json!(!value.as_bool().unwrap_or(false))),
            "dtype" | "groups" | "version" => continue,
            other => (other, value.clone()),
        };
        compared.tags += 1;
        if new.tags.get(new_key) != Some(&expected) {
            differ(format!(
                "tag {key}: {value} vs {new_key} {:?}",
                new.tags.get(new_key)
            ));
        }
    }

    // The presets: the default, and each old preset alone, resolve alike.
    let mut selections = alloc::vec![PresetSelection::default()];
    selections.extend(old.presets.names().map(|n| PresetSelection::of([n])));
    for selection in selections {
        let o = old.presets.select(json!({}), &selection);
        let n = new.presets.select(json!({}), &selection);
        match (o, n) {
            (Ok(o), Ok(n)) if o.document == n.document => {
                compared.selections += 1;
            }
            (Err(_), _) => {}
            (o, n) => differ(format!(
                "presets {:?}: {:?} vs {:?}",
                selection.presets,
                o.map(|a| a.document),
                n.map(|a| a.document)
            )),
        }
    }

    // A stated geometry: the prefab's config, as the build's argument.
    if let (Some(geometry), Some(prefab)) = (kit.geometry, &old.prefab) {
        compared.geometries += 1;
        if new.args.get("geometry") != Some(&geometry(prefab)) {
            differ(format!("geometry: not the prefab {prefab:?}'s config"));
        }
    }
    out
}

/// Every old row has a new row its name finds, the same in every way the
/// new form keeps; and every new row is an old row's.
#[test]
fn test_the_builtin_rows_are_the_old_rows() {
    let mut problems = Vec::new();
    let mut total = 0;
    let mut compared = Compared::default();
    for kit in kits() {
        let new = BuildCatalogProvider::builtin("the built-in catalog", kit.contract, kit.kit)
            .unwrap_or_else(|e| panic!("{}: {e}", kit.kit));
        let old = kit.catalog.table().list();
        if old.len() != new.rows().len() {
            problems.push(format!(
                "{}: {} old rows, {} new",
                kit.kit,
                old.len(),
                new.rows().len()
            ));
        }
        for row in &old {
            total += 1;
            match new.lookup(&row.name) {
                Ok(Some(found)) => problems.extend(differences(&kit, row, &found, &mut compared)),
                Ok(None) => problems.push(format!("{}: no new row", row.name)),
                Err(e) => problems.push(format!("{}: {e}", row.name)),
            }
        }
    }
    assert_eq!(total, 83, "the old catalogs' rows");
    assert!(problems.is_empty(), "{}", problems.join("\n"));
    // Every file, tag and stated geometry was compared, and every preset.
    assert_eq!(compared.geometries, 20, "{compared:?}");
    assert!(
        compared.resources > 300 && compared.tags > 300,
        "{compared:?}"
    );
    assert!(compared.selections > 83, "{compared:?}");
}

/// The rows are named in full, each under its publisher.
#[test]
fn test_the_builtin_rows_are_named_in_full() {
    let qwen3 = BuildCatalogProvider::builtin("", "qwen3-lm", "qwen3").unwrap();
    assert_eq!(
        qwen3.lookup("qwen/qwen3-0.6b").unwrap().unwrap().name,
        "qwen/qwen3/qwen3-0.6b"
    );
    let names: Vec<String> = qwen3.rows().iter().map(|r| r.name.to_string()).collect();
    assert!(
        names.iter().all(|n| n.starts_with("qwen/qwen3/")),
        "{names:?}"
    );
}
