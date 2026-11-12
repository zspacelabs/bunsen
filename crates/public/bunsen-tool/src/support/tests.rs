use bunsen::data::catalog::presets::ResolvedPresetSet;
use clap::Parser;

use super::*;

#[derive(clap::Parser, Debug)]
struct Args {
    #[clap(flatten)]
    presets: PresetArgs,
}

/// A row's presets print as its axes, their members described and the
/// default marked, then the presets on no axis; a row with none prints
/// nothing.
#[test]
fn test_preset_lines() {
    use bunsen::data::catalog::{
        CatalogName,
        document::{
            AxisKind,
            MergeMode,
            PresetEntry,
        },
        presets::{
            AxisInfo,
            ResolvedAxisUse,
        },
    };
    let preset = |name: &str, description: &str| PresetEntry {
        name: name.to_string(),
        merge: MergeMode::Update,
        description: description.to_string(),
        patch: serde_json::json!({}),
    };
    let task = vec!["transcribe".to_string(), "translate".to_string()];
    let set = ResolvedPresetSet {
        presets: vec![
            preset("transcribe", "the speech's own language"),
            preset("translate", "English"),
            preset("loud", ""),
        ],
        axes: vec![ResolvedAxisUse {
            axis: AxisInfo {
                name: CatalogName::root("task").unwrap(),
                kind: AxisKind::Categorical,
                members: task.clone(),
                description: "what to emit".to_string(),
            },
            offers: task,
            default: "transcribe".to_string(),
        }],
        ..ResolvedPresetSet::default()
    };
    assert_eq!(
        preset_lines(&set),
        [
            "  presets (`--preset NAME`, or the model's box: `NAME[PRESET,...]`):",
            "    task: what to emit",
            "      transcribe  the speech's own language (default)",
            "      translate   English",
            "    on no axis:",
            "      loud",
        ]
    );
    assert!(preset_lines(&ResolvedPresetSet::default()).is_empty());
}

/// `--preset` takes a comma list or repeats, in order, and joins the
/// model's box; a checkpoint file has no presets.
#[test]
fn test_preset_args() {
    let args =
        Args::try_parse_from(["x", "--preset", "translate,fast", "--preset", "good"]).unwrap();
    assert_eq!(args.presets.presets, ["translate", "fast", "good"]);
    assert_eq!(
        args.presets.apply("openai/small").unwrap(),
        "openai/small[translate,fast,good]"
    );
    assert_eq!(
        args.presets.apply("openai/small[a]{beam_size: 1}").unwrap(),
        "openai/small[a,translate,fast,good]{beam_size: 1}"
    );
    assert_eq!(
        PresetArgs::default().apply("openai/small").unwrap(),
        "openai/small"
    );

    let file = tempfile::NamedTempFile::new().unwrap();
    let path = file.path().to_str().unwrap();
    assert!(args.presets.apply(path).is_err());
    assert_eq!(PresetArgs::default().apply(path).unwrap(), path);
}
