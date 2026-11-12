//! What the command trees share: a lookup expression with presets or a patch
//! added, a row's presets and a checkpoint's row as the listings print them
//! ([`print_model_row`]), the `--preset` flags, the `--format` choice
//! ([`Format`]), and "did you mean?" on a failed lookup ([`did_you_mean`]).

mod choose_args;
pub mod did_you_mean;
mod model_row;
mod output_format;

use bunsen::{
    data::{
        builds::LookupExpr,
        catalog::presets::ResolvedPresetSet,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
    support::merge_patch,
};
pub use choose_args::*;
pub use model_row::*;
pub use output_format::*;

/// `model`, a lookup expression, with `presets` added to its box.
///
/// # Errors
/// As [`LookupExpr::parse`].
pub fn with_presets(
    model: &str,
    presets: &[&str],
) -> BunsenResult<String> {
    let mut expr = LookupExpr::parse(model)?;
    expr.selection
        .presets
        .extend(presets.iter().map(|p| p.to_string()));
    Ok(expr.to_string())
}

/// `model`, a lookup expression, with `patch` merged over its patch
/// (RFC 7386: a `null` removes a key).
///
/// # Errors
/// As [`LookupExpr::parse`].
pub fn with_patch(
    model: &str,
    patch: &serde_json::Value,
) -> BunsenResult<String> {
    let mut expr = LookupExpr::parse(model)?;
    let mut merged = expr
        .selection
        .patch
        .take()
        .unwrap_or_else(|| serde_json::json!({}));
    merge_patch::apply(&mut merged, patch);
    expr.selection.patch = Some(merged);
    Ok(expr.to_string())
}

/// A row's presets, as the listings print them under the row: each axis
/// with its description, then its members with theirs, the default marked;
/// then the presets on no axis. Nothing for a row with no presets.
///
/// ```text
///   presets (`--preset NAME`, or the model's box: `NAME[PRESET,...]`):
///     task: what to make of the speech
///       transcribe  text in the language spoken (default)
///       translate   English text
/// ```
pub fn preset_lines(set: &ResolvedPresetSet) -> Vec<String> {
    if set.presets.is_empty() {
        return Vec::new();
    }
    let defaults = set.default_names();
    let width = set.presets.iter().map(|p| p.name.len()).max().unwrap_or(0);
    let member = |name: &str| {
        let description = set.get(name).map_or("", |p| p.description.as_str());
        let default = if defaults.iter().any(|d| d == name) {
            " (default)"
        } else {
            ""
        };
        format!("      {name:<width$}  {description}{default}")
            .trim_end()
            .to_string()
    };
    let mut lines =
        vec!["  presets (`--preset NAME`, or the model's box: `NAME[PRESET,...]`):".to_string()];
    for using in &set.axes {
        lines.push(format!(
            "    {}: {}",
            using.axis.name, using.axis.description
        ));
        lines.extend(using.offers.iter().map(|m| member(m)));
    }
    let loose: Vec<&str> = set
        .names()
        .filter(|name| set.axis_of(name).is_none())
        .collect();
    if !loose.is_empty() {
        lines.push("    on no axis:".to_string());
        lines.extend(loose.into_iter().map(member));
    }
    lines
}

/// A model's presets on the command line, for `#[command(flatten)]`:
/// `--preset`, a comma list or repeated, joining the box of the model's
/// lookup expression.
#[derive(clap::Args, Debug, Clone, Default)]
pub struct PresetArgs {
    /// The model's presets, joining its expression's box.
    ///
    /// A comma list, or the flag repeated (`--preset translate --preset
    /// ...`). The model's `list` shows the presets it has.
    #[arg(long = "preset", value_name = "PRESET", value_delimiter = ',')]
    pub presets: Vec<String>,
}

impl PresetArgs {
    /// `model` with the presets joined to its box; `model` as it is without
    /// any.
    ///
    /// # Errors
    /// [`Policy`](bunsen::errors::BunsenErrorKind::Policy) when `model` is a
    /// checkpoint file, which has no presets; as [`with_presets`].
    pub fn apply(
        &self,
        model: &str,
    ) -> BunsenResult<String> {
        if self.presets.is_empty() {
            return Ok(model.to_string());
        }
        if std::path::Path::new(model).is_file() {
            return Err(BunsenError::policy(format!(
                "--preset names a model's presets; the checkpoint file {model} has none"
            )));
        }
        let presets: Vec<&str> = self.presets.iter().map(String::as_str).collect();
        with_presets(model, &presets)
    }
}

#[cfg(test)]
mod tests {
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
}
