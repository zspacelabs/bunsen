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
