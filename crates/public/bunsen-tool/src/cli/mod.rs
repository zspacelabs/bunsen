//! `bunsen-tool`'s command trees, one per branch.

pub mod asr;
pub mod card;
pub mod chat;

use bunsen::{
    errors::BunsenResult,
    kits::support::construct::LookupExpr,
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

/// A model's presets on the command line, for `#[command(flatten)]`:
/// `--preset`, a comma list or repeated, joining the box of the model's
/// lookup expression.
#[derive(clap::Args, Debug, Clone, Default)]
pub struct PresetArgs {
    /// The model's presets, joining its expression's box: a comma list, or
    /// the flag repeated (`--preset translate --preset ...`). The model's
    /// `list` shows the presets it has.
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
            return Err(bunsen::errors::BunsenError::policy(format!(
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
            r#"openai/small[a,translate,fast,good]{"beam_size":1}"#
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
