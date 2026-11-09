//! The model-choosing flags: `--select` and `--patch`.
use bunsen::errors::{
    BunsenError,
    BunsenResult,
};
use serde_json::Value;

use crate::select::merge;

/// How a command line chooses a model and patches its config, for
/// `#[command(flatten)]`; `--preset` is
/// [`PresetArgs`](crate::support::PresetArgs).
#[derive(clap::Args, Debug, Clone, Default)]
pub struct SelectArgs {
    /// The model: a name (a named selection, a row, or a prefix, group or kit
    /// of rows) or a selection object (`{require: {base: false}}`, YAML);
    /// repeatable, at most one name, objects merged in order over the
    /// command's defaults.
    #[arg(long = "select", visible_alias = "model", value_name = "NAME|{..}")]
    pub select: Vec<String>,

    /// A merge patch over the chosen model's config, a YAML mapping
    /// (`{beam_size: 1}`); repeatable, merged in order.
    #[arg(long = "patch", value_name = "{..}")]
    pub patch: Vec<String>,
}

impl SelectArgs {
    /// The `--patch`es merged in order; `None` without any.
    ///
    /// # Errors
    /// [`Policy`](bunsen::errors::BunsenErrorKind::Policy) for a patch that is
    /// not a YAML mapping.
    pub fn patch(&self) -> BunsenResult<Option<Value>> {
        if self.patch.is_empty() {
            return Ok(None);
        }
        let patches = self
            .patch
            .iter()
            .map(|text| {
                let value: Value = serde_saphyr::from_str(text).map_err(|e| {
                    BunsenError::policy(format!("`--patch {text}` is not YAML: {e}"))
                })?;
                if value.is_object() {
                    Ok(value)
                } else {
                    Err(BunsenError::policy(format!(
                        "`--patch {text}` is not a mapping"
                    )))
                }
            })
            .collect::<BunsenResult<Vec<Value>>>()?;
        Ok(Some(merge(&patches)))
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;
    use serde_json::json;

    use super::*;

    #[derive(clap::Parser, Debug)]
    struct Args {
        #[command(flatten)]
        select: SelectArgs,
    }

    #[test]
    fn test_flags() {
        let args = Args::try_parse_from([
            "x",
            "--select",
            "whisper",
            "--model",
            "{require: {base: false}}",
            "--patch",
            "{beam_size: 1, vad: {keep: true}}",
            "--patch",
            "{vad: {keep: null}}",
        ])
        .unwrap();
        assert_eq!(args.select.select, ["whisper", "{require: {base: false}}"]);
        assert_eq!(
            args.select.patch().unwrap(),
            Some(json!({"beam_size": 1, "vad": {}}))
        );
        assert_eq!(SelectArgs::default().patch().unwrap(), None);
        let bad = Args::try_parse_from(["x", "--patch", "[1]"]).unwrap();
        assert!(bad.select.patch().is_err());
    }
}
