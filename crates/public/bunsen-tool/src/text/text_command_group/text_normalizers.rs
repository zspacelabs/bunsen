//! `text normalizers`.

use std::io::Write;

use bunsen::{
    errors::{
        BunsenResult,
        sys_op,
    },
    kits::support::text::{
        TextNormalizationConfig,
        TextNormalizerInfo,
    },
};

/// Lists the normalizers `--normalize` names: each one's spec, what it does,
/// and, for one that needs a bunsen feature, whether this build has it.
#[derive(clap::Args, Debug)]
pub struct TextNormalizersCommand {}

impl TextNormalizersCommand {
    pub fn run(&self) -> BunsenResult<()> {
        let mut out = std::io::stdout().lock();
        write_normalizers(&mut out, &TextNormalizationConfig::builtins())
            .map_err(sys_op("write stdout"))
    }
}

/// `normalizers`, one per line: the spec, padded; what it does; and, for one
/// behind a feature, the feature and whether the build has it.
pub fn write_normalizers(
    out: &mut impl Write,
    normalizers: &[TextNormalizerInfo],
) -> std::io::Result<()> {
    let width = normalizers
        .iter()
        .map(|info| info.spec.len())
        .max()
        .unwrap_or(0);
    for info in normalizers {
        let feature = match info.feature {
            Some(feature) if info.available => format!(" (feature `{feature}`)"),
            Some(feature) => format!(" (feature `{feature}`: not in this build)"),
            None => String::new(),
        };
        writeln!(out, "{:width$}  {}{feature}", info.spec, info.description)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_normalizers() {
        let infos = [
            TextNormalizerInfo {
                spec: "none",
                description: "as written",
                feature: None,
                available: true,
            },
            TextNormalizerInfo {
                spec: "zyphra",
                description: "spoken",
                feature: Some("zyphra-textnorm"),
                available: false,
            },
        ];
        let mut out = Vec::new();
        write_normalizers(&mut out, &infos).unwrap();
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "none    as written\nzyphra  spoken (feature `zyphra-textnorm`: not in this build)\n"
        );
    }
}
