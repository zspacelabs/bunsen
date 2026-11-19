//! `text normalize`.

use std::{
    io::{
        Read,
        Write,
    },
    path::PathBuf,
};

use bunsen::{
    errors::{
        BunsenResult,
        sys_at,
        sys_op,
    },
    kits::support::text::{
        BunsenLanguageTag,
        TextNormalizationConfig,
        TextNormalizer,
        TextNormalizerChain,
    },
};
use bunsen_app::logging::{
    LogArgs,
    LogPolicy,
};
use log::LevelFilter;

/// Normalizes a text, to stdout.
#[derive(clap::Args, Debug)]
pub struct TextNormalizeCommand {
    #[clap(flatten)]
    logging: LogArgs,

    /// A normalizer, by spec; repeated, they run in order.
    ///
    /// `none`, `collapse-line-breaks`, `transcript`, `zyphra`;
    /// `tokenizers:NFKC` (a `tokenizers` normalizer by type) or
    /// `tokenizers:{...}` (one in `tokenizer.json`'s syntax);
    /// `tokenizer-json:PATH` (a `tokenizer.json`'s own normalizer);
    /// `FILE.json` (a normalization config). `text normalizers` lists them.
    #[arg(long = "normalize", required = true, value_name = "SPEC")]
    normalizers: Vec<String>,

    /// The text's language, as a language tag (`en`, `fr-FR`, `en_US`).
    ///
    /// The normalizers that read by language need it (`zyphra`); one with no
    /// grammar for it is an error.
    #[arg(long)]
    language: Option<String>,

    /// Print each normalizer's name and its output in turn, not just the
    /// result.
    #[arg(long)]
    explain: bool,

    /// Read the text from this file.
    #[arg(long, conflicts_with = "text")]
    file: Option<PathBuf>,

    /// The text; stdin without it or `--file`. The words are joined by
    /// spaces.
    text: Vec<String>,
}

impl TextNormalizeCommand {
    pub fn run(&self) -> BunsenResult<()> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Warn, LevelFilter::Info))?;
        let text = self.read_text()?;
        let mut out = std::io::stdout().lock();
        self.normalize_to(&text, &mut out)
    }

    /// The text: `--file`'s, the arguments, or stdin.
    fn read_text(&self) -> BunsenResult<String> {
        if let Some(path) = &self.file {
            return std::fs::read_to_string(path).map_err(sys_at("read", path));
        }
        if !self.text.is_empty() {
            return Ok(self.text.join(" "));
        }
        let mut text = String::new();
        std::io::stdin()
            .read_to_string(&mut text)
            .map_err(sys_at("read", "stdin"))?;
        Ok(text)
    }

    /// The normalizers, one stage each, in order.
    fn chain(&self) -> BunsenResult<TextNormalizerChain> {
        let stages = self
            .normalizers
            .iter()
            .map(|spec| TextNormalizationConfig::parse_spec(spec)?.init())
            .collect::<BunsenResult<Vec<_>>>()?;
        Ok(TextNormalizerChain::new(stages))
    }

    /// `text`, normalized, written to `out`: the result, or with
    /// `--explain` each stage's name and output.
    ///
    /// # Errors
    /// A spec's, the language's, a normalizer's; a write's.
    fn normalize_to(
        &self,
        text: &str,
        out: &mut impl Write,
    ) -> BunsenResult<()> {
        let chain = self.chain()?;
        let language = self
            .language
            .as_deref()
            .map(BunsenLanguageTag::parse)
            .transpose()?;
        chain.check_language(language.as_ref())?;
        let write = |out: &mut dyn Write, text: &str| -> std::io::Result<()> {
            out.write_all(text.as_bytes())?;
            if !text.ends_with('\n') {
                out.write_all(b"\n")?;
            }
            Ok(())
        };
        if self.explain {
            for (name, output) in chain.explain(text, language.as_ref())? {
                writeln!(out, "== {name}").map_err(sys_op("write stdout"))?;
                write(out, &output).map_err(sys_op("write stdout"))?;
            }
            return Ok(());
        }
        let normalized = chain.normalize(text, language.as_ref())?;
        write(out, &normalized).map_err(sys_op("write stdout"))
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;

    #[derive(Parser, Debug)]
    struct Cli {
        #[clap(flatten)]
        normalize: TextNormalizeCommand,
    }

    fn run(argv: &[&str]) -> BunsenResult<String> {
        let cli = Cli::try_parse_from(argv).unwrap();
        let text = cli.normalize.read_text()?;
        let mut out = Vec::new();
        cli.normalize.normalize_to(&text, &mut out)?;
        Ok(String::from_utf8(out).unwrap())
    }

    #[test]
    fn test_normalize() {
        assert_eq!(
            run(&["x", "--normalize", "transcript", "Hello,", "World!"]).unwrap(),
            "hello world\n"
        );
        assert_eq!(
            run(&[
                "x",
                "--normalize",
                "collapse-line-breaks",
                "--normalize",
                "tokenizers:Lowercase",
                "--explain",
                "A\nB",
            ])
            .unwrap(),
            "== collapse-line-breaks\nA B\n== tokenizers(Lowercase)\na b\n"
        );
    }

    #[test]
    fn test_normalize_from_a_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("in.txt");
        std::fs::write(&path, "One\tTwo\n").unwrap();
        assert_eq!(
            run(&[
                "x",
                "--normalize",
                "collapse-line-breaks",
                "--file",
                path.to_str().unwrap(),
            ])
            .unwrap(),
            "One Two \n"
        );
    }

    #[test]
    fn test_refusals() {
        assert!(
            Cli::try_parse_from(["x", "text"]).is_err(),
            "--normalize is required"
        );
        let error = run(&["x", "--normalize", "zyfra", "t"]).unwrap_err();
        assert_eq!(error.kind(), bunsen::errors::BunsenErrorKind::Lookup);
        let error = run(&["x", "--normalize", "none", "--language", "en us", "t"]).unwrap_err();
        assert_eq!(error.kind(), bunsen::errors::BunsenErrorKind::Illegal);
    }

    #[test]
    fn test_zyphra_by_language() {
        let result = run(&[
            "x",
            "--normalize",
            "zyphra",
            "--language",
            "en_US",
            "It costs $5.",
        ]);
        if cfg!(feature = "speech") {
            assert_eq!(result.unwrap(), "It costs five dollars.\n");
            let error = run(&["x", "--normalize", "zyphra", "$5"]).unwrap_err();
            assert_eq!(error.kind(), bunsen::errors::BunsenErrorKind::Illegal);
        } else {
            let error = result.unwrap_err();
            assert_eq!(error.kind(), bunsen::errors::BunsenErrorKind::Unsupported);
        }
    }
}
