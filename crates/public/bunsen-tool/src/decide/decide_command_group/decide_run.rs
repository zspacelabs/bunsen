//! `decide run`.

use std::{
    io::Read,
    path::{
        Path,
        PathBuf,
    },
};

use bunsen::{
    errors::{
        BunsenError,
        BunsenResult,
        sys_at,
    },
    kits::{
        decision::laya::{
            LayaAnswer,
            LayaConstruct,
            LayaRequest,
            LayaSpecials,
            pretrained::default_laya_factory,
        },
        support::{
            construct::LoadPolicy,
            tokens::BunsenTokenizer,
        },
    },
};
use bunsen_app::{
    cache::WeightsCacheArgs,
    device::BackendArgs,
    logging::{
        LogArgs,
        LogPolicy,
    },
};
use burn::tensor::Device;
use clap::ArgMatches;
use log::LevelFilter;

use crate::{
    decide::choose_decide_model,
    support::{
        ChooseArgs,
        Format,
        with_patch,
        with_presets,
    },
};

/// Answers a request's questions.
#[derive(clap::Args, Debug)]
pub struct DecideRunCommand {
    #[clap(flatten)]
    logging: LogArgs,

    #[clap(flatten)]
    cache: WeightsCacheArgs,

    /// The device. The checkpoint loads at the precision it ships in (f16)
    /// where the device accelerates it, else in f32
    /// (`LoadPolicy::performance`).
    #[clap(flatten)]
    device: BackendArgs,

    /// The checkpoint, chosen among Laya's builds by `--model` and the stage
    /// flags; the default checkpoint when nothing is named.
    #[clap(flatten)]
    choose: ChooseArgs,

    /// The results' format: YAML, or JSON on one line.
    #[arg(long, value_enum, default_value_t = Format::Yml)]
    format: Format,

    /// The request; stdin when omitted.
    input: Option<PathBuf>,
}

impl DecideRunCommand {
    /// Answers the request; `matches` are the command's own, for the stage
    /// flags' order.
    pub fn run(
        &self,
        matches: &ArgMatches,
    ) -> BunsenResult<()> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Warn, LevelFilter::Warn))?;
        let request = read_request(self.input.as_deref())?;
        let cache = self.cache.init()?;
        let device = self.device.open()?;
        let flags = self.choose.flags(matches)?;
        let mut model = choose_decide_model(&self.choose.model, &flags)?;
        if !flags.presets.presets.is_empty() {
            let presets: Vec<&str> = flags.presets.presets.iter().map(String::as_str).collect();
            model = with_presets(&model, &presets)?;
        }
        if let Some(patch) = self.choose.patch()? {
            model = with_patch(&model, &patch)?;
        }
        let construct = default_laya_factory()?.load_construct(
            &model,
            &cache,
            &device,
            LoadPolicy::performance(),
        )?;
        let answers = decide(&construct, &request, &device)?;
        print!("{}", render(&answers, self.format)?);
        Ok(())
    }
}

/// A request, from `path` or stdin, as YAML (JSON among it).
///
/// # Errors
/// [`InvalidResource`](bunsen::errors::BunsenErrorKind::InvalidResource) for
/// a document that is not a request; a read's failure.
fn read_request(path: Option<&Path>) -> BunsenResult<LayaRequest> {
    let (text, source) = match path {
        Some(path) => (
            std::fs::read_to_string(path).map_err(sys_at("read", path))?,
            path.display().to_string(),
        ),
        None => {
            let mut text = String::new();
            std::io::stdin()
                .read_to_string(&mut text)
                .map_err(sys_at("read", "stdin"))?;
            (text, "stdin".to_string())
        }
    };
    parse_request(&text).map_err(|e| e.context(source))
}

/// A request, parsed from YAML (JSON among it).
///
/// # Errors
/// [`InvalidResource`](bunsen::errors::BunsenErrorKind::InvalidResource) for
/// a document that is not a request.
fn parse_request(text: &str) -> BunsenResult<LayaRequest> {
    serde_saphyr::from_str(text)
        .map_err(|e| BunsenError::invalid_resource(format!("a decision request: {e}")))
}

/// The request's questions answered by `construct`'s model, its tokenizer
/// and specials read from its files.
///
/// # Errors
/// As the tokenizer's files, and
/// [`LayaModel::decide`](bunsen::kits::decision::laya::LayaModel::decide).
fn decide(
    construct: &LayaConstruct,
    request: &LayaRequest,
    device: &Device,
) -> BunsenResult<Vec<LayaAnswer>> {
    let missing =
        |what: &str| BunsenError::invalid_resource(format!("a Laya checkpoint without its {what}"));
    let tokenizer = BunsenTokenizer::from_file(
        construct
            .tokenizer()
            .ok_or_else(|| missing("tokenizer.json"))?,
    )?;
    let config_path = construct
        .tokenizer_config()
        .ok_or_else(|| missing("tokenizer_config.json"))?;
    let config = std::fs::read_to_string(config_path).map_err(sys_at("read", config_path))?;
    let specials = LayaSpecials::from_tokenizer_config(&config, &tokenizer)?;
    construct.model.decide(
        &tokenizer,
        &specials,
        &construct.agent,
        &request.context,
        &request.choices,
        device,
    )
}

/// The answers, in `format`.
///
/// # Errors
/// [`Internal`](bunsen::errors::BunsenErrorKind::Internal) if they do not
/// serialize.
fn render(
    answers: &[LayaAnswer],
    format: Format,
) -> BunsenResult<String> {
    match format {
        Format::Yml => serde_saphyr::to_string(&answers)
            .map_err(|e| BunsenError::internal(format!("render the results as YAML: {e}"))),
        Format::Json => serde_json::to_string(&answers)
            .map(|json| json + "\n")
            .map_err(|e| BunsenError::internal(format!("render the results as JSON: {e}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REQUEST: &str = r#"
context: "the call failed twice"
choices:
  - type: choice
    question: "which next?"
    options: [retry, {escalate: "hand it to a person"}, close]
  - type: noul
    question: "is the user satisfied?"
  - type: score
    question: "how urgent?"
    options: [low, high]
"#;

    /// A YAML request reads into the kit's own types; a JSON one too.
    #[test]
    fn test_parse_request() {
        let request = parse_request(REQUEST).unwrap();
        assert_eq!(request.context, "the call failed twice");
        assert_eq!(request.choices.len(), 3);
        assert_eq!(
            request.choices[0].rendered_options(),
            ["retry", "escalate: hand it to a person", "close"]
        );
        let json = serde_json::to_string(&request).unwrap();
        assert_eq!(parse_request(&json).unwrap(), request);
        assert!(
            parse_request("choices: []").is_err(),
            "a request has a context"
        );
    }

    /// The results render with each option's probability by its key, in
    /// order.
    #[test]
    fn test_render() {
        let request = parse_request(REQUEST).unwrap();
        let answers = vec![LayaAnswer::new(
            &request.choices[0],
            vec![0.25, 0.5, 0.25],
            0.1,
        )];
        let yaml = render(&answers, Format::Yml).unwrap();
        assert!(yaml.contains("answer: escalate"), "{yaml}");
        let retry = yaml.find("retry").unwrap();
        let close = yaml.find("close").unwrap();
        assert!(retry < close, "in order: {yaml}");
        let json = render(&answers, Format::Json).unwrap();
        assert!(
            json.contains(r#""probabilities":{"retry":0.25,"escalate":0.5,"close":0.25}"#),
            "{json}"
        );
    }
}
