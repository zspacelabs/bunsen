//! `bunsen-tool decide`: typed questions about a context, answered with
//! calibrated probabilities by a Laya decision model. `decide run` reads a
//! request (YAML or JSON: a `context` and its `choices`) and writes one
//! result per choice; `decide list` and `decide fetch` are its checkpoints'.
use std::io::Read;

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
            pretrained::{
                DEFAULT_LAYA_MODEL,
                LAYA_KIT,
                default_laya_factory,
            },
        },
        support::construct::LoadPolicy,
    },
};
use bunsen_app::{
    cache::WeightsCacheArgs,
    device::BackendArgs,
    logging::{
        LevelFilter,
        LogArgs,
        LogPolicy,
    },
};

use crate::{
    card::Format,
    chat::HfTokenizer,
};

/// `bunsen-tool decide`'s commands.
#[derive(clap::Args, Debug)]
pub struct DecideCmd {
    #[clap(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand, Debug)]
enum Command {
    /// Answers a request's questions: a YAML or JSON document, from a file
    /// or stdin, of a `context` and its `choices`, each a `type` (`choice`,
    /// `score` or `noul`), a `question`, and its `options`; one result per
    /// choice, in order.
    Run(RunCmd),

    /// The checkpoints bunsen knows by name: each one's origin, and each of
    /// its files' path in its repo and whether it is cached.
    List(ListCmd),

    /// Brings every file of the named checkpoints into the weights cache.
    Fetch(FetchCmd),
}

impl DecideCmd {
    /// Runs the command.
    pub fn run(&self) -> BunsenResult<()> {
        match &self.command {
            Command::Run(cmd) => cmd.run(),
            Command::List(cmd) => cmd.run(),
            Command::Fetch(cmd) => cmd.run(),
        }
    }
}

#[derive(clap::Args, Debug)]
struct RunCmd {
    #[clap(flatten)]
    logging: LogArgs,

    #[clap(flatten)]
    cache: WeightsCacheArgs,

    /// The device. The checkpoint loads at the precision it ships in (f16)
    /// where the device accelerates it, else in f32
    /// (`LoadPolicy::performance`).
    #[clap(flatten)]
    device: BackendArgs,

    /// The checkpoint: a name `list` shows.
    #[arg(long, default_value = DEFAULT_LAYA_MODEL)]
    model: String,

    /// The results' format: YAML, or JSON on one line.
    #[arg(long, value_enum, default_value_t = Format::Yml)]
    format: Format,

    /// The request; stdin when omitted.
    input: Option<std::path::PathBuf>,
}

impl RunCmd {
    fn run(&self) -> BunsenResult<()> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Warn, LevelFilter::Warn))?;
        let request = read_request(self.input.as_deref())?;
        let cache = self.cache.init()?;
        let device = self.device.open()?;
        let construct = default_laya_factory()?.load_construct(
            &self.model,
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
fn read_request(path: Option<&std::path::Path>) -> BunsenResult<LayaRequest> {
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
    device: &burn::tensor::Device,
) -> BunsenResult<Vec<LayaAnswer>> {
    let missing =
        |what: &str| BunsenError::invalid_resource(format!("a Laya checkpoint without its {what}"));
    let tokenizer = HfTokenizer::from_file(
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

#[derive(clap::Args, Debug)]
struct ListCmd {
    #[clap(flatten)]
    logging: LogArgs,

    #[clap(flatten)]
    cache: WeightsCacheArgs,

    /// Also print each file's first source, where it is fetched from (as
    /// `url: ...`).
    #[arg(long)]
    urls: bool,
}

impl ListCmd {
    fn run(&self) -> BunsenResult<()> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Info, LevelFilter::Info))?;
        let cache = self.cache.init()?;
        println!("cache: {}", cache.cache_dir().display());
        println!();
        println!("# laya checkpoints (the default: {DEFAULT_LAYA_MODEL})");
        let factory = default_laya_factory()?;
        for id in factory.ids() {
            let (_, row) = factory.lookup(&id)?;
            crate::chat::cli::print_row(&cache, LAYA_KIT, &id, &row, DEFAULT_LAYA_MODEL, self.urls);
        }
        Ok(())
    }
}

#[derive(clap::Args, Debug)]
struct FetchCmd {
    #[clap(flatten)]
    logging: LogArgs,

    #[clap(flatten)]
    cache: WeightsCacheArgs,

    /// Checkpoint names, as `--model` takes them (`list` shows them).
    #[arg(required = true)]
    names: Vec<String>,
}

impl FetchCmd {
    fn run(&self) -> BunsenResult<()> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Info, LevelFilter::Info))?;
        let cache = self.cache.init()?;
        let factory = default_laya_factory()?;
        for name in &self.names {
            let model = factory.resolve(name, &cache)?;
            let map = model.plan(&cache)?;
            let loaded = cache.load(LAYA_KIT, &map)?;
            println!("{}:", model.id());
            for (key, part) in loaded.iter() {
                println!("  {key}: {}", part.path.display());
            }
        }
        Ok(())
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
