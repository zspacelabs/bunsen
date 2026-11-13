//! `bunsen-tool summary`: the builds this binary can run, in brief: by
//! command, contract and publisher.
//!
//! One section per command that runs builds (`asr`, `chat`, `decide`); in
//! each, the builds grouped by their contract, then by their publisher (the
//! first segment of their name), each a line of details and its description
//! wrapped below it:
//!
//! ```text
//! # bunsen-tool asr ...
//! # Automated Speech Recognition
//!
//! # contract: parakeet-asr - A Parakeet TDT speech recognizer: ...
//! # publisher: fermion - FermionResearch, Phonon's authors.
//!
//! * fermion/phonon-2: [163.5 MB; aka phonon]
//!   Phonon-2: parakeet-tdt-0.6b-v3 with a five-value encoder, English.
//! ```
//!
//! A build's details are its size, its dtypes, each preset axis with its
//! default starred, any preset on no axis, and its aliases.

#[cfg(feature = "speech")]
use bunsen::kits::speech::{
    asr::{
        parakeet::pretrained::default_parakeet_factory,
        qwen3_asr::pretrained::default_qwen3_asr_factory,
        voxtral::pretrained::default_voxtral_factory,
        whisper::pretrained::default_whisper_factory,
    },
    diarization::sortformer::pretrained::default_sortformer_factory,
    vad::silero_vad::pretrained::default_silero_factory,
};
use bunsen::{
    data::{
        builds::{
            BuildFactory,
            ConstructInit,
        },
        catalog::{
            BunsenCatalogStack,
            CatalogName,
            ResolvedBuild,
            builtin_stack,
            document::CatalogEntry,
            presets::ResolvedPresetSet,
        },
    },
    errors::{
        BunsenResult,
        sys_at,
    },
    kits::{
        decision::laya::pretrained::default_laya_factory,
        language::{
            gemma::pretrained::default_gemma_factory,
            nanochat::pretrained::default_nanochat_factory,
            qwen3::pretrained::default_qwen3_factory,
            qwen3_5::pretrained::default_qwen3_5_factory,
        },
    },
};
use bunsen_app::cache::human_bytes;

/// The commands that run builds, as `--kind` takes them.
#[cfg(feature = "speech")]
const KIND_NAMES: &[&str] = &["asr", "chat", "decide"];
#[cfg(not(feature = "speech"))]
const KIND_NAMES: &[&str] = &["chat", "decide"];

/// `bunsen-tool summary`'s flags.
#[derive(clap::Args, Debug)]
pub struct SummaryCommand {
    /// The column descriptions wrap at; 0 does not wrap.
    #[arg(long, default_value_t = 75)]
    width: usize,

    /// One command's builds alone: `asr`, `chat` or `decide`.
    #[arg(long, value_parser = clap::builder::PossibleValuesParser::new(KIND_NAMES))]
    kind: Option<String>,
}

impl SummaryCommand {
    /// Writes the summary to stdout.
    pub fn run(&self) -> BunsenResult<()> {
        let mut kinds = kinds()?;
        if let Some(only) = &self.kind {
            kinds.retain(|k| k.name == only);
        }
        let text = summary(&kinds, &builtin_stack()?, self.width);
        use std::io::Write;
        std::io::stdout()
            .write_all(text.as_bytes())
            .map_err(sys_at("write to", "stdout"))
    }
}

/// A command that runs builds, `asr`, `chat` or `decide`, and its families.
pub struct Kind {
    /// The command's name.
    pub name: &'static str,

    /// What it does, as a title.
    pub title: &'static str,

    /// Its families, in order.
    pub families: Vec<Family>,
}

/// A family's builds, as its default factory lists them.
pub struct Family {
    /// The family's name.
    pub name: &'static str,

    /// Its first provider's description: what the family loads.
    pub description: String,

    /// The builds its factory lists, in order.
    pub builds: Vec<ResolvedBuild>,
}

impl Family {
    fn of<H: ConstructInit>(
        name: &'static str,
        factory: BuildFactory<H>,
    ) -> Self {
        Self {
            name,
            description: factory
                .providers()
                .first()
                .map(|p| p.description().to_string())
                .unwrap_or_default(),
            builds: factory.list().into_iter().map(|(_, b)| b).collect(),
        }
    }
}

/// The commands that run builds, and the families each one loads.
///
/// # Errors
/// As a family's default factory, if two of its providers share a name.
pub fn kinds() -> BunsenResult<Vec<Kind>> {
    let mut kinds = Vec::new();
    #[cfg(feature = "speech")]
    kinds.push(Kind {
        name: "asr",
        title: "Automated Speech Recognition",
        families: vec![
            Family::of("parakeet", default_parakeet_factory()?),
            Family::of("qwen3-asr", default_qwen3_asr_factory()?),
            Family::of("voxtral", default_voxtral_factory()?),
            Family::of("whisper", default_whisper_factory()?),
            Family::of("sortformer", default_sortformer_factory()?),
            Family::of("silero-vad", default_silero_factory()?),
        ],
    });
    kinds.extend([
        Kind {
            name: "chat",
            title: "Chat and Text Generation",
            families: vec![
                Family::of("qwen3", default_qwen3_factory()?),
                Family::of("qwen3.5", default_qwen3_5_factory()?),
                Family::of("gemma", default_gemma_factory()?),
                Family::of("nanochat", default_nanochat_factory()?),
            ],
        },
        Kind {
            name: "decide",
            title: "Typed Decisions",
            families: vec![Family::of("laya", default_laya_factory()?)],
        },
    ]);
    Ok(kinds)
}

/// A build and the family that lists it.
type Listed<'a> = (&'a Family, &'a ResolvedBuild);

/// Builds under one header: its name and description, and the builds, in
/// the order they were listed.
struct Group<'a> {
    name: String,
    description: String,
    builds: Vec<Listed<'a>>,
}

/// `builds` grouped under `header`'s name and description, in the order
/// each name first appears.
fn group_by<'a>(
    builds: impl IntoIterator<Item = Listed<'a>>,
    header: impl Fn(Listed<'a>) -> (String, String),
) -> Vec<Group<'a>> {
    let mut groups: Vec<Group<'a>> = Vec::new();
    for listed in builds {
        let (name, description) = header(listed);
        match groups.iter_mut().find(|g| g.name == name) {
            Some(group) => group.builds.push(listed),
            None => groups.push(Group {
                name,
                description,
                builds: vec![listed],
            }),
        }
    }
    groups
}

/// The description of the entry `name` in `stack`, when it is a namespace
/// or a contract and has one.
fn entry_description(
    stack: &BunsenCatalogStack,
    name: &CatalogName,
) -> Option<String> {
    match stack.declaration(name)?.body() {
        CatalogEntry::Namespace(ns) => ns.description.clone(),
        CatalogEntry::Contract(c) => Some(c.description.clone()),
        _ => None,
    }
}

/// A build's contract header: the top contract it is under (`qwen3-asr` for
/// `$contracts/qwen3-asr/probed`) and its description; for a build with
/// none, its family's name and description.
fn contract_header(
    stack: &BunsenCatalogStack,
    (family, build): Listed<'_>,
) -> (String, String) {
    let top = build
        .contract
        .as_ref()
        .and_then(|c| c.segments().get(..2))
        .and_then(|s| CatalogName::parse(&s.join("/")).ok());
    match top {
        Some(top) => (
            top.leaf().to_string(),
            entry_description(stack, &top).unwrap_or_default(),
        ),
        None => (family.name.to_string(), family.description.clone()),
    }
}

/// A build's publisher header: the first segment of its name, and that
/// namespace's description.
fn publisher_header(
    stack: &BunsenCatalogStack,
    (_, build): Listed<'_>,
) -> (String, String) {
    let publisher = build.name.segments()[0].clone();
    let description = CatalogName::parse(&publisher)
        .ok()
        .and_then(|name| entry_description(stack, &name))
        .unwrap_or_default();
    (publisher, description)
}

/// The summary's text: for each kind, its command and title, then its
/// builds by contract and publisher, descriptions wrapped at `width`.
pub fn summary(
    kinds: &[Kind],
    stack: &BunsenCatalogStack,
    width: usize,
) -> String {
    let mut out = String::new();
    for (i, kind) in kinds.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str(&format!(
            "# bunsen-tool {} ...\n# {}\n",
            kind.name, kind.title
        ));
        let listed = kind
            .families
            .iter()
            .flat_map(|f| f.builds.iter().map(move |b| (f, b)));
        for contract in group_by(listed, |l| contract_header(stack, l)) {
            out.push('\n');
            out.push_str(&header("contract", &contract.name, &contract.description));
            let publishers = group_by(contract.builds, |l| publisher_header(stack, l));
            for (j, publisher) in publishers.iter().enumerate() {
                if j > 0 {
                    out.push('\n');
                }
                out.push_str(&header(
                    "publisher",
                    &publisher.name,
                    &publisher.description,
                ));
                for (_, build) in &publisher.builds {
                    out.push_str(&format!("\n* {}: [{}]\n", build.name, build_details(build)));
                    for line in wrap(build.description.trim(), width.saturating_sub(2)) {
                        out.push_str(&format!("  {line}\n"));
                    }
                }
            }
        }
    }
    out
}

/// `# {what}: {name} - {description}`, without ` - ` when there is no
/// description.
fn header(
    what: &str,
    name: &str,
    description: &str,
) -> String {
    match description.trim() {
        "" => format!("# {what}: {name}\n"),
        description => format!("# {what}: {name} - {description}\n"),
    }
}

/// A build's details: its size and dtypes, its presets ([`presets_details`]),
/// and its aliases.
fn build_details(build: &ResolvedBuild) -> String {
    let resources = build.resource_set();
    let mut parts = Vec::new();
    let mut files: Vec<String> = resources.size().map(human_bytes).into_iter().collect();
    files.extend(
        resources
            .dtypes()
            .into_iter()
            .map(|d| short_dtype(d).to_string()),
    );
    if !files.is_empty() {
        parts.push(files.join(" "));
    }
    parts.extend(presets_details(&build.presets));
    if !build.aliases.is_empty() {
        let aliases: Vec<&str> = build.aliases.iter().map(|a| a.leaf()).collect();
        parts.push(format!("aka {}", aliases.join(", ")));
    }
    parts.join("; ")
}

/// Each axis, `{axis}: {member}|..`, its default starred when it offers more
/// than one; then the presets on no axis, `+{preset}`, starred when they
/// apply unless selected over.
fn presets_details(set: &ResolvedPresetSet) -> Vec<String> {
    let mut parts: Vec<String> = set
        .axes
        .iter()
        .map(|a| {
            let members: Vec<String> = a
                .offers
                .iter()
                .map(|m| {
                    if !a.is_fixed() && *m == a.default {
                        format!("*{m}")
                    } else {
                        m.clone()
                    }
                })
                .collect();
            format!("{}: {}", a.axis.name.leaf(), members.join("|"))
        })
        .collect();
    let other: Vec<String> = set
        .presets
        .iter()
        .map(|p| &p.name)
        .filter(|n| set.axis_of(n).is_none())
        .map(|n| {
            if set.default.contains(n) {
                format!("+*{n}")
            } else {
                format!("+{n}")
            }
        })
        .collect();
    if !other.is_empty() {
        parts.push(other.join(" "));
    }
    parts
}

/// The usual short name of a dtype (`bf16` for `bfloat16`).
fn short_dtype(dtype: &str) -> &str {
    match dtype {
        "float64" => "f64",
        "float32" => "f32",
        "float16" => "f16",
        "bfloat16" => "bf16",
        other => other,
    }
}

/// `text`'s words in lines of at most `width` characters, a longer word on
/// its own; one line when `width` is 0.
fn wrap(
    text: &str,
    width: usize,
) -> Vec<String> {
    let mut lines = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        if !line.is_empty() && width > 0 && line.chars().count() + 1 + word.chars().count() > width
        {
            lines.push(std::mem::take(&mut line));
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    if !line.is_empty() {
        lines.push(line);
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wrap() {
        assert_eq!(wrap("a bb ccc dddd", 6), ["a bb", "ccc", "dddd"]);
        assert_eq!(wrap("a bb ccc", 0), ["a bb ccc"]);
        assert_eq!(wrap("  ", 10), Vec::<String>::new());
        assert_eq!(wrap("toolongword x", 4), ["toolongword", "x"]);
    }

    /// Every kind heads its section; every build is listed once, by its
    /// full name, under its contract and publisher; descriptions fit the
    /// width; axes star their default.
    #[cfg(feature = "speech")]
    #[test]
    fn test_summary() {
        let kinds = kinds().unwrap();
        let text = summary(&kinds, &builtin_stack().unwrap(), 75);
        for kind in &kinds {
            assert!(
                text.contains(&format!(
                    "# bunsen-tool {} ...\n# {}\n",
                    kind.name, kind.title
                )),
                "{text}"
            );
            for family in &kind.families {
                for build in &family.builds {
                    assert_eq!(
                        text.matches(&format!("\n* {}: [", build.name)).count(),
                        1,
                        "{}",
                        build.name
                    );
                }
            }
        }
        assert!(
            text.contains(
                "# contract: parakeet-asr - A Parakeet TDT speech recognizer: a Phonon-2 \
                 archive, or transformers' safetensors and its tokenizer.\n\
                 # publisher: fermion - FermionResearch, Phonon's authors.\n\n\
                 * fermion/phonon-2: [163.5 MB; aka phonon]\n  Phonon-2: "
            ),
            "{text}"
        );
        // qwen3-asr's sub-contracts fold into one.
        assert_eq!(text.matches("# contract: qwen3-asr ").count(), 1, "{text}");
        assert!(text.contains("task: *transcribe|translate"), "{text}");
        assert!(text.contains("aka sortformer-v2"), "{text}");
        for line in text.lines().filter(|l| l.starts_with("  ")) {
            assert!(line.chars().count() <= 75, "{line}");
        }
    }
}
