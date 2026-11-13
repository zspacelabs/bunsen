//! `bunsen-tool summary`: the builds this binary can run, in brief, by kind,
//! family and catalog namespace.
//!
//! An ASCII bulleted list, one entry per line, its description wrapped
//! below it:
//!
//! ```text
//! - asr: speech to text, who spoke when, and voice activity
//!   - sortformer: [catalog, hf; 3 builds]
//!       A Sortformer speaker diarizer: a `.nemo` archive, which carries its
//!       config, or transformers' safetensors.
//!     - nvidia/sortformer/: [3 builds]
//!       - diar_streaming_sortformer_4spk-v2: [471.4 MB f32; latency: ultra-low|*low|high|offline; ...]
//!           Streaming Sortformer v2: 117 M parameters, up to 4 speakers, 80 ms frames.
//! ```
//!
//! A namespace that holds one entry folds into it (`fermion/phonon-2`); one
//! that holds more is its own entry, with its description. A build's details
//! are its size, its dtypes, each preset axis with its default starred, any
//! preset on no axis, and its aliases.

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
        resources::ResourceCache,
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
use bunsen_app::cache::{
    WeightsCacheArgs,
    human_bytes,
};

/// `bunsen-tool summary`'s flags.
#[derive(clap::Args, Debug)]
pub struct SummaryCommand {
    #[clap(flatten)]
    cache: WeightsCacheArgs,

    /// The column descriptions wrap at; 0 does not wrap.
    #[arg(long, default_value_t = 100)]
    width: usize,

    /// One kind alone: `asr`, `chat` or `decide`.
    #[arg(long)]
    kind: Option<String>,
}

impl SummaryCommand {
    /// Writes the summary to stdout.
    pub fn run(&self) -> BunsenResult<()> {
        let cache = self.cache.init()?;
        let mut kinds = kinds()?;
        if let Some(only) = &self.kind {
            kinds.retain(|k| k.name == only);
        }
        let text = summary(&cache, &kinds, &builtin_stack()?, self.width);
        use std::io::Write;
        std::io::stdout()
            .write_all(text.as_bytes())
            .map_err(sys_at("write to", "stdout"))
    }
}

/// A kind of model, `asr`, `chat` or `decide`, and its families.
pub struct Kind {
    /// Its command's name.
    pub name: &'static str,

    /// What it does, for a person.
    pub description: &'static str,

    /// Its families, in order.
    pub families: Vec<Family>,
}

/// A family's providers and builds, as its default factory lists them.
pub struct Family {
    /// The family's name.
    pub name: &'static str,

    /// The providers its factory searches, in order: name and description.
    pub providers: Vec<(String, String)>,

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
            providers: factory
                .providers()
                .iter()
                .map(|p| (p.name().to_string(), p.description().to_string()))
                .collect(),
            builds: factory.list().into_iter().map(|(_, b)| b).collect(),
        }
    }
}

/// The kinds this build knows, as `card`'s `kits` lists them.
///
/// # Errors
/// As a family's default factory, if two of its providers share a name.
pub fn kinds() -> BunsenResult<Vec<Kind>> {
    let mut kinds = Vec::new();
    #[cfg(feature = "speech")]
    kinds.push(Kind {
        name: "asr",
        description: "speech to text, who spoke when, and voice activity",
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
            description: "chat and text generation",
            families: vec![
                Family::of("qwen3", default_qwen3_factory()?),
                Family::of("qwen3.5", default_qwen3_5_factory()?),
                Family::of("gemma", default_gemma_factory()?),
                Family::of("nanochat", default_nanochat_factory()?),
            ],
        },
        Kind {
            name: "decide",
            description: "typed questions about a context, answered with probabilities",
            families: vec![Family::of("laya", default_laya_factory()?)],
        },
    ]);
    Ok(kinds)
}

/// The summary's text: a header line for the build and its cache, then each
/// kind, family, namespace and build as a bulleted entry.
pub fn summary(
    cache: &ResourceCache,
    kinds: &[Kind],
    stack: &BunsenCatalogStack,
    width: usize,
) -> String {
    let mut out = Out {
        text: String::new(),
        width,
    };
    let features: Vec<&str> = [
        ("speech", cfg!(feature = "speech")),
        ("live", cfg!(feature = "live")),
        ("webhook", cfg!(feature = "webhook")),
        ("mqtt", cfg!(feature = "mqtt")),
        ("cuda", cfg!(feature = "cuda")),
        ("metal", cfg!(feature = "metal")),
        ("vulkan", cfg!(feature = "vulkan")),
        ("wgpu", cfg!(feature = "wgpu")),
    ]
    .into_iter()
    .filter_map(|(name, on)| on.then_some(name))
    .collect();
    out.text.push_str(&format!(
        "{} {}: [{}]\n  cache: {}{}\n",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION"),
        features.join(", "),
        cache.cache_dir().display(),
        if cache.offline() { " (offline)" } else { "" },
    ));
    for kind in kinds {
        out.text.push('\n');
        out.entry(0, kind.name, None, Some(kind.description));
        for family in &kind.families {
            let providers: Vec<&str> = family.providers.iter().map(|(n, _)| n.as_str()).collect();
            let details = format!("{}; {}", providers.join(", "), count(family.builds.len()));
            // The first provider, the catalog's, says what the family loads.
            let description = family.providers.first().map(|(_, d)| d.as_str());
            out.entry(1, family.name, Some(&details), description);
            let mut root = Node::default();
            for (i, build) in family.builds.iter().enumerate() {
                root.insert(build.name.segments(), i);
            }
            for (segment, child) in &root.children {
                out.node(2, &[], segment, child, &family.builds, stack);
            }
        }
    }
    out.text
}

/// `n builds`.
fn count(n: usize) -> String {
    match n {
        1 => "1 build".to_string(),
        n => format!("{n} builds"),
    }
}

/// A namespace's entries, by segment, in the order their first build was
/// listed.
#[derive(Default)]
struct Node {
    children: Vec<(String, Child)>,
}

enum Child {
    Namespace(Node),
    /// A build, by its index in its family's list.
    Build(usize),
}

impl Node {
    fn insert(
        &mut self,
        segments: &[String],
        build: usize,
    ) {
        let [head, rest @ ..] = segments else {
            return;
        };
        if rest.is_empty() {
            self.children.push((head.clone(), Child::Build(build)));
            return;
        }
        let at = match self
            .children
            .iter()
            .position(|(s, c)| s == head && matches!(c, Child::Namespace(_)))
        {
            Some(at) => at,
            None => {
                self.children
                    .push((head.clone(), Child::Namespace(Node::default())));
                self.children.len() - 1
            }
        };
        if let Child::Namespace(node) = &mut self.children[at].1 {
            node.insert(rest, build);
        }
    }

    /// The builds under it, at any depth.
    fn builds(&self) -> usize {
        self.children
            .iter()
            .map(|(_, c)| match c {
                Child::Namespace(n) => n.builds(),
                Child::Build(_) => 1,
            })
            .sum()
    }
}

/// The text being written, and the column descriptions wrap at.
struct Out {
    text: String,
    width: usize,
}

impl Out {
    /// `{indent}- {name}: [{details}]`, then the description wrapped below
    /// it, two levels deeper than the bullet.
    fn entry(
        &mut self,
        depth: usize,
        name: &str,
        details: Option<&str>,
        description: Option<&str>,
    ) {
        let indent = "  ".repeat(depth);
        self.text.push_str(&format!("{indent}- {name}"));
        match details {
            Some(details) if !details.is_empty() => {
                self.text.push_str(&format!(": [{details}]"));
            }
            _ => {}
        }
        self.text.push('\n');
        if let Some(description) = description.map(str::trim).filter(|d| !d.is_empty()) {
            let indent = format!("{indent}    ");
            for line in wrap(description, self.width.saturating_sub(indent.len())) {
                self.text.push_str(&format!("{indent}{line}\n"));
            }
        }
    }

    /// The entry `child`, `segment` under the namespace `scope`: a chain of
    /// namespaces that each hold one entry folds into one label.
    fn node(
        &mut self,
        depth: usize,
        scope: &[String],
        segment: &str,
        child: &Child,
        builds: &[ResolvedBuild],
        stack: &BunsenCatalogStack,
    ) {
        let mut path: Vec<String> = scope.to_vec();
        path.push(segment.to_string());
        let mut label = segment.to_string();
        let mut at = child;
        while let Child::Namespace(node) = at
            && let [(next, only)] = node.children.as_slice()
        {
            label = format!("{label}/{next}");
            path.push(next.clone());
            at = only;
        }
        match at {
            Child::Build(i) => {
                let build = &builds[*i];
                self.entry(
                    depth,
                    &label,
                    Some(&build_details(build)),
                    Some(&build.description),
                );
            }
            Child::Namespace(node) => {
                let description = CatalogName::parse(&path.join("/"))
                    .ok()
                    .and_then(|name| stack.declaration(&name))
                    .and_then(|entry| match entry.body() {
                        CatalogEntry::Namespace(ns) => ns.description.clone(),
                        _ => None,
                    });
                self.entry(
                    depth,
                    &format!("{label}/"),
                    Some(&count(node.builds())),
                    description.as_deref(),
                );
                for (segment, child) in &node.children {
                    self.node(depth + 1, &path, segment, child, builds, stack);
                }
            }
        }
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

    /// The summary lists every kind, every family under it, and every build
    /// once; one-entry namespaces fold, and axes star their default.
    #[cfg(feature = "speech")]
    #[test]
    fn test_summary() {
        let cache = WeightsCacheArgs {
            cache_dir: Some(std::env::temp_dir().join("bunsen-tool-summary-test")),
            offline: true,
        }
        .init()
        .unwrap();
        let kinds = kinds().unwrap();
        let text = summary(&cache, &kinds, &builtin_stack().unwrap(), 100);
        for kind in &kinds {
            assert!(text.contains(&format!("\n- {}\n", kind.name)), "{text}");
            for family in &kind.families {
                assert!(
                    text.contains(&format!("\n  - {}: [", family.name)),
                    "{text}"
                );
                for build in &family.builds {
                    assert!(
                        text.contains(&format!("{}: [", build.name.leaf())),
                        "{}",
                        build.name
                    );
                }
            }
        }
        // fermion holds one parakeet build: it folds.
        assert!(text.contains("    - fermion/phonon-2: ["), "{text}");
        // whisper's builds sit under openai/whisper/, a namespace of many.
        assert!(text.contains("    - openai/whisper/: ["), "{text}");
        assert!(text.contains("task: *transcribe|translate"), "{text}");
        assert!(text.contains("aka sortformer-v2"), "{text}");
        assert!(text.contains("(offline)"), "{text}");
        // No line, but a long build line, runs past the width.
        for line in text.lines().filter(|l| !l.trim_start().starts_with("- ")) {
            assert!(line.chars().count() <= 100 || !line.contains(' '), "{line}");
        }
    }

    /// The summary's families are the card's.
    #[cfg(feature = "speech")]
    #[test]
    fn test_summary_families_are_the_cards() {
        let card = crate::card::card(
            &WeightsCacheArgs {
                cache_dir: Some(std::env::temp_dir().join("bunsen-tool-summary-test")),
                offline: true,
            }
            .init()
            .unwrap(),
            false,
        )
        .unwrap();
        let Some(crate::card::Value::Map(card_kinds)) = card.get("kits") else {
            panic!("kits");
        };
        let ours: Vec<(&str, Vec<&str>)> = kinds()
            .unwrap()
            .iter()
            .map(|k| (k.name, k.families.iter().map(|f| f.name).collect()))
            .collect();
        let theirs: Vec<(&str, Vec<&str>)> = card_kinds
            .iter()
            .map(|(k, v)| {
                let crate::card::Value::Map(families) = v else {
                    panic!("{k}");
                };
                (
                    k.as_str(),
                    families.iter().map(|(f, _)| f.as_str()).collect(),
                )
            })
            .collect();
        assert_eq!(ours, theirs);
    }
}
