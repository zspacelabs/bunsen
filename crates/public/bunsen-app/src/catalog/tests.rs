//! The commands, run over the example catalogs as files.

use serde_json::json;

use super::*;

/// A program of the one command.
#[derive(clap::Parser, Debug)]
#[command(name = "catalog")]
struct Cli {
    #[command(flatten)]
    command: CatalogCommand,
}

/// Parses `argv` (with the program name) and runs it into a string.
fn run_argv(argv: Vec<String>) -> Result<String, String> {
    use clap::{
        CommandFactory,
        FromArgMatches,
    };
    let matches = Cli::command()
        .try_get_matches_from(argv)
        .map_err(|e| e.to_string())?;
    let parsed = Cli::from_arg_matches(&matches).map_err(|e| e.to_string())?;
    let mut out = String::new();
    parsed.command.run_into(&matches, &mut out)?;
    Ok(out)
}

/// Runs the command line `args` (after the program name) over the
/// example catalogs, as files.
fn cli(args: &str) -> Result<String, String> {
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../bunsen/src/data/pretrained/build_catalog/fixtures"
    );
    let mut argv = vec!["catalog".to_string()];
    for file in ["prelude", "commons", "qwen3", "whisper", "supplemental"] {
        argv.push("--catalog".into());
        argv.push(format!("{dir}/{file}.yml"));
    }
    argv.extend(shell_words(args));
    run_argv(argv)
}

/// `args` split on spaces, keeping single-quoted runs together.
fn shell_words(args: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut word = String::new();
    let mut quoted = false;
    for c in args.chars() {
        match c {
            '\'' => quoted = !quoted,
            ' ' if !quoted => {
                if !word.is_empty() {
                    words.push(std::mem::take(&mut word));
                }
            }
            c => word.push(c),
        }
    }
    if !word.is_empty() {
        words.push(word);
    }
    words
}

#[test]
fn test_layers_and_lists() {
    let out = cli("layers").unwrap();
    assert_eq!(out.lines().count(), 5);
    assert!(out.contains("supplemental"));
    let builds = cli("list --kind build").unwrap();
    assert_eq!(builds.lines().count(), 8);
    let families = cli("list --facet family").unwrap();
    assert!(families.contains("qwen/qwen3") && families.contains("openai/whisper"));
    let members = cli("list --members release").unwrap();
    assert_eq!(members.lines().count(), 3);
    let classes = cli("list --children qwen3 --kind template_class").unwrap();
    assert!(classes.contains("qwen/qwen3/hybrid"));
    assert!(
        cli("list --kind family")
            .unwrap_err()
            .contains("no kind `family`")
    );
}

#[test]
fn test_show_tells_the_history() {
    let out = cli("show qwen").unwrap();
    assert!(out.contains("deny     in commons"));
    assert!(out.contains("update   in supplemental"));
    assert!(out.contains("facet: publisher"));
    assert!(cli("show nothing-here").is_err());
}

#[test]
fn test_resolve_with_shortcuts_and_a_patch() {
    let out =
        cli("resolve qwen3-0.6b-base --off thinking --patch '{generation: {temperature: 0.1}}'")
            .unwrap();
    assert!(out.contains("(fixed)"));
    assert!(out.contains("`--off thinking` implies `--preset no-thinking`"));
    assert!(out.contains("\"temperature\":0.1"));
    let err = cli("resolve qwen3-0.6b-base --on thinking").unwrap_err();
    assert!(err.contains("fixed at `no-thinking`"), "{err}");
}

/// The YAML report `select ARGS` prints.
fn report(args: &str) -> Value {
    let out = cli(&format!("select {args}")).unwrap();
    serde_saphyr::from_str(&out).unwrap_or_else(|e| panic!("{e}: {out}"))
}

fn field(
    report: &Value,
    key: &str,
) -> Vec<Value> {
    report["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r[key].clone())
        .collect()
}

#[test]
fn test_select_reports_the_query_why_and_the_result() {
    let all = report("qwen3 --off thinking");
    assert_eq!(
        field(&all, "name"),
        [
            "qwen/qwen3/qwen3-0.6b",
            "qwen/qwen3/qwen3-0.6b-base",
            "qwen/qwen3/qwen3-1.7b",
        ]
    );
    assert_eq!(all["presets"], json!(["no-thinking"]));
    assert_eq!(all["results"][0]["weight"], json!(1.0));
    assert_eq!(all["results"][0]["tags"]["thinking"], json!("switch"));
    assert_eq!(all["results"][0]["contract"], json!("qwen3-lm"));
    // The prefix is stage 0, attributed to the flag.
    assert_eq!(
        all["query"]["run"][0],
        json!({"keep": {"$presets": ["no-thinking"]}})
    );
    assert_eq!(all["why"][0]["origin"], json!("prefix: --off thinking"));
    assert_eq!(all["why"][0]["dropped"], json!(5));
    // The tail orders by weight when nothing else does.
    assert_eq!(all["why"][2]["origin"], json!("tail"));
}

#[test]
fn test_select_flags_are_stages_in_command_line_order() {
    // Ascending parameters: turbo (1.0), v1 (0.5), v2 (0.75), v3 (1.0);
    // the cutoff drops v1, the limit v3.
    let cut = report(
        "whisper --weigh '{release: latest, step: 0.25}' --asc parameters --cutoff 0.6 \
         --limit 2 --rank",
    );
    assert_eq!(field(&cut, "weight"), [1.0, 0.75]);
    let dropped: Vec<(String, String)> = cut["dropped"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|d| d["name"].as_str().unwrap().starts_with("openai"))
        .map(|d| {
            (
                d["name"]
                    .as_str()
                    .unwrap()
                    .rsplit('/')
                    .next()
                    .unwrap()
                    .to_string(),
                d["why"].as_str().unwrap().to_string(),
            )
        })
        .collect();
    assert_eq!(
        dropped,
        [
            (
                "large-v1".to_string(),
                "fails `keep: {$weight: {ge: 0.6}}`".to_string()
            ),
            ("large-v3".to_string(), "past limit".to_string()),
        ]
    );
    assert_eq!(
        cut["results"][1]["release"],
        json!({"group": "openai/whisper/large/release", "released": "2022-12"})
    );
    // Consecutive sort flags join one `order`.
    let sorted = report("whisper --desc '$weight' --desc parameters --weigh '{release: latest}'");
    assert_eq!(
        sorted["query"]["run"][1],
        json!({"order": [{"desc": "$weight"}, {"desc": "parameters"}]})
    );
    let tests = report("--le parameters 8.9e8 --has language en");
    assert_eq!(
        field(&tests, "name"),
        ["openai/whisper/large/large-v3-turbo"]
    );
}

#[test]
fn test_the_preset_prefix_holds_wherever_the_flags_sit() {
    // The prefix runs first, so the limit can't pick a build the preset
    // can't apply to.
    let late = report("qwen3 --limit 1 --off thinking");
    let early = report("qwen3 --off thinking --limit 1");
    assert_eq!(field(&late, "name"), ["qwen/qwen3/qwen3-0.6b"]);
    assert_eq!(field(&late, "name"), field(&early, "name"));
    // A pipeline whose own limit comes first: the same.
    let piped = report("--pipeline '[{from: qwen3}, {limit: 1}]' --off thinking");
    assert_eq!(field(&piped, "name"), ["qwen/qwen3/qwen3-0.6b"]);
    // The printed query, fed back with no preset flags, returns the same
    // rows.
    let query = serde_json::to_string(&late["query"]["run"]).unwrap();
    let again = report(&format!("--pipeline '{query}'"));
    assert_eq!(field(&again, "name"), field(&late, "name"));
    // A build named alone that can't take the presets is an error.
    let err = cli("select qwen3-4b-thinking-2507 --off thinking").unwrap_err();
    assert!(err.contains("was named, but can't be chosen"), "{err}");
    let dropped = report("whisper --preset translate --rank");
    assert_eq!(dropped["dropped"][0]["stage"], json!(0));
    assert!(
        cli("select --on task")
            .unwrap_err()
            .contains("not a bool axis")
    );
}

#[test]
fn test_select_per_and_unset() {
    let largest = report("--per family --desc parameters --limit 1 --end");
    assert_eq!(
        field(&largest, "name"),
        [
            "openai/whisper/large/large-v1",
            "qwen/qwen3/qwen3-4b-thinking-2507"
        ]
    );
    assert_eq!(
        largest["query"]["run"][0]["per"]["run"],
        json!([{"order": {"desc": "parameters"}}, {"limit": 1}])
    );
    assert!(
        cli("select --per family --limit 1")
            .unwrap_err()
            .contains("has no `--end`")
    );
    assert!(
        cli("select --end")
            .unwrap_err()
            .contains("closes no `--per`")
    );
    let all = report("--defaults '[{keep: instruct, as: tuned}]' whisper --unset tuned");
    assert_eq!(field(&all, "name").len(), 4);
}

#[test]
fn test_select_any_kind() {
    let classes = report("--kind template_class --keep thinking");
    assert_eq!(
        field(&classes, "name"),
        ["qwen/qwen3/hybrid", "qwen/qwen3/thinking-only"]
    );
    assert_eq!(classes["results"][0]["kind"], json!("template_class"));
    assert_eq!(classes["query"]["kind"], json!("template_class"));
    let resources = report("--kind resource qwen3-1.7b --eq '$format' json");
    assert_eq!(field(&resources, "name"), ["qwen/qwen3/qwen3-1.7b/config"]);
    assert!(
        cli("select --kind family")
            .unwrap_err()
            .contains("no kind `family`")
    );
}

#[test]
fn test_select_defaults_and_refusals() {
    let defaults = report("--defaults '[{keep: instruct}, {order: {desc: parameters}}]'");
    assert_eq!(
        field(&defaults, "name")[0],
        "qwen/qwen3/qwen3-4b-thinking-2507"
    );
    assert_eq!(defaults["why"][0]["origin"], json!("defaults"));
    // One build named alone skips them.
    let one = report("large-v1 --defaults '[{keep: instruct}]'");
    assert_eq!(field(&one, "name"), ["openai/whisper/large/large-v1"]);
    let err = cli("select --keep '{kit: qwen3}'").unwrap_err();
    assert!(err.contains("nothing matches `kit`"), "{err}");
    let typed = cli("select --keep '{instruct: maybe}'").unwrap_err();
    assert!(typed.contains("is not `true` or `false`"), "{typed}");
    let empty = cli("select qwen3 --cutoff 2").unwrap_err();
    assert!(empty.contains("no build is left"), "{empty}");
    assert!(empty.contains("--cutoff 2"), "{empty}");
    let unknown = cli("select --use nothing").unwrap_err();
    assert!(unknown.contains("no named selection"), "{unknown}");
}

#[test]
fn test_tree() {
    let out = cli("tree").unwrap();
    let lines: Vec<&str> = out.lines().collect();
    // The root's entries are the top level, with no line for the root.
    assert_eq!(lines[0], "qwen: Qwen, Alibaba Cloud's model team.");
    assert!(lines.contains(&"openai: OpenAI."), "{out}");
    // A namespace is its name and description; a build adds its axis
    // uses, default first.
    assert!(out.contains("└── qwen3: Qwen3."), "{out}");
    assert!(
        out.contains(
            "qwen3-0.6b: {thinking: thinking|no-thinking} Qwen3-0.6B: hybrid-thinking chat."
        ),
        "{out}"
    );
    assert!(
        out.contains("qwen3-0.6b-base: {thinking: no-thinking} Qwen3-0.6B-Base"),
        "{out}"
    );
    // Children from the supplemental catalog sit under the group it
    // extends.
    assert!(out.contains("qwen3-1.7b: "), "{out}");
    // Only namespaces and builds: no contracts, releases, template
    // classes, vocabulary or resources.
    for hidden in [
        "qwen3-lm",
        "release",
        "hybrid",
        "template_class_root",
        "parameters",
        "(",
    ] {
        assert!(
            !lines.iter().any(|l| l
                .trim_start_matches(['│', '├', '└', '─', ' '])
                .starts_with(hidden)),
            "{hidden}: {out}"
        );
    }

    let builds = cli("tree whisper --kind build").unwrap();
    let lines: Vec<&str> = builds.lines().collect();
    assert_eq!(lines[0], "openai/whisper: Whisper.");
    assert!(
        lines[1].ends_with("large: The large Whisper models."),
        "{builds}"
    );
    assert_eq!(lines.len(), 6, "{builds}");

    let all = cli("tree --all").unwrap();
    assert!(all.contains("\n$contracts\n"), "{all}");
    assert!(
        all.contains("whisper-asr (contract): [checkpoint, vocabulary] An OpenAI Whisper"),
        "{all}"
    );
    assert!(
        all.contains("checkpoint (resource): model.safetensors"),
        "{all}"
    );
    assert!(all.contains("family (facet): A line of related"), "{all}");
    assert!(all.contains("hybrid (template_class): {thinking"), "{all}");

    let shallow = cli("tree --depth 1").unwrap();
    assert_eq!(
        shallow,
        "qwen: Qwen, Alibaba Cloud's model team.\nopenai: OpenAI.\n"
    );
    assert!(cli("tree --kind nope").is_err());
}

#[test]
fn test_tree_shows_aliases() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/catalog/aliases.yml");
    let out = cli(&format!("--catalog {path} tree")).unwrap();
    // On the build's line, and as entries where they sit.
    assert!(
        out.contains("large-v3: newest, large {task: transcribe|translate} Whisper large-v3."),
        "{out}"
    );
    assert!(
        out.contains("└── large -> openai/whisper/large/large-v3"),
        "{out}"
    );
    assert!(
        out.lines()
            .any(|l| l == "newest -> openai/whisper/large/large-v3"),
        "{out}"
    );
    // An alias to a kind the view leaves out is left out with it.
    let builds = cli(&format!("--catalog {path} tree --kind namespace")).unwrap();
    assert!(!builds.contains("->"), "{builds}");
}

#[test]
fn test_dump_nests_each_catalog_in_order() {
    let yaml = cli("dump").unwrap();
    let docs: Value = serde_saphyr::from_str(&yaml).unwrap();
    let layers: Vec<&str> = docs
        .as_array()
        .unwrap()
        .iter()
        .map(|l| l["layer"].as_str().unwrap())
        .collect();
    assert_eq!(
        layers,
        ["prelude", "commons", "qwen3", "whisper", "supplemental"]
    );
    assert_eq!(docs[0]["catalog"]["thinking"]["$kind"], "axis");
    assert!(docs[0]["path"].as_str().unwrap().ends_with("prelude.yml"));
    // The keyed form, read back in its order, is the catalog imported.
    let OrderedNode::Seq(items) = OrderedNode::from_yaml(&yaml).unwrap() else {
        panic!("a sequence")
    };
    let node = items[2].get("catalog").unwrap().clone();
    let catalog =
        bunsen::data::pretrained::build_catalog::PretrainedCatalog::from_node(node).unwrap();
    let stack = fixtures::supplemented_example_stack().unwrap();
    assert_eq!(&catalog, stack.layers()[2].document());
    let json: Value = serde_json::from_str(&cli("dump --format json").unwrap()).unwrap();
    assert_eq!(json, docs);
    assert!(cli("dump --format toml").is_err());
}

#[test]
fn test_releases() {
    let out = cli("releases release").unwrap();
    let rows: Vec<&str> = out.lines().collect();
    assert_eq!(rows.len(), 3);
    assert!(rows[2].ends_with("(latest)"));
}

#[test]
fn test_needs_catalogs() {
    let err = run_argv(vec!["catalog".into(), "layers".into()]).unwrap_err();
    assert!(err.contains("no catalogs"));
}
