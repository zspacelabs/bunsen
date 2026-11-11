//! Worked examples of the query language over the example catalogs
//! (`catalogs/`): each test writes a query, and the builds it should
//! select, with why.
//!
//! The example builds, in declaration order:
//!
//! | build | family | class chain | thinking | task | parameters | other |
//! |---|---|---|---|---|---|---|
//! | `qwen3-0.6b` | `qwen/qwen3` | `hybrid` | switch | | 6e8 | |
//! | `qwen3-0.6b-base` | `qwen/qwen3` | `hybrid` | never | | 6e8 | `instruct: false` |
//! | `qwen3-4b-thinking-2507` | `qwen/qwen3` | `hybrid > thinking-only` | always | | 4e9 | in `qwen3-2507` |
//! | `large-v1` | `openai/whisper` | `multilingual` | | both | 1.55e9 | release 2022-09 |
//! | `large-v2` | `openai/whisper` | `multilingual` | | both | 1.55e9 | release 2022-12 |
//! | `large-v3` | `openai/whisper` | `multilingual` | | both | 1.55e9 | release 2023-11-06, `timestamps: true` |
//! | `large-v3-turbo` | `openai/whisper` | `multilingual` | | transcribe only | 8.09e8 | |
//!
//! Qwen3's `language` is `["*"]`; Whisper's is `[en, "*"]`. The Qwen3 builds
//! have the contract `qwen3-lm`, the Whisper builds `whisper-asr`.

use pretrained_catalog::{
    PretrainedCatalogStack,
    fixtures,
    presets::PresetSelection,
    select::choose::{
        ChoiceRequest,
        NamedSelection,
        SelectionContext,
    },
};
use serde_json::Value;

/// A YAML query (or query part). A text that starts on a new line is
/// dedented by its common indent, so a query can be written indented.
fn q(yaml: &str) -> Value {
    let text = match yaml.strip_prefix('\n') {
        Some(body) => {
            let indent = body
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| l.len() - l.trim_start().len())
                .min()
                .unwrap_or(0);
            body.lines()
                .map(|l| l.get(indent..).unwrap_or(""))
                .collect::<Vec<_>>()
                .join("\n")
        }
        None => yaml.to_string(),
    };
    serde_saphyr::from_str(&text).expect("a YAML query")
}

/// The builds a request ranks, by leaf, in order.
fn ranked(
    stack: &PretrainedCatalogStack,
    context: &SelectionContext,
    request: ChoiceRequest,
) -> Vec<String> {
    context
        .rank(stack, None, &[], &request)
        .expect("a valid query")
        .selection
        .names()
        .iter()
        .map(|n| n.leaf().to_string())
        .collect()
}

/// The builds a whole query (`--query`) ranks.
fn query(yaml: &str) -> Vec<String> {
    let stack = fixtures::builtin_stack().unwrap();
    ranked(
        &stack,
        &SelectionContext::default(),
        ChoiceRequest::default().query(q(yaml)),
    )
}

/// The error a whole query fails to compile with.
fn refused(yaml: &str) -> String {
    let stack = fixtures::builtin_stack().unwrap();
    SelectionContext::default()
        .rank(&stack, None, &[], &ChoiceRequest::default().query(q(yaml)))
        .expect_err("an invalid query")
        .message
}

const QWEN3: [&str; 3] = ["qwen3-0.6b", "qwen3-0.6b-base", "qwen3-4b-thinking-2507"];
const LARGE: [&str; 4] = ["large-v1", "large-v2", "large-v3", "large-v3-turbo"];

// Membership: `from`, `require`, `exclude`.

#[test]
fn from_a_family_is_its_builds() {
    // `qwen3` resolves to the namespace `qwen/qwen3`; its members are the
    // builds placed in it.
    assert_eq!(query("select: {from: [qwen3]}"), QWEN3);
    assert_eq!(query("select: {from: [whisper]}"), LARGE);
}

#[test]
fn from_is_a_union() {
    // At least one of: a build is itself; a group is its members.
    assert_eq!(
        query("select: {from: [qwen3-0.6b, large-v3]}"),
        ["qwen3-0.6b", "large-v3"]
    );
    let all: Vec<&str> = QWEN3.iter().chain(&LARGE).copied().collect();
    assert_eq!(query("select: {from: [qwen3, whisper]}"), all);
}

#[test]
fn require_is_an_intersection() {
    // Every one of: the release group `release` (under whisper/large) and
    // the contract `whisper-asr`. The turbo build is in no release.
    assert_eq!(
        query("select: {require: [release, whisper-asr]}"),
        ["large-v1", "large-v2", "large-v3"]
    );
    // Two contracts can't both hold: a build has exactly one.
    assert!(query("select: {require: [qwen3-lm, whisper-asr]}").is_empty());
}

#[test]
fn a_template_class_names_the_builds_whose_chain_includes_it() {
    // Every Qwen3 build's chain includes `hybrid`; only 2507's includes
    // `thinking-only`, which is based on it.
    assert_eq!(query("select: {require: [hybrid]}"), QWEN3);
    assert_eq!(
        query("select: {require: [hybrid], exclude: [thinking-only]}"),
        ["qwen3-0.6b", "qwen3-0.6b-base"]
    );
}

#[test]
fn exclude_removes_any_member() {
    // `qwen3-2507` is a plain group 2507 joins; `release` a release group.
    assert_eq!(
        query("select: {from: [qwen3], exclude: [qwen3-2507]}"),
        ["qwen3-0.6b", "qwen3-0.6b-base"]
    );
    assert_eq!(
        query("select: {from: [whisper], exclude: [release]}"),
        ["large-v3-turbo"]
    );
}

// Typed predicates: `where`.

#[test]
fn a_bool_axis_tests_what_its_use_offers() {
    // `false`: offers `no-thinking` (switch or never).
    assert_eq!(
        query("select: {where: {thinking: false}}"),
        ["qwen3-0.6b", "qwen3-0.6b-base"]
    );
    // `true`: offers `thinking` (switch or always).
    assert_eq!(
        query("select: {where: {thinking: true}}"),
        ["qwen3-0.6b", "qwen3-4b-thinking-2507"]
    );
    // The exact states.
    assert_eq!(query("select: {where: {thinking: switch}}"), ["qwen3-0.6b"]);
    assert_eq!(
        query("select: {where: {thinking: always}}"),
        ["qwen3-4b-thinking-2507"]
    );
    assert_eq!(
        query("select: {where: {thinking: never}}"),
        ["qwen3-0.6b-base"]
    );
}

#[test]
fn a_categorical_axis_is_a_set() {
    // A list is "all of": the turbo build offers only `transcribe`.
    assert_eq!(
        query("select: {where: {task: [transcribe, translate]}}"),
        ["large-v1", "large-v2", "large-v3"]
    );
    // `only` is the exact set: a one-member `only` is a fixed mode.
    assert_eq!(
        query("select: {where: {task: {only: [transcribe]}}}"),
        ["large-v3-turbo"]
    );
    // `any` is "at least one of".
    assert_eq!(
        query("select: {where: {task: {any: [translate]}}}").len(),
        3
    );
}

#[test]
fn scalar_tags_compare_and_range() {
    // A count takes bounds; a list of values is "any of".
    assert_eq!(
        query("select: {where: {parameters: {max: 1.0e9}}}"),
        ["qwen3-0.6b", "qwen3-0.6b-base", "large-v3-turbo"]
    );
    assert_eq!(
        query("select: {where: {parameters: [600000000, 809000000]}}"),
        ["qwen3-0.6b", "qwen3-0.6b-base", "large-v3-turbo"]
    );
    // A bool tag compares; only Qwen3 declares `instruct` values.
    assert_eq!(
        query("select: {where: {instruct: false}}"),
        ["qwen3-0.6b-base"]
    );
    // A tag declared in a family's scope is found by its short name.
    assert_eq!(query("select: {where: {timestamps: true}}"), ["large-v3"]);
}

#[test]
fn predicates_and_membership_combine() {
    // The running example: Qwen3, can run without thinking, under 8B, tuned
    // to follow instructions.
    assert_eq!(
        query(
            "select:
               from: [qwen3]
               where: {thinking: false, instruct: true, parameters: {max: 8.0e9}}"
        ),
        ["qwen3-0.6b"]
    );
}

#[test]
fn values_are_checked_against_their_declarations() {
    assert!(refused("select: {where: {instruct: maybe}}").contains("is not `true` or `false`"));
    assert!(refused("select: {where: {parameters: big}}").contains("is not a number"));
    assert!(
        refused("select: {where: {task: summarize}}").contains("not one of the axis's members")
    );
    assert!(refused("select: {where: {thinking: maybe}}").contains("a bool axis takes"));
    // Membership lives in the lists, never in `where`.
    assert!(refused("select: {where: {family: qwen3}}").contains("is a facet"));
    assert!(refused("select: {where: {contract: qwen3-lm}}").contains("is membership"));
    // No free-form keys, and membership references are membership kinds.
    assert!(
        refused("select: {where: {kit: qwen3}}").contains("every key a query tests is declared")
    );
    assert!(refused("select: {from: [parameters]}").contains("nothing matches"));
    assert!(refused("order: [thinking]").contains("not a number to sort by"));
}

// Soft scores, order, cutoff and limit.

#[test]
fn prefer_grades_and_order_sorts_by_weight() {
    // Whisper handles `en`: 1. Qwen3 only wildcards it: `otherwise`.
    let stack = fixtures::builtin_stack().unwrap();
    let ranking = SelectionContext::default()
        .rank(
            &stack,
            None,
            &[],
            &ChoiceRequest::default()
                .prefer(q("where: {language: {is: en, otherwise: 0.5}}"))
                .order(&["-weight"]),
        )
        .unwrap();
    let weights: Vec<(String, f64)> = ranking
        .selection
        .iter()
        .map(|(c, w)| (c.name().leaf().to_string(), w))
        .collect();
    let expected: Vec<(String, f64)> = LARGE
        .iter()
        .map(|b| (b.to_string(), 1.0))
        .chain(QWEN3.iter().map(|b| (b.to_string(), 0.5)))
        .collect();
    assert_eq!(weights, expected);
}

#[test]
fn prefer_latest_release_then_order() {
    // Each release loses `step` per newer release in its group: v1 0.5, v2
    // 0.75, v3 1; the turbo build is in no release, so 1. By weight, then
    // parameters descending (v3 1.55e9 before turbo 8.09e8).
    assert_eq!(
        query(
            "
             select: {from: [whisper]}
             prefer: {release: {step: 0.25}}
             order: [-weight, -parameters]"
        ),
        ["large-v3", "large-v3-turbo", "large-v2", "large-v1"]
    );
}

#[test]
fn cutoff_drops_and_limit_keeps_the_first() {
    // Ascending parameters: turbo (1.0), v1 (0.5), v2 (0.75), v3 (1.0).
    let stack = fixtures::builtin_stack().unwrap();
    let context = SelectionContext::default();
    let whisper = ChoiceRequest::default()
        .name("whisper")
        .prefer(q("release: {step: 0.25}"))
        .order(&["parameters"]);
    let result = |request: &ChoiceRequest| -> Vec<String> {
        context
            .choices(&stack, None, &[], request)
            .unwrap()
            .iter()
            .map(|c| c.candidate.name().leaf().to_string())
            .collect()
    };
    // No cutoff and no limit: the whole ranking.
    assert_eq!(
        result(&whisper),
        ["large-v3-turbo", "large-v1", "large-v2", "large-v3"]
    );
    // `cutoff: 0.6` drops v1; the order of the rest stands.
    let cut = whisper.clone().cutoff(0.6);
    assert_eq!(result(&cut), ["large-v3-turbo", "large-v2", "large-v3"]);
    // `limit: 2` keeps the first two survivors.
    assert_eq!(
        result(&cut.clone().limit(2)),
        ["large-v3-turbo", "large-v2"]
    );
    // Loading one build takes the first, whatever the limit.
    let without_turbo = cut.select(q("exclude: [large-v3-turbo]"));
    let choice = context.choose(&stack, None, &[], &without_turbo).unwrap();
    assert_eq!(choice.candidate.name().leaf(), "large-v2");
}

#[test]
fn limit_is_a_count() {
    assert!(refused("limit: 0").contains("`limit` is a count, 1 or more"));
    assert!(refused("limit: 2.5").contains("`limit` is a count"));
    assert!(refused("cutoff: -1").contains("`cutoff` is a weight"));
}

// Defaults, names and merge patches.

#[test]
fn defaults_merge_under_a_choice_and_patches_override_them() {
    let stack = fixtures::builtin_stack().unwrap();
    let context = SelectionContext {
        defaults: q("
                     select: {where: {instruct: true}}
                     order: [-parameters]"),
        named: vec![],
    };
    // The defaults keep only builds with `instruct: true` (Whisper has no
    // `instruct`), largest first.
    assert_eq!(
        ranked(&stack, &context, ChoiceRequest::default()),
        ["qwen3-4b-thinking-2507", "qwen3-0.6b"]
    );
    // A patch removes one default entry with `null`, keeping the rest.
    assert_eq!(
        ranked(
            &stack,
            &context,
            ChoiceRequest::default()
                .name("whisper")
                .select(q("where: {instruct: null}"))
        ),
        ["large-v1", "large-v2", "large-v3", "large-v3-turbo"]
    );
    // One build named alone skips the defaults altogether.
    assert_eq!(
        ranked(
            &stack,
            &context,
            ChoiceRequest::default().name("qwen3-0.6b-base")
        ),
        ["qwen3-0.6b-base"]
    );
}

#[test]
fn a_named_selection_is_a_query() {
    let stack = fixtures::builtin_stack().unwrap();
    let context = SelectionContext {
        defaults: Value::Null,
        named: vec![NamedSelection {
            name: "speech-en".into(),
            description: "The newest English speech build.".into(),
            query: q("
                      select: {from: [whisper]}
                      prefer: {where: {language: en}, release: latest}
                      order: [-weight]"),
        }],
    };
    let choice = context
        .choose(
            &stack,
            None,
            &[],
            &ChoiceRequest::default().name("speech-en"),
        )
        .unwrap();
    assert_eq!(choice.candidate.name().leaf(), "large-v3");
}

// Presets and axis flags.

#[test]
fn an_axis_flag_is_a_where_entry_and_a_preset() {
    let stack = fixtures::builtin_stack().unwrap();
    // `--off thinking` is `select.where.thinking: false`, so it keeps the
    // builds that offer `no-thinking`, and selects that preset.
    let request = ChoiceRequest::default()
        .name("qwen3")
        .flag("thinking", false);
    let context = SelectionContext::default();
    assert_eq!(
        ranked(&stack, &context, request.clone()),
        ["qwen3-0.6b", "qwen3-0.6b-base"]
    );
    let choice = context.choose(&stack, None, &[], &request).unwrap();
    assert_eq!(choice.presets, PresetSelection::of(["no-thinking"]));
    assert!(
        choice
            .why
            .contains(&"`--off thinking` implies `--preset no-thinking`".to_string())
    );
    // On a build that fixes the other mode, the flag is an explained error.
    let err = context
        .choose(
            &stack,
            None,
            &[],
            &ChoiceRequest::default()
                .name("qwen3-4b-thinking-2507")
                .flag("thinking", false),
        )
        .unwrap_err();
    assert!(err.message.contains("fixed at `thinking`"), "{err}");
}

#[test]
fn a_preset_narrows_with_reasons() {
    let stack = fixtures::builtin_stack().unwrap();
    let ranking = SelectionContext::default()
        .rank(
            &stack,
            None,
            &[],
            &ChoiceRequest::default().name("whisper").preset("translate"),
        )
        .unwrap();
    let leaves: Vec<&str> = ranking.selection.names().iter().map(|n| n.leaf()).collect();
    assert_eq!(leaves, ["large-v1", "large-v2", "large-v3"]);
    assert_eq!(ranking.dropped.len(), 1);
    assert!(ranking.dropped[0].contains("`translate` is not offered"));
}

// Catalogs on top.

#[test]
fn a_supplemental_build_is_selected_like_any_other() {
    // The supplemental catalog adds `qwen3-1.7b` under `qwen/qwen3`, so a
    // family query finds it, in the order it was declared.
    let stack = fixtures::supplemented_stack().unwrap();
    let mut expected: Vec<&str> = QWEN3.to_vec();
    expected.push("qwen3-1.7b");
    assert_eq!(
        ranked(
            &stack,
            &SelectionContext::default(),
            ChoiceRequest::default().name("qwen3")
        ),
        expected
    );
    assert_eq!(
        ranked(
            &stack,
            &SelectionContext::default(),
            ChoiceRequest::default()
                .name("qwen3")
                .select(q("where: {parameters: {min: 1.0e9, max: 2.0e9}}"))
        ),
        ["qwen3-1.7b"]
    );
}

#[test]
fn an_ambiguous_name_is_an_error_that_lists_the_candidates() {
    let mut stack = fixtures::builtin_stack().unwrap();
    stack
        .push_yaml(
            "mirror",
            r#"
openai:
  $kind: namespace
  $merge: update
  whisper:
    $kind: namespace
    $merge: update
    mirror:
      $kind: namespace
      $description: "Mirrored builds."
      large-v2:
        $kind: build
        $description: "Whisper large-v2, mirrored."
        $template_class: multilingual
        $dependencies: { checkpoint: large/large-v2/checkpoint }
        parameters: 1550000000
"#,
        )
        .unwrap();
    let err = SelectionContext::default()
        .rank(
            &stack,
            None,
            &[],
            &ChoiceRequest::default().name("large-v2"),
        )
        .unwrap_err();
    assert!(err.message.contains("ambiguous"), "{err}");
    assert!(
        err.message.contains("openai/whisper/mirror/large-v2"),
        "{err}"
    );
    // A longer spelling picks one.
    assert_eq!(
        ranked(
            &stack,
            &SelectionContext::default(),
            ChoiceRequest::default().name("mirror/large-v2")
        ),
        ["large-v2"]
    );
}
