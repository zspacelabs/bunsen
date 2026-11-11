//! Worked examples of the selection pipeline over the example catalogs
//! (`catalogs/`): each test writes a query's stages, and the builds it
//! should select, with why.
//!
//! The example builds, in full-name order (the order a run starts in):
//!
//! | build | family | class chain | thinking | task | parameters | other |
//! |---|---|---|---|---|---|---|
//! | `large-v1` | `openai/whisper` | `multilingual` | | both | 1.55e9 | release 2022-09 |
//! | `large-v2` | `openai/whisper` | `multilingual` | | both | 1.55e9 | release 2022-12 |
//! | `large-v3` | `openai/whisper` | `multilingual` | | both | 1.55e9 | release 2023-11-06, `timestamps: true` |
//! | `large-v3-turbo` | `openai/whisper` | `multilingual` | | transcribe only | 8.09e8 | |
//! | `qwen3-0.6b` | `qwen/qwen3` | `hybrid` | switch | | 6e8 | |
//! | `qwen3-0.6b-base` | `qwen/qwen3` | `hybrid` | never | | 6e8 | `instruct: false` |
//! | `qwen3-4b-thinking-2507` | `qwen/qwen3` | `hybrid > thinking-only` | always | | 4e9 | in `qwen3-2507` |
//!
//! Qwen3's `language` is `["*"]`; Whisper's is `[en, "*"]`. The Qwen3 builds
//! have the contract `qwen3-lm`, the Whisper builds `whisper-asr`.

use std::collections::BTreeMap;

use pretrained_catalog::{
    document::EntryKind,
    fixtures,
    resolve::ResolvedBuild,
    select::{
        query::SelectQuery,
        state::{
            DeriveEach,
            DeriveTags,
            DerivedKey,
            SelectEnv,
            SelectTable,
        },
    },
};
use serde_json::{
    Value,
    json,
};

/// `text` dedented by its common indent, so YAML can be written indented.
fn dedent(text: &str) -> String {
    let body = text.strip_prefix('\n').unwrap_or(text);
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

/// The query of builds whose `run` is the YAML list `run`.
fn build_query(run: &str) -> SelectQuery {
    let run: Value = serde_saphyr::from_str(&dedent(run)).expect("YAML stages");
    SelectQuery::from_value(&json!({"kind": "build", "run": run})).expect("a query")
}

fn leaves(table: &SelectTable) -> Vec<String> {
    table.names().iter().map(|n| n.leaf().to_string()).collect()
}

/// The builds `run` selects, by leaf, in order.
fn select(run: &str) -> Vec<String> {
    let stack = fixtures::builtin_stack().unwrap();
    let env = SelectEnv::new(&stack);
    let compiled = build_query(run).compile(&env).expect("a valid query");
    let table = compiled
        .run(&env, SelectTable::universe(&env, EntryKind::Build))
        .unwrap();
    leaves(&table)
}

/// The error `run` fails to compile with.
fn refused(run: &str) -> String {
    let stack = fixtures::builtin_stack().unwrap();
    let env = SelectEnv::new(&stack);
    match SelectQuery::from_value(&json!({
        "kind": "build",
        "run": serde_saphyr::from_str::<Value>(&dedent(run)).expect("YAML stages"),
    })) {
        Err(e) => e.message,
        Ok(query) => query.compile(&env).expect_err("an invalid query").message,
    }
}

const QWEN3: [&str; 3] = ["qwen3-0.6b", "qwen3-0.6b-base", "qwen3-4b-thinking-2507"];
const LARGE: [&str; 4] = ["large-v1", "large-v2", "large-v3", "large-v3-turbo"];

// Membership: `from`, names, and `drop`.

#[test]
fn a_run_starts_from_every_build_in_full_name_order() {
    let all: Vec<&str> = LARGE.iter().chain(&QWEN3).copied().collect();
    assert_eq!(select("[]"), all);
}

#[test]
fn from_a_family_is_its_builds() {
    // `qwen3` resolves to the namespace `qwen/qwen3`; its members are the
    // builds placed in it.
    assert_eq!(select("[{from: qwen3}]"), QWEN3);
    assert_eq!(select("[{from: [whisper]}]"), LARGE);
}

#[test]
fn from_is_any_of() {
    // A build is itself; a group is its members.
    assert_eq!(
        select("[{from: [qwen3-0.6b, large-v3]}]"),
        ["large-v3", "qwen3-0.6b"]
    );
    assert_eq!(select("[{from: [qwen3, whisper]}]").len(), 7);
}

#[test]
fn a_list_in_keep_is_all_of() {
    // The release group `release` (under whisper/large) and the contract
    // `whisper-asr`. The turbo build is in no release.
    assert_eq!(
        select("[{keep: [release, whisper-asr]}]"),
        ["large-v1", "large-v2", "large-v3"]
    );
    // Two contracts can't both hold: a build has exactly one.
    assert!(select("[{keep: [qwen3-lm, whisper-asr]}]").is_empty());
}

#[test]
fn a_template_class_names_the_builds_whose_chain_includes_it() {
    assert_eq!(select("[{keep: hybrid}]"), QWEN3);
    assert_eq!(
        select("[{keep: hybrid}, {drop: thinking-only}]"),
        ["qwen3-0.6b", "qwen3-0.6b-base"]
    );
    // `{NAME: false}` is "not in".
    assert_eq!(
        select("[{keep: {hybrid: true, thinking-only: false}}]"),
        ["qwen3-0.6b", "qwen3-0.6b-base"]
    );
}

#[test]
fn drop_is_keep_with_not() {
    // `drop: [a, b]` drops only the builds in both; `{$any: [..]}` either.
    assert_eq!(
        select("[{from: whisper}, {drop: [release, timestamps]}]"),
        ["large-v1", "large-v2", "large-v3-turbo"]
    );
    assert_eq!(
        select("[{from: whisper}, {drop: {$any: [release, large-v3-turbo]}}]"),
        Vec::<String>::new()
    );
    assert_eq!(
        select("[{from: qwen3}, {keep: {$not: qwen3-2507}}]"),
        ["qwen3-0.6b", "qwen3-0.6b-base"]
    );
}

#[test]
fn disjunction_is_a_predicate() {
    assert_eq!(
        select("[{keep: {$any: [[whisper, timestamps], qwen3-0.6b-base]}}]"),
        ["large-v3", "qwen3-0.6b-base"]
    );
}

// Typed tests.

#[test]
fn a_bool_axis_tests_what_its_use_offers() {
    // `false`: offers `no-thinking` (switch or never).
    assert_eq!(
        select("[{keep: {thinking: false}}]"),
        ["qwen3-0.6b", "qwen3-0.6b-base"]
    );
    // `true`, or the bare name: offers `thinking` (switch or always).
    assert_eq!(
        select("[{keep: thinking}]"),
        ["qwen3-0.6b", "qwen3-4b-thinking-2507"]
    );
    assert_eq!(
        select("[{keep: {thinking: true}}]"),
        select("[{keep: thinking}]")
    );
    // The complement of the name is "doesn't offer it": never, and the
    // builds that don't use the axis.
    assert_eq!(
        select("[{keep: {$not: thinking}}, {from: qwen3}]"),
        ["qwen3-0.6b-base"]
    );
    // The axis's implied tag: always, switch or never.
    assert_eq!(select("[{keep: {thinking: switch}}]"), ["qwen3-0.6b"]);
    assert_eq!(
        select("[{keep: {thinking: {in: [always, never]}}}]"),
        ["qwen3-0.6b-base", "qwen3-4b-thinking-2507"]
    );
}

#[test]
fn a_categorical_axis_is_a_set_of_offers() {
    // A member: it offers that one. The turbo build offers only `transcribe`.
    assert_eq!(
        select("[{keep: {task: translate}}]"),
        ["large-v1", "large-v2", "large-v3"]
    );
    assert_eq!(
        select("[{keep: {task: {has_all: [transcribe, translate]}}}]"),
        ["large-v1", "large-v2", "large-v3"]
    );
    assert_eq!(select("[{keep: {task: {has_any: [translate]}}}]").len(), 3);
    assert_eq!(select("[{keep: {task: transcribe}}]").len(), 4);
}

#[test]
fn valued_tags_compare_and_range() {
    assert_eq!(
        select("[{keep: {parameters: {le: 1.0e9}}}]"),
        ["large-v3-turbo", "qwen3-0.6b", "qwen3-0.6b-base"]
    );
    assert_eq!(
        select("[{keep: {parameters: {in: [600000000, 809000000]}}}]"),
        ["large-v3-turbo", "qwen3-0.6b", "qwen3-0.6b-base"]
    );
    // Several tests in one map make a range.
    assert_eq!(
        select("[{keep: {parameters: {gt: 1.0e9, lt: 4.0e9}}}]"),
        ["large-v1", "large-v2", "large-v3"]
    );
    // A bool tag: a bare name is `true`. Only Qwen3 declares `instruct`.
    assert_eq!(select("[{keep: {instruct: false}}]"), ["qwen3-0.6b-base"]);
    assert_eq!(
        select("[{keep: instruct}]"),
        ["qwen3-0.6b", "qwen3-4b-thinking-2507"]
    );
    // A tag declared in a family's scope is found by its short name.
    assert_eq!(select("[{keep: timestamps}]"), ["large-v3"]);
    // A missing value fails every test but `{exists: false}`.
    assert_eq!(select("[{keep: {instruct: {exists: false}}}]"), LARGE);
}

#[test]
fn languages_contain_by_rfc_4647_filtering() {
    // Whisper lists `en`; `en` filters to `en`, not to `en-GB`.
    assert_eq!(select("[{keep: {language: en}}]"), LARGE);
    assert_eq!(
        select("[{keep: {language: {has: en-GB}}}]"),
        Vec::<String>::new()
    );
    assert_eq!(select("[{keep: {language: {has: \"*\"}}}]").len(), 7);
}

#[test]
fn a_facet_tests_the_group_a_build_is_in() {
    assert_eq!(select("[{keep: {family: qwen3}}]"), QWEN3);
    assert_eq!(
        select("[{keep: {publisher: {in: [openai, qwen]}}}]").len(),
        7
    );
}

#[test]
fn field_words_read_the_build() {
    assert_eq!(
        select("[{keep: {$template_class: thinking-only}}]"),
        ["qwen3-4b-thinking-2507"]
    );
    // `$template_class` is the direct class; the bare name is the chain.
    assert_eq!(
        select("[{keep: {$template_class: hybrid}}]"),
        ["qwen3-0.6b", "qwen3-0.6b-base"]
    );
    assert_eq!(select("[{keep: {$contract: qwen3-lm}}]"), QWEN3);
    assert_eq!(select("[{keep: {$parent: large}}]"), LARGE);
    assert_eq!(select("[{keep: {$license: Apache-2.0}}]"), QWEN3);
    assert_eq!(
        select("[{keep: {$released: {ge: 2022-12}}}]"),
        ["large-v2", "large-v3"]
    );
    assert_eq!(
        select("[{keep: {$name: openai/whisper/large/large-v1}}]"),
        ["large-v1"]
    );
}

#[test]
fn presets_test_what_a_build_can_take() {
    // Every Qwen3 build offers `no-thinking` but 2507; `greedy` is
    // `hybrid`'s, on no axis.
    assert_eq!(
        select("[{keep: {$presets: [no-thinking, greedy]}}]"),
        ["qwen3-0.6b", "qwen3-0.6b-base"]
    );
    assert_eq!(
        select("[{keep: {$presets: translate}}]"),
        ["large-v1", "large-v2", "large-v3"]
    );
}

#[test]
fn the_running_example() {
    // Qwen3, can run without thinking, under 8B, tuned to follow
    // instructions, largest first.
    assert_eq!(
        select(
            "
            - from: qwen3
            - keep: {thinking: false, instruct: true, parameters: {le: 8.0e9}}
            - order: {desc: parameters}"
        ),
        ["qwen3-0.6b"]
    );
}

// Order and limit.

#[test]
fn order_is_stable_and_puts_missing_values_last() {
    // By parameters descending; the three 1.55e9 releases keep their order.
    assert_eq!(
        select("[{from: whisper}, {order: {desc: parameters}}]"),
        ["large-v1", "large-v2", "large-v3", "large-v3-turbo"]
    );
    // By release time, newest first: turbo is in no release, so last.
    assert_eq!(
        select("[{from: whisper}, {order: {desc: $released}}]"),
        ["large-v3", "large-v2", "large-v1", "large-v3-turbo"]
    );
    assert_eq!(
        select("[{from: whisper}, {order: {desc: $released, missing: first}}]"),
        ["large-v3-turbo", "large-v3", "large-v2", "large-v1"]
    );
    // Several keys, most significant first.
    assert_eq!(
        select("[{order: [parameters, {desc: $name}]}, {limit: 3}]"),
        ["qwen3-0.6b-base", "qwen3-0.6b", "large-v3-turbo"]
    );
}

#[test]
fn limit_keeps_the_first() {
    assert_eq!(
        select("[{from: whisper}, {limit: 2}]"),
        ["large-v1", "large-v2"]
    );
    assert_eq!(select("[{from: whisper}, {limit: 9}]").len(), 4);
}

// The trace.

#[test]
fn a_traced_run_records_each_stage_and_each_row_removed() {
    let stack = fixtures::builtin_stack().unwrap();
    let env = SelectEnv::new(&stack);
    let compiled = build_query(
        "
        - from: qwen3
        - keep: {$presets: no-thinking}
        - limit: 1",
    )
    .compile(&env)
    .unwrap();
    let run = compiled
        .run_traced(&env, SelectTable::universe(&env, EntryKind::Build))
        .unwrap();
    assert_eq!(leaves(&run.table), ["qwen3-0.6b"]);
    let dropped: Vec<usize> = run.trace.stages.iter().map(|s| s.dropped).collect();
    assert_eq!(dropped, [4, 1, 1]);
    let why: BTreeMap<String, String> = run
        .trace
        .dropped
        .iter()
        .map(|d| (d.entry.path.leaf().to_string(), d.why.clone()))
        .collect();
    assert_eq!(why["large-v1"], "fails `from: qwen3`");
    assert!(
        why["qwen3-4b-thinking-2507"].contains("no-thinking"),
        "{}",
        why["qwen3-4b-thinking-2507"]
    );
    assert_eq!(why["qwen3-0.6b-base"], "past limit");
}

// Derived keys.

#[test]
fn a_derived_key_is_a_dollar_word() {
    let stack = fixtures::builtin_stack().unwrap();
    let size = DeriveEach {
        keys: vec![DerivedKey {
            name: "gigabytes".into(),
            description: "Two bytes a parameter.".into(),
        }],
        tags: |b: &ResolvedBuild| {
            let mut tags = BTreeMap::new();
            if let Some(p) = b.tag("parameters").and_then(Value::as_f64) {
                tags.insert("gigabytes".to_string(), json!(p * 2.0 / 1e9));
            }
            tags
        },
    };
    let steps: Vec<&dyn DeriveTags> = vec![&size];
    let env = SelectEnv::with_derive(&stack, steps).unwrap();
    let compiled = build_query("[{keep: {$gigabytes: {le: 2}}}, {order: {desc: $gigabytes}}]")
        .compile(&env)
        .unwrap();
    let table = compiled
        .run(&env, SelectTable::universe(&env, EntryKind::Build))
        .unwrap();
    assert_eq!(
        leaves(&table),
        ["large-v3-turbo", "qwen3-0.6b", "qwen3-0.6b-base"]
    );
    // A derived key may not be a word the language defines.
    let bad = DeriveEach {
        keys: vec![DerivedKey {
            name: "weight".into(),
            description: "Shadows `$weight`.".into(),
        }],
        tags: |_: &ResolvedBuild| BTreeMap::new(),
    };
    let steps: Vec<&dyn DeriveTags> = vec![&bad];
    assert!(SelectEnv::with_derive(&stack, steps).is_err());
}

// Refusals.

#[test]
fn names_and_values_are_checked_when_the_query_compiles() {
    assert!(refused("[{keep: {instruct: maybe}}]").contains("is not `true` or `false`"));
    assert!(refused("[{keep: {parameters: big}}]").contains("is not a number"));
    assert!(refused("[{keep: {task: summarize}}]").contains("not one of the axis's members"));
    assert!(refused("[{keep: {thinking: maybe}}]").contains("is not one of"));
    assert!(refused("[{keep: {kit: qwen3}}]").contains("nothing matches `kit`"));
    assert!(refused("[{from: parameters}]").contains("nothing matches"));
    assert!(refused("[{keep: {contract: qwen3-lm}}]").contains("`$contract`"));
    assert!(refused("[{keep: {$format: json}}]").contains("a resource's field"));
    assert!(refused("[{keep: {$nope: 1}}]").contains("is not a `$` word"));
    assert!(refused("[{keep: {parameters: [1, 2]}}]").contains("a list is not a value test"));
    assert!(refused("[{keep: {language: {lt: en}}}]").contains("`lt` tests a single value"));
    assert!(refused("[{keep: {instruct: {has: true}}}]").contains("`has` tests a set"));
    assert!(refused("[{keep: {$license: {gt: MIT}}}]").contains("is not ordered"));
    assert!(refused("[{order: thinking}]").contains("is not ordered"));
    assert!(refused("[{limit: 0}]").contains("`limit` is a count, 1 or more"));
    assert!(refused("[{union: [qwen3]}]").contains("unknown variant `union`"));
    assert!(refused("[{keep: parameters}]").contains("is a valued tag"));
}

#[test]
fn a_query_round_trips_through_its_written_form() {
    let query = build_query(
        "
        - from: [qwen3, whisper]
        - keep: {$any: [instruct, {thinking: false}]}
        - drop: qwen3-2507
        - weigh:
            by:
              - {when: {language: en}, else: 0.5}
              - {match: {language: en-GB}, truncated: 0.8}
              - {release: latest, min: 0.5}
              - {mul: 2}
              - {add: -0.5}
              - {clip: [0, 1]}
            floor: 0.25
        - order: [{desc: parameters}, {asc: $name, missing: first}]
        - limit: 2",
    );
    let value = serde_json::to_value(&query).unwrap();
    assert_eq!(SelectQuery::from_value(&value).unwrap(), query);
    assert!(SelectQuery::from_value(&json!({"kind": "build", "floor": -1})).is_err());
    assert!(SelectQuery::from_value(&json!({"run": []})).is_err());
}

// Weighing.

/// The builds `run` selects, by leaf, with their weights, in order.
fn weighed(run: &str) -> Vec<(String, f64)> {
    let stack = fixtures::builtin_stack().unwrap();
    let env = SelectEnv::new(&stack);
    let compiled = build_query(run).compile(&env).expect("a valid query");
    compiled
        .run(&env, SelectTable::universe(&env, EntryKind::Build))
        .unwrap()
        .rows
        .iter()
        .map(|r| (r.entry.path.leaf().to_string(), r.weight))
        .collect()
}

#[test]
fn when_grades_and_order_sorts_by_weight() {
    // Whisper handles `en`: × 1. Qwen3 only wildcards it: × 0.5.
    let expected: Vec<(String, f64)> = LARGE
        .iter()
        .map(|b| (b.to_string(), 1.0))
        .chain(QWEN3.iter().map(|b| (b.to_string(), 0.5)))
        .collect();
    assert_eq!(
        weighed(
            "
            - weigh: {by: [{when: {language: en}, else: 0.5}]}
            - order: {desc: $weight}"
        ),
        expected
    );
}

#[test]
fn match_grades_a_language_by_rfc_4647() {
    // `en-GB` reaches Whisper's `en` only by truncation; Qwen3's `*` is
    // `else`.
    let weights = weighed(
        "
        - weigh: {by: [{match: {language: en-GB}, truncated: 0.8, else: 0.5}]}
        - limit: 5",
    );
    assert_eq!(weights[0], ("large-v1".to_string(), 0.8));
    assert_eq!(weights[4], ("qwen3-0.6b".to_string(), 0.5));
}

#[test]
fn release_latest_grades_against_the_catalog_then_order() {
    // Each release loses `step` per newer release of its group: v1 0.5, v2
    // 0.75, v3 1; the turbo build is in no release, so 1. By weight, then
    // parameters descending (v3 1.55e9 before turbo 8.09e8).
    assert_eq!(
        select(
            "
            - from: whisper
            - weigh: {by: [{release: latest, step: 0.25}]}
            - order: [{desc: $weight}, {desc: parameters}]"
        ),
        ["large-v3", "large-v3-turbo", "large-v2", "large-v1"]
    );
    // Graded in the catalog, not among the rows still in the run: with v3
    // gone, v2 is still one release behind.
    assert_eq!(
        weighed(
            "
            - from: whisper
            - drop: large-v3
            - weigh: {by: [{release: latest, step: 0.25, min: 0.6}]}"
        ),
        [
            ("large-v1".to_string(), 0.6),
            ("large-v2".to_string(), 0.75),
            ("large-v3-turbo".to_string(), 1.0)
        ]
    );
}

#[test]
fn the_cutoff_is_a_weight_test_after_the_weighs() {
    // Ascending parameters: turbo (1.0), v1 (0.5), v2 (0.75), v3 (1.0).
    let run = |tail: &str| {
        select(&format!(
            "
            - from: whisper
            - weigh: {{by: [{{release: latest, step: 0.25}}]}}
            - order: parameters
            {tail}"
        ))
    };
    assert_eq!(
        run(""),
        ["large-v3-turbo", "large-v1", "large-v2", "large-v3"]
    );
    // `cutoff: 0.6` drops v1; the order of the rest stands.
    let cut = "- keep: {$weight: {ge: 0.6}}";
    assert_eq!(run(cut), ["large-v3-turbo", "large-v2", "large-v3"]);
    // A band: the near misses only.
    assert_eq!(run("- keep: {$weight: {ge: 0.6, lt: 1}}"), ["large-v2"]);
    // A running test reads the weight where it sits: before the weigh,
    // every row is at 1.
    assert_eq!(
        select(
            "
            - from: whisper
            - keep: {$weight: {ge: 0.6}}
            - weigh: {by: [{release: latest, step: 0.25}]}"
        )
        .len(),
        4
    );
}

#[test]
fn a_block_is_signed_arithmetic_committed_at_its_end() {
    // `else: 0` then `add` restores the row within one block.
    assert_eq!(
        weighed(
            "
            - from: qwen3
            - weigh: {by: [{when: {thinking: switch}, else: 0}, {add: 0.25}]}"
        ),
        [
            ("qwen3-0.6b".to_string(), 1.25),
            ("qwen3-0.6b-base".to_string(), 0.25),
            ("qwen3-4b-thinking-2507".to_string(), 0.25)
        ]
    );
    // Weights aren't bounded above; `clip` bounds them.
    assert_eq!(
        weighed("[{from: qwen3-0.6b}, {weigh: {by: [{mul: 3}]}}]"),
        [("qwen3-0.6b".to_string(), 3.0)]
    );
    assert_eq!(
        weighed("[{from: qwen3-0.6b}, {weigh: {by: [{mul: 3}, {clip: [0, 1]}]}}]"),
        [("qwen3-0.6b".to_string(), 1.0)]
    );
    // A negative running weight commits as 0, and the floor drops it.
    assert!(weighed("[{from: qwen3-0.6b}, {weigh: {by: [{add: -2}]}}]").is_empty());
}

#[test]
fn a_row_below_the_floor_at_a_block_end_is_dropped_there() {
    // Two blocks: the row falls to 0 at the first one's end, and the second
    // can't raise it. One block holding both updates does.
    let two = "
        - from: qwen3-0.6b-base
        - weigh: {by: [{when: {thinking: switch}, else: 0}]}
        - weigh: {by: [{add: 0.5}]}";
    assert!(select(two).is_empty());
    let one = "
        - from: qwen3-0.6b-base
        - weigh: {by: [{when: {thinking: switch}, else: 0}, {add: 0.5}]}";
    assert_eq!(select(one), ["qwen3-0.6b-base"]);
}

#[test]
fn floors_drop_what_counts_as_zero() {
    let zeroed = "[{from: qwen3}, {weigh: {by: [{when: {thinking: switch}, else: 0}]}}]";
    // The default floor drops a 0.
    assert_eq!(select(zeroed), ["qwen3-0.6b"]);
    // `floor: 0` keeps every row.
    let stack = fixtures::builtin_stack().unwrap();
    let env = SelectEnv::new(&stack);
    let mut query = build_query(zeroed);
    query.floor = Some(0.0);
    let table = query
        .compile(&env)
        .unwrap()
        .run(&env, SelectTable::universe(&env, EntryKind::Build))
        .unwrap();
    assert_eq!(leaves(&table), QWEN3);
    // A block's own floor drops at its end.
    assert_eq!(
        select("[{from: whisper}, {weigh: {by: [{release: latest, step: 0.25}], floor: 0.7}}]"),
        ["large-v2", "large-v3", "large-v3-turbo"]
    );
}

#[test]
fn a_traced_weigh_records_its_weights_and_its_floored_rows() {
    let stack = fixtures::builtin_stack().unwrap();
    let env = SelectEnv::new(&stack);
    let run = build_query(
        "[{from: whisper}, {weigh: {by: [{release: latest, step: 0.25}], floor: 0.7}}]",
    )
    .compile(&env)
    .unwrap()
    .run_traced(&env, SelectTable::universe(&env, EntryKind::Build))
    .unwrap();
    let weigh = &run.trace.stages[1];
    assert_eq!(weigh.op, "weigh");
    assert_eq!(weigh.weights.len(), 3);
    let floored = run
        .trace
        .dropped
        .iter()
        .find(|d| d.entry.path.leaf() == "large-v1")
        .unwrap();
    assert_eq!((floored.stage, floored.weight), (1, 0.5));
    assert!(floored.why.contains("floor"), "{}", floored.why);
}

#[test]
fn weighing_is_checked_when_the_query_compiles() {
    assert!(refused("[{weigh: {by: [{mul: x}]}}]").contains("`mul` is a number"));
    assert!(refused("[{weigh: {by: [{when: instruct, but: 1}]}}]").contains("not `but`"));
    assert!(refused("[{weigh: {by: [{match: {parameters: en}}]}}]").contains("`languages` tag"));
    assert!(refused("[{weigh: {by: [{release: oldest}]}}]").contains("`release` is `latest`"));
    assert!(refused("[{weigh: {by: [{clip: [1, 0]}]}}]").contains("`clip` is `[a, b]`"));
    assert!(refused("[{weigh: {by: [{pow: 2}]}}]").contains("a `by` entry is"));
    assert!(refused("[{weigh: {floor: -1}}]").contains("a block's `floor`"));
}
