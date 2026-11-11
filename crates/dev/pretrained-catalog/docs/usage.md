# Using the pretrained catalog

This guide covers the crate's API, from importing catalogs to choosing a build, and the select
language that does the choosing. The document form (the keyed `$` form, entry kinds, names, references, `$merge`) is
described in the crate's `README.md`. This guide uses as much of it as selection needs.

The examples run against the example catalogs in `catalogs/`, which `fixtures::builtin_stack()`
imports in this order: `prelude`, `commons`, `qwen3`, `whisper`. Their builds (a query's run
starts from them in full-name order: Whisper's `openai/...` first):

| build | family | class chain | thinking | task | parameters | other |
|---|---|---|---|---|---|---|
| `qwen3-0.6b` | `qwen/qwen3` | `hybrid` | switch | | 6e8 | |
| `qwen3-0.6b-base` | `qwen/qwen3` | `hybrid` | never | | 6e8 | `instruct: false` |
| `qwen3-4b-thinking-2507` | `qwen/qwen3` | `hybrid > thinking-only` | always | | 4e9 | in `qwen3-2507` |
| `large-v1` | `openai/whisper` | `multilingual` | | both | 1.55e9 | release 2022-09 |
| `large-v2` | `openai/whisper` | `multilingual` | | both | 1.55e9 | release 2022-12 |
| `large-v3` | `openai/whisper` | `multilingual` | | both | 1.55e9 | release 2023-11-06, `timestamps: true` |
| `large-v3-turbo` | `openai/whisper` | `multilingual` | | transcribe only | 8.09e8 | |

Qwen3's `language` is `["*"]`; Whisper's is `[en, "*"]`. The Qwen3 builds have the contract
`qwen3-lm`, the Whisper builds `whisper-asr`. The supplemental catalog
(`fixtures::supplemented_stack()`) adds `qwen3-1.7b`.

## Contents

1. [The pipeline](#the-pipeline)
2. [Importing catalogs](#importing-catalogs)
3. [Finding entries](#finding-entries)
4. [Resolving a build](#resolving-a-build)
5. [Presets](#presets)
6. [Choosing builds](#choosing-builds)
7. [The select language](#the-select-language)
8. [The CLI](#the-cli)
9. [Lower layers: a query by hand](#lower-layers-a-query-by-hand)
10. [Errors](#errors)

## The pipeline

```text
catalog documents ──push──▶ PretrainedCatalogStack ──resolve_build──▶ ResolvedBuild
                              (validated state)                         (tags, presets, deps)
                                     │
                     SelectEnv: the stack, derive steps, resolved builds
                                     │
     a tool's preset prefix ++ defaults ++ request (`use` spliced) ++ tail
                                     │ SelectionContext::compose
                                SelectQuery ──compile──▶ CompiledQuery
                                     │ run
   universe (full-name order, weight 1) ─▶ stage ─▶ stage ─▶ … ─▶ SelectRun (table, trace)
```

- A **stack** holds the imported catalogs as one validated state. Every query reads it.
- A **resolved build** is everything the state says about one build: its template-class chain,
  its memberships and facets, its tags (written and implied), its presets folded along the
  chain, and its dependencies.
- A **query** is a kind and a list of stages, written only in YAML particles. Each stage maps a
  table of weighted rows to a table; no stage adds a row. The result is the last table.
- A **choice** is the first row of the result. A query never selects presets: the caller applies
  the presets it chose, and a prefix keeps only the builds they apply to.

## Importing catalogs

A catalog document is the namespace tree, in the keyed form the crate's `README.md` describes:
every key the catalog language defines starts with `$`, and a bare key is a name (a child, a
tag, or a preset).

`PretrainedCatalogStack` imports catalogs one after another. A push applies the catalog's
entries, in order, to a copy of the state, validates the whole copy, and commits it only if
nothing is wrong. A refused catalog leaves the stack as it was, and the error lists every
problem found.

```rust
use pretrained_catalog::{
    CatalogErrorKind,
    PretrainedCatalogStack,
    fixtures,
};

let mut stack = fixtures::builtin_stack().unwrap();
let labels: Vec<&str> = stack.layers().iter().map(|l| l.label()).collect();
assert_eq!(labels, ["prelude", "commons", "qwen3", "whisper"]);

// A build whose template class doesn't exist is refused, and so is its catalog.
let err = stack
    .push_yaml(
        "broken",
        r#"
orphan:
  $kind: build
  $description: "Names a class nobody declared."
  $template_class: no-such-class
"#,
    )
    .unwrap_err();
assert!(err.has(CatalogErrorKind::Reference), "{err}");
assert_eq!(stack.layers().len(), 4);

// An empty stack, filled from files, or from documents built in code.
let mut empty = PretrainedCatalogStack::new();
empty.push_yaml("prelude", fixtures::PRELUDE).unwrap();
```

| call | does |
|---|---|
| `PretrainedCatalogStack::new()` | an empty stack |
| `push(label, PretrainedCatalog)` | imports a document built in code |
| `push_yaml(label, text)` | parses YAML, then pushes |
| `push_file(path)` | reads YAML (or JSON, by a `.json` extension), labelled by its file stem |
| `layers()` | the imported catalogs, in order, with their labels, paths and documents |
| `occurrences(name)` | every occurrence of a name across the catalogs, in import order |
| `declaration(name).history()` | which catalogs declared, updated and replaced an entry |

The first catalog is usually a prelude: the facets, registered axes and tags, and
`template_class_root`. A catalog can refer to anything below it in the stack, and its extending
occurrences (`$merge: update`) add to groups an earlier catalog declared. `catalogs/supplemental.yml`
shows a local addition.

## Finding entries

Names are `/`-separated paths, such as `openai/whisper/large/large-v3`. Lookups accept any
spelling that identifies one entry: the leaf, or a longer path.

```rust
use pretrained_catalog::{
    document::EntryKind,
    fixtures,
    lookup::{
        LeafMatch,
        SEARCHABLE,
    },
};

let stack = fixtures::builtin_stack().unwrap();

let v3 = stack.find_build("large-v3").unwrap();
assert_eq!(v3.to_string(), "openai/whisper/large/large-v3");
// The shortest spelling that still identifies it, for people.
assert_eq!(stack.display_name(&v3).short, "large-v3");

// Every build has a resource called `checkpoint`, so the leaf alone is ambiguous.
// The error lists the candidates. A longer spelling picks one.
let err = stack
    .find("checkpoint", &SEARCHABLE, LeafMatch::Exact)
    .unwrap_err();
assert!(err.message.contains("large-v3/checkpoint"), "{err}");
let one = stack
    .find("large-v3/checkpoint", &[EntryKind::Resource], LeafMatch::Exact)
    .unwrap();
assert_eq!(one.leaf(), "checkpoint");

// Groups and their members.
let whisper_asr = stack
    .find("whisper-asr", &[EntryKind::Contract], LeafMatch::Exact)
    .unwrap();
assert_eq!(stack.members_of(&whisper_asr).len(), 4);
let contracts: Vec<String> = stack
    .of_kind(EntryKind::Contract)
    .iter()
    .map(|c| c.name().to_string())
    .collect();
assert_eq!(contracts, ["qwen3-lm", "whisper-asr"]);

// Release groups are ordered by release time.
let release = stack
    .find("large/release", &[EntryKind::Release], LeafMatch::Exact)
    .unwrap();
assert_eq!(stack.latest(&release).unwrap(), Some(v3.clone()));
```

| call | does |
|---|---|
| `find(query, kinds, LeafMatch)` | one entry of `kinds`; `LeafMatch::Prefix` lets the last segment be a prefix |
| `find_build(query)` | `find` over builds, with an exact leaf |
| `matches(query, kinds, LeafMatch)` | every entry a query could mean |
| `display_name(name)` | `{short, full}`: the least specified spelling that resolves to the entry |
| `declaration(name)` | the entry: `kind()`, `body()`, `history()`, `declared_in()` |
| `of_kind(kind)`, `children(name)` | entries of a kind; the entries in a scope |
| `members_of(group)` | the builds in a group, directly or through other groups and class chains |
| `facet_groups(facet)`, `builds_in_facet(facet)` | "all families"; "builds in some family" |
| `build_memberships(build)` | every group a build is in |
| `template_chain(start, from)` | a class chain, root first |
| `releases(group)`, `latest(group)` | a release group's members by time; its newest |
| `resolve_ref(reference, from)` | a reference as an entry written in `from` would resolve it |

## Resolving a build

`resolve_build` gathers everything the state says about a build into a `ResolvedBuild`.

```rust
use pretrained_catalog::{
    CatalogName,
    fixtures,
};
use serde_json::json;

let stack = fixtures::builtin_stack().unwrap();
let name = stack.find_build("large-v3").unwrap();
let build = stack.resolve_build(&name).unwrap();

// Written tags, and the tags the catalog implies: facets, the contract, axes.
assert_eq!(build.tag("parameters"), Some(&json!(1550000000)));
// A tag declared in a namespace is keyed by its full name.
assert_eq!(build.tag("openai/whisper/timestamps"), Some(&json!(true)));
assert_eq!(build.tag("family"), Some(&json!("openai/whisper")));
assert_eq!(build.tag("contract"), Some(&json!("whisper-asr")));

let chain: Vec<&str> = build.template_chain.iter().map(|c| c.leaf()).collect();
assert_eq!(chain, ["template_class_root", "multilingual"]);
assert_eq!(build.contract, Some(CatalogName::parse("whisper-asr").unwrap()));
let release = build.release.as_ref().unwrap();
assert_eq!(release.released.to_string(), "2023-11-06");
assert!(build.dependencies.contains_key("checkpoint"));
```

| field | holds |
|---|---|
| `name`, `description`, `license`, `origin` | as written |
| `template_chain` | the class chain, root first |
| `memberships`, `facets`, `contract` | every group the build is in; its group per facet; its one contract |
| `release` | `{group, released}`, when it joins a release group |
| `tags` | each tag's value and source, by full name (`TagSource`: `Written` by the build or the class that writes it lowest in the chain, or implied by an `Axis`, a `Facet` or the `Contract`) |
| `presets` | the folded `ResolvedPresetSet` |
| `dependencies` | each dependency key's resource |

## Presets

A build's presets fold down its class chain. An axis use (`{use, offers, default}`) may narrow
what the use above it offered, and when it narrows, the presets of the dropped members are
pruned. A use that offers one member is a fixed mode.

A preset with the name of an inherited one meets it by its `$merge`:

- `update`, the default, merges its patch over the inherited patch, and a description replaces
  the inherited one;
- `replace` takes the inherited preset's place;
- `deny` refuses an inherited preset of its name.

With nothing inherited, the preset is declared, whatever its mode.

```rust
use pretrained_catalog::{
    CatalogName,
    fixtures,
    presets::PresetSelection,
};
use serde_json::json;

let stack = fixtures::builtin_stack().unwrap();
let thinking = CatalogName::parse("thinking").unwrap();

let hybrid = stack
    .resolve_build(&stack.find_build("qwen3-0.6b").unwrap())
    .unwrap();
// The negative member of the bool axis `thinking`.
let off = hybrid.presets.bool_member(&thinking, false).unwrap();
assert_eq!(off, "no-thinking");
let applied = hybrid
    .presets
    .select(json!({}), &PresetSelection::of([off.as_str()]))
    .unwrap();
assert_eq!(applied.document["chat"]["enable_thinking"], json!(false));
assert_eq!(applied.document["generation"]["temperature"], json!(0.7));

// A build that fixes the mode explains why the other member doesn't apply.
let always = stack
    .resolve_build(&stack.find_build("qwen3-4b-thinking-2507").unwrap())
    .unwrap();
let err = always.presets.bool_member(&thinking, false).unwrap_err();
assert!(err.message.contains("fixed at `thinking`"), "{err}");
```

- `select(base, &selection)` resolves a document: the caller's `base`, the set's `common` patch, each
  applied preset in order (each axis's default unless a member of it is selected, the `default`
  list, then the selected), and the selection's final `patch`.
- `check_selection(&selection)` is the same check without the document: unknown presets, two
  members of one axis, a member the build doesn't offer.
- `applied(&selection)` lists the presets `select` would apply, in order.

## Choosing builds

A tool chooses with a `SelectionContext` (its defaults and named selections), the presets it will
apply, and a `SelectRequest` (names, then stages). `preset_prefix` turns the presets into the
first stage, which keeps only the builds that can take them.

```rust
use pretrained_catalog::{
    fixtures,
    presets::PresetSelection,
    select::{
        choose::{SelectRequest, SelectionContext, bool_axis_member, preset_prefix},
        query::Stage,
        state::{SelectEnv, StageOrigin},
    },
};
use serde_json::json;

let stack = fixtures::builtin_stack().unwrap();
let env = SelectEnv::new(&stack);
let context = SelectionContext {
    defaults: serde_saphyr::from_str("[{keep: instruct}]").unwrap(),
    named: Vec::new(),
};

// `--off thinking`: the member `no-thinking`, read from the axis's declaration.
let off = bool_axis_member(&stack, "thinking", false).unwrap();
let presets = PresetSelection::of([off.as_str()]);
let prefix = || preset_prefix(&presets, vec!["--off thinking".into()]);

// Every Qwen3 build that can answer without thinking, largest first.
let request = SelectRequest::default().name("qwen3").stage(
    serde_saphyr::from_str::<Stage>("{order: {desc: parameters}}").unwrap(),
    StageOrigin::Flag("--desc parameters".into()),
);
let run = context.rank(&env, None, prefix(), &request).unwrap();
let leaves: Vec<&str> = run.table.names().iter().map(|n| n.leaf()).collect();
assert_eq!(leaves, ["qwen3-0.6b"]);

// Choosing takes the first; the caller applies the presets.
let (build, _) = context.choose(&env, None, prefix(), &request).unwrap();
let applied = build.build.presets.select(json!({}), &presets).unwrap();
assert_eq!(applied.document["chat"]["enable_thinking"], json!(false));

// One build named alone skips the defaults; one the presets can't apply to is an error.
let named = SelectRequest::default().name("qwen3-4b-thinking-2507");
let err = context.rank(&env, None, prefix(), &named).unwrap_err();
assert!(err.message.contains("was named, but can't be chosen"), "{err}");
```

### `compose`, `rank` and `choose`

| call | does |
|---|---|
| `compose(env, prefix, request, choosing)` | the `ComposedQuery`: the stages, each one's `StageOrigin`, and the build named alone, if any |
| `run_composed(env, universe, composed)` | runs it, traced, over every build (or `universe`) |
| `rank(env, universe, prefix, request)` | `compose` without the limit, then `run_composed`: every row of the result |
| `choose(env, universe, prefix, request)` | `compose` with `limit: 1`: the first build, resolved, and the run; `empty_error` when none is left |

The composed pipeline is a concatenation; nothing is merged:

1. **The prefix**, when the tool applies presets: `keep: {$presets: [..]}`.
2. **The defaults**, unless the request's only name is one build.
3. **The request's names:** a named selection's name splices its stages in (`use`); the other
   names are one `from` stage.
4. **The request's stages**, each `use` spliced in place.
5. **The tail:** `order: {desc: $weight}` when nothing orders, and `limit: 1` when choosing.

An `unset: L` anywhere in the request (or in a named selection) removes the earlier stage
labeled `L`, a default's among them.

A `SelectRun` is the result `table` and its `trace`: one `StageTrace` per stage (its operation,
origin, how many rows it removed, and the weights a `weigh` committed), and every `DroppedRow`,
with the stage and why.

### Derived keys

A derive step adds tags the catalog doesn't hold. A query reads a derived key `k` as the `$`
word `$k`, so it can't collide with a catalog name. `SelectEnv::with_derive` refuses a key that
is a `$` word the language defines.

```rust
use std::collections::BTreeMap;

use pretrained_catalog::{
    document::EntryKind,
    fixtures,
    resolve::ResolvedBuild,
    select::{
        query::SelectQuery,
        state::{DeriveEach, DeriveTags, DerivedKey, SelectEnv, SelectTable},
    },
};
use serde_json::{Value, json};

let stack = fixtures::builtin_stack().unwrap();
let gigabytes = DeriveEach {
    keys: vec![DerivedKey { name: "gigabytes".into(), description: "At bf16.".into() }],
    tags: |b: &ResolvedBuild| {
        let mut tags = BTreeMap::new();
        if let Some(p) = b.tag("parameters").and_then(Value::as_f64) {
            tags.insert("gigabytes".to_string(), json!(p * 2.0 / 1e9));
        }
        tags
    },
};
let steps: Vec<&dyn DeriveTags> = vec![&gigabytes];
let env = SelectEnv::with_derive(&stack, steps).unwrap();
let query = SelectQuery::from_yaml(
    "{kind: build, run: [{keep: {$gigabytes: {le: 2}}}, {order: {asc: $gigabytes}}]}",
)
.unwrap();
let table = query
    .compile(&env)
    .unwrap()
    .run(&env, SelectTable::universe(&env, EntryKind::Build))
    .unwrap();
assert_eq!(table.names()[0].leaf(), "qwen3-0.6b");
```

## The select language

A query names one kind, then a list of stages run in order:

```yaml
kind: build              # required, first
floor: 0.001             # optional: the least weight a row keeps between stages
run:
  - from: [qwen3, whisper]
  - keep: {instruct: true, thinking: false, parameters: {le: 8.9e9}}
  - weigh: {by: [{when: {language: en}, else: 0.5}]}
  - order: [{desc: $weight}, {desc: parameters}, {asc: $name}]
  - limit: 1
```

**No string is ever parsed.** A string is a name (a reference: `qwen3`, `a/b`, `/root`), a `$`
word, or a literal value in a value position. A map is an operator from a closed vocabulary, or a
test keyed by a name or a `$` word. **Wherever a catalog name can appear, anything that isn't one
starts with `$`.** A `$` word can never be a name, so nothing is reserved.

| `$` word | is |
|---|---|
| `$any`, `$all`, `$not` | the predicate operators |
| `$weight` | the row's weight at that point in the pipeline |
| `$name` | the build's full name |
| `$presets` | can take these presets together |
| `$released` | the release time the build joined its release group with |
| `$license`, `$origin` | the build's fields |
| `$parent` | the naming scope, exactly (a bare name tests membership by closure) |
| `$template_class` | the direct class (a bare class name tests the whole chain) |
| `$contract` | the build's contract |
| `$k` | a derived key `k` |

Words where no names appear stay bare: stage names, value-test words and weighing words.

### Stages

| stage | argument | does |
|---|---|---|
| `from` | a name or a list of names | keeps the rows in any of them |
| `keep` | a predicate | keeps the rows that satisfy it |
| `drop` | a predicate | `keep: {$not: P}`: `drop: [a, b]` drops the rows in both, `drop: {$any: [a, b]}` either |
| `weigh` | `{by: [..], floor: f}` | updates each row's weight |
| `order` | a sort key or a list of them | sorts, stably |
| `limit` | an integer, 1 or more | keeps the first N |
| `per` | `{group: G, run: [..]}` | partitions the rows, runs `run` on each part, and concatenates the parts |
| `use` | a name | splices in a named selection's stages, when the query is composed |
| `unset` | a label | removes the earlier stage with that label, when the query is composed |

A stage may carry a label, the one key allowed beside its operation: `{keep: instruct, as: tuned}`.
A later `unset: tuned` removes it, so a request can drop a hard default. An `unset` of a label no
earlier stage has, or two stages with one label, is an error.

`per`'s `group` is a facet (for builds: a part per group of it) or a list of names (a part per
name; a row goes in the first it is in). The parts come out in the order of their first row, and
the rows in no group form one more part, last. Each part runs `run` on its own:

```yaml
- per:
    group: family
    run:
      - order: [{desc: parameters}, {asc: $name}]
      - limit: 1
# the largest build of each family: large-v1, qwen3-4b-thinking-2507
```

There is no cutoff stage: a cutoff is `keep: {$weight: {ge: W}}`, at the position the query
chooses.

### Predicates

| written | means |
|---|---|
| `qwen3`, `hybrid`, `qwen3-0.6b` | in a group (by closure), a template class (by chain), a build (itself) |
| `[a, b, ..]` | all of them |
| `{$any: [..]}`, `{$all: [..]}`, `{$not: P}` | the operators |
| `{NAME: true}` / `{NAME: false}` | in / not in the set a name denotes |
| `thinking`, `{thinking: true}` / `{thinking: false}` | a bool axis: it offers the on / the off member |
| `{thinking: switch}` | the axis's implied tag: `always`, `switch` or `never` |
| `{task: translate}`, `{task: {has_any: [..]}}` | a categorical axis: what its use offers |
| `instruct`, `{TAG: VALUE_TEST}` | a bool tag is `true`; a tag's value |
| `{family: qwen3}` | a facet: the build's group in it |
| `{$presets: [P, ..]}` | it can take those presets together |
| `{$WORD: VALUE_TEST}` | a field, `$weight` or a derived key |
| a map with several keys | all of its entries |

**Value tests**, typed by the tag's declaration; a value that doesn't fit is an error:

| written | for | means |
|---|---|---|
| `v` | any | equals; for a set-valued tag, contains |
| `{eq: v}`, `{ne: v}` | single-valued | equals / doesn't |
| `{lt: n}`, `{le: n}`, `{gt: n}`, `{ge: n}` | counts, times, `$weight`, derived keys | compares; several make a range, `{ge: 1e9, le: 4e9}` |
| `{in: [..]}` | single-valued | is one of these |
| `{has: v}`, `{has_all: [..]}`, `{has_any: [..]}` | set-valued (`languages`, `text_list`, categorical offers) | contains |
| `{exists: true}` | any | has a value |

A build without a value fails every test but `{exists: false}`. A `languages` tag contains a
language by RFC 4647 filtering: `en` contains `en-US`, not the other way round.

### Weighing

Every row starts at weight 1, and only `weigh` changes it. A block applies its `by` entries in
order, as signed arithmetic, commits the result clipped to 0 or more, and drops the row below its
own `floor`. The query's `floor` (1e-9 unless set; `floor: 0` keeps everything) applies after each
block: a row below it is dropped there, even if a later block would raise it again. An update that
should recover a row goes in the same block.

| entry | does to the weight |
|---|---|
| `{when: P, else: w, then: v}` | × `v` (default 1) if the row satisfies P, else × `w` (default 0) |
| `{match: {TAG: L}, truncated: t, else: w}` | × 1 when `L` filters to one of the tag's languages, × `t` (default 1) when only a truncation of `L` does, else × `w` (default 0) |
| `{release: latest, step: s, min: m}` | × 1 for the newest of its release group in the catalog, `s` (default 0.05) less per newer release, never below `m` (default 0); × 1 in no release group |
| `{mul: k}`, `{add: k}`, `{clip: [a, b]}` | × k; + k; clamps |

Weights aren't bounded above, and nothing renormalizes.

```yaml
- from: whisper
- weigh: {by: [{release: latest, step: 0.25}]}
# large-v1 (0.5), large-v2 (0.75), large-v3 (1), large-v3-turbo (1, in no release)
- keep: {$weight: {ge: 0.6}}
# large-v2, large-v3, large-v3-turbo
```

### Kinds

| kind | name tests | tags, axis tests, `$presets` | `$` fields |
|---|---|---|---|
| `build` | its groups (by closure), its template-class chain, itself | its own, its chain's, the implied ones, derived keys | `$released`, `$license`, `$origin`, `$template_class`, `$contract` |
| `template_class` | the groups it joins through its chain (by closure), its base chain, itself | its chain's, and its folded axis uses and presets | |
| `namespace`, `contract`, `release` | the groups in its closure, itself | none | |
| `tag`, `axis`, `facet`, `resource` | the scopes that enclose it, itself | none | `$format`, `$dtype` (a resource's) |

`$weight`, `$name`, `$parent`, `weigh` (`when`, `mul`, `add`, `clip`), `order` and `limit` apply to
every kind; `match` to builds and template classes, `release` to builds. A test the kind doesn't
admit is an error when the query compiles. Choosing takes a build; a query of another kind is
ranked.

```yaml
kind: resource
run:
  - keep: qwen3-0.6b              # the resources in the build's scope
  - keep: {$format: json}
# qwen/qwen3/qwen3-0.6b/config
```

### `order` and `limit`

A sort key is `K` (ascending), `{asc: K}` or `{desc: K}`, with an optional `missing: first` (the
default puts rows without a value last). The keys are `$weight`, `$name`, `$released`, count tags
and derived keys. The sort is stable, so a trailing `{asc: $name}` makes the order total. `limit:
N` keeps the first N.

### Recipes

```yaml
# The largest instruction-tuned Qwen3 that can answer without thinking.
- from: qwen3
- keep: {instruct: true, thinking: false}
- order: {desc: parameters}
# qwen3-0.6b

# English speech, newest release, without the pruned turbo build.
- keep: whisper-asr
- drop: large-v3-turbo
- weigh: {by: [{when: {language: en}}, {release: latest}]}
- order: {desc: $weight}
# large-v3 (1), large-v2 (0.95), large-v1 (0.9)

# Prefer timestamps, but take any Whisper build.
- from: whisper
- weigh: {by: [{when: timestamps, else: 0.5}]}
- order: {desc: $weight}
# large-v3 (1), then large-v1, large-v2, large-v3-turbo (0.5 each)

# The two smallest builds that can think.
- keep: thinking
- order: parameters
- limit: 2
# qwen3-0.6b, qwen3-4b-thinking-2507

# British English, accepting builds that list only `en`, and none that merely wildcard it.
- weigh: {by: [{match: {language: en-GB}, truncated: 0.8, else: 0.4}]}
- keep: {$weight: {ge: 0.5}}
- order: {desc: $weight}
# the Whisper builds (0.8 each); Qwen3's `"*"` weighs 0.4, below the cut
```

## The CLI

```sh
cargo run -p pretrained-catalog -- --builtin select <names> <flags>
cargo run -p pretrained-catalog -- --catalog a.yml --catalog b.yml select <names> <flags>
```

The bin stands in for a tool that chooses builds. Each flag is one stage, in command-line order,
and each flag value is one YAML particle. `--on`, `--off` and `--preset` are the exception: they
are the presets the tool applies, and the prefix that keeps only the builds that can take them,
wherever they sit on the command line.

| flag | stage |
|---|---|
| `--kind K` | the query's kind; `build` by default |
| bare names | one `from: [..]`; a named selection's name is a `use` |
| `--keep P`, `--drop P` | that stage; P is a name or a flow map |
| `--lt/--le/--gt/--ge/--eq/--ne KEY V` | `keep: {KEY: {le: V}}`, and so on |
| `--has TAG V` | `keep: {TAG: {has: V}}` |
| `--weigh E` | `weigh: {by: [E]}`; `--weigh-block '{by: [..], floor: f}'` for a whole block |
| `--cutoff W` | `keep: {$weight: {ge: W}}` |
| `--asc K`, `--desc K` | one sort key; consecutive ones join one `order` stage |
| `--limit N`, `--use NAME`, `--unset LABEL` | that stage |
| `--per GROUP` … `--end` | a `per` stage; the flags between are its `run` |
| `--pipeline FILE_OR_FLOW` | a whole list of stages, spliced in |
| `--floor F` | the query's floor |
| `--on X`, `--off X`, `--preset P` | the presets applied, and the prefix |
| `--defaults FILE_OR_FLOW` | the context's defaults: a list of stages |
| `--rank` | also prints every row a stage removed, with why |

A bare name or a number goes as it is; anything else goes in single quotes (a flow map, a `$`
word).

```sh
pretrained-catalog --builtin select qwen3 --off thinking --keep instruct --desc parameters --limit 1
pretrained-catalog --builtin select whisper --weigh '{release: latest}' --cutoff 0.9 --rank
```

`select` prints YAML: the composed `query`, `why` (each stage: its operation, origin, how many
rows it removed, and a `weigh`'s weights), `dropped` (with `--rank`), the `presets` applied, and
the `results`: each build resolved with its weight, or, for another kind, each entry's name,
kind and weight.

## Lower layers: a query by hand

A `SelectQuery` compiles against a `SelectEnv` (which checks every name and test against the
stack) and runs on a `SelectTable`, traced or not. Nothing composes it: no defaults, no prefix,
no tail.

```rust
use pretrained_catalog::{
    document::EntryKind,
    fixtures,
    select::{
        query::SelectQuery,
        state::{SelectEnv, SelectTable},
    },
};

let stack = fixtures::builtin_stack().unwrap();
let env = SelectEnv::new(&stack);
let query = SelectQuery::from_yaml(
    r#"
kind: build
run:
  - from: whisper
  - weigh: {by: [{release: latest, step: 0.25}]}
  - keep: {$weight: {ge: 0.6}}
  - order: [{desc: $weight}, {asc: $name}]
"#,
)
.unwrap();
let compiled = query.compile(&env).unwrap();
let run = compiled
    .run_traced(&env, SelectTable::universe(&env, EntryKind::Build))
    .unwrap();
let leaves: Vec<&str> = run.table.names().iter().map(|n| n.leaf()).collect();
assert_eq!(leaves, ["large-v3", "large-v3-turbo", "large-v2"]);
// The trace says where each row went, and why.
let v1 = run
    .trace
    .dropped
    .iter()
    .find(|d| d.entry.path.leaf() == "large-v1")
    .unwrap();
assert_eq!((v1.stage, v1.weight), (2, 0.5));
```

`select::typed` holds `match_language`, the RFC 4647 matcher that the tests and `match` share.

## Errors

Every problem is a `CatalogError` with a `kind`, a message, and, where known, the entry and the
catalog it was found in. An import returns `CatalogErrors`, every problem it found, which has
`has(kind)` and `mentions(text)`.

| kind | raised by |
|---|---|
| `Parse`, `Name`, `Structure`, `Conflict`, `Reference`, `Merge`, `Facet`, `Contract`, `Tag`, `Preset`, `Release`, `Dependency` | importing a catalog: the document's rules |
| `Lookup` | `find`, `find_build`: a name that matches nothing, or several entries |
| `Selection` | a query (a stage or key of the wrong shape, a name that resolves to nothing, a value that doesn't fit its declaration), a preset selection a build can't take, an axis flag on something that isn't a bool axis, a build named alone that the run removed, and an empty result |

Selection messages name the stage and the key at fault: ``"stage 1 (`keep`): `instruct`: "maybe"
is not `true` or `false`"``, ``"stage 0 (`from`): `qwen3-1.7b`: nothing matches `qwen3-1.7b`"``,
``"stage 2 (`limit`): `limit` is a count, 1 or more"``. An empty result names the stages that
removed rows: ``"no build is left to choose: stage 0 (`keep`, prefix: --off thinking) dropped
5"``.
