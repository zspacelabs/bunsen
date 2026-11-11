# Using the pretrained catalog

This guide covers the crate's API, from importing catalogs to choosing a build, and the select
language that does the choosing. The document form (the keyed `$` form, entry kinds, names, references, `$merge`) is
described in the crate's `README.md`. This guide uses as much of it as selection needs.

The examples run against the example catalogs in `catalogs/`, which `fixtures::builtin_stack()`
imports in this order: `prelude`, `commons`, `qwen3`, `whisper`. Their builds:

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
9. [Lower layers: queries and selections by hand](#lower-layers-queries-and-selections-by-hand)
10. [Errors](#errors)

## The pipeline

```text
catalog documents ──push──▶ PretrainedCatalogStack ──resolve_build──▶ ResolvedBuild
                              (validated state)                         (tags, presets, deps)
                                     │
                    SelectionContext::rank / choices / choose
                                     │
             names + defaults + named selections + patches + flags
                         merged into one query
                                     │
  universe ─▶ derive ─▶ preset/flag narrowing ─▶ select ─▶ prefer ─▶ order ─▶ cutoff ─▶ limit
                                     │
                   Ranking ─▶ result ─▶ Choice (presets, why)
```

- A **stack** holds the imported catalogs as one validated state. Every query reads it.
- A **resolved build** is everything the state says about one build: its template-class chain,
  its memberships and facets, its tags (written and implied), its presets folded along the
  chain, and its dependencies.
- A **query** is a document in the select language. It ranks the builds, and the result is the
  ranked list after the cutoff and the limit.
- A **choice** is one build of that result, with the presets to apply and the reasons it was
  chosen. Loading a model takes one build, the first of the result.

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

`select::choose` is the layer a command uses. A `SelectionContext` holds what a command chooses
with: a default query, and the selections it names. A `ChoiceRequest` holds what one command line
asks for. Its builders mirror the CLI flags.

```rust
use pretrained_catalog::{
    fixtures,
    presets::PresetSelection,
    select::choose::{
        ChoiceRequest,
        NamedSelection,
        SelectionContext,
    },
};
use serde_json::Value;

fn yaml(text: &str) -> Value {
    serde_saphyr::from_str(text).unwrap()
}

let stack = fixtures::builtin_stack().unwrap();
let context = SelectionContext {
    // Applied under every choice, unless the only name is one build.
    defaults: yaml("{select: {where: {instruct: true}}, order: [-parameters]}"),
    named: vec![NamedSelection {
        name: "en".into(),
        description: "English speech.".into(),
        // Whisper has no `instruct` tag, so this selection removes that default with `null`.
        query: yaml(
            "{select: {from: [whisper], where: {instruct: null}},
              prefer: {where: {language: en}, release: latest},
              order: [-weight]}",
        ),
    }],
};

// `--select qwen3 --off thinking`
let choice = context
    .choose(
        &stack,
        None,
        &[],
        &ChoiceRequest::default().name("qwen3").flag("thinking", false),
    )
    .unwrap();
assert_eq!(choice.candidate.name().leaf(), "qwen3-0.6b");
assert_eq!(choice.presets, PresetSelection::of(["no-thinking"]));
assert_eq!(
    choice.why,
    [
        "from `qwen/qwen3` (namespace)",
        "`--off thinking` is `select.where.thinking: false`",
        "`--off thinking` implies `--preset no-thinking`",
        "took `qwen/qwen3/qwen3-0.6b` at weight 1",
    ]
);

// `--select en`: a named selection.
let en = context
    .choose(&stack, None, &[], &ChoiceRequest::default().name("en"))
    .unwrap();
assert_eq!(en.candidate.name().leaf(), "large-v3");

// One build, named exactly: the defaults don't apply, so `instruct: true` doesn't refuse it.
let base = context
    .choose(&stack, None, &[], &ChoiceRequest::default().name("qwen3-0.6b-base"))
    .unwrap();
assert_eq!(base.candidate.name().leaf(), "qwen3-0.6b-base");
```

### `choose`, `choices` and `rank`

All three take the same arguments, `(stack, universe, derive, request)`:

| call | returns | for |
|---|---|---|
| `choose` | the first `Choice` of the result | loading a model: an implicit limit of 1 |
| `choices` | every `Choice` in the result, in order | listing; the query's `limit` caps it |
| `rank` | the `Ranking`: every ranked candidate, before the cutoff and limit | explaining |

- `universe: Option<&[CatalogName]>` restricts the candidates, for example to the builds a
  kit can load. `None` means every build in the stack.
- `derive: &[&dyn DeriveOp]` adds computed tags before the query runs (see
  [Derived keys](#derived-keys)).
- `choose` and `choices` fail when the result is empty. The message names the builds that the
  presets or flags dropped, and why.

A `Choice` carries the `candidate` (its `build` and derived tags), its `weight`, the `presets` to
apply (each `--preset`, then the member each axis flag names), and `why`, one line per step.

A `Ranking` carries the full ordering (`selection`), the `cutoff` and `limit`, the merged `query`,
`why`, the builds the presets or flags `dropped` with their reasons, the `flags` and the
`presets`. Use `result()` to get the cut list, `choices()` to turn it into choices, and
`excluded(i, w)` to see why the `i`th ranked candidate is not in the result
(`"below cutoff"` or `"past limit"`). `report(every)` turns the ranking into a serializable
`SelectionReport`, which is what the CLI prints (see [the CLI](#the-cli)). Its `results` are
`ChoiceRecord`s: each a resolved build, with tags as plain values, plus `weight`, `presets` and,
when `every` lists the whole ranking, `excluded`:

```rust
use pretrained_catalog::{
    fixtures,
    select::choose::{
        ChoiceRequest,
        SelectionContext,
    },
};
use serde_json::json;

let stack = fixtures::builtin_stack().unwrap();
let request = ChoiceRequest::default()
    .name("whisper")
    .prefer(json!({"release": {"step": 0.25}}))
    .order(&["parameters"])
    .cutoff(0.6)
    .limit(2);
let ranking = SelectionContext::default()
    .rank(&stack, None, &[], &request)
    .unwrap();
let rows: Vec<(&str, f64, Option<&str>)> = ranking
    .selection
    .iter()
    .enumerate()
    .map(|(i, (c, w))| (c.name().leaf(), w, ranking.excluded(i, w)))
    .collect();
assert_eq!(
    rows,
    [
        ("large-v3-turbo", 1.0, None),
        ("large-v1", 0.5, Some("below cutoff")),
        ("large-v2", 0.75, None),
        ("large-v3", 1.0, Some("past limit")),
    ]
);
let result = ranking.result();
let leaves: Vec<&str> = result.names().iter().map(|n| n.leaf()).collect();
assert_eq!(leaves, ["large-v3-turbo", "large-v2"]);
```

### `ChoiceRequest`

| builder | CLI | fills |
|---|---|---|
| `.name(n)` | `--select NAME` | a named selection's query, or `select.from += [n]` |
| `.select(patch)` | `--select {..}` | merged into `select` |
| `.prefer(patch)` | `--prefer {..}` | merged into `prefer` |
| `.order(&keys)` | `--order KEY`... | `order` |
| `.cutoff(w)` | `--cutoff W` | `cutoff` |
| `.limit(n)` | `--limit N` | `limit` |
| `.query(patch)` | `--query {..}` | merged into the whole query |
| `.preset(p)` | `--preset P` | the build must take preset `p` |
| `.flag(axis, on)` | `--on AXIS` / `--off AXIS` | `select.where.AXIS: on`, and that member's preset |

Patches merge in the order they were added. The CLI adds them in a fixed order (see
[the CLI](#the-cli)).

### How a request becomes a ranking

1. **Names.** A name that a named selection has stands for that selection's query. Every
   other name is a reference, appended to `select.from`, so several names form a union. When
   the only name is a single build, the defaults are skipped: naming a build exactly gets that
   build.
2. **The query.** The defaults, the named selections, then the patches merge in order, as RFC
   7386 merge patches. Each axis flag then sets its `select.where` entry, so a flag overrides a
   patch's value for its axis.
3. **Candidates.** The universe is resolved, and the derive steps run. A build that doesn't
   offer a flag's member, or can't take the `--preset`s and the flags' members together, is
   dropped, and the reason is recorded. If the one build named exactly is dropped, the error
   says why.
4. **The query runs.** It compiles against the stack, so every reference and key is checked,
   and then `select`, `prefer`, `order`, `cutoff` and `limit` apply in turn.

### Derived keys

A derive step adds tags the catalog doesn't hold, such as a size estimate from the parameter
count and a load policy, or "already in the cache". It declares the keys it writes, so a query
may test and order by them.

```rust
use std::collections::BTreeMap;

use pretrained_catalog::{
    fixtures,
    select::{
        choose::{
            ChoiceRequest,
            SelectionContext,
        },
        core::{
            DeriveEach,
            DerivedKey,
            SelectionCandidate,
        },
    },
};
use serde_json::json;

let stack = fixtures::builtin_stack().unwrap();
let bytes = DeriveEach {
    keys: vec![DerivedKey {
        name: "bf16_bytes".into(),
        description: "The checkpoint's size in bf16.".into(),
    }],
    tags: |c: &SelectionCandidate| {
        let mut tags = BTreeMap::new();
        if let Some(p) = c.number("parameters") {
            tags.insert("bf16_bytes".to_string(), json!(p * 2.0));
        }
        tags
    },
};
// The largest build that fits in 2 GB.
let choice = SelectionContext::default()
    .choose(
        &stack,
        None,
        &[&bytes],
        &ChoiceRequest::default()
            .select(json!({"where": {"bf16_bytes": {"max": 2e9}}}))
            .order(&["-bf16_bytes"]),
    )
    .unwrap();
assert_eq!(choice.candidate.name().leaf(), "large-v3-turbo");
```

A derived key is untyped. In `where`, it is tested as a single value (equality, a list of values
as "any of", or `{min, max}`). A candidate's lookup of a key checks its derived tags first, then
its build's tags.

## The select language

A query is a map with five optional keys:

```yaml
select:                            # the hard filter: a build passes or is out
  from:    [qwen3]                 # a member of at least one
  require: [hybrid]                # a member of every one
  exclude: [qwen3-2507]            # a member of none
  where:   {thinking: switch, instruct: true, parameters: {max: 8e9}}
prefer:                            # soft scores, multiplied into the weight
  where:   {language: {is: en, otherwise: 0.5}}
  release: latest                  # or {step, floor}
order: [-weight, -parameters]      # most significant first; `-` descends
cutoff: 0.5                        # the least weight kept; absent, 0
limit: 3                           # how many are kept; absent, all
```

Every candidate starts at weight 1, and the steps run in this order:

1. **`select`** keeps a build only if it passes every part. The others leave the ranking.
2. **`prefer`** multiplies each candidate's weight by each entry's score, a number from 0 to 1. A
   build whose weight reaches 0 leaves the ranking.
3. **`prefer.release`** scores each release group's builds by recency.
4. **`order`** sorts, stably, by the listed keys. With no `order`, the builds stay in declaration
   order.
5. **`cutoff`** drops the candidates whose weight is below it.
6. **`limit`** keeps the first so many of the rest.

The list that's left is the **result**. Loading a model takes its first build.

Unknown keys at any level are errors, and so is every reference or key that doesn't resolve.
A query is checked against the stack before it runs.

### Membership: `from`, `require`, `exclude`

Each list holds catalog references, resolved by lookup like any name. A leading `/` resolves
from the root. What it means for a build to be "in" a reference depends on the kind the
reference names:

| the reference names | a build is in it when |
|---|---|
| a namespace, contract or release group | it is in the group's membership closure: directly, through a nested group, or through its class chain's joins |
| a template class | the class is in the build's chain |
| a build | it is that build |

```yaml
select: {from: [qwen3, large-v3]}                 # a union: every Qwen3 build, and large-v3
select: {require: [qwen3-lm, hybrid]}             # Qwen3 LMs whose chain includes `hybrid`
select: {from: [whisper], exclude: [release]}     # Whisper builds in no release group
select: {require: [thinking-only]}                # qwen3-4b-thinking-2507
```

- With no `from`, every build in the universe is a candidate.
- Membership is the only way to test facets and contracts. `where: {family: ..}` and
  `where: {contract: ..}` are refused, and the error points to the lists.
- A reference that matches several entries is an error that lists them. A longer spelling picks
  one: `large/release`, `openai/whisper`.
- Requiring two contracts selects nothing, because a build is in exactly one.

### `where`: typed predicates

Each `where` key is a declared axis or tag, or a key a derive step declares. Keys resolve by
lookup like any name, so `timestamps` finds `openai/whisper/timestamps`. A key that isn't
declared is an error, so a misspelling fails rather than matching nothing. Each value is checked
against its key's declaration when the query compiles.

| the key is | `x` | `[x, y]` | object forms |
|---|---|---|---|
| a bool axis (`thinking`) | `true`: offers the positive member; `false`: offers the negative | n/a | the values `switch` (offers both), `always` (positive only), `never` (negative only) |
| set-valued: a categorical axis (`task`, its offers), `text_list`, `languages` | contains `x` | contains **all** of them | `{any: [..]}` contains any; `{only: [..]}` is exactly that set |
| single-valued: a `bool`, `count`, `text` or `one_of` tag, or a derived key | equals `x` | equals **any** of them | `{min, max}` for a `count` or derived key, either bound alone |

`{is: x}` means the same as `x` everywhere. It exists so that `prefer` can put `x` beside
`otherwise` and `truncated`.

The list rule: **a list means "all of" when the value is a set, and "any of" when it is a single
value**, because one value can't equal two things.

```yaml
where: {thinking: true}                  # can think: qwen3-0.6b, qwen3-4b-thinking-2507
where: {thinking: false}                 # can answer directly: qwen3-0.6b, qwen3-0.6b-base
where: {thinking: switch}                # both, switchable: qwen3-0.6b
where: {thinking: always}                # thinks, can't be switched off: qwen3-4b-thinking-2507
where: {task: translate}                 # offers translate: large-v1, -v2, -v3
where: {task: [transcribe, translate]}   # offers both: the same three
where: {task: {any: [transcribe, translate]}}   # offers either: all four
where: {task: {only: [transcribe]}}      # transcribe and nothing else: large-v3-turbo
where: {instruct: true}                  # the tag is true; Whisper has no `instruct`, so it is out
where: {instruct: [true, false]}         # any value: has the tag at all
where: {parameters: {min: 1e9, max: 2e9}}
where: {timestamps: true}                # large-v3 (a tag declared under openai/whisper)
where: {language: en}                    # handles English: the Whisper builds
```

- **A build without the key fails** a `where` entry. To accept builds that lack the key, use
  `prefer`, or remove the entry with `null` (see [Merging](#merging)).
- **Bool axes.** `true` and `false` ask whether the build *offers* a member. They don't ask
  about its default. Use `switch`, `always` or `never` to ask about the whole offer. A bool axis
  (and a bool tag) is named by its positive side, so it's `timestamps`, never
  `no_timestamps`.
- **Languages** follow RFC 4647 filtering, case-insensitively. A request `en` matches `en` and
  `en-US`. In `where`, only a real match counts: `en-GB` doesn't hold for a build that lists
  only `en`, and the build's `"*"` ("other languages, unmeasured") holds for nothing. `prefer`
  grades the near misses instead (below).
- **Values are typed.** Each of these is an error when the query compiles: `thinking: maybe`,
  `{min}` on a text tag, a member the axis doesn't have, `limit: 0`, and `{any}` on a
  single-valued key. A `scores` tag can't be tested as a whole.
- YAML reads `yes` and `no` as booleans: `instruct: yes` means `instruct: true`.

### `prefer`: soft scores

`prefer.where` takes the same keys and value forms as `select.where`. Each entry scores a
candidate, and the score multiplies its weight. Two fields shape the score:

| field | default | the score when |
|---|---|---|
| (the entry holds) | 1 | the predicate holds |
| `otherwise` | 0 | it doesn't, or the build lacks the key |
| `truncated` | 1 | a `languages` request matches only once truncated (`en-GB` to `en`) |

A `languages` entry matched only through the build's `"*"` scores `otherwise`. With several
requested languages, the score is the least of their grades.

```yaml
prefer:
  where:
    language: {is: en-GB, truncated: 0.8, otherwise: 0.3}
    timestamps: {is: true, otherwise: 0.5}
```

An entry with the default `otherwise: 0` is a hard filter with one difference: it applies after
`select`, in the weight. Write it in `select.where` unless the weight is the point.

`prefer.release` ranks each release group's builds by recency:

| value | means |
|---|---|
| `latest` | `{step: 0.05, floor: 0}` |
| `{step, floor}` | each newer release in the build's group, among the current candidates, costs `step`; the score never goes below `floor` |

A build in no release group scores 1. With `floor: 0`, a release `1 / step` or more behind its
group's newest is dropped. Only candidates still in the ranking count, so the newest release a
query keeps scores 1. To choose the newest release, combine it with `order: [-weight]`:

```yaml
select: {from: [whisper]}
prefer: {release: {step: 0.25}}
order: [-weight, -parameters]
# large-v3 (1), large-v3-turbo (1, in no release), large-v2 (0.75), large-v1 (0.5)
```

### `order`

A list of keys, most significant first. A leading `-` sorts in descending order.

- `weight` is the candidate's weight after `prefer`.
- A `count` tag, or a derived key, sorts by its number.
- A candidate that lacks a key sorts after those that have it, in either direction.
- The sort is stable. Ties, and a query with no `order` at all, keep declaration order.

Sorting by any other key, such as a bool axis or a text tag, is an error.

### `cutoff` and `limit`

- `cutoff: W` (a weight, 0 or more) drops every candidate whose weight is below `W`. Order
  doesn't matter to it: it removes builds wherever they sit, and leaves the rest in order.
- `limit: N` (a whole number, 1 or more) keeps the first `N` candidates that are left.
- With neither, the result is the whole ranking. Loading a model takes the first build, an
  implicit limit of 1. A command that lists builds (the CLI's `select`) prints the whole result
  unless a limit is given.

```yaml
select: {from: [whisper]}
prefer: {release: {step: 0.25}}
order: [parameters]
cutoff: 0.6
limit: 2
# ranked: turbo (1), v1 (0.5, below cutoff), v2 (0.75), v3 (1, past limit)
# result: turbo, v2
```

### Merging

Queries combine as RFC 7386 merge patches, applied in order:

- the context's **defaults** (skipped when the only name is one build);
- each **named selection** used;
- each **patch**: `--query`, `--select {..}`, `--prefer`, then `--order`, `--cutoff` and
  `--limit`;
- each **axis flag's** `select.where` entry.

Every level is a map and merges key by key, except `from`, `require`, `exclude` and `order`.
Those are lists, and a later patch replaces each one whole. A `null` removes a key.

```yaml
# defaults
select: {where: {instruct: true, thinking: switch}}
order: [-parameters]
# patch
select: {where: {thinking: null}}
order: [-weight]
# merged
select: {where: {instruct: true}}
order: [-weight]
```

Names given on the command line are appended to `select.from` after the merge, so they add to a
`from` that a patch wrote.

### Axis flags

`--on AXIS` and `--off AXIS` are shortcuts, not modes of their own. A flag on the bool axis `X`
means:

1. `select.where.X: true` (or `false`), so a build that doesn't offer the member is not a
   candidate; and
2. the member, `X` or `no-X`, selected as a preset on the chosen build.

The ranking records both: ``"`--off thinking` is `select.where.thinking: false`"``, and the
choice's ``"`--off thinking` implies `--preset no-thinking`"``. A flag on an axis that isn't a
registered bool axis is an error. When the one build named exactly can't take the flag, the error
says why: ``the build `qwen3-4b-thinking-2507` can't be chosen: … fixed at `thinking` ``.

### Recipes

```yaml
# The largest instruction-tuned Qwen3 that can answer without thinking.
select: {from: [qwen3], where: {instruct: true, thinking: false}}
order: [-parameters]
# qwen3-0.6b

# English speech, newest release, without the pruned turbo build.
select: {require: [whisper-asr], exclude: [large-v3-turbo]}
prefer: {where: {language: en}, release: latest}
order: [-weight]
# large-v3, large-v2, large-v1

# Prefer timestamps, but take any Whisper build.
select: {from: [whisper]}
prefer: {where: {timestamps: {is: true, otherwise: 0.5}}}
order: [-weight]
# large-v3 (1), then large-v1, large-v2, large-v3-turbo (0.5 each)

# The two smallest builds that can think.
select: {where: {thinking: true}}
order: [parameters]
limit: 2
# qwen3-0.6b, qwen3-4b-thinking-2507

# British English, accepting builds that list only `en`, and none that merely wildcard it.
prefer: {where: {language: {is: en-GB, truncated: 0.8, otherwise: 0.4}}}
order: [-weight]
cutoff: 0.5
# the Whisper builds (0.8 each); Qwen3's `"*"` scores 0.4, below the cutoff
```

## The CLI

```sh
cargo run -p pretrained-catalog -- --builtin select <flags>
cargo run -p pretrained-catalog -- --catalog a.yml --catalog b.yml select <flags>
```

| flag | fills |
|---|---|
| `--select NAME` (repeatable) | a named selection, or `select.from += [NAME]` |
| `--select '{..}'` (repeatable) | merged into `select` |
| `--prefer '{..}'` (repeatable) | merged into `prefer` |
| `--order KEY` (repeatable) | `order`, most significant first |
| `--cutoff W` | `cutoff` |
| `--limit N` | `limit` |
| `--query '{..}'` (repeatable) | merged into the whole query |
| `--preset P` (repeatable) | the build must take `P` |
| `--on AXIS` / `--off AXIS` | `select.where.AXIS: true/false`, plus that member's preset |
| `--defaults '{..}'` | the default query |
| `--rank` | list every ranked build, not only the result |

Patches merge in this order: `--defaults`, then each `--query`, each `--select {..}`, each
`--prefer`, and then `--order`, `--cutoff` and `--limit`. Values are YAML or JSON.

`select` prints a YAML report, `SelectionReport`:

| key | holds |
|---|---|
| `query` | the merged query that ran |
| `why` | how the candidates were found, a line per step |
| `dropped` | the builds the presets or flags ruled out, with the reasons (absent when none) |
| `results` | the result, in order: each a `ChoiceRecord` |

Each result is the resolved build plus what the choice adds:

| key | holds |
|---|---|
| `name` | the build's full name |
| `weight` | its weight after `prefer`, a float |
| `excluded` | with `--rank` only, for a build outside the result: `below cutoff` or `past limit` |
| `presets` | the presets the choice selects: each `--preset`, then each flag's member (axis defaults apply beneath them) |
| `description`, `license`, `origin` | as written; `license` and `origin` are absent when unset |
| `template_chain`, `memberships`, `facets`, `contract` | as resolved |
| `release` | `{group, released}`, absent when the build is in no release group |
| `tags` | every tag, written and implied, by full name, as plain values (`resolve` shows their sources) |
| `derived` | the tags a derive step added, absent when none |
| `dependencies` | each dependency key's resource |

```text
$ pretrained-catalog --builtin select --select qwen3 --off thinking --limit 1
query:
  limit: 1
  select:
    from:
    - qwen3
    where:
      thinking: false
why:
- from `qwen/qwen3` (namespace)
- "`--off thinking` is `select.where.thinking: false`"
dropped:
- "`qwen/qwen3/qwen3-4b-thinking-2507`: `--off thinking` doesn't apply: `no-thinking` is not offered: on this build the axis `thinking` is fixed at `thinking`"
results:
- name: qwen/qwen3/qwen3-0.6b
  weight: 1.0
  presets:
  - no-thinking
  description: "Qwen3-0.6B: hybrid-thinking chat."
  license: Apache-2.0
  origin: https://huggingface.co/Qwen/Qwen3-0.6B
  template_chain:
  - template_class_root
  - qwen/qwen3/hybrid
  memberships:
  - qwen
  - qwen/qwen3
  - qwen3-lm
  facets:
    family: qwen/qwen3
    publisher: qwen
  contract: qwen3-lm
  tags:
    contract: qwen3-lm
    family: qwen/qwen3
    instruct: true
    language:
    - "*"
    parameters: 600000000
    publisher: qwen
    thinking: switch
  dependencies:
    checkpoint: qwen/qwen3/qwen3-0.6b/checkpoint
    config: qwen/qwen3/qwen3-0.6b/config
    tokenizer: qwen/qwen3/qwen3-0.6b/tokenizer
```

Without `--limit`, `results` holds the whole result. With `--rank`, it holds every ranked build,
and those outside the result say why. Abridged to the fields that differ:

```text
$ pretrained-catalog --builtin select --select whisper --prefer '{release: {step: 0.25}}' \
    --order parameters --cutoff 0.6 --limit 2 --rank
results:
- name: openai/whisper/large/large-v3-turbo
  weight: 1.0
- name: openai/whisper/large/large-v1
  weight: 0.5
  excluded: below cutoff
  release:
    group: openai/whisper/large/release
    released: 2022-09
- name: openai/whisper/large/large-v2
  weight: 0.75
- name: openai/whisper/large/large-v3
  weight: 1.0
  excluded: past limit
```

An empty result is an error, not an empty list. Its message names the builds the presets or flags
dropped.

The other commands help in writing queries: `list --kind tag` and `list --kind axis` show the
keys `where` accepts, `tree` shows the groups to reference, `releases GROUP` shows a release
group's order, and `resolve BUILD` shows a build's tags and presets.

## Lower layers: queries and selections by hand

`choose` is built from two lower layers, which can be used directly:

- `select::expr`: `Query::from_value` reads a query, `compile` checks it against the stack
  (through `TagKeys`, which knows the declared keys and any derived ones), and the
  `CompiledQuery` ranks with `apply` and cuts with `cut`. `merge` combines query values as merge
  patches.
- `select::core`: a `Selection` is a list of weighted candidates, worked by its steps:
  - `derive` adds tags;
  - `weigh` multiplies each weight by a `WeighOp`'s score, and drops a candidate whose score is
    0 or whose weight falls below the epsilon;
  - `weigh_each` does the same with a closure;
  - `order` is a stable sort by a key tuple;
  - `cutoff`, `limit` and `first`.

```rust
use pretrained_catalog::{
    document::EntryKind,
    fixtures,
    select::{
        core::{
            Selection,
            SelectionCandidate,
        },
        expr::{
            Query,
            TagKeys,
        },
    },
};
use serde_json::Value;

let stack = fixtures::builtin_stack().unwrap();
let query: Value = serde_saphyr::from_str(
    "{select: {from: [whisper]}, prefer: {release: latest}, order: [-weight], limit: 1}",
)
.unwrap();
let compiled = Query::from_value(&query)
    .unwrap()
    .compile(&TagKeys::new(&stack, vec![]))
    .unwrap();

let every_build = Selection::new(
    stack
        .of_kind(EntryKind::Build)
        .iter()
        .map(|b| SelectionCandidate::new(stack.resolve_build(b.name()).unwrap())),
);
let ranked = compiled.apply(every_build);
assert_eq!(ranked.len(), 4);

// Any further weighing composes: here, halve the builds without timestamps.
let ranked = ranked
    .weigh_each(|c| if c.is_true("timestamps") { 1.0 } else { 0.5 })
    .order(&[pretrained_catalog::select::core::Order::Descending], |_, w| [Some(w)]);
let result = compiled.cut(ranked);
assert_eq!(result.first().unwrap().0.name().leaf(), "large-v3");
```

The by-hand path skips what `rank` adds on top: names, the defaults, named selections, axis
flags, and dropping builds that can't take the presets.

`select::predicate` holds the typed tests: `Subject` (what a key resolved to), `Predicate`
(`compile`, `holds`, `grade`), and `match_language`, the RFC 4647 matcher. `select::latest::Latest`
is the release-recency `WeighOp`.

## Errors

Every problem is a `CatalogError` with a `kind`, a message, and, where known, the entry and the
catalog it was found in. An import returns `CatalogErrors`, every problem it found, which has
`has(kind)` and `mentions(text)`.

| kind | raised by |
|---|---|
| `Parse`, `Name`, `Structure`, `Conflict`, `Reference`, `Merge`, `Facet`, `Contract`, `Tag`, `Preset`, `Release`, `Dependency` | importing a catalog: the document's rules |
| `Lookup` | `find`, `find_build`: a name that matches nothing, or several entries |
| `Selection` | a query (an unknown key, a reference or key that resolves to nothing, a value that doesn't fit its declaration), a preset selection a build can't take, an axis flag on something that isn't a bool axis, and an empty result |

Selection messages name the part of the query at fault: ``"`where.thinking`: …"``,
``"`select.from`: `qwen3-1.7b`: nothing matches `qwen3-1.7b`"``,
``"`limit` is a count, 1 or more, not 0"``.
