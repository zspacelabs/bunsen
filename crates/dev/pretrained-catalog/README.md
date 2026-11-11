# pretrained-catalog

The pretrained catalog's runtime logic on its own: how a catalog is read, validated, queried
and resolved. It covers the document types and their YAML/JSON form, loading catalogs as an
ordered stack with validation, fall-through queries, and template-class resolution with the
preset merge. It doesn't depend on bunsen. How catalogs are built, how resources are cached
and how contracts load are out of its scope, so the catalog rules can be worked on apart from
the builder and the kits.

## The document

A catalog document is the namespace tree. **Every key the catalog language defines starts
with `$`, and a bare key is a name.** Abridged from `catalogs/qwen3.yml`:

```yaml
$description: "Qwen3 chat models."
qwen:
  $kind: namespace
  $merge: update
  $facet: publisher
  qwen3:
    $kind: namespace
    $facet: family
    $description: "Qwen3."
    hybrid:
      $kind: template_class
      $description: "Qwen3 hybrid-thinking chat."
      $join: [qwen3-lm]
      instruct: true                     # a tag
      $presets:
        $common: { generation: { max_new_tokens: 2048 } }
        $axes:
          thinking: { $offers: [thinking, no-thinking], $default: thinking }
        thinking: { $patch: { generation: { temperature: 0.6 } } }
    qwen3-0.6b:
      $kind: build
      $description: "Qwen3-0.6B."
      $template_class: hybrid
      parameters: 600000000              # a tag
      $dependencies: { checkpoint: checkpoint }
      checkpoint: { $kind: resource, $file: model.safetensors, $format: safetensors, $sources: [] }
```

- **An entry** is a map with `$kind`, under its segment. Its name is its path of keys: nesting
  is the only placement.
- **Bare keys** under a namespace or contract are children; under a template class, tags; under
  a build, a child when the value is a map with `$kind`, and a tag otherwise. Other kinds
  have none.
- **`$presets`** holds `$common` (a patch every selection keeps), `$axes` (axis uses, keyed by
  the axis), `$default`, and the presets, keyed by name.
- **Foreign documents are data:** a preset's `$patch` and `$common` keep the model's own keys.
- **Refused:** a `$` word the kind doesn't define, a child without `$kind`, a bare key where
  the kind has none, and a repeated key. So one catalog states a name once.

The set of kinds is closed:

| kind | what it is |
|---|---|
| `namespace` | A plain group: a naming scope, and a membership. |
| `contract` | A group that also lists the `$dependency_keys` its loader reads. Every build is in exactly one, and carries a `contract` tag naming it. |
| `release` | A group whose members are releases of one sequence. A build joins it as `{$group, $released}`, and `latest` orders the members by time. It holds no children. |
| `build` | Something we can build. It names a `$template_class` and its tags, presets and dependencies. It can hold its own resources, tags and axes. |
| `template_class` | A preamble that builds inherit along a chain of `$base` classes, ending at `template_class_root`. It may sit in a namespace, but its placement confers no membership. |
| `tag` | A declared tag, with a `$type`. No tag is undeclared. |
| `axis` | A registered preset axis. A `bool` axis `X` has the members `X` and `no-X`, with a negation between them; a `categorical` axis has declared `$members` and no negation. An axis is also its own tag's declaration. |
| `facet` | A declared, exclusive set of groups (`publisher`, `family`). A build is in at most one group of each facet. A facet becomes required through a template class's `$required_facets`. |
| `resource` | One artifact: a `$file`, its `$format`, an optional `$sha256` and `$dtype`, and its `$sources`. |

**Names** are `/`-separated paths. Each segment is a plain, unquoted YAML key, and only
ASCII is allowed. Collisions are kind-flat: a scope holds one entry per name, whatever
the kind, and a second occurrence of a name in its scope meets the first by its `$merge`. A
name may repeat in other scopes, above, below or beside, and a lookup by a shorter name takes
the spelling only one entry matches.

**References** resolve from the entry they are written in. A relative reference written in
`a/b/c` names the first of `a/b/c/ref`, `a/b/ref`, `a/ref` and `ref` that is declared. A
leading `/` resolves from the root only. References resolve against the state, and a catalog
may not change what an earlier catalog's reference resolves to: a name nearer to it than what
it names would capture it, and is refused. Only a `replace` changes an entry. A template
class's references resolve where the class is written, not where its builds are.

**Memberships** are declared only by the member, through its naming scope and `$join`. They
are transitive. A template class's `$join` applies to every build of the class.

**Presets** fold down the chain, root first. A preset with an inherited preset's name meets
it by its `$merge`: `update` (the default) merges its patch over the inherited one, `replace`
takes its place, and `deny` refuses it. A use of an axis
(`{$offers, $default}`, under `$axes`) may only narrow what the use above it offered. When it narrows,
the presets of the dropped members are pruned. An offer of one member is a fixed mode. The
tags implied by the axes used (`thinking: always | switch | never`) and by the facets
(`family: qwen/qwen3`) are computed, never written.

## The stack

`PretrainedCatalogStack` imports catalogs one after another into one materialized state.
Importing a catalog applies its occurrences, in order, to a copy of the state, then validates
the copy as a whole, and commits it only if it is clean. The state before was valid, so any
problem is the importing catalog's, and it is refused whole. Queries read the state. The
imported catalogs are kept as provenance: each entry records the catalogs that declared,
updated and replaced it.

An occurrence's `$merge` says how it meets an existing entry of its name:

- `deny`, the default, declares the name; an existing entry is a conflict.
- `update` updates it, as its kind defines:
  - a `namespace` adds its children, each of which meets any existing child by its own
    `$merge`, and may assert the `$facet` it expects;
  - a `resource` must match every field but `$sources`, and its new sources are tried first;
  - every other kind refuses it. A contract is the interface its loader is built against.
- `replace` substitutes the occurrence's own fields for the entry's, for every reference to
  it. The kind stays the same, and children are entries of their own.

A build can depend on any resource in the state.

## Selection

A query is a kind and a list of stages, written only in YAML particles:

```yaml
kind: build
run:
  - from: [qwen3, whisper]                       # in any of them
  - keep: {instruct: true, thinking: false, parameters: {le: 8.9e9}}
  - drop: qwen3-2507                             # keep: {$not: ..}
  - weigh: {by: [{when: {language: en}, else: 0.5}, {release: latest}]}
  - keep: {$weight: {ge: 0.5}}                   # a cutoff, where it sits
  - order: [{desc: $weight}, {desc: parameters}]
  - limit: 3
```

- **No string is parsed.** A string is a name, a `$` word or a literal; a map is an operator or
  a test. Wherever a catalog name can appear, anything that isn't one starts with `$`
  (`$any`, `$not`, `$weight`, `$name`, `$presets`, `$released`, the field words `$license`,
  `$origin`, `$parent`, `$template_class`, `$contract`, and derived keys).
- **Each stage maps a table of weighted rows to a table.** The run starts from every build, in
  full-name order, at weight 1, and no stage adds a row.
- **Names** test membership: a group by closure, a template class by chain, a build itself. A
  bool axis (`thinking`) tests what its use offers; tags take value tests (`eq`, `ne`, `lt`,
  `le`, `gt`, `ge`, `in`, `has`, `has_all`, `has_any`, `exists`), typed by their declaration.
- **`weigh`** is a block of updates (`when`, `match`, `release`, `mul`, `add`, `clip`),
  committed clipped to 0 or more at its end; the query's `floor` drops what counts as zero.
- **A query only filters, weighs and orders.** It never selects presets. A tool composes its
  pipeline as its preset prefix (`keep: {$presets: [..]}`, so every row can take the presets it
  will apply), then its defaults, then the request (`use` splices in a named selection), then a
  tail (`order: {desc: $weight}`, and `limit: 1` when choosing).

`tests/select_examples.rs` works through the language over the example catalogs, each query
with the builds it selects, and why.

[`docs/usage.md`](docs/usage.md) is the full guide to the API and the select language. It is
also the crate's `usage` module in rustdoc, and its Rust examples run as doctests.

## The CLI

```sh
cargo run -p pretrained-catalog -- --catalog a.yml --catalog b.yml <command>   # in order
cargo run -p pretrained-catalog -- --builtin <command>                         # the examples
```

| command | does |
|---|---|
| `layers` | the imported catalogs, in order |
| `list [--kind K] [--facet F] [--members G] [--children S]` | entries, by their shortest unique names |
| `show NAME` | an entry's kind, history (which catalog declared, updated or replaced it) and merged fields |
| `resolve BUILD [--preset P] [--on AXIS] [--off AXIS] [--patch {..}]` | the resolved build, and the document its presets make |
| `select [NAME]... [--keep P] [--drop P] [--le KEY V]... [--weigh E] [--cutoff W] [--asc K] [--desc K] [--limit N] [--use NAME] [--pipeline ..] [--floor F] [--preset P] [--on/--off AXIS] [--defaults ..] [--rank]` | a YAML report: the composed query, why (each stage, where it came from, and what it removed), the presets applied, and the result in order, each build resolved with its `weight`; each flag is one stage, in command-line order, but the preset flags, which are the prefix; `--rank` adds every removed row with why |
| `tree [NAME] [--kind K]... [--depth N]` | the namespace as a tree, from the root or one entry, annotated (facets, releases, base classes, files); `--kind` keeps only those kinds and the containers on the way to them |
| `releases GROUP` | a release group's members, oldest first |

## Running it

```sh
cargo test -p pretrained-catalog
cargo run -p pretrained-catalog -- --builtin select qwen3 --off thinking
cargo run -p pretrained-catalog --example catalog_tour   # layers, facets, names, releases
cargo run -p pretrained-catalog --example think_flags    # --think / --no-think as presets
cargo run -p pretrained-catalog --example supplemental   # extensions, fall-through, refusals
```

The example catalogs in `catalogs/` (`prelude`, `commons`, `qwen3`, `whisper`, and the
`supplemental`) are what the tests and examples use, through `fixtures`. Their
SHA-256s and revisions are placeholders.
