# pretrained-catalog

The pretrained catalog's runtime logic on its own: how a catalog is read, validated, queried
and resolved. It covers the document types and their YAML/JSON form, loading catalogs as an
ordered stack with validation, fall-through queries, and template-class resolution with the
preset merge. It doesn't depend on bunsen. How catalogs are built, how resources are cached
and how contracts load are out of its scope, so the catalog rules can be worked on apart from
the builder and the kits.

## The document

A catalog is a list of entries, and every entry names its `kind`. The set of kinds is closed:

| kind | what it is |
|---|---|
| `namespace` | A plain group: a naming scope, and a membership. |
| `contract` | A group that also lists the `dependency_keys` its loader reads. Every build is in exactly one, and carries a `contract` tag naming it. |
| `release` | A group whose members are releases of one sequence. A build joins it as `{group, released}`, and `latest` orders the members by time. It holds no children. |
| `build` | Something we can build. It names a `template_class` and its tags, presets and dependencies. It can hold its own resources, tags and axes. |
| `template_class` | A preamble that builds inherit along a chain of `base` classes, ending at `template_class_root`. It may sit in a namespace, but its placement confers no membership. |
| `tag` | A declared tag, with a type. No tag is undeclared. |
| `axis` | A registered preset axis. A `bool` axis `X` has the members `X` and `no-X`, with a negation between them; a `categorical` axis has declared members and no negation. An axis is also its own tag's declaration. |
| `facet` | A declared, exclusive set of groups (`publisher`, `family`). A build is in at most one group of each facet. A facet becomes required through a template class's `required_facets`. |
| `resource` | One artifact: a file, its format, an optional SHA-256, and its sources. |

**Names** are `/`-separated paths. Each segment is a plain, unquoted YAML key, and only
ASCII is allowed. A group or build may nest its `children`, which is the same as each child
naming it as `parent`. Collisions are kind-flat: a scope holds one entry per name, whatever
the kind, and a second occurrence of a name in its scope meets the first by its `merge`. A
name may repeat in other scopes, above, below or beside, and a lookup by a shorter name takes
the spelling only one entry matches.

**References** resolve from the entry they are written in. A relative reference written in
`a/b/c` names the first of `a/b/c/ref`, `a/b/ref`, `a/ref` and `ref` that is declared. A
leading `/` resolves from the root only. References resolve against the state, and a catalog
may not change what an earlier catalog's reference resolves to: a name nearer to it than what
it names would capture it, and is refused. Only a `replace` changes an entry. A template class's references resolve where
the class is written, not where its builds are.

**Memberships** are declared only by the member, through `parent` (its naming scope) and
`join`. They are transitive. A template class's `join` applies to every build of the class.

**Presets** fold down the chain, root first. A use of an axis
(`{use, offers, default}`) may only narrow what the use above it offered. When it narrows,
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

An occurrence's `merge` says how it meets an existing entry of its name:

- `deny`, the default, declares the name; an existing entry is a conflict.
- `update` updates it, as its kind defines:
  - a `namespace` adds its children, each of which meets any existing child by its own
    `merge`, and may assert the facet it expects;
  - a `resource` must match every field but `sources`, and its new sources are tried first;
  - every other kind refuses it. A contract is the interface its loader is built against.
- `replace` substitutes the occurrence's own fields for the entry's, for every reference to
  it. The kind stays the same, and children are entries of their own.

A build can depend on any resource in the state.

## Selection

A query is `{select, prefer, order, cutoff, limit}`:

```yaml
select:                            # the hard filter
  from:    [qwen3]                 # a member of at least one
  require: [hybrid]                # a member of every one
  exclude: [qwen3-2507]            # a member of none
  where:   {thinking: false, base: false, parameters: {max: 8e9}}
prefer:                            # soft scores
  where:   {language: {is: en, otherwise: 0.5}}
  release: latest
order: [-weight, -parameters]
cutoff: 0.5
limit: 3
```

- **Membership** lists references to groups (namespaces, contracts and release groups, through
  their membership closure), template classes (the build's chain includes the class), or builds.
- **`where`** keys are declared axes and tags, and each value is typed by its declaration:
  - a bool axis takes `true`, `false`, `switch`, `always` or `never`;
  - for a set-valued key (categorical axis offers, `text_list`, `languages`), a list means
    "all of", and `{any}` and `{only}` are available;
  - for a single-valued key, a list means "any of", and a count takes `{min, max}`.

  A value that doesn't fit its declaration is an error. Facet keys and `contract` are refused,
  because membership is in the lists.
- **`prefer`** mirrors `where`, with `otherwise` and `truncated` per entry; a `languages` tag is
  graded by RFC 4647. `release` ranks each release group by release time.
- **`order`** sorts by `weight` and numeric tags; **`cutoff`** drops the builds whose weight is
  below it, and **`limit`** keeps the first so many of the rest. That list is the result. Loading
  a model takes one build, the first: an implicit limit of 1.
- Queries merge as RFC 7386 merge patches: the defaults, then named selections, then each patch.
- **Flags fill the keys of their name:**
  - `--select NAME` adds to `select.from`, and several names are a union; when the only name is
    one build, the defaults don't apply;
  - `--select {..}`, `--prefer {..}`, `--order`, `--cutoff` and `--limit` fill their own keys, and
    `--query {..}` patches the whole query;
  - `--on`/`--off AXIS` is `select.where.AXIS: true/false`, and selects that member's preset; a
    build that can't take it is dropped with the reason.

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
| `select [--select NAME\|{..}]... [--prefer {..}]... [--order KEY]... [--cutoff W] [--limit N] [--query {..}]... [--preset P] [--on/--off AXIS] [--defaults {..}] [--rank]` | a YAML report: the merged query, why, and the result in order, each build resolved with its `weight` and `presets` (all of it unless `--limit`); `--rank` lists every ranked build, with `excluded: below cutoff` or `past limit` on those outside the result |
| `tree [NAME] [--kind K]... [--depth N]` | the namespace as a tree, from the root or one entry, annotated (facets, releases, base classes, files); `--kind` keeps only those kinds and the containers on the way to them |
| `releases GROUP` | a release group's members, oldest first |

## Running it

```sh
cargo test -p pretrained-catalog
cargo run -p pretrained-catalog -- --builtin select --select qwen3 --off thinking
cargo run -p pretrained-catalog --example catalog_tour   # layers, facets, names, releases
cargo run -p pretrained-catalog --example think_flags    # --think / --no-think as presets
cargo run -p pretrained-catalog --example supplemental   # extensions, fall-through, refusals
```

The example catalogs in `catalogs/` (`prelude`, `commons`, `qwen3`, `whisper`, and the
`supplemental`) are what the tests and examples use, through `fixtures`. Their
SHA-256s and revisions are placeholders.
