# pretrained-catalog

A CLI, examples and integration tests over bunsen's build catalogs
(`bunsen::data::pretrained::build_catalog`): import catalogs as a stack, list, show and resolve
their entries, and run selection pipelines over them, with each stage explained. The module's
`catalog_format` page describes the catalog document, and its `usage` page the API and the select
language.

## The CLI

```sh
cargo run -p pretrained-catalog -- --catalog a.yml --catalog b.yml <command>   # in order
cargo run -p pretrained-catalog -- --examples <command>                         # the examples
```

| command | does |
|---|---|
| `layers` | the imported catalogs, in order |
| `list [--kind K] [--facet F] [--members G] [--children S]` | entries, by their shortest unique names |
| `show NAME` | an entry's kind, history (which catalog declared, updated or replaced it) and merged fields |
| `resolve BUILD [--preset P] [--on AXIS] [--off AXIS] [--patch {..}]` | the resolved build, and the document its presets make |
| `select [--kind K] [NAME]... [--keep P] [--drop P] [--le KEY V]... [--weigh E] [--cutoff W] [--asc K] [--desc K] [--limit N] [--use NAME] [--unset L] [--per G .. --end] [--pipeline ..] [--floor F] [--preset P] [--on/--off AXIS] [--defaults ..] [--rank]` | a YAML report: the composed query, why (each stage, where it came from, and what it removed), the presets applied, and the result in order, each build resolved with its `weight`; each flag is one stage, in command-line order, but the preset flags, which are the prefix; `--rank` adds every removed row with why |
| `tree [NAME] [--kind K]... [--all] [--depth N]` | the namespace as a directory view, from the root's entries or one entry: `name: description` for a namespace, `name: aliases {axis: default\|other} description` for a build, `name -> target` for an alias. It shows namespaces, builds and the aliases that name them; `--kind` keeps only those kinds (and the containers on the way to them), `--all` shows every kind, as `name (kind): ..` |
| `dump [--format yml\|json]` | the imported catalogs, in order, as one sequence: each item's `layer`, `path` and `catalog`, the document in its keyed form |
| `releases GROUP` | a release group's members, oldest first |

## Running it

```sh
cargo test -p pretrained-catalog
cargo run -p pretrained-catalog -- --examples select qwen3 --off thinking
cargo run -p pretrained-catalog --example catalog_tour   # layers, facets, names, releases
cargo run -p pretrained-catalog --example think_flags    # --think / --no-think as presets
cargo run -p pretrained-catalog --example supplemental   # extensions, fall-through, refusals
```

`--examples` imports the module's example catalogs (`build_catalog::fixtures`: `prelude`,
`commons`, `qwen3`, `whisper`), which the tests and examples use too. Their SHA-256s and
revisions are placeholders. `tests/select_examples.rs` works through the select language over
them, each query with the builds it selects, and why.
