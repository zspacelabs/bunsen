# The catalog

The catalog is what bunsen can build, as data: a stack of YAML layers that
name contracts, builds, resources, tags and presets, imported one after
another into one materialized state. The reference is
[`bunsen::data::catalog`]: its
[format](bunsen::data::catalog::catalog_format) describes the document and
the stack, and its [usage guide](bunsen::data::catalog::usage) the API, from
importing layers to choosing a build.

## Entries

Every entry has a full name, a `/`-separated path (`qwen/qwen3/qwen3-0.6b`),
and a kind:

- a **contract** is the interface a loader implements: the resource slots it
  reads, each with the formats it accepts, and the argument slots it is
  told. A kit's contract is its own `contract.yml`, gathered into the base
  layer's `$contracts`; a dataset contract, such as `parquet-shards`, is
  `data::datasets`'.
- a **build** is one thing a loader builds: a model, or a data source. It
  names its contract, binds the contract's slots to resources and arguments
  (a stated geometry, for a kit whose weights carry none), and carries its
  tags and presets. A **template class** is what builds of one family
  share, and a build's presets fold down its chain.
- a **resource** is one artifact: a file, a sharded checkpoint (its index
  and shards), or a templated set of numbered files, a dataset's shards. Its
  sources are places (`$hub`, `$base_url`, `$local_dir`) or URLs.
- **namespaces**, **facets** and **releases** group builds: a publisher, a
  family, a release sequence. **Tags** and **axes** are declared once, in the
  base layer, and builds state them; a tag's declaration is its type.

Names are looked up by shorter spellings too (`qwen3-0.6b`), and an alias
is another name for one entry. A Hugging Face repo that no catalog names is
a build under the reserved root `hf` (`hf/openai/whisper-tiny`), whose names
keep the Hub's spelling.

## The built-in catalog

The catalog bunsen ships is
[`bunsen-catalog`](https://github.com/zspacelabs/bunsen/tree/main/crates/public/bunsen-catalog):
the prelude (the vocabulary and the contracts), the commons (the
publishers), the datasets, and a layer per kit, each with the bunsen feature
it needs. [`BuiltinCatalog::get`](bunsen::data::catalog::BuiltinCatalog::get)
imports and resolves it once in a process; its
[`build`](bunsen::data::catalog::BuiltinCatalog::build) is what a name
means, and its [`provider`](bunsen::data::catalog::BuiltinCatalog::provider)
is a kit's `catalog:` provider.

The layers are built, not hand-written. `cargo catalog`, in the validation
workspace, reads each layer's source under
`validation/catalog_builder/probe_contracts/`, pins every file from its
remote (an LFS digest from the Hub's listing, a small file hashed as it is
fetched), runs the probes a source names over a model's files to fill its
presets, and writes the built layers. `cargo catalog check` fails when the
checked-in catalog is not what the sources build.

## Presets

A build's presets are named, described
[merge patches](bunsen::support::merge_patch) over the config its kit builds
the model under: a Whisper policy, a Sortformer streaming setup, a Qwen3
generation policy
([`PresetEntry`](bunsen::data::catalog::document::PresetEntry)). Which fields
a preset may set is the config's business. A field the construction reads
changes what is built, and one a driver reads changes how it runs.

A build's [`ResolvedPresetSet`](bunsen::data::catalog::presets::ResolvedPresetSet)
groups them. An axis holds presets that exclude one another, such as
Whisper's `task` (`transcribe`, `translate`) or Sortformer's `latency`; a
selection names at most one member of each. The default selection is
explicit: each axis has a default member, and the set names the presets on
no axis that apply by default, so the build's config with nothing selected is
its config under the default presets. A `common` patch holds what every
selection keeps, such as the stop tokens a checkpoint's
`generation_config.json` names.

Resolution stacks the layers, each a patch over the one before
([`resolve_presets`](bunsen::kits::support::model_config::resolve_presets)):

1. the family's config, serialized;
2. the set's `common` patch;
3. the default presets, except where the selection names another member of
   the same axis;
4. the selected presets, as one layer: two that set the same path are an
   error, so their order never matters;
5. the lookup expression's patch.

Every layer is checked against the config it patches, so a misspelled field
fails, naming the preset or the patch, and the result must deserialize as
the config. A kit's hook resolves presets before it builds, and
[`BuildFactory::resolve_presets`](bunsen::data::builds::BuildFactory::resolve_presets)
resolves a build's config with nothing fetched or built.

A catalog states presets once per template class, and its builds inherit
them under their own. An `hf:` repo that no build describes gets its
template class's. Some presets are read from a model's own files: a source
names probes, such as one that reads `generation_config.json`, and `cargo
catalog` runs them when it builds the catalog, so a build's presets are
fixed data and need no probing when the model loads. A probe may read a
token by name: Gemma's instruction-tuned builds read the id
`tokenizer_config.json` gives `<end_of_turn>` into their stop tokens. A chat
model's template is a preset field too (`generation.chat_template`): a file
of the checkpoint's, which the `chat_template` probe finds, or a template's
source.

What a build's probes find accumulates: two probes' stop ids join. The
hand-written layers are merge patches, which replace. The build's own
presets lay over what its probes found, and the family's lie under the
build's. Then a lookup lays its selected presets (`--preset` too) over the
build's, and last its own patch, `name{generation: {stop_ids: [1]}}`.
`bunsen-tool card --policies` prints the preset sets as data, and the config
each build's default selection resolves to.

## Choosing a build

The catalog also chooses among builds: a selection is a pipeline of stages
over the builds a command may pick from, which keep, weigh and order them by
their tags ([`select`](bunsen::data::catalog::select)). `bunsen-tool chat`'s
`--model`, `--keep` and `--weigh` are stages, and `--think` is a shortcut
into a build's presets: it keeps only the builds that can think, and selects
the preset that makes them.

## Extension layers

A stack takes layers beyond the built-in ones. A later layer may declare new
entries, or meet an existing one by its `$merge`: `update` adds to a
namespace, retargets an alias, or adds sources to a resource, tried first;
`replace` substitutes an entry. A build is reached as a scope: an extending
build occurrence carries only its kind and children, so a layer can add a
source to a resource the build declares.

That is how a deployment ships its files: a layer that updates its builds'
resources with local-dir sources, so the same name loads from the shipped
files. `whisper-model-validation` builds its bundle's layer that way, and
loads `openai/whisper/base` through two factories, with the layer and
without it.
