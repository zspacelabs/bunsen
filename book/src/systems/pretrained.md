# Files, caches and pretrained models

This chapter covers how a name such as `tiny` or `hf:openai/whisper-tiny`
becomes a loaded model: what a name means, who answers it, where the files
land, and why bunsen trusts them once they are there. The reference is
[`bunsen::data`], whose
[lifecycle diagram](bunsen::data#lifecycle) links every step,
[`bunsen::data::pretrained`] for the data side (rows, resource bundles,
providers, the cache), and
[`bunsen::kits::support::construct`] for the construct side (the hook, the
deferred model, the factory, the loaded handle).

The kits show three sizes of the same machinery.
[`default_resnet_factory`](bunsen::kits::images::resnet::pretrained::default_resnet_factory)
is the simple case: one compiled-in table of `torchvision` and `timm`
checkpoints.
[`default_whisper_factory`](bunsen::kits::speech::asr::whisper::pretrained::default_whisper_factory)
is the full one: a compiled-in table, and Hugging Face repos.
[`default_silero_factory`](bunsen::kits::speech::vad::silero_vad::pretrained::default_silero_factory)
is the smallest: one compiled-in row, upstream's 16 kHz checkpoint.

## The pathway

```text
spec ("tiny", "hf:org/repo")
  │  the kit's factory asks its providers
  ▼
row: a name over a resource bundle of pinned resources
  │  paired with the kit's hook, optionally overlaid
  ▼
deferred model
  │  plan, bring the files local through the cache, construct
  ▼
loaded model, behind an Arc
```

A [`PretrainedFactory`](bunsen::kits::support::construct::PretrainedFactory) is the
one object a caller holds for a kit, and its
[`load`](bunsen::kits::support::construct::PretrainedFactory::load) runs the whole
pathway. A file on disk does not go through the factory at all: it is a
given resource bundle ([`ResourceBundle::given`](bunsen::data::pretrained::ResourceBundle::given)),
loaded the same way from the deferred step on.

## Prefabs, rows and resources

Three things are kept apart, because they vary independently:

- A **prefab** is a geometry by name, with no weights, so a name means a
  shape before anything is fetched. A kit keeps its prefabs in a
  [`StaticPreFabMap`](bunsen::data::pretrained::StaticPreFabMap).
- A **row**, a [`Pretrained`](bunsen::data::pretrained::Pretrained), is a
  name in a provider, with its aliases, the prefab it instantiates, and the
  resource bundles it is made of.
- A **resource**, a [`Resource`](bunsen::data::pretrained::Resource), is one
  file: the key the kit reads it by, the SHA-256 that pins it, a `kind`
  label, and the places it can be had from, in order.

One prefab has many rows, in many providers. One file can serve many rows:
Whisper's two vocabulary files serve its twelve checkpoints, each as a
resource bundle of its own that the rows reference. Rows, resource bundles and resources
know nothing about how a model is built: that is the kit's business. They
are plain serde data, so a manifest can carry them ([prefabs, rows and
resources](bunsen::data::pretrained#prefabs-rows-and-resources)).

## Providers

A [`PretrainedProvider`](bunsen::data::pretrained::PretrainedProvider) is a
namespace of rows: the `provider` in `provider:ref`. Three are in use.
`well-known:` holds the compiled-in rows. `bundled:` holds the rows a build
ships with, a table its caller registers. `hf:`, an
[`HfProvider`](bunsen::data::pretrained::HfProvider), is a Hugging Face repo,
by `org/repo`.

A factory holds a kit's providers in search order. A spec whose prefix
names a registered provider goes to that provider, and nowhere else. Any
other spec is offered whole to each provider that answers bare names, in
order, and the first row wins. A hub
never answers a bare name, so `hf:` is only reached when the spec names it.
The two kinds of "no" are kept distinct. A provider that doesn't have the
row says so (`Ok(None)`), and the search moves on. A provider that fails
returns an error, and the error aborts the lookup, because a hub that cannot
be reached is not the same as a hub that has no such row
([dispatch](bunsen::kits::support::construct::PretrainedFactory#dispatch)).

A spec is a lookup expression: `[provider:]name`, then an optional box of
presets and an optional YAML patch over the resolved config, as in
`'openai/small[translate]{beam_size: 1}'`. Quote it on a shell command
line, since `[` and `]` glob. Only the name is dispatched. The box and the
patch are checked against the row found before anything is fetched, and a
kit that takes no presets refuses them rather than ignore them
([lookup expressions](bunsen::kits::support::construct::PretrainedFactory#lookup-expressions)).
Well-known names compare ignoring case, so `Qwen/Qwen3-0.6B` finds
`qwen/qwen3-0.6b`. `hf:` names compare as given, since they are paths in
URLs.

There is no process-wide registry. Each kit ships a
`default_{kit}_factory()`, and a caller that wants another provider adds it
to that factory ([no registry](bunsen::kits::support::construct::PretrainedFactory#no-registry)).

## Presets

A row's presets are named, described
[merge patches](bunsen::support::merge_patch) over the config its kit
builds the model under: a Whisper policy, a Sortformer streaming setup, a
Qwen3 generation policy ([`Preset`](bunsen::data::pretrained::Preset)). Which
fields a preset may set is the config's business. A field the construction
reads changes what is built, and one a driver reads changes how it runs.

A row's [`PresetSet`](bunsen::data::pretrained::PresetSet) groups them. An
axis holds presets that exclude one another, such as Whisper's `task`
(`transcribe`, `translate`) or Sortformer's `latency`; a selection names at
most one member of each. The default selection is explicit: the row's
config with nothing selected is its config under the default presets. A
`base` patch holds what every selection keeps, such as the stop tokens a
checkpoint's `generation_config.json` names.

Resolution stacks the layers, each a patch over the one before
([`resolve_presets`](bunsen::kits::support::model_config::resolve_presets)):

1. the family's config, serialized;
2. the set's `base`;
3. the default presets, except where the selection names another member of
   the same axis;
4. the selected presets, as one layer: two that set the same path are an
   error, so their order never matters;
5. the lookup expression's patch.

Every layer is checked against the config it patches, so a misspelled
field fails, naming the preset or the patch, and the result must
deserialize as the config. A kit's hook resolves presets before it builds,
and [`PretrainedFactory::resolve_presets`](bunsen::kits::support::construct::PretrainedFactory::resolve_presets)
resolves a row's config with nothing fetched or built.

A catalog states presets once per template class, and its builds inherit
them under their own. An `hf:` repo that no build describes gets its
template class's. Some presets are read from a model's own files. A
catalog source names probes, such as one that reads
`generation_config.json`, and `cargo catalog` runs them when it builds the
built-in catalog, so a well-known row's presets are fixed data in the build
and need no probing when the model loads. A probe may read a token by name: Gemma's
instruction-tuned rows read the id `tokenizer_config.json` gives
`<end_of_turn>` into their stop tokens. A chat model's template is a preset
field too (`generation.chat_template`): a file of the checkpoint's, which
the `chat_template` probe finds, or a template's source. Gemma's family
states its own rendering of Gemma's turn format, which a row's probed
template lies over, so the pretrained rows, which ship none, chat through
the family's.

What a row's probes find accumulates: two probes' stop ids join. The
hand-written layers are merge patches, which replace. The row's own
presets lay over what its probes found, and the family's lie under the
row's. Then a lookup lays its selected presets (`--preset` too) over the
row's, and last its own patch, `name{generation: {stop_ids: [1]}}`.

`bunsen-tool` shows them. Each listing prints a row's axes, members and
default; `bunsen-tool card --policies` prints the preset sets as data, and
the config each row's default selection resolves to.

## The cache and its trust model

A [`PretrainedCache`](bunsen::data::pretrained::PretrainedCache) keeps each
pinned file at `<cache>/pretrained/<namespace>/<sha256>/<file>`, the namespace a
catalog resource's full name (`qwen/qwen3/qwen3-0.6b/checkpoint`), so a file two
builds share is cached once; files off the catalog sit under `hf` and `given`. **The
digest in the path is the pin.** A file there was verified when it was
written, so later runs trust it without re-hashing gigabytes, and a model
whose pin changes can never collide with a stale copy. The cache trusts the
path whoever wrote it: a download, bundled bytes, or a build script or a
deployment that laid the directory out ahead of time
([trust model](bunsen::data::pretrained::PretrainedCache#trust-model)).

The cache looks at that path first, then at a resource's local sources, and
only then at its URLs
([source order](bunsen::data::pretrained::PretrainedCache#source-order)). A
local directory, such as `openai-whisper`'s own cache, is a "trust me"
source: its file is used in place and hashed only when the cache's options
ask for verification. Nothing is linked or copied into the cache, and only
a download or bundled bytes write there. Downloads need the `fetch` feature.
Without it, or on an offline cache, a file that only a URL can supply is
reported as not found ([features](bunsen::data#features)).

## Bundling is a populated cache directory

Because the cache trusts its layout, shipping a model with a build does not
need a special kind of source. It needs a cache directory that is already
full. [`bind-whisper-onnx`](https://github.com/zspacelabs/bunsen/tree/main/validation/onnx/bind-whisper-onnx),
the Whisper validation crate's asset crate, fetches Whisper's
`base` files at build time into its build directory, laid out as a cache,
and a `PretrainedCache` rooted there finds them without knowing they were
bundled. An offline deployment does the same thing at install time: fill a
cache directory ahead of time, and point the cache at it.

The other form links the bytes into the binary: a
[`Source::Bundled`](bunsen::data::pretrained::Source::Bundled), checked
against its digest and written into the cache on first use. Both forms are
described under
[the cache and bundles](bunsen::data::pretrained#the-cache-and-bundles).

## Deferred models and the kit's hook

What a name resolves to is a
[`Deferred`](bunsen::kits::support::construct::Deferred) model: the row's resource bundle, plus
the kit's [`ConstructInit`](bunsen::kits::support::construct::ConstructInit) hook. The hook
is chosen for the resource bundle by
[`for_resource_bundle`](bunsen::kits::support::construct::ConstructInit::for_resource_bundle), from the
resources' `kind`s, because rows from different providers can need
different readers: an OpenAI `.pt` checkpoint and a Hugging Face safetensors
repo are both Whisper, read two ways.

Loading runs three steps. The hook's
[`plan`](bunsen::kits::support::construct::ConstructInit::plan) completes the resource bundle and
checks what the row promises, such as its prefab, before any bytes are
fetched. The cache then brings the files local. Finally the hook's
[`try_init`](bunsen::kits::support::construct::ConstructInit::try_init) builds the
model from them. Before loading, a caller can lay its own resources over
the row's with
[`with_overlay`](bunsen::kits::support::construct::Deferred::with_overlay), where the
caller's resource wins by key. `bunsen-tool asr family whisper`'s `--vocab` flag works this way.

## PyTorch checkpoints

An OpenAI `.pt` checkpoint is read through burn-store's PyTorch reader, and
a Hugging Face safetensors repo through its safetensors reader, both behind
the `store` feature ([features](../organization/features.md)). The PyTorch reader honours each
tensor's strides, so a weight that PyTorch saved as a transposed view of
another tensor loads as the weight it is.

## The operational view

`bunsen-tool` shows the machinery from the outside. Each family's `list`
shows its rows, their presets, and where each one's files stand, and its
`fetch` brings a model's resources local. Every `--model` takes a lookup
expression, and `--preset` adds to its box; a name no table has fails with
the names nearest it, as "did you mean?"; `bunsen-tool asr family whisper inspect` checks a
checkpoint against its prefab, and `bunsen-tool asr family whisper prefabs` lists
the geometries; `bunsen-tool cache` lists and clears what the cache holds. Its
[README](https://github.com/zspacelabs/bunsen/blob/main/crates/public/bunsen-tool/README.md#whisper-models)
explains Whisper's index.
