# Files, caches and pretrained models

This chapter covers how a name such as `tiny` or `hf:openai/whisper-tiny`
becomes a loaded model: what a name means, who answers it, where the files
land, and why bunsen trusts them once they are there. The reference is
[`bunsen::data`], whose
[lifecycle diagram](bunsen::data#lifecycle) links every step,
[`bunsen::data::builds`] for the pathway (the providers, the hook, the
deferred model, the factory, the loaded handle), and
[`bunsen::data::resources`] for the files (resource sets and the cache).

The kits show three sizes of the same machinery.
[`default_resnet_factory`](bunsen::kits::images::resnet::pretrained::default_resnet_factory)
is the simple case: one compiled-in table of `torchvision` and `timm`
checkpoints.
[`default_whisper_factory`](bunsen::kits::speech::asr::whisper::pretrained::default_whisper_factory)
is the full one: a compiled-in table, and Hugging Face repos.
[`default_silero_factory`](bunsen::kits::speech::vad::silero_vad::pretrained::default_silero_factory)
is the smallest: one compiled-in build, upstream's 16 kHz checkpoint.

## The pathway

```text
spec ("tiny", "hf:org/repo")
  │  the kit's factory asks its providers
  ▼
build: a name over a resource set of pinned resources
  │  paired with the kit's hook, optionally overlaid
  ▼
deferred model
  │  plan, bring the files local through the cache, construct
  ▼
loaded model, behind an Arc
```

A [`BuildFactory`](bunsen::data::builds::BuildFactory) is the
one object a caller holds for a kit, and its
[`load`](bunsen::data::builds::BuildFactory::load) runs the whole
pathway. A file on disk does not go through the factory at all: it is a
given resource set ([`ResourceSet::given`](bunsen::data::resources::ResourceSet::given)),
loaded the same way from the deferred step on.

## Prefabs, builds and resources

Three things are kept apart, because they vary independently:

- A **prefab** is a geometry by name, with no weights, so a name means a
  shape before anything is fetched. A kit keeps its prefabs in a
  [`StaticPreFabMap`](bunsen::data::pretrained::StaticPreFabMap).
- A **build**, a [`ResolvedBuild`](bunsen::data::catalog::ResolvedBuild), is
  what the catalog says one model is: its name and aliases, its tags and
  presets, the geometry it states, and the resources its contract's slots are
  bound to.
- A **resource**, a [`Resource`](bunsen::data::resources::Resource), is one
  file: the slot the kit reads it by, the SHA-256 that pins it, its format,
  and the places it can be had from, in order.

A build states its geometry rather than naming a prefab, and a kit's tests
check that every stated geometry is a prefab's. One file can serve many
builds: Whisper's two vocabulary files serve its twelve checkpoints, each a
resource the builds bind. Builds and resources know nothing about how a model
is built: that is the kit's business ([builds and
resources](bunsen::data::builds#builds-and-resources)).

## Providers

A [`BuildProvider`](bunsen::data::builds::BuildProvider) is a
namespace of builds: the `provider` in `provider:ref`. Two are in use.
`catalog:` holds the builds of the kit's contract in the built-in catalog, and
in any extension layer imported over it: a build that ships its files is a
layer that adds them as sources. `hf:`, an
[`HfProvider`](bunsen::data::builds::HfProvider), is a Hugging Face repo,
by `org/repo`.

A factory holds a kit's providers in search order. A spec whose prefix
names a registered provider goes to that provider, and nowhere else. Any
other spec is offered whole to each provider that answers bare names, in
order, and the first build wins. A hub
never answers a bare name, so `hf:` is only reached when the spec names it.
The two kinds of "no" are kept distinct. A provider that doesn't have the
build says so (`Ok(None)`), and the search moves on. A provider that fails
returns an error, and the error aborts the lookup, because a hub that cannot
be reached is not the same as a hub that has no such build
([dispatch](bunsen::data::builds::BuildFactory#dispatch)).

A spec is a lookup expression: `[provider:]name`, then an optional box of
presets and an optional YAML patch over the resolved config, as in
`'openai/small[translate]{beam_size: 1}'`. Quote it on a shell command
line, since `[` and `]` glob. Only the name is dispatched. The box and the
patch are checked against the build found before anything is fetched, and a
kit that takes no presets refuses them rather than ignore them
([lookup expressions](bunsen::data::builds::BuildFactory#lookup-expressions)).
Well-known names compare ignoring case, so `Qwen/Qwen3-0.6B` finds
`qwen/qwen3-0.6b`. `hf:` names compare as given, since they are paths in
URLs.

There is no process-wide registry. Each kit ships a
`default_{kit}_factory()`, and a caller that wants another provider adds it
to that factory ([no registry](bunsen::data::builds::BuildFactory#no-registry)).

## Presets

A build's presets are named, described
[merge patches](bunsen::support::merge_patch) over the config its kit
builds the model under: a Whisper policy, a Sortformer streaming setup, a
Qwen3 generation policy
([`PresetEntry`](bunsen::data::catalog::document::PresetEntry)).
Which fields a preset may set is the config's business. A field the
construction reads changes what is built, and one a driver reads changes how
it runs.

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

Every layer is checked against the config it patches, so a misspelled
field fails, naming the preset or the patch, and the result must
deserialize as the config. A kit's hook resolves presets before it builds,
and [`BuildFactory::resolve_presets`](bunsen::data::builds::BuildFactory::resolve_presets)
resolves a build's config with nothing fetched or built.

A catalog states presets once per template class, and its builds inherit
them under their own. An `hf:` repo that no build describes gets its
template class's. Some presets are read from a model's own files. A
catalog source names probes, such as one that reads
`generation_config.json`, and `cargo catalog` runs them when it builds the
built-in catalog, so a catalog build's presets are fixed data in the build
and need no probing when the model loads. A probe may read a token by name: Gemma's
instruction-tuned builds read the id `tokenizer_config.json` gives
`<end_of_turn>` into their stop tokens. A chat model's template is a preset
field too (`generation.chat_template`): a file of the checkpoint's, which
the `chat_template` probe finds, or a template's source. Gemma's family
states its own rendering of Gemma's turn format, which a build's probed
template lies over, so the pretrained builds, which ship none, chat through
the family's.

What a build's probes find accumulates: two probes' stop ids join. The
hand-written layers are merge patches, which replace. The build's own
presets lay over what its probes found, and the family's lie under the
build's. Then a lookup lays its selected presets (`--preset` too) over the
build's, and last its own patch, `name{generation: {stop_ids: [1]}}`.

`bunsen-tool` shows them. Each listing prints a build's axes, members and
default; `bunsen-tool card --policies` prints the preset sets as data, and
the config each build's default selection resolves to.

## The cache and its trust model

A [`ResourceCache`](bunsen::data::resources::ResourceCache) keeps each
pinned file at `<cache>/resources/<name>/<sha256>/<file>`, the name a
catalog resource's full name (`qwen/qwen3/qwen3-0.6b/checkpoint`), so a file two
builds share is cached once; an `hf:` repo's files sit under `hf/<org>/<repo>/<slot>`, and a given path's under `given`. **The
digest in the path is the pin.** A file there was verified when it was
written, so later runs trust it without re-hashing gigabytes, and a model
whose pin changes can never collide with a stale copy. The cache trusts the
path whoever wrote it: a download, bundled bytes, or a build script or a
deployment that laid the directory out ahead of time
([trust model](bunsen::data::resources::ResourceCache#trust-model)).

The cache looks at that path first, then at a resource's local sources, and
only then at its URLs
([source order](bunsen::data::resources::ResourceCache#source-order)). A
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
and a `ResourceCache` rooted there finds them without knowing they were
bundled. An offline deployment does the same thing at install time: fill a
cache directory ahead of time, and point the cache at it.

The other form links the bytes into the binary: a
[`Source::Bundled`](bunsen::data::resources::Source::Bundled), checked
against its digest and written into the cache on first use. Both forms are
described under
[the cache and bundles](bunsen::data::resources#the-cache-and-bundles).

## Deferred models and the kit's hook

What a name resolves to is a
[`Deferred`](bunsen::data::builds::Deferred) model: the build's resource set, plus
the kit's [`ConstructInit`](bunsen::data::builds::ConstructInit) hook. The hook
is chosen for the resource set by
[`for_resource_set`](bunsen::data::builds::ConstructInit::for_resource_set), from the
resources' `format`s, because builds from different providers can need
different readers: an OpenAI `.pt` checkpoint and a Hugging Face safetensors
repo are both Whisper, read two ways.

Loading runs three steps. The hook's
[`plan`](bunsen::data::builds::ConstructInit::plan) completes the resource set and
checks what the build promises, such as its stated geometry, before any bytes are
fetched. The cache then brings the files local. Finally the hook's
[`try_init`](bunsen::data::builds::ConstructInit::try_init) builds the
model from them. Before loading, a caller can lay its own resources over
the build's with
[`with_overlay`](bunsen::data::builds::Deferred::with_overlay), where the
caller's resource wins by key. `bunsen-tool asr family whisper`'s `--vocab` flag works this way.

## PyTorch checkpoints

An OpenAI `.pt` checkpoint is read through burn-store's PyTorch reader, and
a Hugging Face safetensors repo through its safetensors reader, both behind
the `store` feature ([features](../organization/features.md)). The PyTorch reader honours each
tensor's strides, so a weight that PyTorch saved as a transposed view of
another tensor loads as the weight it is.

## The operational view

`bunsen-tool` shows the machinery from the outside. Each family's `list`
shows its builds, their presets, and where each one's files stand, and its
`fetch` brings a model's resources local. Every `--model` takes a lookup
expression, and `--preset` adds to its box; a name no table has fails with
the names nearest it, as "did you mean?"; `bunsen-tool asr family whisper inspect` checks a
checkpoint against its prefab, and `bunsen-tool asr family whisper prefabs` lists
the geometries; `bunsen-tool cache` lists and clears what the cache holds. Its
[README](https://github.com/zspacelabs/bunsen/blob/main/crates/public/bunsen-tool/README.md#whisper-models)
explains Whisper's index.
