# Files, caches and pretrained models

This chapter covers how a name such as `tiny` or `hf:openai/whisper-tiny`
becomes a loaded model: what a name means, who answers it, where the files
land, and why bunsen trusts them once they are there. The reference is
[`bunsen::data`], whose
[lifecycle diagram](bunsen::data#lifecycle) links every step, and
[`bunsen::data::pretrained`].

The kits show three sizes of the same machinery.
[`default_resnet_factory`](bunsen::kits::images::resnet::default_resnet_factory)
is the simple case: one compiled-in table of `torchvision` and `timm`
checkpoints.
[`default_whisper_factory`](bunsen::kits::speech::whisper::pretrained::default_whisper_factory)
is the full one: a compiled-in table, a bundled row, and Hugging Face repos.
[`default_silero_factory`](bunsen::kits::speech::silero_vad::pretrained::default_silero_factory)
is bundled-only: one row whose bytes are linked into the binary by the
`silero-weights` feature.

## The pathway

```text
spec ("tiny", "hf:org/repo")
  │  the kit's factory asks its providers
  ▼
row: a name over a map of pinned resources
  │  paired with the kit's hook, optionally overlaid
  ▼
deferred model
  │  plan, bring the files local through the cache, construct
  ▼
loaded model, behind an Arc
```

A [`PretrainedFactory`](bunsen::data::pretrained::PretrainedFactory) is the
one object a caller holds for a kit, and its
[`load`](bunsen::data::pretrained::PretrainedFactory::load) runs the whole
pathway. A file on disk does not go through the factory at all: it is a
given map ([`ResourceMap::given`](bunsen::data::pretrained::ResourceMap::given)),
loaded the same way from the deferred step on.

## Prefabs, rows and resources

Three things are kept apart, because they vary independently:

- A **prefab** is a geometry by name, with no weights, so a name means a
  shape before anything is fetched. A kit keeps its prefabs in a
  [`StaticPreFabMap`](bunsen::data::pretrained::StaticPreFabMap).
- A **row**, a [`Pretrained`](bunsen::data::pretrained::Pretrained), is a
  name in a provider, with its aliases, the prefab it instantiates, and the
  resource maps it is made of.
- A **resource**, a [`Resource`](bunsen::data::pretrained::Resource), is one
  file: the key the kit reads it by, the SHA-256 that pins it, a `kind`
  label, and the places it can be had from, in order.

One prefab has many rows, in many providers. One file can serve many rows:
Whisper's two vocabulary files serve its twelve checkpoints, each as a
resource map of its own that the rows reference. Rows, maps and resources
know nothing about how a model is built: that is the kit's business. They
are plain serde data, so a manifest can carry them ([prefabs, rows and
resources](bunsen::data::pretrained#prefabs-rows-and-resources)).

## Providers

A [`PretrainedProvider`](bunsen::data::pretrained::PretrainedProvider) is a
namespace of rows: the `provider` in `provider:ref`. Three are in use.
`well-known:` holds the compiled-in rows. `bundled:` holds the rows a build
ships with, registered under a kit's `*-weights` feature. `hf:`, an
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
([dispatch](bunsen::data::pretrained::PretrainedFactory#dispatch)).

There is no process-wide registry. Each kit ships a
`default_{kit}_factory()`, and a caller that wants another provider adds it
to that factory ([no registry](bunsen::data::pretrained::PretrainedFactory#no-registry)).

## The cache and its trust model

A [`PretrainedCache`](bunsen::data::pretrained::PretrainedCache) keeps each
pinned file at `<cache>/pretrained/<kit>/<namespace>/<sha256>/<file>`. **The
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
full. [`bunsen_bundled_whisper`] fetches Whisper's
`base` files at build time into its build directory, laid out as a cache,
and a `PretrainedCache` rooted there finds them without knowing they were
bundled. An offline deployment does the same thing at install time: fill a
cache directory ahead of time, and point the cache at it.

The other form links the bytes into the binary.
[`bunsen_bundled_silero`]'s burnpack is a
[`Source::Bundled`](bunsen::data::pretrained::Source::Bundled), checked
against its digest and written into the cache on first use. Both forms are
described under
[the cache and bundles](bunsen::data::pretrained#the-cache-and-bundles).

## Deferred models and the kit's hook

What a name resolves to is a
[`Deferred`](bunsen::data::pretrained::Deferred) model: the row's map, plus
the kit's [`Construct`](bunsen::data::pretrained::Construct) hook. The hook
is chosen for the map by
[`for_map`](bunsen::data::pretrained::Construct::for_map), from the
resources' `kind`s, because rows from different providers can need
different readers: an OpenAI `.pt` checkpoint and a Hugging Face safetensors
repo are both Whisper, read two ways.

Loading runs three steps. The hook's
[`plan`](bunsen::data::pretrained::Construct::plan) completes the map and
checks what the row promises, such as its prefab, before any bytes are
fetched. The cache then brings the files local. Finally the hook's
[`construct`](bunsen::data::pretrained::Construct::construct) builds the
model from them. Before loading, a caller can lay its own resources over
the row's with
[`with_overlay`](bunsen::data::pretrained::Deferred::with_overlay), where the
caller's resource wins by key. whisper-cli's `--vocab` flag works this way.

## PyTorch checkpoints

Some checkpoints need repair on the way in. PyTorch may save a weight as a
transposed view of another tensor, and `burn-store`'s PyTorch reader reads
the raw storage as if it were row-major, which scrambles a non-square
weight. [`bunsen::burner::store`] attaches the repair
to a built module's parameters, so it runs only as the weights cross the
store boundary.
[`FixPytorchLoadMappers`](bunsen::burner::store::FixPytorchLoadMappers)
walks a module tree and attaches it wherever it is needed, and Whisper's
loader applies it before reading an OpenAI checkpoint.

## The operational view

whisper-cli's `models` subcommand shows the machinery from the outside. It
lists the rows and where each one stands (cached, in a local directory, or
remote), lists the prefabs, fetches a model's resources, and inspects a
checkpoint against its prefab. Its
[README](https://github.com/zspacelabs/bunsen/blob/main/examples/whisper-cli/README.md#models)
walks through a session.
