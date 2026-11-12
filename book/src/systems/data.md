# Data: from files to models

This section covers how a name such as `tiny`, `hf:openai/whisper-tiny` or
`fineweb-edu-100b-shuffle` becomes a loaded model or a set of training
files: what a name means, who answers it, where the files land, and why
bunsen trusts them once they are there. The reference is
[`bunsen::data`], whose [lifecycle diagram](bunsen::data#lifecycle) links
every step.

`bunsen::data` is split along what each part is about, each built on the
one before:

| Module | What it holds | Chapter |
|---|---|---|
| [`cache`](bunsen::data::cache) | the disk cache: where files land, digests, links, downloads | [Resources and the cache](./data/resources.md) |
| [`resources`](bunsen::data::resources) | pinned files, the places they come from, the resource cache, the Hub client | [Resources and the cache](./data/resources.md) |
| [`catalog`](bunsen::data::catalog) | what there is to build, as data: a layered catalog of contracts, builds, resources, tags and presets | [The catalog](./data/catalog.md) |
| [`formats`](bunsen::data::formats) | reading the files: safetensors, `PyTorch`, MLX, the Phonon formats | [Formats](./data/formats.md) |
| [`builds`](bunsen::data::builds) | a name to a model: providers, the factory, the kit's hook | [Loading builds](./data/builds.md) |
| [`pretrained`](bunsen::data::pretrained) | the kits' prefabs, geometries by name | [Loading builds](./data/builds.md) |
| [`datasets`](bunsen::data::datasets) | dataset shard sets | [Datasets](./data/datasets.md) |

The kits show three sizes of the same machinery.
[`default_resnet_factory`](bunsen::kits::images::resnet::pretrained::default_resnet_factory)
is the simple case: the catalog's `torchvision` and `timm` checkpoints.
[`default_whisper_factory`](bunsen::kits::speech::asr::whisper::pretrained::default_whisper_factory)
is the full one: the catalog's builds, and Hugging Face repos.
[`default_silero_factory`](bunsen::kits::speech::vad::silero_vad::pretrained::default_silero_factory)
is the smallest: one build, upstream's 16 kHz checkpoint.

## The pathway

```text
spec ("tiny", "hf:org/repo")
  │  the kit's factory asks its providers
  ▼
build: what the catalog says the model is, and its resources
  │  paired with the kit's hook, optionally overlaid
  ▼
deferred model
  │  plan, bring the files local through the cache, construct
  ▼
loaded model, behind an Arc
```

A [`BuildFactory`](bunsen::data::builds::BuildFactory) is the one object a
caller holds for a kit, and its
[`load`](bunsen::data::builds::BuildFactory::load) runs the whole pathway. A
file on disk does not go through the factory at all: it is a given resource
set ([`ResourceSet::given`](bunsen::data::resources::ResourceSet::given)),
loaded the same way from the deferred step on. A dataset takes a shorter
path: its build is a shard set, which a data loader reads
([Datasets](./data/datasets.md)).

## The operational view

`bunsen-tool` shows the machinery from the outside. `bunsen-tool catalog`
reads, checks and queries the catalog. Each family's `list` shows its
builds, their presets, and where each one's files stand, and its `fetch`
brings a model's resources local. Every `--model` takes a lookup expression,
and `--preset` adds to its box; a name no provider has fails with the names
nearest it, as "did you mean?"; `bunsen-tool asr family whisper inspect`
checks a checkpoint against the geometry its build states, and
`bunsen-tool asr family whisper prefabs` lists the geometries;
`bunsen-tool cache` lists and clears what the cache holds. Its
[README](https://github.com/zspacelabs/bunsen/blob/main/crates/public/bunsen-tool/README.md#whisper-models)
explains Whisper's builds.
