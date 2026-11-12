# Loading builds

A kit loads a model by name through
[`bunsen::data::builds`]: providers answer names with builds, a factory
dispatches a spec to them, and the kit's hook builds from the files the
cache brings local.

## Prefabs, builds and resources

Three things are kept apart, because they vary independently:

- A **prefab** is a geometry by name, with no weights, so a name means a
  shape before anything is fetched. A kit's prefabs are constructors on its
  config (`WhisperApiConfig::tiny_en()`), listed by
  [`Prefabs`](bunsen::data::pretrained::Prefabs).
- A **build**, a [`ResolvedBuild`](bunsen::data::catalog::ResolvedBuild), is
  what the catalog says one model is: its name and aliases, its tags and
  presets, the geometry it states, and the resources its contract's slots are
  bound to.
- A **resource**, a [`Resource`](bunsen::data::resources::Resource), is one
  file ([Resources and the cache](./resources.md)).

A build states its geometry rather than naming a prefab, and a kit's tests
check that every stated geometry is a prefab's. One file can serve many
builds: Whisper's two vocabulary files serve its twelve checkpoints, each a
resource the builds bind. Builds and resources know nothing about how a
model is built: that is the kit's business ([builds and
resources](bunsen::data::builds#builds-and-resources)).

## Providers

A [`BuildProvider`](bunsen::data::builds::BuildProvider) is a namespace of
builds: the `provider` in `provider:ref`. Two are in use. `catalog:`, a
[`CatalogBuildProvider`](bunsen::data::builds::CatalogBuildProvider), holds
the builds of the kit's contract in the built-in catalog, and in any
extension layer imported over it. `hf:`, an
[`HfProvider`](bunsen::data::builds::HfProvider), is a Hugging Face repo, by
`org/repo`, answered with a build named `hf/<org>/<repo>` from the repo's
listing, fetched once through the cache.

A factory holds a kit's providers in search order. A spec whose prefix names
a registered provider goes to that provider, and nowhere else. Any other
spec is offered whole to each provider that answers bare names, in order,
and the first build wins. A hub never answers a bare name, so `hf:` is only
reached when the spec names it. The two kinds of "no" are kept distinct. A
provider that doesn't have the build says so (`Ok(None)`), and the search
moves on. A provider that fails returns an error, and the error aborts the
lookup, because a hub that cannot be reached is not the same as a hub that
has no such build ([dispatch](bunsen::data::builds::BuildFactory#dispatch)).

A spec is a lookup expression: `[provider:]name`, then an optional box of
presets and an optional YAML patch over the resolved config, as in
`'openai/small[translate]{beam_size: 1}'`. Quote it on a shell command line,
since `[` and `]` glob. Only the name is dispatched. The box and the patch
are checked against the build found before anything is fetched, and a kit
that takes no presets refuses them rather than ignore them
([lookup expressions](bunsen::data::builds::BuildFactory#lookup-expressions)).
Catalog names compare ignoring case, so `Qwen/Qwen3-0.6B` finds
`qwen/qwen3/qwen3-0.6b`. `hf:` names compare as given, since they are paths
in URLs.

There is no process-wide registry. Each kit ships a
`default_{kit}_factory()`, and a caller that wants another provider adds it
to that factory ([no registry](bunsen::data::builds::BuildFactory#no-registry)).

## Deferred models and the kit's hook

What a name resolves to is a [`Deferred`](bunsen::data::builds::Deferred)
model: a [`BuildRef`](bunsen::data::builds::BuildRef) (the build, or a given
resource set), plus the kit's
[`ConstructInit`](bunsen::data::builds::ConstructInit) hook. The hook is
chosen for the resource set by
[`for_resource_set`](bunsen::data::builds::ConstructInit::for_resource_set),
from the resources' `format`s, because builds from different providers can
need different readers: an OpenAI `.pt` checkpoint and a Hugging Face
safetensors repo are both Whisper, read two ways.

Loading runs three steps. The hook's
[`plan`](bunsen::data::builds::ConstructInit::plan) completes the resource
set and checks what the build promises, such as its stated geometry, before
any bytes are fetched. The cache then brings the files local. Finally the
hook's [`try_init`](bunsen::data::builds::ConstructInit::try_init) builds the
model from them. Before loading, a caller can lay its own resources over the
build's with [`with_overlay`](bunsen::data::builds::Deferred::with_overlay),
where the caller's resource wins by slot. `bunsen-tool asr family whisper`'s
`--vocab` flag works this way.
