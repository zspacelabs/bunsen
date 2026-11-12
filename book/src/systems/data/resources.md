# Resources and the cache

A [`Resource`](bunsen::data::resources::Resource) is one file: the slot a
kit reads it by (`checkpoint`, `vocabulary`), the file name it lands as, the
SHA-256 that pins it, its format, its full name, and the
[`Source`](bunsen::data::resources::Source)s it can be had from, in order. A
[`ResourceSet`](bunsen::data::resources::ResourceSet) is a named set of them,
what a loader reads, and two sets
[fuse](bunsen::data::resources::ResourceSet::fuse) by slot: strictly for
authoring, where a repeated slot is a mistake, or as an overlay, where the
right-hand set wins. None of this is about models: a model's build and a
dataset's shards both load through it
([`bunsen::data::resources`]).

## The cache and its trust model

A [`ResourceCache`](bunsen::data::resources::ResourceCache) keeps each
pinned file at `<cache>/resources/<name>/<sha256>/<file>`. The name is the
resource's full name: a catalog resource's (`qwen/qwen3/qwen3-0.6b/checkpoint`),
so a file two builds share is cached once; an `hf:` repo's
(`hf/<org>/<repo>/<slot>`); `given` for a path's. **The digest in the path is
the pin.** A file there was verified when it was written, so later runs
trust it without re-hashing gigabytes, and a model whose pin changes can
never collide with a stale copy. The cache trusts the path whoever wrote it:
a download, bundled bytes, or a build script or a deployment that laid the
directory out ahead of time
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
the Whisper validation crate's asset crate, fetches Whisper's `base` files
at build time into its build directory, laid out as a cache, and a
`ResourceCache` rooted there finds them without knowing they were bundled.
An offline deployment does the same thing at install time: fill a cache
directory ahead of time, and point the cache at it. Files a deployment ships
in place are a catalog layer instead, which adds them to their builds as
local-dir sources ([extension layers](./catalog.md#extension-layers)).

The other form links the bytes into the binary: a
[`Source::Bundled`](bunsen::data::resources::Source::Bundled), checked
against its digest and written into the cache on first use. Both forms are
described under
[the cache and bundles](bunsen::data::resources#the-cache-and-bundles).

## The Hub

[`resources::hub`](bunsen::data::resources::hub) is what a client of the
Hugging Face Hub needs to name its files: a
[`HubRepo`](bunsen::data::resources::hub::HubRepo) at a revision, a model's
or a dataset's, the URLs it serves a file and its tree listing at, and the
listing's entries, whose LFS digests pin the files. The `hf:` provider reads
a repo's listing through the cache, and `cargo catalog` pins the catalog's
Hub files from the same listings.

The Hub may be mirrored. A mirror is a build of the `hf-mirror` contract,
which states the origin that serves the Hub's paths, and a catalog's mirror
set is its `hf-mirror` builds in the catalog's order: the built-in catalog's
`hub` layer declares `huggingface/hub`, the Hub itself, and a layer above can
add more. Every Hub fetch tries them in order: a catalog build's `$hub` files
and an `hf:` repo's files and listing each carry one URL per mirror, and the
cache moves to the next when one is unreachable, lacks the file or serves the
wrong bytes. A file is pinned and cached the same whichever mirror served it.
