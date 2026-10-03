# Assets, bundling and the network

bunsen's kits and their validation need files too large to commit:
checkpoints, reference exports, vocabularies. This page explains how the
workspace gets them without every build reaching the network, and why the
rules are what they are. The rules themselves are in STYLE.md's
[Network access](https://github.com/zspacelabs/bunsen/blob/main/STYLE.md#network-access)
and
[Fetched assets live in `OUT_DIR`](https://github.com/zspacelabs/bunsen/blob/main/STYLE.md#fetched-assets-live-in-out_dir);
the features that carry them are listed in
[`bunsen::data`](bunsen::data#features).

## Build time and run time

There are two switches, and neither implies the other:

- **`download`**: the *build* may reach the network. A build script fetches
  an asset, and the crate compiles against it. It is off by default, so a
  plain `cargo build --workspace` never fetches. `download` is a
  conventional name rather than a bunsen feature: a crate that offers it
  turns on the per-asset features of the crates that own the assets.
- **`fetch`**: the *program* may reach the network at run time, to bring a
  pretrained model or a dataset shard into the disk cache
  ([pretrained models](../systems/pretrained.md),
  [shard sets](../systems/shards.md)).

A test that needs a checkpoint usually wants `download`: the bytes are in
place before it runs, and the run itself is offline. A program that loads
models by name wants `fetch`.

## Digest pins

Every fetched asset is pinned to a SHA-256, written beside its URL in the
build script and re-verified on every build. A file that changes upstream is
a build error, not a silently different model.

An environment variable can point the build at a local file instead:
`WHISPER_BASE_PT` and its siblings, listed in the Whisper validation
[README](https://github.com/zspacelabs/bunsen/blob/main/crates/validation/whisper-model-validation/README.md#running-it).
An override skips the pin, so use it for a copy you already trust.

## Why `OUT_DIR`

`OUT_DIR` is the one directory a build script may write to. `cargo publish`
builds the packaged crate and fails if the build touched anything else in
the package, so an asset cache beside the manifest would make the crate
unpublishable. A crate unpacked from crates.io must not write into the
registry either. The cost is that `cargo clean` discards the assets along
with everything else.

## The two bundled crates

Shipping a model with a build doesn't need a special kind of source. It
needs a cache directory that is already full
([bundling](../systems/pretrained.md#bundling-is-a-populated-cache-directory)).
The two bundled crates show the two forms:

- [`bunsen_bundled_whisper`] hosts nothing. Its
  build script fetches Whisper's `base` checkpoint, the vocabularies and,
  for validation, the ONNX reference; pins each one; and lays them out in
  `OUT_DIR` as a pretrained cache. A cache rooted there finds them offline
  without knowing they were bundled. bunsen's `whisper-weights` feature
  pulls this crate in, so that feature makes the build reach the network on
  a cold cache.
- [`bunsen_bundled_silero`] needs no network.
  Silero's ONNX graph ships in the crate, and its build script converts the
  graph to a burnpack that is linked into the binary. The burnpack's digest
  is computed at build time, and the Silero kit's `bundled:silero/vad` row
  is pinned to it.

A deployment does the same thing at install time: fill the runtime cache
ahead of time, and the program never needs `fetch`.

## Why CI doesn't cache them

A cold CI run fetches the Whisper assets again, and that is deliberate.
CI's Rust cache prunes the workspace's own crates from `target/` before it
saves, so the assets don't ride along. Asking it to keep workspace crates
would cache the workspace's build output too, test binaries included. A
separate cache over an `OUT_DIR` path goes stale as soon as the hash in that
path changes, which every toolchain or lockfile update does.

So don't add a cache step for them. To avoid the download locally, point the
override variables at copies you already have.
