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

The build scripts fetch through the `bunsen-fetch` crate's `Asset`, and
bunsen's disk cache fetches through the same crate at run time: one
implementation of the streamed, hashed, renamed-into-place transfer, and one
statement of the pretrained cache layout, so what a build lays out is what the
cache reads.

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

## The bundled crates

Shipping a model with a build doesn't need a special kind of source. It
needs a cache directory that is already full
([bundling](../systems/pretrained.md#bundling-is-a-populated-cache-directory)).
Every bundled crate hosts nothing: its build script fetches the files, pins
each one, and lays them out in `OUT_DIR` as a pretrained cache. A cache rooted
there finds them offline without knowing they were bundled.

- [`bunsen_bundled_whisper`] fetches Whisper's `base` checkpoint, the
  vocabularies and the ONNX reference, for `whisper-model-validation`.
- [`bunsen_bundled_silero`] fetches upstream's 16 kHz Silero VAD
  checkpoint and its ONNX graph, from which the reference model is
  generated, for `silero-model-validation`.
- `bunsen-bundled-parakeet` and `bunsen-bundled-qwen3-asr` fetch the Phonon
  checkpoints, the base models and the ONNX references for their validation
  crates only, and are not published.
- `bunsen-bundled-sortformer` fetches NVIDIA's `Nemotron-3-Diarization` and
  streaming Sortformer v2 and v2.1 checkpoints, and ONNX exports of Nemotron
  (which its build script rewrites where `burn-onnx` cannot import it) and of
  v2.1, for its validation crate only, and is not published.
- `bunsen-bundled-qwen3` fetches Qwen's `Qwen3-0.6B`, `-1.7B`, `-4B` and
  `-4B-Instruct-2507` checkpoints, and an ONNX export of `Qwen3-0.6B` (which
  its build script rewrites where `burn-onnx` cannot import it), for its
  validation crate only, and is not published.

Only the validation crates depend on them, through their `weights` and
`download` features, so no other build reaches the network. bunsen's own tests
read no fetched asset: where a test needs what a real model does, it replays
a committed recording of it, and the validation crate checks the recording
against the model.

A deployment does the same thing at install time: fill the runtime cache
ahead of time, and the program never needs `fetch`.

## How CI caches them

CI caches compiled units, not `target/`, so cargo runs every build script on
every run, and a build script's fetch is not a compiled unit. A cache over an
`OUT_DIR` path would go stale as soon as the hash in that path changes, which
every toolchain or lockfile update does.

So assets are cached by digest instead. `BUNSEN_ASSET_CACHE` names a
directory every build script keeps what it fetches in, as `<sha256>`, and
takes a pinned file from before fetching it, verified as any asset is. Its
paths never move. The pull-request jobs fetch nothing at all; `validation.yml`,
which runs only when asked (from the Actions tab, or for a pull request with
the `validate` label), runs Silero's and Whisper's whole suites on the CPU over
about 730 MB of assets, and caches that directory under a key that hashes the
pins. The other families' assets run to GBs; their suites run by hand.

Locally, set `BUNSEN_ASSET_CACHE` to keep assets across `cargo clean` and
toolchain updates, or point the override variables at copies you already
have.
