# whisper-cli example

Transcribes an audio file with an OpenAI Whisper checkpoint and its vocabulary, through bunsen's Whisper stream
driver. The audio is pushed in chunks as a live loop would feed it, and segments are printed with their times as they
become final; under the responsive preset, drafts come first and are marked `~`.

The model is named, as `openai-whisper`'s `load_model` names it: `--model openai/tiny.en`, `--model large`, or a path
to a checkpoint. Weights are fetched on demand into bunsen's cache, pinned to their SHA-256, and used in place from
`openai-whisper`'s own `~/.cache/whisper` when it already has them. The default is `openai/base`. A deployment that
must not reach the network populates the cache ahead of time: `models fetch openai/base` in a Dockerfile, or
`--cache-dir` pointed at a directory laid out as the cache.

The vocabulary follows the checkpoint: the token layout its vocabulary size implies selects `multilingual.tiktoken` or
`gpt2.tiktoken`, and that rank file is a resource of the same model. A named model declares it, a path derives it, and
it comes through the same cache as the checkpoint, from the cache or one 800 KB fetch. This example is flags, the
driver and a `models` subcommand; the [model index](#models) and the name-to-model pathway are bunsen's.

## Bunsen features exercised

- `data::pretrained::{StaticPretrained, StaticResourceMap}` — the model index: each `openai` row fuses a checkpoint
  map with the vocabulary map its token layout selects, so a model is two keyed resources, `checkpoint` and
  `vocabulary`, and `PretrainedCache::load` brings both local, together, under
  `<cache>/pretrained/whisper/openai/<sha256>/<file>`.
- `kits::speech::whisper::pretrained::WhisperConstruct` — the hook that builds a `WhisperBundle` from a model's
  resources: it settles the vocabulary by the checkpoint's token layout (derived for a path, checked for a name),
  checks the geometry a name promised, and reads both files. From the vocabulary come the text, via a `wordchipper`
  detokenizer, and upstream's default suppress list; the driver takes both from the bundle (`init_from_bundle`).
  `--vocab` overlays a file by path, trusted as given.
- `kits::speech::whisper::pretrained::PytorchWhisperScanner` — scans a checkpoint's geometry and loads its weights;
  `load_named` checks the scan against the prefab the name promised before the weights are read. `--state-dict-key`
  configures it.
- `data::pretrained::StaticPreFabMap` — the prefab table, `WHISPER_PREFABS`, is bunsen's prefab type over
  `WhisperApiConfig`, as `PREFAB_RESNET_MAP` is over the ResNet config.
- `data::pretrained::PretrainedCache` over `data::cache::BunsenDiskCache` — the cache directory (`--cache-dir`,
  `$BUNSEN_CACHE_DIR`, the platform's cache dir), and the digest-pinned resolve of every resource through it
  (`--offline`, `--upstream-cache-dir`).
- `silero-weights` — `SileroVad::load_16khz_pretrained`, the voice-activity model the `conservative` and `responsive`
  presets gate on.
- `kits::speech::whisper::driver` — `WhisperStreamDriverConfig`, `WhisperStreamContext`: the stream driver, with its
  emission presets, the timestamp seek loop, beams, per-stream language detection, and the fallback ladder behind
  flags.

## The backend

The example computes on `bunsen::support::testing::PerformanceBackend`: the backend bunsen's own compute-heavy tests run
on. This crate does not choose it. **bunsen's backend feature** does, at build time, so the flag that picks the backend
for `cargo test` picks it here too, and what the example runs on is what the tests ran on.

| build with                           | backend                     |
|--------------------------------------|-----------------------------|
| `--features bunsen/cuda`             | `burn::backend::Cuda`       |
| `--features bunsen/metal`            | `burn::backend::Metal`      |
| `--features bunsen/wgpu`             | `burn::backend::Wgpu`       |
| `--features bunsen/flex`, or nothing | `burn::backend::Flex` (CPU) |

When several are on, the first of `cuda`, `metal`, `wgpu`, `flex` wins. bunsen's default `testing` feature enables
`flex`, which is why a bare build runs on the CPU. The `dependency/feature` form of `--features` works from any package
in the workspace, so `cargo run -p whisper-cli --features bunsen/wgpu`
from the root and `cargo run --features bunsen/wgpu` from this directory are the same build.

## Running the example

The first run fetches the model it names (145 MB for `openai/base`) and its vocabulary into bunsen's cache. To run
without the network, populate the cache first: `models fetch openai/base`, or point `--cache-dir` at a directory laid
out as the cache (the layout is under [Models](#models)).

```bash
$ cargo run --release -p whisper-cli --features bunsen/wgpu -- \
   transcribe --model openai/tiny.en --timestamps /path/to/clip.wav
```

Model options:

- `--model` — `provider/name` or a bare name from `models list` (`openai/tiny.en`, `large`, `turbo`), or a path to a
  checkpoint (default `openai/base`).
- `--cache-dir` — where fetched weights live; `$BUNSEN_CACHE_DIR`, then the platform's cache directory, when omitted.
- `--offline` — never reach the network; a model that is not already local is an error.
- `--upstream-cache-dir` — `openai-whisper`'s download root, whose files are used in place (default `~/.cache/whisper`).
- `--vocab` — a `.tiktoken` vocabulary by path, in place of the one the checkpoint's token layout selects.
- `--state-dict-key` — the key the checkpoint keeps its tensors under (default `model_state_dict`, as OpenAI's do);
  an empty string for a checkpoint whose tensors are at the top level.

Decode options:

- `--chunk-ms` — milliseconds of audio per push (default `1000`).
- `--language` — a Whisper language code; detected from the first window when omitted. `--task` — `transcribe` (default)
  or `translate`, to English.
- `--timestamps` — emit timestamp tokens and split segments on them, seeking to the last closed timestamp as upstream's
  `transcribe()` does.
- `--beam-size` — beams per window (default `5`).
- `--max-tokens` — cap per window (default `224`).
- `--prompt-carry` — prompt each window with the transcript so far (default `true`).
- `--fallback` — climb upstream's temperature ladder when a window's decode fails its thresholds (default `true`);
  `--fallback false` for temperature zero alone.
- `--preset` — `offline` (default: whole windows, all final), `conservative`
  (speech regions as well, all final), `responsive` (drafts every 600 ms of speech besides). The last two load the
  bundled VAD.
- `--ids` — print each segment's ids beside its text.

## Models

The index keeps three things apart, because they vary independently:

- A **prefab** is a public *configuration*: a geometry with no weights (`n_mels`, vocabulary size, `d_model`, layer
  counts). The table is upstream's `ModelDimensions`, typed in, so a name means a shape before anything is fetched.
  The vocabulary size is part of it: `tiny.en` (51864) and `tiny` (51865) are different prefabs; `large-v1` and
  `large-v2` share `large`; `large-v3` has 128 mels and a hundredth language; `large-v3-turbo` a four-layer decoder.
- A **pretrained** is a row: a name and its aliases (`large` is `large-v3`, `turbo` is `large-v3-turbo`), the prefab
  it instantiates, and the **resource maps** it is made of. An `openai` row fuses a checkpoint map with the
  vocabulary map its token layout selects, so a model is two keyed **resources**, `checkpoint` and `vocabulary`, and
  the two vocabularies are shared by the twelve checkpoints. One prefab may have several rows.
- A **resource** is one file: pinned to a SHA-256, labeled for a listing (`pytorch fp16`, `tiktoken`), and reachable
  from its map's **bases** in order. For a checkpoint those are `openai-whisper`'s `~/.cache/whisper`, a "trust me"
  directory whose file is used in place (hashed only under `--verify`), and then upstream's digest-addressed URL,
  which is streamed to a `.partial`, hashed as it lands, and renamed into place only on a match. Nothing is ever
  linked or copied into the cache.

The two `.tiktoken` vocabularies are resource maps of their own, listed under `vocabularies`. A named model declares
the one its layout selects; a path model derives it from the scanned checkpoint; `models fetch` brings every resource
of a model's map.

Fetched resources live at `<cache>/pretrained/whisper/<namespace>/<sha256>/<file>`. The digest in the path is the
pin, as it is in upstream's URLs: a file there was verified when written, so later runs trust it without re-hashing
3 GB, and a re-pinned model cannot collide with a stale one. A directory laid out this way *is* the cache: a
deployment that wants to run offline populates one ahead of time, and `--cache-dir` points at it.

The `models` subcommand takes the same cache options as `transcribe`:

```terminaloutput
$ cargo run -q -p whisper-cli -- models list
cache: /home/alice/.cache/bunsen
upstream cache: /home/alice/.cache/whisper

openai: OpenAI's Whisper checkpoints, as `openai-whisper` names and pins them (MIT; https://github.com/openai/whisper/blob/main/whisper/__init__.py)
  NAME                   PREFAB           STATUS     RESOURCES              DESCRIPTION
  openai/tiny.en         tiny.en          remote     checkpoint+vocabulary  39 M parameters, English-only
  openai/tiny            tiny             remote     checkpoint+vocabulary  39 M parameters, multilingual
  openai/base.en         base.en          remote     checkpoint+vocabulary  74 M parameters, English-only
  openai/base            base             cached     checkpoint+vocabulary  74 M parameters, multilingual
  ...
  openai/large-v3        large-v3         local dir  checkpoint+vocabulary  1550 M parameters, multilingual, 128 mels (also: large)
  openai/large-v3-turbo  large-v3-turbo   remote     checkpoint+vocabulary  809 M parameters, multilingual, 128 mels, four-layer decoder (also: turbo)

vocabularies:
  NAME                          KEY          KIND           STATUS     DESCRIPTION
  openai/multilingual.tiktoken  vocabulary   tiktoken       cached     the multilingual vocabulary: GPT-2's ranks plus one, as every multilingual checkpoint numbers them
  openai/gpt2.tiktoken          vocabulary   tiktoken       remote     the English-only vocabulary: GPT-2's ranks, as the `*.en` checkpoints number them

$ cargo run -q -p whisper-cli -- models prefabs
whisper: OpenAI Whisper geometries, as `whisper.model.ModelDimensions` has them
  NAME             MELS  VOCAB D_MODEL HEADS  ENC  DEC AUDIO_CTX TEXT_CTX  DESCRIPTION
  tiny               80  51865     384     6    4    4      3000      448  d_model 384, 4 + 4 layers, multilingual
  ...
  large-v3-turbo    128  51866    1280    20   32    4      3000      448  d_model 1280, 32 + 4 layers, 128 mels, 100 languages

$ cargo run -q -p whisper-cli -- models fetch --verify openai/tiny.en
openai/tiny.en:
  checkpoint: /home/alice/.cache/bunsen/pretrained/whisper/openai/d3dd.../tiny.en.pt (downloaded)
    sha256 d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03 ok
  vocabulary: /home/alice/.cache/bunsen/pretrained/whisper/openai/306c.../gpt2.tiktoken (downloaded)
    sha256 306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930 ok

$ cargo run -q -p whisper-cli -- models inspect large
model: openai/large-v3
  provider: openai (OpenAI's Whisper checkpoints, as `openai-whisper` names and pins them)
  description: 1550 M parameters, multilingual, 128 mels
resources:
  checkpoint: large-v3.pt (pytorch fp16, sha256 e5b1...) [local dir]
    local dir openai-whisper (/home/alice/.cache/whisper)
    url https://openaipublic.azureedge.net/main/whisper/models/e5b1.../large-v3.pt
  vocabulary: multilingual.tiktoken (tiktoken, sha256 b34b...) [cached]
    url https://raw.githubusercontent.com/openai/whisper/839639a2.../whisper/assets/multilingual.tiktoken
prefab: large-v3 (d_model 1280, 32 + 32 layers, 128 mels, 100 languages)
  WhisperGeometry { n_mels: 128, vocab_size: 51866, d_model: 1280, max_audio_ctx: 3000, n_encoder_layers: 32, max_text_ctx: 448, n_decoder_layers: 32, d_head: 64 }
checkpoint: /home/alice/.cache/whisper/large-v3.pt (local dir)
scanned: WhisperGeometry { n_mels: 128, vocab_size: 51866, d_model: 1280, max_audio_ctx: 3000, n_encoder_layers: 32, max_text_ctx: 448, n_decoder_layers: 32, d_head: 64 }
  heads: 20
  front end: WhisperFrontEndConfig { sample_rate: 16000, hop_ms: 10, window_ms: 25, range_clamp_db: 8.0 }
  matches prefab large-v3
```

`inspect` on a path reports which prefab, if any, the checkpoint's geometry is, and which vocabulary that geometry's
layout selects; on a name whose checkpoint does not scan as its prefab, it reports the mismatch instead of loading.
`transcribe` makes the same check before reading any tensors.

### Where the index lives

All of it is bunsen's: `kits::speech::whisper::pretrained::{WHISPER_PREFABS, OPENAI, WHISPER_PROVIDERS,
OPENAI_CHECKPOINTS, MULTILINGUAL_VOCABULARY, GPT2_VOCABULARY}` over `data::pretrained::{StaticPretrained,
StaticPretrainedProvider, StaticResourceMap, PretrainedCache, PretrainedRef}`, with `WhisperGeometry` beside
`WhisperApiConfig`, `pretrained::{resolve_model, scan_model, load_model, load_named}` (and their `_with` forms, which
take a configured scanner) as the name-to-model pathway, and `pretrained::{vocabulary_map, vocabulary_for}` as the
layout-to-vocabulary rule and its resolve. This crate resolves `--model` with `load_named_with` and reports through
the `models` subcommand; it keeps no index of its own.

## Benchmarks

### Setting up the SLR45 Dataset

See the [SLR45](https://www.openslr.org/45/) dataset.

```terminaloutput
cd $DATA_DIR
mkdir SLR45
cd SLR45
wget https://openslr.trmal.net/resources/45/ST-AEDS-20180100_1-OS.tgz
tar xzf ST-AEDS-20180100_1-OS.tgz
rm ST-AEDS-20180100_1-OS.tgz
```

### Running a small benchmark

```terminaloutput
$ cargo run --release -p whisper-cli --features bunsen/wgpu -- transcribe  --print-filename $DATA_DIR/SLR45/f0001_us_f0001_0000{1,2,3,4,5}.wav
...
INFO model: 80 n_mels, vocabulary 51865, d_model 512, 6 + 6 layers
INFO language: detected from the first window
INFO  [    0.00 -->     4.68]
f0001_us_f0001_00001.wav        The world needs opportunities for new leaders and new ideas.
INFO  [    0.00 -->     3.08]
f0001_us_f0001_00002.wav        along with all the other references that I had.
INFO  [    0.00 -->     2.64]
f0001_us_f0001_00003.wav        wouldn't have hesitated for a second.
INFO  [    0.00 -->     2.56]
f0001_us_f0001_00004.wav        I would always examine the patient.
INFO  [    0.00 -->     2.52]
f0001_us_f0001_00005.wav        Will we like to play with stuff?
INFO mean sample: 3.1s      
INFO mean decode: 1.5s      
INFO mean sample/decode: 3.77

$ head -5 $DATA_DIR/SLR45/text.txt 
f0001_us_f0001_00001.wav        the world needs opportunities for new leaders and new ideas.
f0001_us_f0001_00002.wav        along with all the other reference that I had
f0001_us_f0001_00003.wav        I wouldn't have hesitated for a second.
f0001_us_f0001_00004.wav        I would always examine the patient.
f0001_us_f0001_00005.wav        Well we like to play with stuff.
```
