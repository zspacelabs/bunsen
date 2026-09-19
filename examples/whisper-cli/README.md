# whisper-cli example

Transcribes an audio file with an OpenAI Whisper checkpoint and its vocabulary, through bunsen's Whisper stream
driver. The audio is pushed in chunks as a live loop would feed it, and segments are printed with their times as they
become final; under the responsive preset, drafts come first and are marked `~`.

The model is named, as `openai-whisper`'s `load_model` names it: `--model openai/tiny.en`, `--model large`, or a path
to a checkpoint. Weights are fetched on demand into bunsen's cache, pinned to their SHA-256, and read from the bundled
checkpoint or from `openai-whisper`'s own `~/.cache/whisper` when either already has them. The default, `openai/base`,
is the checkpoint bunsen bundles, so it needs no network.

Where [`whisper-dev`](../../crates/dev/whisper-dev) takes a checkpoint and a vocabulary by path and drives the mel and
decode ops by hand, this example takes a model name and the audio: everything else comes from bunsen's features and
the [model index](#models) in `src/models/`.

## Bunsen features exercised

- `whisper-weights` — `bunsen-bundled-whisper` fetches the multilingual `base` checkpoint at build time (pinned to a
  SHA-256, cached), which the index lists as the bundled source of `openai/base`; and `pretrained::bundled_vocabulary`
  picks the `.tiktoken` rank file that matches a checkpoint's token layout, for every model. From the vocabulary come
  the text, via a `wordchipper` detokenizer, and upstream's default suppress list.
- `kits::speech::whisper::pretrained::PytorchWhisperScanner` — scans a checkpoint's geometry and loads its weights;
  the index checks the scan against the prefab the name promised before the weights are read.
- `data::pretrained::StaticPreFabMap` — the prefab table (`src/models/prefab.rs`) is bunsen's prefab type over
  `WhisperApiConfig`, as `PREFAB_RESNET_MAP` is over the ResNet config.
- `data::cache::BunsenDiskCache` — resolves the cache directory (`--cache-dir`, `$BUNSEN_CACHE_DIR`, the platform's
  cache dir); the digest-pinned transfer is this crate's (`src/models/weights_cache.rs`).
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

The first build fetches the bundled checkpoint (145 MB) and the two vocabularies into the build cache.

```bash
$ cargo run --release -p whisper-cli --features bunsen/wgpu -- \
   transcribe --model openai/tiny.en --timestamps /path/to/clip.wav
```

Model options:

- `--model` — `provider/name` or a bare name from `models list` (`openai/tiny.en`, `large`, `turbo`), or a path to a
  checkpoint (default `openai/base`, the bundled checkpoint).
- `--cache-dir` — where fetched weights live; `$BUNSEN_CACHE_DIR`, then the platform's cache directory, when omitted.
- `--offline` — never reach the network; a model that is not already local is an error.
- `--upstream-cache-dir` — `openai-whisper`'s download root, read as a local source (default `~/.cache/whisper`).

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
- A **pretrained** is a trained set of weights for a prefab, with a format (the quantization axis; OpenAI ships one,
  fp16 PyTorch), a SHA-256, and a list of **sources**. One prefab may have several pretrained; one pretrained may have
  several names (`large` is `large-v3`, `turbo` is `large-v3-turbo`) and several sources, all the same bytes.
- A **source** is a URL, `openai-whisper`'s `~/.cache/whisper`, or the file `bunsen-bundled-whisper` fetched at build
  time. Sources are tried in order: the bundled file is used in place (the build verified it); a file in upstream's
  cache is hashed and, on a match, linked into the cache; a URL is streamed to a `.partial`, hashed as it lands, and
  renamed into place only on a match.

Fetched weights live at `<cache>/weights/whisper/<provider>/<sha256>/<file>`. The digest in the path is the pin, as it is in
upstream's URLs: a file there was verified when written, so later runs trust it without re-hashing 3 GB, and a
re-pinned model cannot collide with a stale one.

The `models` subcommand takes the same cache options as `transcribe`:

```terminaloutput
$ cargo run -q -p whisper-cli -- models list
cache: /home/alice/.cache/bunsen
upstream cache: /home/alice/.cache/whisper

openai: OpenAI's Whisper checkpoints, as `openai-whisper` names and pins them (MIT; https://github.com/openai/whisper/blob/main/whisper/__init__.py)
  NAME                   PREFAB           FORMAT         STATUS          DESCRIPTION
  openai/tiny.en         tiny.en          pytorch fp16   remote          39 M parameters, English-only
  openai/tiny            tiny             pytorch fp16   remote          39 M parameters, multilingual
  openai/base.en         base.en          pytorch fp16   remote          74 M parameters, English-only
  openai/base            base             pytorch fp16   bundled         74 M parameters, multilingual; the checkpoint bunsen bundles
  ...
  openai/large-v3        large-v3         pytorch fp16   upstream cache  1550 M parameters, multilingual, 128 mels (also: large)
  openai/large-v3-turbo  large-v3-turbo   pytorch fp16   remote          809 M parameters, multilingual, 128 mels, four-layer decoder (also: turbo)

$ cargo run -q -p whisper-cli -- models prefabs
whisper: OpenAI Whisper geometries, as `whisper.model.ModelDimensions` has them
  NAME             MELS  VOCAB D_MODEL HEADS  ENC  DEC AUDIO_CTX TEXT_CTX  DESCRIPTION
  tiny               80  51865     384     6    4    4      3000      448  d_model 384, 4 + 4 layers, multilingual
  ...
  large-v3-turbo    128  51866    1280    20   32    4      3000      448  d_model 1280, 32 + 4 layers, 128 mels, 100 languages

$ cargo run -q -p whisper-cli -- models fetch --verify openai/tiny.en
INFO openai/tiny.en: fetching https://openaipublic.azureedge.net/main/whisper/models/d3dd.../tiny.en.pt
openai/tiny.en: /home/alice/.cache/bunsen/whisper/openai/d3dd.../tiny.en.pt (downloaded)
  sha256 d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03 ok

$ cargo run -q -p whisper-cli -- models inspect large
model: openai/large-v3
  ...
prefab: large-v3 (d_model 1280, 32 + 32 layers, 128 mels, 100 languages)
  WhisperGeometry { n_mels: 128, vocab_size: 51866, d_model: 1280, max_audio_ctx: 3000, n_encoder_layers: 32, max_text_ctx: 448, n_decoder_layers: 32, d_head: 64 }
INFO openai/large-v3: verifying /home/alice/.cache/whisper/large-v3.pt
checkpoint: /home/alice/.cache/bunsen/whisper/openai/e5b1.../large-v3.pt (upstream cache)
scanned: WhisperGeometry { n_mels: 128, vocab_size: 51866, d_model: 1280, max_audio_ctx: 3000, n_encoder_layers: 32, max_text_ctx: 448, n_decoder_layers: 32, d_head: 64 }
  heads: 20
  front end: WhisperFrontEndConfig { sample_rate: 16000, hop_ms: 10, window_ms: 25, range_clamp_db: 8.0 }
  matches prefab large-v3
```

`inspect` on a path reports which prefab, if any, the checkpoint's geometry is; on a name whose checkpoint does not
scan as its prefab, it reports the mismatch instead of loading. `transcribe` makes the same check before reading any
tensors.

### What is bunsen's, and what is the CLI's

The index is bunsen's: `kits::speech::whisper::pretrained::{WHISPER_PREFABS, OPENAI, WHISPER_PROVIDERS}` over
`data::pretrained::{StaticPretrainedWeightsDescriptor, StaticPretrainedProvider, WeightsCache}`, with
`WhisperGeometry` beside `WhisperApiConfig`. The CLI keeps `src/models/loader.rs`: `ModelRef` (a name, an alias, or
a path), the geometry check against the prefab a name promised, and the load through bunsen's scanner. That is the
name-to-model pathway, and it moves into the kit next.

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
