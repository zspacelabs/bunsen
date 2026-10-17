# bunsen-asr

`bunsen-asr`: speech to text on [bunsen](https://crates.io/crates/bunsen), one subcommand per model family, each with
its own `list` and `transcribe` and only the flags that apply to it:

```sh
bunsen-asr parakeet  (list|fetch|transcribe|live) [FLAGS]   # NVIDIA's Parakeet TDT; FermionResearch's Phonon-2 by default
bunsen-asr qwen3-asr (list|fetch|transcribe|live) [FLAGS]   # Qwen's Qwen3-ASR; FermionResearch's Phonon-1 by default
bunsen-asr whisper   (list|fetch|transcribe|live) [FLAGS]   # OpenAI's Whisper; openai/base by default
```

```sh
cargo run --release -p bunsen-asr --features wgpu -- parakeet transcribe recording.wav
cargo run --release -p bunsen-asr --features wgpu -- qwen3-asr transcribe --model qwen/qwen3-asr-1.7b --json recording.wav
cargo run --release -p bunsen-asr --features wgpu -- whisper transcribe --preset conservative --language en recording.wav
cargo run --release -p bunsen-asr -- qwen3-asr list
```

Installed from crates.io, with the GPU backends you want (the CPU is always there):

```sh
cargo install bunsen-asr --features wgpu
bunsen-asr parakeet transcribe recording.wav
```

`list` shows the family's named models, with each one's licence, its files, and whether each is cached; every family
also takes `hf:org/repo`, a Hugging Face repo in `transformers`' layout, whose `config.json` gives the geometry.
`fetch NAME...` brings every file of the named models into the cache and prints where each one is (`--verify`
re-hashes each pinned file): a deployment that runs `--offline` fills its cache this way ahead of time.

`transcribe` loads `--model` (the family's default unless named) through bunsen's pretrained cache, fetching it on
first use, logs its info card (name, licence, origin, every file and where it came from) at `INFO` (`-v`), and prints
each file's text, or with `--json` one object per file: the text, the language when the model names one (Qwen3-ASR,
Whisper), each word's start and end in seconds (Parakeet), and each segment's (Whisper; Qwen3-ASR's are its committed
windows), every word and segment with the token ids it was decoded as. After each file it logs how
long the transcription took at `INFO`, and after several, the total. Audio is read at the model's rate, 16 kHz mono,
and streamed through the family's stream driver (bunsen's `asr_driver`), about a second a push.

`--preset` says when the driver decodes, for every family: `offline` (the default) decodes whole windows, one after
another; `conservative` also finds speech regions with Silero's VAD (`silero/vad-16k`, through the same cache) and
decodes each as it ends, so silence is never decoded; `responsive` adds a draft of the speech so far every 600 ms of
it, which `live` shows. `--chunk-ms` (1000) is how much audio each push carries.

`live` transcribes the microphone as it speaks, through the same driver: the host's default input device, or
`--input-device` (a substring of an id or name from `--list-devices`), opened at the model's rate when it offers that
and resampled otherwise. Each decode's committed text is a line on stdout; under `--preset responsive` the draft of the
speech so far is on stderr, redrawn in place on a terminal. `--preset` defaults to `conservative` here, `--chunk-ms`
to 250; `--seconds` stops after that much audio, Ctrl-C otherwise, and either way the speech past the last commit is
decoded before the process exits. The default `live` feature builds it: it captures through `cpal`, which on Linux
links ALSA (`libasound2-dev`); `--no-default-features` builds without it.

```sh
cargo run --release -p bunsen-asr --features wgpu -- parakeet live --preset responsive
cargo run --release -p bunsen-asr -- whisper live --list-devices
```

| Family | Flags of its own |
|---|---|
| `parakeet` | `--window-seconds` (60): a full window is committed up to a pause between words, and the rest decoded again with the audio after it |
| `qwen3-asr` | `--window-seconds` (30, the most Qwen3-ASR was trained on): a full window is cut at the quietest point of its last third |
| `whisper` | `--language` (detected when omitted), `--task` (`transcribe`, `translate`), `--beam-size` (5), `--timestamps` (split segments on timestamp tokens), `--max-tokens` (224 a window), `--prompt-carry` (`true`: prompt each window with the transcript so far), `--fallback` (`true`: upstream's temperature ladder), `--vocab` (a vocabulary file in place of the layout's), `--upstream-cache-dir` (`openai-whisper`'s download cache, used in place; `~/.cache/whisper`); `--model` also takes a checkpoint's path |

`bunsen-asr whisper inspect NAME|PATH` scans a Whisper checkpoint and checks it against its prefab (its resources and
where each stands, the geometry it promised, and the one its shapes say); `bunsen-asr whisper prefabs` lists the
geometries. Neither needs a backend.

`bunsen-asr cache` manages the weights cache every family loads through (`bunsen_app::cache`):

```sh
bunsen-asr cache path                         # the cache directory
bunsen-asr cache list [KIT[/NAMESPACE]]       # each file's kit, namespace, size and pin, then the total
bunsen-asr cache clear [KIT[/NAMESPACE]] [--partial] [--yes]
```

The kits are `parakeet`, `qwen3-asr`, `whisper` and `silero_vad` (the VAD the `conservative` preset loads).
`clear` without `--yes` only lists what it would remove; `--partial` selects unfinished downloads.

`transcribe` and `live` take `--preset` and `--chunk-ms`; every command takes `--cache-dir`, `--offline`, `--device` (from the backends compiled in: `--features wgpu`, `vulkan`,
`metal`, `cuda`; the CPU is always there) and `-v`/`-q`.

| model | | licence |
|---|---|---|
| `parakeet`: `fermion/phonon-2` | 164 MB, `parakeet-tdt-0.6b-v3` with a five-value encoder, English | CC-BY-4.0 |
| `parakeet`: `nvidia/parakeet-tdt-0.6b-v3` | 2.5 GB, NVIDIA's base model, 25 European languages | CC-BY-4.0 |
| `qwen3-asr`: `fermion/phonon-1` | 415 MB, `Qwen3-ASR-0.6B` re-packed, 30 languages; `-micro` (285 MB) and `-big` (581 MB) builds | Apache-2.0 |
| `qwen3-asr`: `qwen/qwen3-asr-0.6b`, `qwen/qwen3-asr-1.7b` | 1.6 and 4.1 GB, Qwen's base models | Apache-2.0 |
| `whisper`: `openai/*` | every `openai-whisper` checkpoint, `tiny` to `large-v3-turbo` | MIT |

## Whisper models

Whisper's index keeps three things apart, because they vary independently:

- A **prefab** is a public configuration: a geometry with no weights (`n_mels`, vocabulary size, `d_model`, layer
  counts), upstream's `ModelDimensions` typed in, so a name means a shape before anything is fetched. The vocabulary
  size is part of it: `tiny.en` (51864) and `tiny` (51865) are different prefabs; `large-v3` has 128 mels and a
  hundredth language; `large-v3-turbo` a four-layer decoder. `bunsen-asr whisper prefabs` lists them.
- A **pretrained** is a row: a name and its aliases (`large` is `large-v3`, `turbo` is `large-v3-turbo`), the prefab
  it instantiates, and its resources. An `openai` row is two, `checkpoint` and the `vocabulary` its token layout
  selects; the two `.tiktoken` vocabularies are shared by the twelve checkpoints.
- A **resource** is one file, pinned to a SHA-256 and reachable from its sources in order: for a checkpoint,
  `openai-whisper`'s `~/.cache/whisper` (`--upstream-cache-dir`), whose file is used in place, then upstream's
  digest-addressed URL, streamed to a `.partial`, hashed as it lands, and renamed into place only on a match.

Fetched files live at `<cache>/pretrained/whisper/<namespace>/<sha256>/<file>`: the digest in the path is the pin, so
a file there is trusted without re-hashing, and a directory laid out this way is the cache (`fetch` fills one ahead
of time for a deployment that runs `--offline`). `hf:openai/whisper-large-v3` is a Hugging Face repo's safetensors
checkpoint at `main`, sharded or not, each file pinned by the digest the hub's listing carries; its vocabulary is the
one its layout selects, as for a checkpoint's path.

`bunsen-asr whisper inspect NAME|PATH` prints a model's resources and where each stands, the prefab it promised, and
the geometry its checkpoint's shapes say; on a path, which prefab, if any, that geometry is, and which vocabulary its
layout selects; on a name whose checkpoint does not scan as its prefab, the mismatch. `transcribe` makes the same
check before it reads any tensors. The index is bunsen's (`kits::speech::whisper::pretrained`), and this binary keeps
none of its own.

## Benchmarks

At `-v`, `transcribe` logs each file's audio length, how long it took and the realtime factor, then the totals over
the files. Over a few utterances of [SLR45](https://www.openslr.org/45/):

```sh
cd "$DATA_DIR" && mkdir -p SLR45 && cd SLR45
wget https://openslr.trmal.net/resources/45/ST-AEDS-20180100_1-OS.tgz && tar xzf ST-AEDS-20180100_1-OS.tgz
cargo run --release -p bunsen-asr --features wgpu -- whisper transcribe -v --json "$DATA_DIR"/SLR45/f0001_us_f0001_0000{1,2,3,4,5}.wav
```

`$DATA_DIR/SLR45/text.txt` holds the transcripts. The pushes are sized to the stream driver's grain, so the
autotuned kernels see one shape; the first file of a run pays for the tuning.
