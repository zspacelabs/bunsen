# bunsen-asr

`bunsen-asr`: speech to text on [bunsen](https://crates.io/crates/bunsen). Name a **profile**, a model chosen by what
it is for, and `bunsen-asr` picks the family, the model and its settings, for files (`transcribe`) or the microphone
(`live`):

```sh
bunsen-asr transcribe [--profile en] [FLAGS] FILE...   # audio files
bunsen-asr live       [--profile en] [FLAGS]           # the microphone, as it speaks
bunsen-asr live       [--speech S] [--task T] [--quality Q] [FLAGS]   # or by what is needed
bunsen-asr profiles   (list|show|fetch)                # what each profile is for, and what it runs
```

```sh
cargo run --release -p bunsen-asr --features wgpu -- transcribe recording.wav
cargo run --release -p bunsen-asr --features wgpu -- transcribe --profile multi --json recording.wav
cargo run --release -p bunsen-asr --features wgpu -- live --profile translate-en
cargo run --release -p bunsen-asr --features wgpu -- transcribe --speech multi --quality fast recording.wav
```

| profile | for | `transcribe` runs | `live` runs |
|---|---|---|---|
| `en` (`default`) | English, each word timed | `parakeet` `fermion/phonon-2` | `parakeet` `fermion/phonon-2`, `conservative` |
| `multi` | any of 30 languages, found from the speech | `qwen3-asr` `qwen/qwen3-asr-1.7b` | `qwen3-asr` `qwen/qwen3-asr-0.6b`, `conservative` |
| `translate-en` | any of Whisper's languages, as English text | `whisper` `openai/large-v3`, `--task translate` | `whisper` `openai/small`, `--task translate --beam-size 1`, `conservative` |

The profiles are this binary's, and improve with it: when a better model arrives, a profile's row changes, and the
command lines that name the profile do not. `profiles show [NAME...]` prints each recipe as the `bunsen-asr family`
command it runs, `-v` logs it before a run, and `transcribe --json` records `profile` and `recipe` (family, model,
preset, flags) with each transcript. `profiles fetch NAME...` brings every file a profile runs into the cache, as
`family <family> fetch` does for a model.

Each profile is a point in a table of what is needed: `--speech` (`en`, `multi`) x `--task` (`transcribe`, `translate`)
x `--quality` (`fast`, `balanced`, `good`), with a recipe for `transcribe` and one for `live` where a model was seen to
do the job. `en` is `en`/`transcribe`/`balanced`, `multi` is `multi`/`transcribe`/`balanced`, and `translate-en` is
`multi`/`translate`/`balanced`. A point with no recipe falls through to the nearest that has one (another quality, a
better before a worse; then `multi` for `en`), logged at `INFO`.

- Without `--profile`, the three flags choose the point, over `default`'s; `--task translate` alone means `multi`
  speech. `--json` records the point as the profile (`multi/transcribe/fast`).
- With `--profile`, the profile's model is kept, and the flags may only adjust it: `--quality` moves along the
  profile's row, `--task` is set on a model that takes a task (Whisper's `--task`; an `--opt` still wins), and
  `--speech` must be one the profile hears (`multi` hears English). Anything else is an error:
  `--profile multi --task translate` fails, as Qwen3-ASR cannot translate.

A profile's run takes every flag below that is not a family's own (`--json`, `--rttm`, `--diarize`, `--device`, the
cache's and `live`'s capture flags), and:

- `--language CODE`: the spoken language, for a model that can be told it (Whisper); one that finds the language itself
  goes on doing so.
- `--preset` and `--chunk-ms`, in place of the profile's.
- `--opt [FAMILY.]KEY[=VALUE]`, repeated: one of the family's own flags, without its `--` (`KEY` alone is `KEY=true`).
  With `FAMILY.` (`--opt whisper.beam-size=1`) it applies only while the profile runs that family and is logged as not
  applied otherwise, so a script keeps working when the profile moves to another family; several families' can be
  given at once. Without, the profile's family must have it. `model=NAME` replaces the profile's model. A key its
  family does not have is an error either way.

## Choosing a family yourself

`bunsen-asr family` runs a model family directly, each with its own `list` and `transcribe` and only the flags that
apply to it:

```sh
bunsen-asr family parakeet  (list|fetch|transcribe|live) [FLAGS]   # NVIDIA's Parakeet TDT; FermionResearch's Phonon-2 by default
bunsen-asr family qwen3-asr (list|fetch|transcribe|live) [FLAGS]   # Qwen's Qwen3-ASR; FermionResearch's Phonon-1 (English) by default
bunsen-asr family whisper   (list|fetch|transcribe|live) [FLAGS]   # OpenAI's Whisper; openai/base by default
bunsen-asr diarize          (list|fetch|run|live)        [FLAGS]   # who spoke when: NVIDIA's Nemotron-3-Diarization
```

```sh
cargo run --release -p bunsen-asr --features wgpu -- family parakeet transcribe recording.wav
cargo run --release -p bunsen-asr --features wgpu -- family qwen3-asr transcribe --model qwen/qwen3-asr-1.7b --json recording.wav
cargo run --release -p bunsen-asr --features wgpu -- family whisper transcribe --preset conservative --language en recording.wav
cargo run --release -p bunsen-asr -- family qwen3-asr list
```

Installed from crates.io, with the GPU backends you want (the CPU is always there):

```sh
cargo install bunsen-asr --features wgpu
bunsen-asr transcribe recording.wav
```

`list` shows the family's named models, with each one's licence, its files, and whether each is cached; every family
also takes `hf:org/repo`, a Hugging Face repo in `transformers`' layout, whose `config.json` gives the geometry;
`diarize` also takes one holding a Sortformer v2 `.nemo` alone, whose own config does
(`hf:nvidia/diar_streaming_sortformer_4spk-v2.1`), read from the archive's head before the archive is fetched. `fetch
NAME...` brings every file of the named models into the cache and prints where each one is (`--verify` re-hashes each
pinned file): a deployment that runs `--offline` fills its cache this way ahead of time.

`transcribe` loads `--model` (the family's default unless named) through bunsen's pretrained cache, fetching it on
first use, logs its info card (name, licence, origin, every file and where it came from) at `INFO` (`-v`), and prints
each file's text, or with `--json` one object per file: the text, the language when the model names one (Qwen3-ASR,
Whisper), each word's start and end in seconds (Parakeet), and each segment's (Whisper; Qwen3-ASR's are its committed
windows), every word and segment with the token ids it was decoded as. After each file it logs how
long the transcription took at `INFO`, and after several, the total. Audio is read at the model's rate, 16 kHz mono,
and streamed through the family's stream driver (bunsen's `asr_driver`), about a second a push.

`--preset` says when the driver decodes, for every family: `offline` (the default) decodes whole windows, one after
another (a family's `transcribe` default; a profile names its own); `conservative` also finds speech regions with Silero's VAD (`silero/vad-16k`, through the same cache) and
decodes each as it ends, so silence is never decoded; `responsive` adds a draft of the speech so far every 600 ms of
it, which `live` shows. `--chunk-ms` (1000) is how much audio each push carries.

`live` transcribes the microphone as it speaks, through the same driver: the host's default input device, or
`--input-device` (a substring of an id or name from `--list-devices`), opened at the model's rate when it offers that
and resampled otherwise. Each decode's committed text is a line on stdout; under `--preset responsive` the draft of the
speech so far is on stderr, redrawn in place on a terminal. A family's `--preset` defaults to `conservative` here, `--chunk-ms`
to 250; `--seconds` stops after that much audio, Ctrl-C otherwise, and either way the speech past the last commit is
decoded before the process exits. The default `live` feature builds it: it captures through `cpal`, which on Linux
links ALSA (`libasound2-dev`); `--no-default-features` builds without it.

```sh
cargo run --release -p bunsen-asr --features wgpu -- family parakeet live --preset responsive
cargo run --release -p bunsen-asr -- live --list-devices
```

`diarize` says who spoke when, through NVIDIA's Sortformer diarizers: `nvidia/nemotron-3-diarization` (up to 8 speakers,
10 ms frames, the default), or the streaming Sortformer v2 and v2.1 (`--model sortformer-v2`, `--model sortformer-v2.1`;
up to 4 speakers, 80 ms frames). `diarize run FILE...` prints each file's speaker turns as lines of `{start} {end}
speaker_{k}` in seconds, speakers numbered in the order they first speak; `--rttm` prints NIST RTTM instead (the file's
stem as its recording id), and `--json` one object per file, its turns and each speaker's total. `diarize live` prints
each turn as it closes, from the microphone, opened as the families' `live` opens it. `--latency` picks the model's
streaming preset: `low` (1.04 s, the default), `ultra-low` (0.32 s), `offline` (30.4 s chunks, the most accurate on
files), and the presets only one family has, `very-low` (0.64 s, Nemotron-3-Diarization) and `high` (10 s, Sortformer
v2). `--postprocess` picks how activity becomes turns: `none` (thresholds of 0.5, the default), or NeMo's tuned
`dihard3` or `callhome` parameters. `list` and `fetch` work as for the families.

```sh
cargo run --release -p bunsen-asr --features wgpu -- diarize run --latency offline meeting.wav
cargo run --release -p bunsen-asr --features wgpu -- diarize run --rttm meeting.wav > meeting.rttm
```

`--diarize` on `transcribe` or `live`, a profile's or any family's, says who said what: the diarizer (`--diarize` alone for
`nvidia/nemotron-3-diarization`, `--diarize=MODEL` for another) runs beside the recognizer, through the stream
driver's speaker seam, and labels each word (Parakeet) or segment with its speaker. Qwen3-ASR and Whisper also cut
their windows at speaker turns, so a segment is one speaker's. The text is printed a line per turn, `[speaker_k] text`
(`[?]` where the diarizer heard nobody); `live` prefixes a line only where the speaker changes. `--json` adds each
word's or segment's `speaker`, and the `turns`: each speaker's runs with their start, end and text. `transcribe
--rttm` prints the turns as NIST RTTM. `--diarize-latency` picks the diarizer's streaming preset, as `diarize
--latency` does; a decode waits until the diarizer has caught up with it.

```sh
cargo run --release -p bunsen-asr --features wgpu -- live --diarize
cargo run --release -p bunsen-asr --features wgpu -- transcribe --profile multi --diarize --json meeting.wav
```

| Family | Flags of its own |
|---|---|
| `parakeet` | `--window-seconds` (60): a full window is committed up to a pause between words, and the rest decoded again with the audio after it |
| `qwen3-asr` | `--window-seconds` (30, the most Qwen3-ASR was trained on): a full window is cut at the quietest point of its last third |
| `whisper` | `--language` (detected when omitted), `--task` (`transcribe`, `translate`), `--beam-size` (5), `--timestamps` (split segments on timestamp tokens), `--max-tokens` (224 a window), `--prompt-carry` (`true`: prompt each window with the transcript so far), `--fallback` (`true`: upstream's temperature ladder), `--vocab` (a vocabulary file in place of the layout's), `--upstream-cache-dir` (`openai-whisper`'s download cache, used in place; `~/.cache/whisper`); `--model` also takes a checkpoint's path |

`bunsen-asr family whisper inspect NAME|PATH` scans a Whisper checkpoint and checks it against its prefab (its resources and
where each stands, the geometry it promised, and the one its shapes say); `bunsen-asr family whisper prefabs` lists
the geometries. Neither needs a backend.

`bunsen-asr cache` manages the weights cache every family loads through (`bunsen_app::cache`):

```sh
bunsen-asr cache path                         # the cache directory
bunsen-asr cache list [KIT[/NAMESPACE]]       # each file's kit, namespace, size and pin, then the total
bunsen-asr cache clear [KIT[/NAMESPACE]] [--partial] [--yes]
```

The kits are `parakeet`, `qwen3-asr`, `whisper`, `sortformer` (the diarizer) and `silero_vad` (the VAD the
`conservative` preset loads).
`clear` without `--yes` only lists what it would remove; `--partial` selects unfinished downloads.

`transcribe` and `live` take `--preset` and `--chunk-ms` (`diarize live` takes `--chunk-ms`); every command takes `--cache-dir`, `--offline`, `--device` (from the backends compiled in: `--features wgpu`, `vulkan`,
`metal`, `cuda`; the CPU is always there) and `-v`/`-q`.

| model | | licence |
|---|---|---|
| `parakeet`: `fermion/phonon-2` | 164 MB, `parakeet-tdt-0.6b-v3` with a five-value encoder, English | CC-BY-4.0 |
| `parakeet`: `nvidia/parakeet-tdt-0.6b-v3` | 2.5 GB, NVIDIA's base model, 25 European languages | CC-BY-4.0 |
| `qwen3-asr`: `fermion/phonon-1` | 415 MB, `Qwen3-ASR-0.6B` fine-tuned for English (other speech comes out as broken English); `-micro` (285 MB) and `-big` (581 MB) builds | Apache-2.0 |
| `qwen3-asr`: `qwen/qwen3-asr-0.6b`, `qwen/qwen3-asr-1.7b` | 1.6 and 4.1 GB, Qwen's base models, 30 languages | Apache-2.0 |
| `whisper`: `openai/*` | every `openai-whisper` checkpoint, `tiny` to `large-v3-turbo` | MIT |
| `diarize`: `nvidia/nemotron-3-diarization` | 397 MB, NVIDIA's streaming Sortformer, up to 8 speakers | OpenMDW-1.1 |

## Whisper models

Whisper's index keeps three things apart, because they vary independently:

- A **prefab** is a public configuration: a geometry with no weights (`n_mels`, vocabulary size, `d_model`, layer
  counts), upstream's `ModelDimensions` typed in, so a name means a shape before anything is fetched. The vocabulary
  size is part of it: `tiny.en` (51864) and `tiny` (51865) are different prefabs; `large-v3` has 128 mels and a
  hundredth language; `large-v3-turbo` a four-layer decoder. `bunsen-asr family whisper prefabs` lists them.
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

`bunsen-asr family whisper inspect NAME|PATH` prints a model's resources and where each stands, the prefab it promised, and
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
cargo run --release -p bunsen-asr --features wgpu -- family whisper transcribe -v --json "$DATA_DIR"/SLR45/f0001_us_f0001_0000{1,2,3,4,5}.wav
```

`$DATA_DIR/SLR45/text.txt` holds the transcripts. The pushes are sized to the stream driver's grain, so the
autotuned kernels see one shape; the first file of a run pays for the tuning.
