# bunsen-tool

`bunsen-tool`: [bunsen](https://crates.io/crates/bunsen)'s models on the command line, a branch for each kind of model:

- `bunsen-tool asr`: speech to text, by profile or by model family, and who spoke when ([below](#asr-speech-to-text)).
- `bunsen-tool chat`: chat with a Qwen3 or nanochat checkpoint ([below](#chat-qwen3-and-nanochat)).
- `bunsen-tool cache <path|list|clear>`: the weights cache every command loads through, by kit.

`bunsen-tool` replaces `bunsen-asr`, whose last release is 0.41.0: its commands are `bunsen-tool asr`'s, and its
`cache` is `bunsen-tool cache`. The chat driver under `bunsen-tool chat` is also this crate's library,
[`bunsen_tool::chat`](https://docs.rs/bunsen-tool/latest/bunsen_tool/chat/).

## Models

`bunsen-tool card` describes the build (`build`: its version and features), its weights cache (`config`), and the
models it knows by name (`kits`), as YAML or, with `--format json`, JSON on one line; `--section` prints one. The
models, as `bunsen-tool card --section kits` prints them (`cargo make cards` rebuilds this card):

<!-- card: bunsen-tool card --section kits -->

```yaml
kits:
  asr:
    parakeet:
    - provider:
        name: well-known
    - provider:
        name: hf
    - model:
        name: fermion/phonon-2
        aliases:
        - phonon
        description: "Phonon-2: parakeet-tdt-0.6b-v3 with a five-value encoder, 164 MB, English"
    - model:
        name: nvidia/parakeet-tdt-0.6b-v3
        description: "parakeet-tdt-0.6b-v3: 600 M parameters, f32, 25 European languages"
    qwen3-asr:
    - provider:
        name: well-known
    - provider:
        name: hf
    - model:
        name: fermion/phonon-1
        description: "Phonon-1: Qwen3-ASR-0.6B fine-tuned for English, re-packed, 415 MB"
    - model:
        name: fermion/phonon-1-micro
        description: "Phonon-1-Micro: Qwen3-ASR-0.6B fine-tuned for English, re-packed, 285 MB"
    - model:
        name: fermion/phonon-1-big
        description: "Phonon-1-Big: Qwen3-ASR-0.6B fine-tuned for English, re-packed, 581 MB"
    - model:
        name: qwen/qwen3-asr-0.6b
        description: "Qwen3-ASR-0.6B: 0.8 B parameters, bf16, 30 languages"
    - model:
        name: qwen/qwen3-asr-1.7b
        description: "Qwen3-ASR-1.7B: 2.0 B parameters, bf16, 30 languages"
    voxtral:
    - provider:
        name: well-known
    - provider:
        name: hf
    - model:
        name: mistralai/voxtral-mini-4b-realtime-2602
        aliases:
        - voxtral-mini-4b-realtime
        description: "Voxtral-Mini-4B-Realtime-2602: 4.4 B parameters, bf16, 13 languages, streaming"
    whisper:
    - provider:
        name: well-known
    - provider:
        name: hf
    - model:
        name: openai/tiny.en
        description: 39 M parameters, English-only
    - model:
        name: openai/tiny
        description: 39 M parameters, multilingual
    - model:
        name: openai/base.en
        description: 74 M parameters, English-only
    - model:
        name: openai/base
        description: 74 M parameters, multilingual
    - model:
        name: openai/small.en
        description: 244 M parameters, English-only
    - model:
        name: openai/small
        description: 244 M parameters, multilingual
    - model:
        name: openai/medium.en
        description: 769 M parameters, English-only
    - model:
        name: openai/medium
        description: 769 M parameters, multilingual
    - model:
        name: openai/large-v1
        description: 1550 M parameters, multilingual
    - model:
        name: openai/large-v2
        description: 1550 M parameters, multilingual
    - model:
        name: openai/large-v3
        aliases:
        - large
        description: 1550 M parameters, multilingual, 128 mels
    - model:
        name: openai/large-v3-turbo
        aliases:
        - turbo
        description: 809 M parameters, multilingual, 128 mels, four-layer decoder
    sortformer:
    - provider:
        name: well-known
    - provider:
        name: hf
    - model:
        name: nvidia/nemotron-3-diarization
        description: "Nemotron-3-Diarization: 100 M parameters, up to 8 speakers, 10 ms frames, streaming"
    - model:
        name: nvidia/diar_streaming_sortformer_4spk-v2
        aliases:
        - sortformer-v2
        description: "Streaming Sortformer v2: 117 M parameters, up to 4 speakers, 80 ms frames"
    - model:
        name: nvidia/diar_streaming_sortformer_4spk-v2.1
        aliases:
        - sortformer-v2.1
        description: "Streaming Sortformer v2.1: 117 M parameters, up to 4 speakers, 80 ms frames"
    silero-vad:
    - provider:
        name: well-known
    - model:
        name: silero/vad-16k
        description: Silero VAD, the 16 kHz model, from upstream's safetensors (1.2 MB)
  chat:
    qwen3:
    - provider:
        name: well-known
    - provider:
        name: hf
    - model:
        name: qwen/qwen3-0.6b
        description: "Qwen3-0.6B: thinking and non-thinking chat, bf16, 1.50 GB"
    - model:
        name: qwen/qwen3-0.6b-base
        description: "Qwen3-0.6B-Base: the pretrained model, no chat tuning, bf16, 1.19 GB"
    - model:
        name: qwen/qwen3-1.7b
        description: "Qwen3-1.7B: thinking and non-thinking chat, bf16, 4.06 GB"
    - model:
        name: qwen/qwen3-1.7b-base
        description: "Qwen3-1.7B-Base: the pretrained model, no chat tuning, bf16, 3.44 GB"
    - model:
        name: qwen/qwen3-4b
        description: "Qwen3-4B: thinking and non-thinking chat, bf16, 8.04 GB"
    - model:
        name: qwen/qwen3-4b-base
        description: "Qwen3-4B-Base: the pretrained model, no chat tuning, bf16, 8.04 GB"
    - model:
        name: qwen/qwen3-4b-instruct-2507
        description: "Qwen3-4B-Instruct-2507: non-thinking chat, 256K context, bf16, 8.04 GB"
    - model:
        name: qwen/qwen3-4b-thinking-2507
        description: "Qwen3-4B-Thinking-2507: thinking chat, 256K context, bf16, 8.04 GB"
    - model:
        name: qwen/qwen3-8b
        description: "Qwen3-8B: thinking and non-thinking chat, bf16, 16.38 GB"
    - model:
        name: qwen/qwen3-8b-base
        description: "Qwen3-8B-Base: the pretrained model, no chat tuning, bf16, 16.38 GB"
    - model:
        name: qwen/qwen3-4b-saferl
        description: "Qwen3-4B-SafeRL: Qwen3-4B tuned for safety by reinforcement learning, thinking and non-thinking chat, bf16, 8.04 GB"
    - model:
        name: qwen/qwen3guard-gen-0.6b
        description: "Qwen3Guard-Gen-0.6B: safety moderation, rating a prompt or a reply Safe, Unsafe or Controversial, with its categories, bf16, 1.50 GB"
    - model:
        name: qwen/qwen3guard-gen-4b
        description: "Qwen3Guard-Gen-4B: safety moderation, rating a prompt or a reply Safe, Unsafe or Controversial, with its categories, bf16, 8.82 GB"
    - model:
        name: qwen/qwen3guard-gen-8b
        description: "Qwen3Guard-Gen-8B: safety moderation, rating a prompt or a reply Safe, Unsafe or Controversial, with its categories, bf16, 16.38 GB"
    nanochat:
    - provider:
        name: well-known
    - model:
        name: nanochat-students/base-d20
        aliases:
        - nanochat-d20-base
        description: "nanochat-students' d20 after pretraining (step 21400): upstream's 2025 model, 561M, base"
    - model:
        name: nanochat-students/mid-d20
        aliases:
        - nanochat-d20-mid
        description: "nanochat-students' d20 after midtraining (step 765): upstream's 2025 model, 561M, chat format"
    - model:
        name: nanochat-students/rl-d20
        aliases:
        - nanochat-d20-rl
        description: "nanochat-students' d20 after its RL stage (step 650): upstream's 2025 model, 561M, chat"
    - model:
        name: nitishpandey04/nanochat-d18-base
        description: "nitishpandey04's d18 after pretraining (step 3712): upstream's 2026 model, 702M, base"
    - model:
        name: nitishpandey04/nanochat-d18-sft
        description: "nitishpandey04's d18 after SFT (step 478): upstream's 2026 model, 702M, chat"
```

<!-- /card -->

## asr: speech to text

Name a **profile**, a model chosen by what it is for, and `bunsen-tool asr` picks the family, the model and its
settings, for files (`transcribe`) or the microphone (`live`):

```sh
bunsen-tool asr transcribe [--profile en] [FLAGS] FILE...   # audio files
bunsen-tool asr live       [--profile en] [FLAGS]           # the microphone, as it speaks
bunsen-tool asr live       [--speech S] [--task T] [--quality Q] [FLAGS]   # or by what is needed
bunsen-tool asr profiles   (list|show|fetch)                # what each profile is for, and what it runs
```

```sh
cargo run --release -p bunsen-tool --features wgpu -- asr transcribe recording.wav
cargo run --release -p bunsen-tool --features wgpu -- asr transcribe --profile multi --json recording.wav
cargo run --release -p bunsen-tool --features wgpu -- asr live --profile translate-en
cargo run --release -p bunsen-tool --features wgpu -- asr transcribe --speech multi --quality fast recording.wav
```

| profile | for | `transcribe` runs | `live` runs |
|---|---|---|---|
| `en` (`default`) | English, each word timed | `parakeet` `fermion/phonon-2` | `parakeet` `fermion/phonon-2`, `conservative` |
| `multi` | any of 30 languages, found from the speech | `qwen3-asr` `qwen/qwen3-asr-1.7b` | `qwen3-asr` `qwen/qwen3-asr-0.6b`, `conservative` |
| `translate-en` | any of Whisper's languages, as English text | `whisper` `openai/large-v3`, `--task translate` | `whisper` `openai/small`, `--task translate --beam-size 1`, `conservative` |

The profiles are this binary's, and improve with it: when a better model arrives, a profile's row changes, and the
command lines that name the profile do not. `profiles show [NAME...]` prints each recipe as the `bunsen-tool asr family`
command it runs, `-v` logs it before a run, and `transcribe --json` records `profile`, `need` (the point served:
`speech`, `task`, `quality`; its `task` is the one the model ran) and `recipe` (family, model, preset, flags) with each
transcript. `profiles fetch NAME...` brings every file a profile runs into the cache, as `family <family> fetch` does
for a model.

Each profile is a point in a table of what is needed: `--speech` (`en`, `multi`) x `--task` (`transcribe`, `translate`)
x `--quality` (`fast`, `balanced`, `good`), with a recipe for `transcribe` and one for `live` where a model was seen to
do the job. `en` is `en`/`transcribe`/`balanced`, `multi` is `multi`/`transcribe`/`balanced`, and `translate-en` is
`multi`/`translate`/`balanced`. A point with no recipe falls through to the nearest that has one (another quality, a
better before a worse; then `multi` for `en`), logged at `INFO`.

The flags choose, in this order:

1. **`--language` implies `--speech`**: `en` means `en` speech, any other code `multi`. `--speech en` with another
   language is an error.
2. **The point.** Without `--profile`, each of `--speech`, `--task` and `--quality` not given is `default`'s
   (`en`/`transcribe`/`balanced`), except that `--task translate` makes the speech `multi`; `--json` records the point
   as the profile (`multi/transcribe/fast`) whenever one of them, or `--language`, is given. With `--profile`, the
   ones not given are the profile's.
3. **English speech translated to English is transcribed**, logged at `INFO`: Whisper's output for it is its
   transcript, so no cell translates English.
4. **What runs.** Without `--profile`, the table's cell for the point. With it, the profile's recipe, and the flags
   may only adjust it: `--quality` moves along the profile's row; `--speech` (or the speech `--language` implies) must
   be one the profile hears (`multi` hears English); a task other than the profile's is set on a model that takes one
   (Whisper's `--task`; an `--opt` still wins). Anything else is an error that names another profile, or none.

So the task is `transcribe` unless `translate-en` or `--task translate` asks to translate speech that is not English.

| command line | runs | task |
|---|---|---|
| (nothing) | `en`: Parakeet | transcribe |
| `--language de` | `multi/transcribe/balanced`: Qwen3-ASR | transcribe |
| `--task translate` | `multi/translate/balanced`: Whisper | translate |
| `--speech en --task translate` | `en/transcribe/balanced`: Parakeet | transcribe |
| `--profile translate-en --task transcribe` | `translate-en`: Whisper, `--task transcribe` | transcribe |
| `--profile translate-en --speech en` | `translate-en`: Whisper, `--task transcribe` | transcribe |
| `--profile en --task translate` | `en`: Parakeet | transcribe |
| `--profile en --language de` | an error: `en` hears only English | |
| `--profile multi --task translate` | an error: Qwen3-ASR cannot translate | |

A profile's run takes every flag below that is not a family's own (`--publish`, `--json`, `--rttm`, `--diarize`,
`--device`, the cache's and `live`'s capture flags), and:

- `--language CODE`: the spoken language. It implies `--speech` (above), and is given to a model that can be told it
  (Whisper); one that finds the language itself goes on doing so.
- `--preset` and `--chunk-ms`, in place of the profile's.
- `--opt [FAMILY.]KEY[=VALUE]`, repeated: one of the family's own flags, without its `--` (`KEY` alone is `KEY=true`).
  With `FAMILY.` (`--opt whisper.beam-size=1`) it applies only while the profile runs that family and is logged as not
  applied otherwise, so a script keeps working when the profile moves to another family; several families' can be
  given at once. Without, the profile's family must have it. `model=NAME` replaces the profile's model. A key its
  family does not have is an error either way.

### Choosing a family yourself

`bunsen-tool asr family` runs a model family directly, each with its own `list` and `transcribe` and only the flags that
apply to it:

```sh
bunsen-tool asr family parakeet  (list|fetch|transcribe|live) [FLAGS]   # NVIDIA's Parakeet TDT; FermionResearch's Phonon-2 by default
bunsen-tool asr family qwen3-asr (list|fetch|transcribe|live) [FLAGS]   # Qwen's Qwen3-ASR; FermionResearch's Phonon-1 (English) by default
bunsen-tool asr family voxtral   (list|fetch|transcribe|live) [FLAGS]   # Mistral's Voxtral Realtime; Voxtral-Mini-4B-Realtime-2602 by default
bunsen-tool asr family whisper   (list|fetch|transcribe|live) [FLAGS]   # OpenAI's Whisper; openai/base by default
bunsen-tool asr diarize          (list|fetch|run|live)        [FLAGS]   # who spoke when: NVIDIA's Nemotron-3-Diarization
```

```sh
cargo run --release -p bunsen-tool --features wgpu -- asr family parakeet transcribe recording.wav
cargo run --release -p bunsen-tool --features wgpu -- asr family qwen3-asr transcribe --model qwen/qwen3-asr-1.7b --json recording.wav
cargo run --release -p bunsen-tool --features wgpu -- asr family voxtral transcribe --delay-ms 960 --json recording.wav
cargo run --release -p bunsen-tool --features wgpu -- asr family whisper transcribe --preset conservative --language en recording.wav
cargo run --release -p bunsen-tool -- asr family qwen3-asr list
```

Installed from crates.io, with the GPU backends you want (the CPU is always there):

```sh
cargo install bunsen-tool --features wgpu
bunsen-tool asr transcribe recording.wav
```

`list` shows the family's named models, with each one's licence, its files, and whether each is cached; every family
also takes `hf:org/repo`, a Hugging Face repo in `transformers`' layout, whose `config.json` gives the geometry;
`diarize` also takes one holding a Sortformer v2 `.nemo` alone, whose own config does
(`hf:nvidia/diar_streaming_sortformer_4spk-v2.1`), read from the archive's head before the archive is fetched. `fetch
NAME...` brings every file of the named models into the cache and prints where each one is (`--verify` re-hashes each
pinned file): a deployment that runs `--offline` fills its cache this way ahead of time.

`transcribe` loads `--model` (the family's default unless named) through bunsen's pretrained cache, fetching it on
first use, logs its info card (name, licence, origin, every file and where it came from) at `INFO` (`-v`), and prints
each file's text. `--json` prints JSON lines instead ([where the transcript goes](#where-the-transcript-goes)): each
file's statements, then its `transcript`: the text, the language when the model names one (Qwen3-ASR, Whisper), each
word's start and end in seconds (Parakeet, Voxtral), and each segment's (Whisper; Qwen3-ASR's are its committed windows), every
word and segment with the token ids it was decoded as. After each file it logs how
long the transcription took at `INFO`, and after several, the total. Audio is read at the model's rate, 16 kHz mono,
and streamed through the family's stream driver (bunsen's `asr_driver`), about a second a push.

`--preset` says when the driver decodes, for every family: `offline` (the default) decodes whole windows, one after
another (a family's `transcribe` default; a profile names its own); `conservative` also finds speech regions with Silero's VAD (`silero/vad-16k`, through the same cache) and
decodes each as it ends, so silence is never decoded; `responsive` adds a draft of the speech so far every 600 ms of
it, which `live` shows. `--chunk-ms` (1000) is how much audio each push carries.

`live` transcribes the microphone as it speaks, through the same driver: the host's default input device, or
`--input-device` (a substring of an id or name from `--list-devices`), opened at the model's rate when it offers that
and resampled otherwise. Each decode's committed text is a line on stdout (a statement per speaker under `--diarize`),
its times, where a target shows them, the wall clock's; under `--preset responsive` the draft of the speech so far is
on stderr, redrawn in place on a terminal. A family's `--preset` defaults to `conservative` here, `--chunk-ms`
to 250; `--seconds` stops after that much audio, Ctrl-C otherwise, and either way the speech past the last commit is
decoded before the process exits. The default `live` feature builds it: it captures through `cpal`, which on Linux
links ALSA (`libasound2-dev`); `--no-default-features` builds without it.

```sh
cargo run --release -p bunsen-tool --features wgpu -- asr family parakeet live --preset responsive
cargo run --release -p bunsen-tool -- asr live --list-devices
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
cargo run --release -p bunsen-tool --features wgpu -- asr diarize run --latency offline meeting.wav
cargo run --release -p bunsen-tool --features wgpu -- asr diarize run --rttm meeting.wav > meeting.rttm
```

`--diarize` on `transcribe` or `live`, a profile's or any family's, says who said what: the diarizer (`--diarize` alone for
`nvidia/nemotron-3-diarization`, `--diarize=MODEL` for another) runs beside the recognizer, through the stream
driver's speaker seam, and labels each word (Parakeet) or segment with its speaker. Qwen3-ASR and Whisper also cut
their windows at speaker turns, so a segment is one speaker's. The text is printed a line per statement, a run of one
speaker's words, `[speaker_k] text` (`[?]` where the diarizer heard nobody). Under `--json` each statement is wrapped
under its speaker, and the `transcript` adds each word's or segment's `speaker`, and the `turns`: each speaker's runs
with their start, end and text. `--rttm` prints the statements as NIST RTTM. `--diarize-latency` picks the diarizer's streaming preset, as `diarize
--latency` does; a decode waits until the diarizer has caught up with it.

```sh
cargo run --release -p bunsen-tool --features wgpu -- asr live --diarize
cargo run --release -p bunsen-tool --features wgpu -- asr transcribe --profile multi --diarize --json meeting.wav
```

### Where the transcript goes

`transcribe` and `live` push what they hear into a publisher, which delivers it to every target `--publish CONFIG`
names, at once. Without it, the target is stdout: text, or with `--json` JSON lines, or with `--rttm` NIST RTTM.
`CONFIG` is inline or a file's path, in JSON or YAML (`.json`, `.yaml`, `.yml`; inline, either): the list of targets,
or `{targets: [...]}` with `times`, `speakers` and `events` set for every target, each of which a target may set for
itself.

```yaml
times: true
targets:
  - type: stdout                    # text: [00:00:01.200 --> 00:00:03.400] [speaker_0] hello there
  - type: file
    path: meeting.jsonl
    format: jsonl                   # text (the default), jsonl, rttm (needs --diarize)
    append: false
  - type: log                       # a JSON line per record, at `level` under `target`
    level: info
    target: bunsen_tool::transcript
  - type: webhook                   # each record POSTed as JSON (the `webhook` feature)
    url: https://example.com/hook
    headers: {Authorization: Bearer TOKEN}
    events: [statement]
  - type: mqtt                      # each record published to {prefix}/{event} (the `mqtt` feature)
    broker: mqtt://broker.lan:1883  # plain TCP; 1883 by default
    prefix: bunsen-tool/room-1
    qos: 1                          # 0, 1 (the default) or 2
    retain: false
    client_id: bunsen-tool-room-1    # bunsen-tool-{pid} by default
    username: asr                   # and password, both optional
```

```sh
cargo run --release -p bunsen-tool --features wgpu -- asr live --diarize \
  --publish '[{type: stdout}, {type: mqtt, broker: broker.lan, prefix: office/asr}]'
```

The records are the events: a `statement`, a run of committed text one speaker spoke (undiarized, a decode's text on
`live`, a file's on `transcribe`); a `draft`, `live`'s latest draft under `--preset responsive`; and a `transcript`,
a file's whole transcript after its statements, with the profile and recipe that chose its model. A statement is flat;
under `--diarize`, it is wrapped under its speaker (`null` where the diarizer heard nobody):

```json
{"event":"statement","file":"meeting.wav","start":1.2,"end":3.4,"text":"hello there"}
{"event":"statement","file":"meeting.wav","speaker":{"id":"speaker_0","confidence":0.91},"statement":{"start":1.2,"end":3.4,"text":"hello there"}}
{"event":"draft","start":"2026-10-23T16:37:35.948Z","end":"2026-10-23T16:37:36.548Z","text":"hello th"}
```

- `times` (off for text, on otherwise): each statement's and draft's start and end. For a file, seconds from its start
  (`HH:MM:SS.mmm` in text); for the microphone, the wall clock, RFC 3339 in UTC (`HH:MM:SS.mmmZ` in text). RTTM always
  has seconds from the start.
- `speakers` (on): under `--diarize`, each statement's speaker; off, a diarized statement is a plain one.
- `events` (all): the records a target takes, of `statement`, `draft` and `transcript`. Text shows statements, and on
  stdout drafts on stderr; RTTM shows the statements with a speaker.

The default `webhook` and `mqtt` features build the network targets (through `ureq` and `rumqttc`); a build without
them refuses a config that names one. A local write that fails ends the run. A network target delivers from a thread of its own, so a slow one never holds
the audio up: a failed POST, or a publish while the broker is gone (`rumqttc` reconnects), is logged and the next
record tried, and at the end each waits for what is queued (MQTT up to 5 s for the broker's acknowledgements). The
log is filtered as every log is: `RUST_LOG=bunsen_tool::transcript=info` shows a `log` target's records alone.

| Family | Flags of its own |
|---|---|
| `parakeet` | `--window-seconds` (60): a full window is committed up to a pause between words, and the rest decoded again with the audio after it |
| `qwen3-asr` | `--window-seconds` (30, the most Qwen3-ASR was trained on): a full window is cut at the quietest point of its last third |
| `voxtral` | `--window-seconds` (30): a full window is committed up to a pause between words, and the rest decoded again with the audio after it; `--delay-ms` (the model's, 480): how far the transcript runs behind the audio, a multiple of 80 up to 1200, or 2400 |
| `whisper` | `--language` (detected when omitted), `--task` (`transcribe`, `translate`), `--beam-size` (5), `--timestamps` (split segments on timestamp tokens), `--max-tokens` (224 a window), `--prompt-carry` (`true`: prompt each window with the transcript so far), `--fallback` (`true`: upstream's temperature ladder), `--vocab` (a vocabulary file in place of the layout's), `--upstream-cache-dir` (`openai-whisper`'s download cache, used in place; `~/.cache/whisper`); `--model` also takes a checkpoint's path |

`bunsen-tool asr family whisper inspect NAME|PATH` scans a Whisper checkpoint and checks it against its prefab (its resources and
where each stands, the geometry it promised, and the one its shapes say); `bunsen-tool asr family whisper prefabs` lists
the geometries. Neither needs a backend.

`bunsen-tool cache` manages the weights cache every family loads through (`bunsen_app::cache`):

```sh
bunsen-tool cache path                         # the cache directory
bunsen-tool cache list [KIT[/NAMESPACE]]       # each file's kit, namespace, size, fetched and last used (UTC), pin
bunsen-tool cache clear [KIT[/NAMESPACE]] [--partial] [--age AGE] [--yes]
```

The kits are `parakeet`, `qwen3-asr`, `whisper`, `sortformer` (the diarizer), `silero_vad` (the VAD the
`conservative` preset loads), `qwen3` and `nanochat`.
`clear` without `--yes` only lists what it would remove; `--partial` selects unfinished downloads, and `--age 30d` the
files not loaded for 30 days (`s`, `m`, `h`, `d`, `w`). A load that finds a file in the cache marks it used; listing,
and checking what is cached, do not.

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

### Whisper models

Whisper's index keeps three things apart, because they vary independently:

- A **prefab** is a public configuration: a geometry with no weights (`n_mels`, vocabulary size, `d_model`, layer
  counts), upstream's `ModelDimensions` typed in, so a name means a shape before anything is fetched. The vocabulary
  size is part of it: `tiny.en` (51864) and `tiny` (51865) are different prefabs; `large-v3` has 128 mels and a
  hundredth language; `large-v3-turbo` a four-layer decoder. `bunsen-tool asr family whisper prefabs` lists them.
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

`bunsen-tool asr family whisper inspect NAME|PATH` prints a model's resources and where each stands, the prefab it promised, and
the geometry its checkpoint's shapes say; on a path, which prefab, if any, that geometry is, and which vocabulary its
layout selects; on a name whose checkpoint does not scan as its prefab, the mismatch. `transcribe` makes the same
check before it reads any tensors. The index is bunsen's (`kits::speech::whisper::pretrained`), and this binary keeps
none of its own.

### Benchmarks

At `-v`, `transcribe` logs each file's audio length, how long it took and the realtime factor, then the totals over
the files. Over a few utterances of [SLR45](https://www.openslr.org/45/):

```sh
cd "$DATA_DIR" && mkdir -p SLR45 && cd SLR45
wget https://openslr.trmal.net/resources/45/ST-AEDS-20180100_1-OS.tgz && tar xzf ST-AEDS-20180100_1-OS.tgz
cargo run --release -p bunsen-tool --features wgpu -- asr family whisper transcribe -v --json "$DATA_DIR"/SLR45/f0001_us_f0001_0000{1,2,3,4,5}.wav
```

`$DATA_DIR/SLR45/text.txt` holds the transcripts. The pushes are sized to the stream driver's grain, so the
autotuned kernels see one shape; the first file of a run pays for the tuning.

## chat: Qwen3 and nanochat

`bunsen-tool chat` answers one prompt (`run`) or holds a conversation on stdin (`chat`: `/reset` starts over, `/quit`
or the end of input leaves), on the KV cache: each turn prefills only what the cache does not already hold.

```sh
bunsen-tool chat list [--urls]                       # every family's checkpoints, and what of each is cached
bunsen-tool chat fetch NAME...                       # bring checkpoints into the cache
bunsen-tool chat run  [--model NAME] [FLAGS] [PROMPT]   # one reply; the prompt from stdin when omitted
bunsen-tool chat chat [--model NAME] [FLAGS]            # a conversation
bunsen-tool chat family (qwen3|nanochat) (list|fetch|run|chat) [FLAGS]   # one family's, its own default by default
```

```sh
cargo run --release -p bunsen-tool --features wgpu -- chat run --no-think "What is the capital of France?"
cargo run --release -p bunsen-tool --features wgpu -- chat chat --model qwen3-4b-instruct-2507 --stats
cargo run --release -p bunsen-tool --features wgpu -- chat family nanochat run "What is the capital of France?"
cargo run --release -p bunsen-tool --features cuda -- chat family nanochat chat --device cuda --precision half
```

Outside `family`, the name `--model` gives picks the family, and Qwen3's default (`qwen/qwen3-0.6b`) is the default.
The families:

| family | checkpoints | default |
|---|---|---|
| `qwen3` | Qwen's Qwen3 by name, or `hf:org/repo`, any Hugging Face repo in `transformers`' `Qwen3ForCausalLM` layout | `qwen/qwen3-0.6b` |
| `nanochat` | community checkpoints of upstream nanochat's 2025 and 2026 models (`nanochat-students/{base,mid,rl}-d20`, `nitishpandey04/nanochat-d18-{base,sft}`) | `nanochat-students/rl-d20` |

Each family takes only the flags that apply to it: thinking (`--think`, `--no-think`, `--show-thinking`) is Qwen3's.
Sampling follows the checkpoint's generation defaults; `--greedy`, `--temperature`, `--top-k`, `--top-p` and `--seed`
override them. `--precision half` runs in bf16 where the device does (cuda, metal), the dtype Qwen3 ships and
nanochat trains in; f16 overflows Qwen3. `--context` sizes the KV cache (the checkpoint's trained length by default),
`--kv-storage` picks how it stores its history, and `--stats` prints each reply's token counts and speeds.
