# bunsen-tool

`bunsen-tool`: [bunsen](https://crates.io/crates/bunsen)'s models on the command line, a branch for each kind of model:

- `bunsen-tool asr`: speech to text, by ASR preset or by model family, and who spoke when ([below](#asr-speech-to-text)).
- `bunsen-tool chat`: chat with a Qwen3, Gemma or nanochat checkpoint ([below](#chat-qwen3-gemma-and-nanochat)).
- `bunsen-tool decide`: typed questions about a context, answered with calibrated probabilities by a Laya decision
  model ([below](#decide-typed-questions-answered-by-laya)).
- `bunsen-tool cache <path|list|clear>`: the weights cache every command loads through, by kit.

`bunsen-tool` replaces `bunsen-asr`, whose last release is 0.41.0: its commands are `bunsen-tool asr`'s, and its
`cache` is `bunsen-tool cache`. The chat driver under `bunsen-tool chat` is also this crate's library,
[`bunsen_tool::chat`](https://docs.rs/bunsen-tool/latest/bunsen_tool/chat/).

## Models

`bunsen-tool card` describes the build (`build`: its version and features), its weights cache (`config`), and the
models it knows by name (`kits`: each one's presets in brief, its axes' members and its default), as YAML or, with
`--format json`, JSON on one line; `--section` prints one. `--policies` adds each model's presets as data, their
patches included, and the policy its default selection resolves to. The
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
        description: "Phonon-2: parakeet-tdt-0.6b-v3 with a five-value encoder, English"
        size: 163.5 MB
    - model:
        name: nvidia/parakeet-tdt-0.6b-v3
        description: "parakeet-tdt-0.6b-v3: 600 M parameters, 25 European languages"
        size: 2.51 GB
        dtype: float32
    qwen3-asr:
    - provider:
        name: well-known
    - provider:
        name: hf
    - model:
        name: fermion/phonon-1
        description: "Phonon-1: Qwen3-ASR-0.6B fine-tuned for English, re-packed"
        size: 415.1 MB
    - model:
        name: fermion/phonon-1-micro
        description: "Phonon-1-Micro: Qwen3-ASR-0.6B fine-tuned for English, re-packed"
        size: 285.1 MB
    - model:
        name: fermion/phonon-1-big
        description: "Phonon-1-Big: Qwen3-ASR-0.6B fine-tuned for English, re-packed"
        size: 580.9 MB
    - model:
        name: qwen/qwen3-asr-0.6b
        description: "Qwen3-ASR-0.6B: 0.8 B parameters, 30 languages"
        size: 1.58 GB
        dtype: bfloat16
    - model:
        name: qwen/qwen3-asr-1.7b
        description: "Qwen3-ASR-1.7B: 2.0 B parameters, 30 languages"
        size: 4.09 GB
        dtype: bfloat16
    voxtral:
    - provider:
        name: well-known
    - provider:
        name: hf
    - model:
        name: mistralai/voxtral-mini-4b-realtime-2602
        aliases:
        - voxtral-mini-4b-realtime
        description: "Voxtral-Mini-4B-Realtime-2602: 4.4 B parameters, 13 languages, streaming"
        size: 8.87 GB
        dtype: bfloat16
    whisper:
    - provider:
        name: well-known
    - provider:
        name: hf
    - model:
        name: openai/tiny.en
        description: 39 M parameters, English-only
        size: 76.4 MB
        dtype: float16
    - model:
        name: openai/tiny
        description: 39 M parameters, multilingual
        size: 76.4 MB
        dtype: float16
        presets:
          task:
          - transcribe
          - translate
          default:
          - transcribe
    - model:
        name: openai/base.en
        description: 74 M parameters, English-only
        size: 146.1 MB
        dtype: float16
    - model:
        name: openai/base
        description: 74 M parameters, multilingual
        size: 146.1 MB
        dtype: float16
        presets:
          task:
          - transcribe
          - translate
          default:
          - transcribe
    - model:
        name: openai/small.en
        description: 244 M parameters, English-only
        size: 484.5 MB
        dtype: float16
    - model:
        name: openai/small
        description: 244 M parameters, multilingual
        size: 484.4 MB
        dtype: float16
        presets:
          task:
          - transcribe
          - translate
          default:
          - transcribe
    - model:
        name: openai/medium.en
        description: 769 M parameters, English-only
        size: 1.53 GB
        dtype: float16
    - model:
        name: openai/medium
        description: 769 M parameters, multilingual
        size: 1.53 GB
        dtype: float16
        presets:
          task:
          - transcribe
          - translate
          default:
          - transcribe
    - model:
        name: openai/large-v1
        description: 1550 M parameters, multilingual
        size: 3.09 GB
        dtype: float16
        presets:
          task:
          - transcribe
          - translate
          default:
          - transcribe
    - model:
        name: openai/large-v2
        description: 1550 M parameters, multilingual
        size: 3.09 GB
        dtype: float16
        presets:
          task:
          - transcribe
          - translate
          default:
          - transcribe
    - model:
        name: openai/large-v3
        aliases:
        - large
        description: 1550 M parameters, multilingual, 128 mels
        size: 3.09 GB
        dtype: float16
        presets:
          task:
          - transcribe
          - translate
          default:
          - transcribe
    - model:
        name: openai/large-v3-turbo
        aliases:
        - turbo
        description: 809 M parameters, multilingual, 128 mels, four-layer decoder
        size: 1.62 GB
        dtype: float16
    sortformer:
    - provider:
        name: well-known
    - provider:
        name: hf
    - model:
        name: nvidia/nemotron-3-diarization
        description: "Nemotron-3-Diarization: 100 M parameters, up to 8 speakers, 10 ms frames, streaming"
        size: 397.0 MB
        dtype: float32
        presets:
          latency:
          - ultra-low
          - very-low
          - low
          - offline
          postprocess:
          - none
          - dihard3
          - callhome
          default:
          - low
          - none
    - model:
        name: nvidia/diar_streaming_sortformer_4spk-v2
        aliases:
        - sortformer-v2
        description: "Streaming Sortformer v2: 117 M parameters, up to 4 speakers, 80 ms frames"
        size: 471.4 MB
        dtype: float32
        presets:
          latency:
          - ultra-low
          - low
          - high
          - offline
          postprocess:
          - none
          - dihard3
          - callhome
          default:
          - low
          - none
    - model:
        name: nvidia/diar_streaming_sortformer_4spk-v2.1
        aliases:
        - sortformer-v2.1
        description: "Streaming Sortformer v2.1: 117 M parameters, up to 4 speakers, 80 ms frames"
        size: 471.4 MB
        dtype: float32
        presets:
          latency:
          - ultra-low
          - low
          - high
          - offline
          postprocess:
          - none
          - dihard3
          - callhome
          default:
          - low
          - none
    silero-vad:
    - provider:
        name: well-known
    - model:
        name: silero/vad-16k
        description: Silero VAD, the 16 kHz model, from upstream's safetensors
        size: 1.2 MB
        dtype: float32
        presets:
          vad-policy:
          - silero
          - faster-whisper
          default:
          - silero
  chat:
    qwen3:
    - provider:
        name: well-known
    - provider:
        name: hf
    - model:
        name: qwen/qwen3-0.6b
        description: "Qwen3-0.6B: thinking and non-thinking chat"
        size: 1.51 GB
        dtype: bfloat16
        presets:
          thinking-mode:
          - thinking
          - no-thinking
          default:
          - thinking
    - model:
        name: qwen/qwen3-0.6b-base
        description: "Qwen3-0.6B-Base: the pretrained model, no chat tuning"
        size: 1.20 GB
        dtype: bfloat16
    - model:
        name: qwen/qwen3-1.7b
        description: "Qwen3-1.7B: thinking and non-thinking chat"
        size: 4.07 GB
        dtype: bfloat16
        presets:
          thinking-mode:
          - thinking
          - no-thinking
          default:
          - thinking
    - model:
        name: qwen/qwen3-1.7b-base
        description: "Qwen3-1.7B-Base: the pretrained model, no chat tuning"
        size: 3.45 GB
        dtype: bfloat16
    - model:
        name: qwen/qwen3-4b
        description: "Qwen3-4B: thinking and non-thinking chat"
        size: 8.06 GB
        dtype: bfloat16
        presets:
          thinking-mode:
          - thinking
          - no-thinking
          default:
          - thinking
    - model:
        name: qwen/qwen3-4b-base
        description: "Qwen3-4B-Base: the pretrained model, no chat tuning"
        size: 8.05 GB
        dtype: bfloat16
    - model:
        name: qwen/qwen3-4b-instruct-2507
        description: "Qwen3-4B-Instruct-2507: non-thinking chat, 256K context"
        size: 8.06 GB
        dtype: bfloat16
    - model:
        name: qwen/qwen3-4b-thinking-2507
        description: "Qwen3-4B-Thinking-2507: thinking chat, 256K context"
        size: 8.06 GB
        dtype: bfloat16
    - model:
        name: qwen/qwen3-8b
        description: "Qwen3-8B: thinking and non-thinking chat"
        size: 16.39 GB
        dtype: bfloat16
        presets:
          thinking-mode:
          - thinking
          - no-thinking
          default:
          - thinking
    - model:
        name: qwen/qwen3-8b-base
        description: "Qwen3-8B-Base: the pretrained model, no chat tuning"
        size: 16.39 GB
        dtype: bfloat16
    - model:
        name: qwen/qwen3-4b-saferl
        description: "Qwen3-4B-SafeRL: Qwen3-4B tuned for safety by reinforcement learning, thinking and non-thinking chat"
        size: 8.06 GB
        dtype: bfloat16
        presets:
          thinking-mode:
          - thinking
          - no-thinking
          default:
          - thinking
    - model:
        name: qwen/qwen3guard-gen-0.6b
        description: "Qwen3Guard-Gen-0.6B: safety moderation, rating a prompt or a reply Safe, Unsafe or Controversial, with its categories"
        size: 1.51 GB
        dtype: bfloat16
    - model:
        name: qwen/qwen3guard-gen-4b
        description: "Qwen3Guard-Gen-4B: safety moderation, rating a prompt or a reply Safe, Unsafe or Controversial, with its categories"
        size: 8.83 GB
        dtype: bfloat16
    - model:
        name: qwen/qwen3guard-gen-8b
        description: "Qwen3Guard-Gen-8B: safety moderation, rating a prompt or a reply Safe, Unsafe or Controversial, with its categories"
        size: 16.39 GB
        dtype: bfloat16
    qwen3.5:
    - provider:
        name: well-known
    - provider:
        name: hf
    - model:
        name: qwen/qwen3.5-0.8b
        description: "Qwen3.5-0.8B: thinking and non-thinking chat (its text model)"
        size: 1.76 GB
        dtype: bfloat16
        presets:
          thinking-mode:
          - thinking
          - no-thinking
          default:
          - no-thinking
    - model:
        name: qwen/qwen3.5-0.8b-base
        description: "Qwen3.5-0.8B-Base: the pretrained model, no chat tuning (its text model)"
        size: 1.76 GB
        dtype: bfloat16
    - model:
        name: qwen/qwen3.5-2b
        description: "Qwen3.5-2B: thinking and non-thinking chat (its text model)"
        size: 4.56 GB
        dtype: bfloat16
        presets:
          thinking-mode:
          - thinking
          - no-thinking
          default:
          - no-thinking
    - model:
        name: qwen/qwen3.5-2b-base
        description: "Qwen3.5-2B-Base: the pretrained model, no chat tuning (its text model)"
        size: 4.56 GB
        dtype: bfloat16
    - model:
        name: qwen/qwen3.5-4b
        description: "Qwen3.5-4B: thinking and non-thinking chat (its text model)"
        size: 9.33 GB
        dtype: bfloat16
        presets:
          thinking-mode:
          - thinking
          - no-thinking
          default:
          - thinking
    - model:
        name: qwen/qwen3.5-4b-base
        description: "Qwen3.5-4B-Base: the pretrained model, no chat tuning (its text model)"
        size: 9.33 GB
        dtype: bfloat16
    - model:
        name: qwen/qwen3.5-9b
        description: "Qwen3.5-9B: thinking and non-thinking chat (its text model)"
        size: 19.32 GB
        dtype: bfloat16
        presets:
          thinking-mode:
          - thinking
          - no-thinking
          default:
          - thinking
    - model:
        name: qwen/qwen3.5-9b-base
        description: "Qwen3.5-9B-Base: the pretrained model, no chat tuning (its text model)"
        size: 19.32 GB
        dtype: bfloat16
    - model:
        name: qwen/qwen3.5-27b
        description: "Qwen3.5-27B: thinking and non-thinking chat (its text model); 54 GB in bf16, more than a card holds"
        size: 55.58 GB
        dtype: bfloat16
        presets:
          thinking-mode:
          - thinking
          - no-thinking
          default:
          - thinking
    gemma:
    - provider:
        name: well-known
    - provider:
        name: hf
    - model:
        name: gemma/gemma-3-270m
        description: "Gemma 3 270M: the pretrained model"
        size: 570.8 MB
        dtype: bfloat16
    - model:
        name: gemma/gemma-3-270m-it
        description: Gemma 3 270M, instruction-tuned
        size: 570.8 MB
        dtype: bfloat16
    - model:
        name: gemma/gemma-3-1b-pt
        description: "Gemma 3 1B: the pretrained model"
        size: 2.03 GB
        dtype: bfloat16
    - model:
        name: gemma/gemma-3-1b-it
        description: Gemma 3 1B, instruction-tuned
        size: 2.03 GB
        dtype: bfloat16
    - model:
        name: gemma/gemma-3-4b-pt
        description: "Gemma 3 4B: the pretrained model's text model"
        size: 8.63 GB
        dtype: bfloat16
    - model:
        name: gemma/gemma-3-4b-it
        description: "Gemma 3 4B, instruction-tuned: its text model"
        size: 8.63 GB
        dtype: bfloat16
    - model:
        name: gemma/gemma-2-2b
        description: "Gemma 2 2B: the pretrained model, stored in f32"
        size: 5.25 GB
        dtype: bfloat16
    - model:
        name: gemma/gemma-2-2b-it
        description: Gemma 2 2B, instruction-tuned
        size: 5.25 GB
        dtype: bfloat16
    - model:
        name: gemma/gemma-2-9b
        description: "Gemma 2 9B: the pretrained model"
        size: 18.52 GB
        dtype: bfloat16
    - model:
        name: gemma/gemma-2-9b-it
        description: Gemma 2 9B, instruction-tuned
        size: 18.50 GB
        dtype: bfloat16
    - model:
        name: gemma/gemma-2b
        description: "Gemma 2B (Gemma 1): the pretrained model"
        size: 5.03 GB
        dtype: bfloat16
    - model:
        name: gemma/gemma-2b-it
        description: Gemma 2B (Gemma 1), instruction-tuned
        size: 5.03 GB
        dtype: bfloat16
    - model:
        name: gemma/gemma-7b
        description: "Gemma 7B (Gemma 1): the pretrained model"
        size: 17.11 GB
        dtype: bfloat16
    - model:
        name: gemma/gemma-7b-it
        description: Gemma 7B (Gemma 1), instruction-tuned
        size: 17.09 GB
        dtype: bfloat16
    nanochat:
    - provider:
        name: well-known
    - model:
        name: nanochat-students/base-d20
        aliases:
        - nanochat-d20-base
        description: "nanochat-students' d20 after pretraining (step 21400): upstream's 2025 model, 561M, base"
        size: 2.08 GB
    - model:
        name: nanochat-students/mid-d20
        aliases:
        - nanochat-d20-mid
        description: "nanochat-students' d20 after midtraining (step 765): upstream's 2025 model, 561M, chat format"
        size: 2.08 GB
    - model:
        name: nanochat-students/rl-d20
        aliases:
        - nanochat-d20-rl
        description: "nanochat-students' d20 after its RL stage (step 650): upstream's 2025 model, 561M, chat"
        size: 2.08 GB
    - model:
        name: nitishpandey04/nanochat-d18-base
        description: "nitishpandey04's d18 after pretraining (step 3712): upstream's 2026 model, 702M, base"
        size: 2.05 GB
    - model:
        name: nitishpandey04/nanochat-d18-sft
        description: "nitishpandey04's d18 after SFT (step 478): upstream's 2026 model, 702M, chat"
        size: 2.05 GB
  decide:
    laya:
    - provider:
        name: well-known
    - model:
        name: laya/laya
        aliases:
        - laya-en
        description: "Laya, English: ModernBERT-large (28 layers, 1024 wide), 421M parameters, prompts of 512 tokens"
        size: 846.2 MB
        dtype: float16
    - model:
        name: laya/laya-multilingual
        description: "Laya, multilingual: mmBERT-base (22 layers, 768 wide, a 256k vocabulary), 322M parameters, prompts of 1024 tokens"
        size: 678.2 MB
        dtype: float16
    - model:
        name: laya/laya-typed-decisions
        description: "Laya, typed decisions: the English model trained on typed questions, prompts of 1024 tokens"
        size: 846.2 MB
        dtype: float16
```

<!-- /card -->

`bunsen-tool devices` lists the backends the build compiled in, each one's default device, and what the device does
with each float dtype, as burn reports it: `storage` (holds and converts it), `arithmetic` (computes in it) and
`accelerated` (runs it on tensor cores). On an RTX 3090, CUDA accelerates f32, f16 and bf16; wgpu accelerates none and
has no bf16; the CPU computes in each and accelerates none. `--format json` prints JSON.

## asr: speech to text

Name an **ASR preset**, a model chosen by what it is for, and `bunsen-tool asr` picks the family, the model and its
settings, for files (`transcribe`) or the microphone (`live`):

```sh
bunsen-tool asr transcribe [--preset en] [FLAGS] FILE...   # audio files
bunsen-tool asr live       [--preset en] [FLAGS]           # the microphone, as it speaks
bunsen-tool asr live       [--speech S] [--task T] [--quality Q] [FLAGS]   # or by what is needed
bunsen-tool asr presets   (list|show|fetch)                # what each ASR preset is for, and what it runs
```

```sh
cargo run --release -p bunsen-tool --features wgpu -- asr transcribe recording.wav
cargo run --release -p bunsen-tool --features wgpu -- asr transcribe --preset multi --json recording.wav
cargo run --release -p bunsen-tool --features wgpu -- asr live --preset translate-en
cargo run --release -p bunsen-tool --features wgpu -- asr transcribe --speech multi --quality fast recording.wav
```

| ASR preset | for | selects | `transcribe` runs | `live` runs |
|---|---|---|---|---|
| `en` (`default`) | English | `require: {language: en}` | `parakeet` `fermion/phonon-2` | `parakeet` `fermion/phonon-2`, `conservative` |
| `multi` | many languages, found from the speech | `require: {multilingual: true}` | `qwen3-asr` `qwen/qwen3-asr-1.7b` | `qwen3-asr` `qwen/qwen3-asr-0.6b`, `conservative` |
| `translate-en` | other speech, as English text | `require: {translate: true}` | `whisper` `openai/large-v3[translate]{beam_size: 5}` | `whisper` `openai/tiny[translate]{beam_size: 1}`, `conservative` |

An ASR preset's selection chooses its model among every speech family's checkpoints, by their tags ([Choosing a
model](#choosing-a-model)), over the table's own: the newest of a release group, then the lowest published English
word error rate (`scores: {wer.en}`), a model without one after those with. `multilingual` is a checkpoint whose
`language` lists more than one language, or `"*"`. `--quality fast` and `live` add `require: {parameters: {max: 1e9}}`,
a model small enough to keep up; `--quality good` takes the larger among equals. The ASR presets are this binary's, and
improve with it: when a better model arrives, its tags choose it, and the command lines that name the ASR preset do not
change. `presets show [NAME...]` prints each recipe as the `bunsen-tool asr family`
command it runs, `-v` logs it before a run, and `transcribe --json` records `preset`, `need` (the point served:
`speech`, `task`, `quality`; its `task` is the one the model ran) and `recipe` (family, model expression, emission, push size, VAD)
with each transcript. `presets fetch NAME...` brings every file an ASR preset runs into the cache, as `family <family> fetch` does
for a model.

Each ASR preset serves a need: `--speech` (`en`, `multi`) x `--task` (`transcribe`, `translate`) x `--quality`
(`fast`, `balanced`, `good`). `en` is `en`/`transcribe`/`balanced`, `multi` is `multi`/`transcribe`/`balanced`, and
`translate-en` is `multi`/`translate`/`balanced`.

The flags choose, in this order:

1. **`--language` implies `--speech`**, by the table's `language` entry: `en` means `en` speech, any other code
   `multi`. A `--speech` that does not hear it is an error (`--speech en --language de`); `multi` hears English.
2. **The ASR preset.** Without `--preset`, the flags name it: `translate-en` for `--task translate` (unless the
   speech is English), else `multi` for `multi` speech, else `en`; the axes not given are its need's, and `--json`
   records the need (`multi/transcribe/fast`) whenever one of them, or `--language`, is given. With `--preset`, the
   axes not given are the ASR preset's, and `--speech` (or the speech `--language` implies) must be one it hears.
3. **English speech translated to English is transcribed**, logged at `INFO`: Whisper's output for it is its
   transcript, so nothing translates English.
4. **What runs.** The ASR preset's selection, adjusted by the need: a translator for a need that translates
   (`--preset multi --task translate` runs Whisper), the size cap at `--quality fast` and `live`, and `--language`
   preferred; then its family's recipe for the mode (emission, push size, and Whisper's `beam_size`). A task other than
   the model's default is set on it (Whisper's `[translate]` preset in its model's box).

So the task is `transcribe` unless `translate-en` or `--task translate` asks to translate speech that is not English.

| command line | runs | task |
|---|---|---|
| (nothing) | `en`: Parakeet | transcribe |
| `--language de` | `multi/transcribe/balanced`: Qwen3-ASR | transcribe |
| `--task translate` | `multi/translate/balanced`: Whisper | translate |
| `--speech en --task translate` | `en/transcribe/balanced`: Parakeet | transcribe |
| `--preset translate-en --task transcribe` | `translate-en`: Whisper, without `[translate]` | transcribe |
| `--preset translate-en --speech en` | `translate-en`: Whisper, without `[translate]` | transcribe |
| `--preset en --task translate` | `en`: Parakeet | transcribe |
| `--preset en --language de` | an error: `en` hears only English | |
| `--preset multi --task translate` | `multi`, translating: Whisper | translate |

An ASR preset's run takes every flag below that is not a family's own (`--publish`, `--json`, `--rttm`, `--diarize`,
`--device`, the cache's and `live`'s capture flags), and:

- `--language CODE`: the spoken language. It implies `--speech` (above), and is given to a model that can be told it
  (Whisper); one that finds the language itself goes on doing so.
- `--model EXPR`: the model, as a lookup expression (`--model 'openai/small[translate]{beam_size: 1}'`), in place of
  the ASR preset's. The family that lists it runs, logged at `INFO` when it is not the ASR preset's
  (`asr transcribe --model openai/small` runs Whisper); a name no family lists (`hf:org/repo`) runs in the ASR preset's.
- `--emission` and `--chunk-ms`, in place of the ASR preset's.
- `--vad EXPR`: the VAD that finds the speech regions, in place of the table's (`silero/vad-16k[faster-whisper]`,
  which `asr presets show` prints). It is an error under `--emission offline`, which finds none.

A model's settings are its expression's, not flags: its presets in the box and any field in the patch
(`--model 'openai/small[translate]{beam_size: 1, timestamps: true}'`; the [table below](#family-settings) has each
family's). `--language` on Whisper joins the patch, and a task the ASR preset sets joins the box.

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
cargo run --release -p bunsen-tool --features wgpu -- asr family voxtral transcribe --model 'mistralai/voxtral-mini-4b-realtime-2602{delay_ms: 960}' --json recording.wav
cargo run --release -p bunsen-tool --features wgpu -- asr family whisper transcribe --emission conservative --model 'openai/base{language: en}' recording.wav
cargo run --release -p bunsen-tool -- asr family qwen3-asr list
```

Installed from crates.io, with the GPU backends you want (the CPU is always there):

```sh
cargo install bunsen-tool --features wgpu
bunsen-tool asr transcribe recording.wav
```

`list` shows the family's named models, with each one's licence, its presets (each axis and its members, described,
the default marked), its files, and whether each is cached; `diarize list` and `chat list` show theirs the same way. A name that
no table has, a model's, a model's preset or an ASR preset's, fails with the names nearest it (`did you mean
`openai/tiny`?`), by edit distance, case aside. Every family
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

`--emission` says when the driver decodes, for every family: `offline` (the default) decodes whole windows, one after
another (a family's `transcribe` default; an ASR preset names its own); `conservative` also finds speech regions with a VAD (`--vad EXPR`; Silero's
`silero/vad-16k[faster-whisper]`, through the same cache, by default, and an error under `offline`) and decodes each
as it ends, so silence is never decoded; `drafting` adds a draft of the speech so far every 600 ms of
it, which `live` shows. `--chunk-ms` (1000) is how much audio each push carries.

`live` transcribes the microphone as it speaks, through the same driver: the host's default input device, or
`--input-device` (a substring of an id or name from `--list-devices`), opened at the model's rate when it offers that
and resampled otherwise. Each decode's committed text is a line on stdout (a statement per speaker under `--diarize`),
its times, where a target shows them, the wall clock's; under `--emission drafting` the draft of the speech so far is
on stderr, redrawn in place on a terminal. A family's `--emission` defaults to `conservative` here, `--chunk-ms`
to 250; `--seconds` stops after that much audio, Ctrl-C otherwise, and either way the speech past the last commit is
decoded before the process exits. The default `live` feature builds it: it captures through `cpal`, which on Linux
links ALSA (`libasound2-dev`); `--no-default-features --features speech` builds `asr` without it. The default `speech`
feature builds `asr` itself; `--no-default-features` alone builds a tool with only `chat`, `decide`, `cache`, `card`
and `devices`.

```sh
cargo run --release -p bunsen-tool --features wgpu -- asr family parakeet live --emission drafting
cargo run --release -p bunsen-tool -- asr live --list-devices
```

`diarize` says who spoke when, through NVIDIA's Sortformer diarizers: `nvidia/nemotron-3-diarization` (up to 8 speakers,
10 ms frames, the default), or the streaming Sortformer v2 and v2.1 (`--model sortformer-v2`, `--model sortformer-v2.1`;
up to 4 speakers, 80 ms frames). `diarize run FILE...` prints each file's speaker turns as lines of `{start} {end}
speaker_{k}` in seconds, speakers numbered in the order they first speak; `--rttm` prints NIST RTTM instead (the file's
stem as its recording id), and `--json` one object per file, its turns and each speaker's total. `diarize live` prints
each turn as it closes, from the microphone, opened as the families' `live` opens it. The model's presets
(`--preset`, a comma list) choose its `latency`: `low` (1.04 s, the default), `ultra-low` (0.32 s), `offline` (30.4 s
chunks, the most accurate on files), and the presets only one family has, `very-low` (0.64 s, Nemotron-3-Diarization)
and `high` (10 s, Sortformer v2); and its `postprocess`, how activity becomes turns: `none` (thresholds of 0.5, the
default), or NeMo's tuned `dihard3` or `callhome` parameters. `list` and `fetch` work as for the families.

```sh
cargo run --release -p bunsen-tool --features wgpu -- asr diarize run --preset offline,dihard3 meeting.wav
cargo run --release -p bunsen-tool --features wgpu -- asr diarize run --rttm meeting.wav > meeting.rttm
```

`--diarize` on `transcribe` or `live`, an ASR preset's or any family's, says who said what: the diarizer (`--diarize` alone for
`nvidia/nemotron-3-diarization`, `--diarize=EXPR` for another, or for its presets) runs beside the recognizer, through the stream
driver's speaker seam, and labels each word (Parakeet) or segment with its speaker. Qwen3-ASR and Whisper also cut
their windows at speaker turns, so a segment is one speaker's. The text is printed a line per statement, a run of one
speaker's words, `[speaker_k] text` (`[?]` where the diarizer heard nobody). Under `--json` each statement is wrapped
under its speaker, and the `transcript` adds each word's or segment's `speaker`, and the `turns`: each speaker's runs
with their start, end and text. `--rttm` prints the statements as NIST RTTM. The diarizer's latency is its expression's preset, as for `diarize`
(`--diarize='nvidia/nemotron-3-diarization[very-low]'`); a decode waits until the diarizer has caught up with it.

```sh
cargo run --release -p bunsen-tool --features wgpu -- asr live --diarize
cargo run --release -p bunsen-tool --features wgpu -- asr transcribe --preset multi --diarize --json meeting.wav
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
`live`, a file's on `transcribe`); a `draft`, `live`'s latest draft under `--emission drafting`; and a `transcript`,
a file's whole transcript after its statements, with the ASR preset and recipe that chose its model. A statement is flat;
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

<a id="family-settings"></a>A family's settings are its model's policy, set in the model's patch (`--model
'NAME{key: value}'`) or by its presets; the flags left are paths:

| Family | Settings (patch keys) | Flags of its own |
|---|---|---|
| `parakeet` | `window_seconds` (60): a full window is committed up to a pause between words, and the rest decoded again with the audio after it | |
| `qwen3-asr` | `window_seconds` (30, the most Qwen3-ASR was trained on): a full window is cut at the quietest point of its last third | |
| `voxtral` | `window_seconds` (30), as Parakeet's; `delay_ms` (the model's, 480): how far the transcript runs behind the audio, a multiple of 80 up to 1200, or 2400 | |
| `whisper` | `language` (detected when unset), `task` (`Transcribe`, `Translate`; or the `[translate]` preset), `beam_size` (5), `timestamps` (split segments on timestamp tokens), `max_tokens` (224 a window), `condition_on_previous_text` (`true`: prompt each window with the transcript so far), `fallback` (`true`: upstream's temperature ladder) | `--vocab` (a vocabulary file in place of the layout's), `--upstream-cache-dir` (`openai-whisper`'s download cache, used in place; `~/.cache/whisper`); `--model` also takes a checkpoint's path |

`bunsen-tool asr family whisper inspect NAME|PATH` scans a Whisper checkpoint and checks it against its prefab (its resources and
where each stands, the geometry it promised, and the one its shapes say); `bunsen-tool asr family whisper prefabs` lists
the geometries. Neither needs a backend.

`bunsen-tool cache` manages the weights cache every family loads through (`bunsen_app::cache`):

```sh
bunsen-tool cache path                                 # the cache directory
bunsen-tool cache list [NAME]                          # each file's namespace, size, fetched and last used (UTC), pin
bunsen-tool cache clear [NAME] [--partial] [--age AGE] [--yes]
```

A file is cached under its namespace: its resource's full name in the built-in catalog
(`openai/whisper/base/checkpoint`), or `hf` and `given` for the files no catalog names. `NAME` selects the namespaces
under it: `qwen`, `openai/whisper`, `openai/whisper/base`, `hf`.
`clear` without `--yes` only lists what it would remove; `--partial` selects unfinished downloads, and `--age 30d` the
files not loaded for 30 days (`s`, `m`, `h`, `d`, `w`). A load that finds a file in the cache marks it used; listing,
and checking what is cached, do not.

`transcribe` and `live` take `--emission` and `--chunk-ms` (`diarize live` takes `--chunk-ms`); every command takes `--cache-dir`, `--offline`, `--device` (from the backends compiled in: `--features wgpu`, `vulkan`,
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

Fetched files live at `<cache>/pretrained/<resource>/<sha256>/<file>`, `<resource>` the catalog resource's full name
(`openai/whisper/base/checkpoint`): the digest in the path is the pin, so
a file there is trusted without re-hashing, and a directory laid out this way is the cache (`fetch` fills one ahead
of time for a deployment that runs `--offline`). `hf:openai/whisper-large-v3` is a Hugging Face repo's safetensors
checkpoint at `main`, sharded or not, each file pinned by the digest the hub's listing carries; its vocabulary is the
one its layout selects, as for a checkpoint's path.

`bunsen-tool asr family whisper inspect NAME|PATH` prints a model's resources and where each stands, the prefab it promised, and
the geometry its checkpoint's shapes say; on a path, which prefab, if any, that geometry is, and which vocabulary its
layout selects; on a name whose checkpoint does not scan as its prefab, the mismatch. `transcribe` makes the same
check before it reads any tensors. The index is bunsen's (`kits::speech::asr::whisper::pretrained`), and this binary keeps
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

## chat: Qwen3, Gemma and nanochat

`bunsen-tool chat` answers one prompt (`run`) or holds a conversation on stdin (`repl`: `/reset` starts over, `/quit`
or the end of input leaves), on the KV cache: each turn prefills only what the cache does not already hold.

```sh
bunsen-tool chat list [--urls]                       # every family's checkpoints, and what of each is cached
bunsen-tool chat fetch NAME...                       # bring checkpoints into the cache
bunsen-tool chat run  [--select NAME|{..}]... [FLAGS] [PROMPT]   # one reply; the prompt from stdin when omitted
bunsen-tool chat repl [--select NAME|{..}]... [FLAGS]            # a conversation
bunsen-tool chat family (qwen3|gemma|nanochat) (list|fetch|run|repl) [FLAGS]   # one family's, its own default by default
```

```sh
cargo run --release -p bunsen-tool --features wgpu -- chat run --no-think "What is the capital of France?"
cargo run --release -p bunsen-tool --features wgpu -- chat repl --select qwen3-4b-instruct-2507 --stats
cargo run --release -p bunsen-tool --features wgpu -- chat run --select gemma-3-270m-it "What is the capital of France?"
cargo run --release -p bunsen-tool --features wgpu -- chat run --select '{require: {thinking: always}}' "Why is the sky blue?"
cargo run --release -p bunsen-tool --features wgpu -- chat family nanochat run "What is the capital of France?"
cargo run --release -p bunsen-tool --features cuda -- chat family nanochat repl --device cuda
```

`--select` (`--model` is an alias) chooses the checkpoint, among every family's (one family's under `family`), by
the rows' tags ([Choosing a model](#choosing-a-model)); outside `family`, the chosen checkpoint's family runs it, and
Qwen3's default (`qwen/qwen3-0.6b`) is the default. An `hf:org/repo` is a Qwen3, or under `family` the family's.
The families:

| family | checkpoints | default |
|---|---|---|
| `qwen3` | Qwen's Qwen3 by name, or `hf:org/repo`, any Hugging Face repo in `transformers`' `Qwen3ForCausalLM` layout | `qwen/qwen3-0.6b` |
| `gemma` | Google's Gemma 1, 2 and 3 text models by name (`gemma-3-{270m,1b,4b}`, `gemma-2-{2b,9b}`, `gemma-{2b,7b}`, each with its `-it` build), fetched from their `unsloth/` mirrors; under `family gemma`, `hf:org/repo` in a `transformers` Gemma's layout | `gemma/gemma-3-1b-it` |
| `nanochat` | community checkpoints of upstream nanochat's 2025 and 2026 models (`nanochat-students/{base,mid,rl}-d20`, `nitishpandey04/nanochat-d18-{base,sft}`) | `nanochat-students/rl-d20` |

Each family takes only the flags that apply to it: thinking (`--think`, `--no-think`, `--show-thinking`) is Qwen3's.
Sampling follows the checkpoint's generation defaults; `--greedy`, `--temperature`, `--top-k`, `--top-p` and `--seed`
override them. A checkpoint loads at the precision it ships in where the device accelerates it, else in f32: Qwen3's
bf16 on cuda and metal, f32 on vulkan, wgpu and the CPU; `bunsen-tool devices` shows what a device accelerates.
`--context` sizes the KV cache (the checkpoint's trained length by default),
`--kv-storage` picks how it stores its history, and `--stats` prints, after each reply, the model's lookup expression (its name with the presets the flags add), then the reply's token counts and speeds.

## decide: typed questions answered by Laya

`bunsen-tool decide` answers questions about a context with calibrated probabilities over each question's options,
in one forward pass of a Laya decision model (`convaiinnovations/laya`, Apache-2.0: a `ModernBERT` encoder and a
decision head). It generates no text.

```sh
bunsen-tool decide list [--urls]                         # the Laya checkpoints, and what of each is cached
bunsen-tool decide fetch NAME...                         # bring checkpoints into the cache
bunsen-tool decide run [--select NAME|{..}]... [--format yml|json] [REQUEST]   # the request from stdin when omitted
```

`--select` (`--model` is an alias) chooses the checkpoint among Laya's, as chat's does ([Choosing a
model](#choosing-a-model)); `laya/laya` is the default.

A request is a YAML (or JSON) document: a `context`, and its `choices`, each a `type`, a `question` and its
`options`:

- `choice`: a pick among named options, each a name or `{name: what it means}`;
- `score`: a level on a scale, the options its levels in order; the answer also gives the expected level (`score`);
- `noul`: a yes/no probability; its options, when given, say what `{false: ...}` and `{true: ...}` mean.

```yaml
context: "Customer: my kettle arrived with a cracked lid and I asked for a refund twice with no answer."
choices:
  - type: choice
    question: "What should the agent do next?"
    options: [refund, replace, {escalate: "hand the case to a supervisor"}, close]
  - type: noul
    question: "Is the customer satisfied?"
```

Each result has the question, its type, each option's probability (a score's by level, a yes/no question's as
`false` and `true`), the most probable option (`answer`), the `confidence` (one less the distribution's normalized
entropy), and the act head's probability (`act_probability`, which upstream's model card says carries no usable
signal yet):

```yaml
- question: What should the agent do next?
  type: choice
  probabilities:
    refund: 0.606
    replace: 0.343
    escalate: 0.024
    close: 0.027
  answer: refund
  confidence: 0.383
  act_probability: 1.0
```

```sh
cargo run --release -p bunsen-tool --features wgpu -- decide run request.yml
cargo run --release -p bunsen-tool --features wgpu -- decide run --select laya-multilingual --format json < request.yml
```

## Choosing a model

`chat` and `decide` choose their checkpoint with `--select`, repeatable: at most one name, and any number of
selection objects, YAML. A checkpoint's facts are its tags, which its build in the built-in catalog carries:
`instruct`, `thinking` (`always`, `switch` or `never`), `reasoning`, `language` (BCP 47 tags; `"*"` for unspecified
others), `parameters`, `release` (its release group and date), `kit`, `contract`, `family`, `publisher`.

- A name that is a checkpoint's (its full id, its bare name, or an alias) is that checkpoint, as named.
- Any other name is a set of checkpoints: those whose names it begins (`--select qwen3`), else those of the group it
  names, else the kit's (`--select nanochat`). The command's defaults choose within it: a tuned model
  (`instruct: true`), each release group's newest, the heaviest first.
- An object narrows and orders, merged over the defaults: `require` (a tag's value, one of a list, `{min, max}`, or a
  language), `prefer` (each a tag with the score it keeps when it is not met: `{language: de, otherwise: 0.5}`),
  `latest`, `absent` (`drop`, or `{default: value}`, for a checkpoint without the tag), `order` (`[-parameters]`),
  `take` (the least weight taken). `null` removes a default: `{require: {instruct: null}}`.
- `--preset` keeps only the checkpoints that have the preset; `--patch {..}` merges over the chosen one's config.

```sh
bunsen-tool chat run --select qwen3 --select '{require: {instruct: null, thinking: never}, order: [parameters]}' "Hi"
bunsen-tool chat run --select '{require: {parameters: {max: 2e9}}, prefer: [{language: zh, otherwise: 0}]}' "你好"
```

The choice, and why, is logged at `INFO` (`RUST_LOG=bunsen_tool=info`).

The checkpoints: `laya/laya/laya` (English, on `ModernBERT-large`; the default), `laya/laya/laya-multilingual` (on
`mmBERT-base`) and `laya/laya/laya-typed-decisions`. A checkpoint loads at the precision it ships in (f16) where the
device accelerates it, else in f32.

## catalog: build catalogs

`bunsen-tool catalog` queries bunsen's built-in model catalog, the one the kits load from
(`bunsen::data::pretrained::build_catalog`; its `catalog_format` page describes the document,
and its `usage` page the select language). `--catalog` imports an extension over it, YAML or
JSON, repeatable, in order: an extension may add builds, template classes and resources, but
not contracts.

```sh
bunsen-tool catalog tree
bunsen-tool catalog --catalog local.yml select qwen3 --off thinking --limit 1
```

| command | does |
|---|---|
| `layers` | the imported catalogs, in order |
| `show NAME` | an entry's kind, history (which catalog declared, updated or replaced it) and merged fields |
| `resolve BUILD [--preset P] [--on AXIS] [--off AXIS] [--patch {..}]` | the resolved build, and the document its presets make |
| `select [--kind K] [NAME]... [--keep P] [--drop P] [--le KEY V]... [--weigh E] [--cutoff W] [--asc K] [--desc K] [--limit N] [--use NAME] [--unset L] [--per G .. --end] [--pipeline ..] [--floor F] [--preset P] [--on/--off AXIS] [--defaults ..] [--rank]` | a YAML report: the composed query, why (each stage, where it came from, and what it removed), the presets applied, and the result in order, each build resolved with its `weight`; each flag is one stage, in command-line order, but the preset flags, which are the prefix; `--rank` adds every removed row with why |
| `tree [NAME] [--kind K]... [--all] [--aliases inline\|entries\|both\|none] [--depth N]` | the namespace as a directory view, from the root's entries or one entry: `name: description` for a namespace, `name: aliases {axis: default\|other} description` for a build, `name -> target` for an alias. It shows namespaces, builds and the aliases that name them; `--kind` keeps only those kinds (and the containers on the way to them), `--all` shows every kind, as `name (kind): ..`, and the base catalog's `$contracts`; `--aliases` says where aliases show (both, by default) |
| `dump [--format yml\|json]` | the imported catalogs, in order, as one sequence: each item's `layer`, `path` and `catalog`, the document in its keyed form |
| `releases GROUP` | a release group's members, oldest first |
