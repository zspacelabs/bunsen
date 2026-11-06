# bunsen - tackling tomorrow's burn problems today!

*by [ZSpaceLabs](https://zspacelabs.ai)*

[![Crates.io Version](https://img.shields.io/crates/v/bunsen)](https://crates.io/crates/bunsen)
[![Documentation](https://img.shields.io/docsrs/bunsen)](https://docs.rs/bunsen/latest/bunsen/)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](#license)
[![Discord](https://img.shields.io/discord/1475229838754316502?label=discord)](https://discord.gg/vBgXHWCeah)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/zspacelabs/bunsen)

The [burn](https://burn.dev) team is working hard to give us a stable long term foundation
for developing high-performance and maintainable tensor programming in
Rust; The "in Rust" aspect requires a degree of type and runtime R&D research
and that requires a conservative posture to api extension.

`bunsen` provides a "batteries included" complementary community standard library
for research extensions to `burn`; ideas which are useful enough
to collect into one place, document, and build upon; but front-run
the `burn` interface frontier.
Every idea in `bunsen` seeks to either be refined and ported into `burn`,
or supplanted by the eventual `burn` solution to that problem.

- [The bunsen book](https://zspacelabs.ai/bunsen/book/) explains how bunsen is organized, the systems that cut across
  it, and how it is developed. Start with
  [Installing and first use](https://zspacelabs.ai/bunsen/book/organization/install.html).
- [The API docs](https://docs.rs/bunsen/latest/bunsen/) document every type and lifecycle, with worked examples, and
  list the crate's features.

## Kits

A kit is a whole model, with its configs and the weights it loads by name
([Kits](https://zspacelabs.ai/bunsen/book/organization/kits.html)):

- [`decision::laya`](https://docs.rs/bunsen/latest/bunsen/kits/decision/laya/index.html): Laya, an open decision
  model: typed questions about a state answered with calibrated probabilities, with its checkpoints by name.
- [`images::resnet`](https://docs.rs/bunsen/latest/bunsen/kits/images/resnet/index.html): ResNet image classifiers,
  with torchvision's and `timm`'s checkpoints by name.
- [`images::swin::v2`](https://docs.rs/bunsen/latest/bunsen/kits/images/swin/v2/index.html): Swin Transformer V2
  image classifiers, without pretrained weights.
- [`language::nanochat`](https://docs.rs/bunsen/latest/bunsen/kits/language/nanochat/index.html): a port of karpathy's
  nanochat GPT, and the corpus it trains on. Work in progress.
- [`speech::asr::whisper`](https://docs.rs/bunsen/latest/bunsen/kits/speech/asr/whisper/index.html): OpenAI's Whisper, with
  a pretrained index, decoders, and a stream driver.
- [`speech::vad::silero_vad`](https://docs.rs/bunsen/latest/bunsen/kits/speech/vad/silero_vad/index.html): the Silero
  voice-activity detector, with weights a build can bundle.
- [`text::encoders::modernbert`](https://docs.rs/bunsen/latest/bunsen/kits/text/encoders/modernbert/index.html):
  Answer.AI's ModernBERT text encoder, which Laya is built on.
- [`support`](https://docs.rs/bunsen/latest/bunsen/kits/support/index.html): not a model; what kits of more than one
  family share: the ids-to-text seam, and the decoding search machinery.

The models the kits know by name, by kind and family, as `bunsen-tool card --section kits` prints them
(`cargo make cards` rebuilds this card):

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


## Simulations

Simulations written as tensor programs, built the way the kits are, a config and a state you step
([`bunsen::sims`](https://docs.rs/bunsen/latest/bunsen/sims/index.html)):

- [`sims::conway`](https://docs.rs/bunsen/latest/bunsen/sims/conway/index.html): Conway's Game of Life on
  toroidal 2D and 3D boards.
- [`sims::lbm`](https://docs.rs/bunsen/latest/bunsen/sims/lbm/index.html): a D2Q9 lattice-Boltzmann fluid.

## The workspace

Most users depend on `bunsen` alone. The [bunsen repository](https://github.com/zspacelabs/bunsen) also publishes its
companion crates (the firehose data pipeline, a Parquet-to-tokens data loader, the shape-contract macro, and bundled
model assets), and holds the development, validation, and example crates.
[The workspace](https://zspacelabs.ai/bunsen/book/organization/workspace.html) describes each one.

## Examples

The repository's [`examples/`](https://github.com/zspacelabs/bunsen/tree/main/examples) directory holds runnable
programs, one or more per kit or system. The book's
[Examples](https://zspacelabs.ai/bunsen/book/organization/examples.html) page says what each one shows.

## License

`bunsen` is distributed under the terms of both the MIT license and the Apache License (Version 2.0).
