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

A kit is a whole model or simulation, with its configs and, for a model, the weights it loads by name
([Kits](https://zspacelabs.ai/bunsen/book/organization/kits.html)):

- [`images::resnet`](https://docs.rs/bunsen/latest/bunsen/kits/images/resnet/index.html): ResNet image classifiers,
  with torchvision's and `timm`'s checkpoints by name.
- [`images::swin::v2`](https://docs.rs/bunsen/latest/bunsen/kits/images/swin/v2/index.html): Swin Transformer V2
  image classifiers, without pretrained weights.
- [`gpts::nanochat`](https://docs.rs/bunsen/latest/bunsen/kits/gpts/nanochat/index.html): a port of karpathy's
  nanochat GPT, and the corpus it trains on. Work in progress.
- [`sims::conway`](https://docs.rs/bunsen/latest/bunsen/kits/sims/conway/index.html): Conway's Game of Life on
  toroidal 2D and 3D boards.
- [`sims::lbm`](https://docs.rs/bunsen/latest/bunsen/kits/sims/lbm/index.html): a D2Q9 lattice-Boltzmann fluid.
- [`speech::whisper`](https://docs.rs/bunsen/latest/bunsen/kits/speech/whisper/index.html): OpenAI's Whisper, with
  a pretrained index, decoders, and a stream driver.
- [`speech::silero_vad`](https://docs.rs/bunsen/latest/bunsen/kits/speech/silero_vad/index.html): the Silero
  voice-activity detector, with weights a build can bundle.
- [`tokens`](https://docs.rs/bunsen/latest/bunsen/kits/tokens/index.html): not a model; the ids-to-text seam the
  model kits share.

The models the kits know by name, by kind and family, as `bunsen-tool card --section kits` prints them
(`cargo make cards` rebuilds this card):

<!-- card: bunsen-tool card --section kits -->

```yaml
kits:
  asr:
    parakeet:
      provider:
      - name: well-known
      - name: hf
      model:
      - name: fermion/phonon-2
        aliases:
        - phonon
        description: "Phonon-2: parakeet-tdt-0.6b-v3 with a five-value encoder, 164 MB, English"
      - name: nvidia/parakeet-tdt-0.6b-v3
        description: "parakeet-tdt-0.6b-v3: 600 M parameters, f32, 25 European languages"
    qwen3-asr:
      provider:
      - name: well-known
      - name: hf
      model:
      - name: fermion/phonon-1
        description: "Phonon-1: Qwen3-ASR-0.6B fine-tuned for English, re-packed, 415 MB"
      - name: fermion/phonon-1-micro
        description: "Phonon-1-Micro: Qwen3-ASR-0.6B fine-tuned for English, re-packed, 285 MB"
      - name: fermion/phonon-1-big
        description: "Phonon-1-Big: Qwen3-ASR-0.6B fine-tuned for English, re-packed, 581 MB"
      - name: qwen/qwen3-asr-0.6b
        description: "Qwen3-ASR-0.6B: 0.8 B parameters, bf16, 30 languages"
      - name: qwen/qwen3-asr-1.7b
        description: "Qwen3-ASR-1.7B: 2.0 B parameters, bf16, 30 languages"
    voxtral:
      provider:
      - name: well-known
      - name: hf
      model:
      - name: mistralai/voxtral-mini-4b-realtime-2602
        aliases:
        - voxtral-mini-4b-realtime
        description: "Voxtral-Mini-4B-Realtime-2602: 4.4 B parameters, bf16, 13 languages, streaming"
    whisper:
      provider:
      - name: well-known
      - name: hf
      model:
      - name: openai/tiny.en
        description: 39 M parameters, English-only
      - name: openai/tiny
        description: 39 M parameters, multilingual
      - name: openai/base.en
        description: 74 M parameters, English-only
      - name: openai/base
        description: 74 M parameters, multilingual
      - name: openai/small.en
        description: 244 M parameters, English-only
      - name: openai/small
        description: 244 M parameters, multilingual
      - name: openai/medium.en
        description: 769 M parameters, English-only
      - name: openai/medium
        description: 769 M parameters, multilingual
      - name: openai/large-v1
        description: 1550 M parameters, multilingual
      - name: openai/large-v2
        description: 1550 M parameters, multilingual
      - name: openai/large-v3
        aliases:
        - large
        description: 1550 M parameters, multilingual, 128 mels
      - name: openai/large-v3-turbo
        aliases:
        - turbo
        description: 809 M parameters, multilingual, 128 mels, four-layer decoder
    sortformer:
      provider:
      - name: well-known
      - name: hf
      model:
      - name: nvidia/nemotron-3-diarization
        description: "Nemotron-3-Diarization: 100 M parameters, up to 8 speakers, 10 ms frames, streaming"
      - name: nvidia/diar_streaming_sortformer_4spk-v2
        aliases:
        - sortformer-v2
        description: "Streaming Sortformer v2: 117 M parameters, up to 4 speakers, 80 ms frames"
      - name: nvidia/diar_streaming_sortformer_4spk-v2.1
        aliases:
        - sortformer-v2.1
        description: "Streaming Sortformer v2.1: 117 M parameters, up to 4 speakers, 80 ms frames"
    silero-vad:
      provider:
      - name: well-known
      model:
      - name: silero/vad-16k
        description: Silero VAD, the 16 kHz model, from upstream's safetensors (1.2 MB)
  chat:
    qwen3:
      provider:
      - name: well-known
      - name: hf
      model:
      - name: qwen/qwen3-0.6b
        description: "Qwen3-0.6B: thinking and non-thinking chat, bf16, 1.50 GB"
      - name: qwen/qwen3-0.6b-base
        description: "Qwen3-0.6B-Base: the pretrained model, no chat tuning, bf16, 1.19 GB"
      - name: qwen/qwen3-1.7b
        description: "Qwen3-1.7B: thinking and non-thinking chat, bf16, 4.06 GB"
      - name: qwen/qwen3-1.7b-base
        description: "Qwen3-1.7B-Base: the pretrained model, no chat tuning, bf16, 3.44 GB"
      - name: qwen/qwen3-4b
        description: "Qwen3-4B: thinking and non-thinking chat, bf16, 8.04 GB"
      - name: qwen/qwen3-4b-base
        description: "Qwen3-4B-Base: the pretrained model, no chat tuning, bf16, 8.04 GB"
      - name: qwen/qwen3-4b-instruct-2507
        description: "Qwen3-4B-Instruct-2507: non-thinking chat, 256K context, bf16, 8.04 GB"
      - name: qwen/qwen3-4b-thinking-2507
        description: "Qwen3-4B-Thinking-2507: thinking chat, 256K context, bf16, 8.04 GB"
    nanochat:
      provider:
      - name: well-known
      model:
      - name: nanochat-students/base-d20
        aliases:
        - nanochat-d20-base
        description: "nanochat-students' d20 after pretraining (step 21400): upstream's 2025 model, 561M, base"
      - name: nanochat-students/mid-d20
        aliases:
        - nanochat-d20-mid
        description: "nanochat-students' d20 after midtraining (step 765): upstream's 2025 model, 561M, chat format"
      - name: nanochat-students/rl-d20
        aliases:
        - nanochat-d20-rl
        description: "nanochat-students' d20 after its RL stage (step 650): upstream's 2025 model, 561M, chat"
      - name: nitishpandey04/nanochat-d18-base
        description: "nitishpandey04's d18 after pretraining (step 3712): upstream's 2026 model, 702M, base"
      - name: nitishpandey04/nanochat-d18-sft
        description: "nitishpandey04's d18 after SFT (step 478): upstream's 2026 model, 702M, chat"
```

<!-- /card -->

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
