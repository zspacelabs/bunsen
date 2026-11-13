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

The builds the kits run, by command, contract and publisher, as `bunsen-tool summary` prints them
(`cargo make cards` rebuilds this card):

<!-- card: bunsen-tool summary -->

```text
# bunsen-tool asr ...
# Automated Speech Recognition

# contract: parakeet-asr
# A Parakeet TDT speech recognizer: a Phonon-2 archive, or transformers'
# safetensors and its tokenizer.
# publisher: fermion - FermionResearch, Phonon's authors.

* fermion/phonon-2: [163.5 MB; aka phonon]
  Phonon-2: parakeet-tdt-0.6b-v3 with a five-value encoder, English.

# publisher: nvidia - NVIDIA.

* nvidia/parakeet-tdt-0.6b-v3: [2.51 GB f32]
  parakeet-tdt-0.6b-v3: 600 M parameters, 25 European languages.

# contract: qwen3-asr
# A Qwen3-ASR speech recognizer: a Phonon-1 archive or transformers'
# safetensors, and its tokenizer.
# publisher: fermion - FermionResearch, Phonon's authors.

* fermion/phonon-1: [415.1 MB]
  Phonon-1: Qwen3-ASR-0.6B fine-tuned for English, re-packed.

* fermion/phonon-1-micro: [285.1 MB]
  Phonon-1-Micro: Qwen3-ASR-0.6B fine-tuned for English, re-packed.

* fermion/phonon-1-big: [580.9 MB]
  Phonon-1-Big: Qwen3-ASR-0.6B fine-tuned for English, re-packed.

# publisher: qwen - Qwen, Alibaba Cloud's model team.

* qwen/qwen3-asr-0.6b: [1.58 GB bf16]
  Qwen3-ASR-0.6B: 0.8 B parameters, 30 languages.

* qwen/qwen3-asr-1.7b: [4.09 GB bf16]
  Qwen3-ASR-1.7B: 2.0 B parameters, 30 languages.

# contract: voxtral-asr
# A Voxtral Realtime speech recognizer: its safetensors and its tekken
# vocabulary.
# publisher: mistralai - Mistral AI.

* mistralai/voxtral/voxtral-mini-4b-realtime-2602: [8.87 GB bf16; aka voxtral-mini-4b-realtime]
  Voxtral-Mini-4B-Realtime-2602: 4.4 B parameters, 13 languages, streaming.

# contract: whisper-asr
# An OpenAI Whisper checkpoint, as openai-whisper saves it, and its
# tiktoken vocabulary.
# publisher: openai - OpenAI.

* openai/whisper/large/large-v1: [3.09 GB f16; task: *transcribe|translate]
  1550 M parameters, multilingual.

* openai/whisper/large/large-v2: [3.09 GB f16; task: *transcribe|translate]
  1550 M parameters, multilingual.

* openai/whisper/large/large-v3: [3.09 GB f16; task: *transcribe|translate; aka large]
  1550 M parameters, multilingual, 128 mels.

* openai/whisper/large/large-v3-turbo: [1.62 GB f16; aka turbo]
  809 M parameters, multilingual, 128 mels, four-layer decoder.

* openai/whisper/tiny.en: [76.4 MB f16]
  39 M parameters, English-only.

* openai/whisper/tiny: [76.4 MB f16; task: *transcribe|translate]
  39 M parameters, multilingual.

* openai/whisper/base.en: [146.1 MB f16]
  74 M parameters, English-only.

* openai/whisper/base: [146.1 MB f16; task: *transcribe|translate]
  74 M parameters, multilingual.

* openai/whisper/small.en: [484.5 MB f16]
  244 M parameters, English-only.

* openai/whisper/small: [484.4 MB f16; task: *transcribe|translate]
  244 M parameters, multilingual.

* openai/whisper/medium.en: [1.53 GB f16]
  769 M parameters, English-only.

* openai/whisper/medium: [1.53 GB f16; task: *transcribe|translate]
  769 M parameters, multilingual.

# contract: sortformer-diarizer
# A Sortformer speaker diarizer: a `.nemo` archive, which carries its
# config, or transformers' safetensors.
# publisher: nvidia - NVIDIA.

* nvidia/sortformer/nemotron-3-diarization: [397.0 MB f32; latency: ultra-low|very-low|*low|offline; postprocess: *none|dihard3|callhome]
  Nemotron-3-Diarization: 100 M parameters, up to 8 speakers, 10 ms frames,
  streaming.

* nvidia/sortformer/diar_streaming_sortformer_4spk-v2: [471.4 MB f32; latency: ultra-low|*low|high|offline; postprocess: *none|dihard3|callhome; aka sortformer-v2]
  Streaming Sortformer v2: 117 M parameters, up to 4 speakers, 80 ms
  frames.

* nvidia/sortformer/diar_streaming_sortformer_4spk-v2.1: [471.4 MB f32; latency: ultra-low|*low|high|offline; postprocess: *none|dihard3|callhome; aka sortformer-v2.1]
  Streaming Sortformer v2.1: 117 M parameters, up to 4 speakers, 80 ms
  frames.

# contract: silero-vad
# The Silero voice-activity detector at 16 kHz: upstream's state dict.
# publisher: silero - Silero, snakers4/silero-vad's authors.

* silero/silero_vad/vad-16k: [1.2 MB f32; vad-policy: *silero|faster-whisper]
  Silero VAD, the 16 kHz model, from upstream's safetensors.

# bunsen-tool chat ...
# Chat and Text Generation

# contract: qwen3-lm
# A dense Qwen3 causal LM, as transformers lays it out: its geometry from
# `config.json`.
# publisher: qwen - Qwen, Alibaba Cloud's model team.

* qwen/qwen3/qwen3-0.6b: [1.51 GB bf16; thinking: *thinking|no-thinking]
  Qwen3-0.6B: thinking and non-thinking chat.

* qwen/qwen3/qwen3-0.6b-base: [1.20 GB bf16; thinking: no-thinking]
  Qwen3-0.6B-Base: the pretrained model, no chat tuning.

* qwen/qwen3/qwen3-1.7b: [4.07 GB bf16; thinking: *thinking|no-thinking]
  Qwen3-1.7B: thinking and non-thinking chat.

* qwen/qwen3/qwen3-1.7b-base: [3.45 GB bf16; thinking: no-thinking]
  Qwen3-1.7B-Base: the pretrained model, no chat tuning.

* qwen/qwen3/qwen3-4b: [8.06 GB bf16; thinking: *thinking|no-thinking]
  Qwen3-4B: thinking and non-thinking chat.

* qwen/qwen3/qwen3-4b-base: [8.05 GB bf16; thinking: no-thinking]
  Qwen3-4B-Base: the pretrained model, no chat tuning.

* qwen/qwen3/qwen3-4b-instruct-2507: [8.06 GB bf16; thinking: no-thinking]
  Qwen3-4B-Instruct-2507: non-thinking chat, 256K context.

* qwen/qwen3/qwen3-4b-thinking-2507: [8.06 GB bf16; thinking: thinking]
  Qwen3-4B-Thinking-2507: thinking chat, 256K context.

* qwen/qwen3/qwen3-8b: [16.39 GB bf16; thinking: *thinking|no-thinking]
  Qwen3-8B: thinking and non-thinking chat.

* qwen/qwen3/qwen3-8b-base: [16.39 GB bf16; thinking: no-thinking]
  Qwen3-8B-Base: the pretrained model, no chat tuning.

* qwen/qwen3/qwen3-4b-saferl: [8.06 GB bf16; thinking: *thinking|no-thinking]
  Qwen3-4B-SafeRL: Qwen3-4B tuned for safety by reinforcement learning,
  thinking and non-thinking chat.

* qwen/qwen3/qwen3guard-gen-0.6b: [1.51 GB bf16; thinking: no-thinking]
  Qwen3Guard-Gen-0.6B: safety moderation, rating a prompt or a reply Safe,
  Unsafe or Controversial, with its categories.

* qwen/qwen3/qwen3guard-gen-4b: [8.83 GB bf16; thinking: no-thinking]
  Qwen3Guard-Gen-4B: safety moderation, rating a prompt or a reply Safe,
  Unsafe or Controversial, with its categories.

* qwen/qwen3/qwen3guard-gen-8b: [16.39 GB bf16; thinking: no-thinking]
  Qwen3Guard-Gen-8B: safety moderation, rating a prompt or a reply Safe,
  Unsafe or Controversial, with its categories.

# contract: qwen3.5-lm
# A dense Qwen3.5 checkpoint's text model, as transformers lays it out: its
# geometry from `config.json`.
# publisher: qwen - Qwen, Alibaba Cloud's model team.

* qwen/qwen3.5/qwen3.5-0.8b: [1.76 GB bf16; thinking: thinking|*no-thinking]
  Qwen3.5-0.8B: thinking and non-thinking chat (its text model).

* qwen/qwen3.5/qwen3.5-0.8b-base: [1.76 GB bf16; thinking: no-thinking]
  Qwen3.5-0.8B-Base: the pretrained model, no chat tuning (its text model).

* qwen/qwen3.5/qwen3.5-2b: [4.56 GB bf16; thinking: thinking|*no-thinking]
  Qwen3.5-2B: thinking and non-thinking chat (its text model).

* qwen/qwen3.5/qwen3.5-2b-base: [4.56 GB bf16; thinking: no-thinking]
  Qwen3.5-2B-Base: the pretrained model, no chat tuning (its text model).

* qwen/qwen3.5/qwen3.5-4b: [9.33 GB bf16; thinking: *thinking|no-thinking]
  Qwen3.5-4B: thinking and non-thinking chat (its text model).

* qwen/qwen3.5/qwen3.5-4b-base: [9.33 GB bf16; thinking: no-thinking]
  Qwen3.5-4B-Base: the pretrained model, no chat tuning (its text model).

* qwen/qwen3.5/qwen3.5-9b: [19.32 GB bf16; thinking: *thinking|no-thinking]
  Qwen3.5-9B: thinking and non-thinking chat (its text model).

* qwen/qwen3.5/qwen3.5-9b-base: [19.32 GB bf16; thinking: no-thinking]
  Qwen3.5-9B-Base: the pretrained model, no chat tuning (its text model).

* qwen/qwen3.5/qwen3.5-27b: [55.58 GB bf16; thinking: *thinking|no-thinking]
  Qwen3.5-27B: thinking and non-thinking chat (its text model); 54 GB in
  bf16, more than a card holds.

# contract: gemma-lm
# A Gemma 1, 2 or 3 text model, as transformers lays it out: its geometry
# from `config.json`.
# publisher: gemma - Google's Gemma models, from their unsloth/ mirrors.

* gemma/gemma/gemma-3-270m: [570.8 MB bf16; thinking: no-thinking]
  Gemma 3 270M: the pretrained model.

* gemma/gemma/gemma-3-270m-it: [570.8 MB bf16; thinking: no-thinking]
  Gemma 3 270M, instruction-tuned.

* gemma/gemma/gemma-3-1b-pt: [2.03 GB bf16; thinking: no-thinking]
  Gemma 3 1B: the pretrained model.

* gemma/gemma/gemma-3-1b-it: [2.03 GB bf16; thinking: no-thinking]
  Gemma 3 1B, instruction-tuned.

* gemma/gemma/gemma-3-4b-pt: [8.63 GB bf16; thinking: no-thinking]
  Gemma 3 4B: the pretrained model's text model.

* gemma/gemma/gemma-3-4b-it: [8.63 GB bf16; thinking: no-thinking]
  Gemma 3 4B, instruction-tuned: its text model.

* gemma/gemma/gemma-2-2b: [5.25 GB bf16; thinking: no-thinking]
  Gemma 2 2B: the pretrained model, stored in f32.

* gemma/gemma/gemma-2-2b-it: [5.25 GB bf16; thinking: no-thinking]
  Gemma 2 2B, instruction-tuned.

* gemma/gemma/gemma-2-9b: [18.52 GB bf16; thinking: no-thinking]
  Gemma 2 9B: the pretrained model.

* gemma/gemma/gemma-2-9b-it: [18.50 GB bf16; thinking: no-thinking]
  Gemma 2 9B, instruction-tuned.

* gemma/gemma/gemma-2b: [5.03 GB bf16; thinking: no-thinking]
  Gemma 2B (Gemma 1): the pretrained model.

* gemma/gemma/gemma-2b-it: [5.03 GB bf16; thinking: no-thinking]
  Gemma 2B (Gemma 1), instruction-tuned.

* gemma/gemma/gemma-7b: [17.11 GB bf16; thinking: no-thinking]
  Gemma 7B (Gemma 1): the pretrained model.

* gemma/gemma/gemma-7b-it: [17.09 GB bf16; thinking: no-thinking]
  Gemma 7B (Gemma 1), instruction-tuned.

# contract: nanochat-lm
# A nanochat GPT, as upstream's training run writes it: its geometry from
# the run's `meta_*.json`.
# publisher: nanochat-students - nanochat-students, who publish nanochat training runs.

* nanochat-students/base-d20: [2.08 GB bf16; thinking: no-thinking; aka nanochat-d20-base]
  nanochat-students' d20 after pretraining (step 21400): upstream's 2025
  model, 561M, base.

* nanochat-students/mid-d20: [2.08 GB bf16; thinking: no-thinking; aka nanochat-d20-mid]
  nanochat-students' d20 after midtraining (step 765): upstream's 2025
  model, 561M, chat format.

* nanochat-students/rl-d20: [2.08 GB bf16; thinking: no-thinking; aka nanochat-d20-rl]
  nanochat-students' d20 after its RL stage (step 650): upstream's 2025
  model, 561M, chat.

# publisher: nitishpandey04 - nitishpandey04, who publishes nanochat training runs.

* nitishpandey04/nanochat-d18-base: [2.05 GB; thinking: no-thinking]
  nitishpandey04's d18 after pretraining (step 3712): upstream's 2026
  model, 702M, base.

* nitishpandey04/nanochat-d18-sft: [2.05 GB; thinking: no-thinking]
  nitishpandey04's d18 after SFT (step 478): upstream's 2026 model, 702M,
  chat.

# bunsen-tool decide ...
# Typed Decisions

# contract: laya-decision
# A Laya decision model: a ModernBERT encoder and a decision head, in one
# safetensors file.
# publisher: laya - convaiinnovations, Laya's authors.

* laya/laya/laya: [846.2 MB f16; aka laya-en]
  Laya, English: ModernBERT-large (28 layers, 1024 wide), 421M parameters,
  prompts of 512 tokens.

* laya/laya/laya-multilingual: [678.2 MB f16]
  Laya, multilingual: mmBERT-base (22 layers, 768 wide, a 256k vocabulary),
  322M parameters, prompts of 1024 tokens.

* laya/laya/laya-typed-decisions: [846.2 MB f16]
  Laya, typed decisions: the English model trained on typed questions,
  prompts of 1024 tokens.
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
