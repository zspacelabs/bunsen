# bunsen - inventing tomorrow's burn problems today!

*by [ZSpaceLabs](https://zspacelabs.ai)*

[![Crates.io Version](https://img.shields.io/crates/v/bunsen)](https://crates.io/crates/bunsen)
[![Documentation](https://img.shields.io/docsrs/bunsen)](https://docs.rs/bunsen/latest/bunsen/)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](#license)
[![Discord](https://img.shields.io/discord/1475229838754316502?label=discord)](https://discord.gg/vBgXHWCeah)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/zspacelabs/bunsen)

- [bunsen-tool](https://crates.io/crates/bunsen-tool) - cli model tool: chat, asr, tts, and decision models.
- [The bunsen book](https://zspacelabs.ai/bunsen/book/) explains how bunsen is organized, the systems that cut across
  it, and how it is developed. Start with
  [Installing and first use](https://zspacelabs.ai/bunsen/book/organization/install.html).
- [The API docs](https://docs.rs/bunsen/latest/bunsen/) document every type and lifecycle, with worked examples.

#### Pretrained Catalog

`bunsen` ships with an extensive catalog of pretrained models, and the caching and loading mechanisms needed to load and
run them.

<details>
<summary>Click to expand the pretrained catalog summary.</summary>

<!-- card: bunsen-tool summary -->

```text
# bunsen-tool asr ...
# Automated Speech Recognition

# contract: parakeet-asr
# A Parakeet TDT speech recognizer: a Phonon-2 archive, or transformers'
# safetensors and its tokenizer.
# publisher: fermion - FermionResearch, Phonon's authors.

* fermion/phonon-2: [163.5 MB]
  - phonon
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

* mistralai/voxtral/voxtral-mini-4b-realtime-2602: [8.87 GB bf16]
  - voxtral-mini-4b-realtime
  Voxtral-Mini-4B-Realtime-2602: 4.4 B parameters, 13 languages, streaming.

# contract: whisper-asr
# An OpenAI Whisper checkpoint, as openai-whisper saves it, and its
# tiktoken vocabulary.
# publisher: openai - OpenAI.

* openai/whisper/large/large-v1: [3.09 GB f16; task: *transcribe|translate]
  1550 M parameters, multilingual.

* openai/whisper/large/large-v2: [3.09 GB f16; task: *transcribe|translate]
  1550 M parameters, multilingual.

* openai/whisper/large/large-v3: [3.09 GB f16; task: *transcribe|translate]
  - large
  1550 M parameters, multilingual, 128 mels.

* openai/whisper/large/large-v3-turbo: [1.62 GB f16]
  - turbo
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

* nvidia/sortformer/diar_streaming_sortformer_4spk-v2: [471.4 MB f32; latency: ultra-low|*low|high|offline; postprocess: *none|dihard3|callhome]
  - sortformer-v2
  Streaming Sortformer v2: 117 M parameters, up to 4 speakers, 80 ms
  frames.

* nvidia/sortformer/diar_streaming_sortformer_4spk-v2.1: [471.4 MB f32; latency: ultra-low|*low|high|offline; postprocess: *none|dihard3|callhome]
  - sortformer-v2.1
  Streaming Sortformer v2.1: 117 M parameters, up to 4 speakers, 80 ms
  frames.

# contract: silero-vad
# The Silero voice-activity detector at 16 kHz: upstream's state dict.
# publisher: silero - Silero, snakers4/silero-vad's authors.

* silero/silero_vad/vad-16k: [1.2 MB f32; vad-policy: *silero|faster-whisper]
  Silero VAD, the 16 kHz model, from upstream's safetensors.

# bunsen-tool tts ...
# Text to Speech

# contract: zonos3-tts
# A ZONOS3 text-to-speech model as Zyphra's transformers packaging lays it
# out: the language model and the codec decoder in one safetensors file,
# their geometry from `config.json`.
# publisher: zyphra - Zyphra.

* zyphra/zonos3/zonos3-mini: [2.39 GB bf16 f32]
  ZONOS3-Mini: 800 M parameters (100 M active), 44.1 kHz speech in 9
  languages, 27 stock voices.

# contract: qwen3-tts
# A Qwen3-TTS model as Qwen's repositories lay it out: the talker and its
# code predictor in one safetensors file, the 12 Hz codec in another, their
# geometry from their `config.json`s; the tokenizer as its slow files.
# publisher: qwen - Qwen, Alibaba Cloud's model team.

* qwen/qwen3-tts/qwen3-tts-12hz-0.6b-customvoice: [2.50 GB f32]
  Qwen3-TTS-12Hz-0.6B-CustomVoice: 9 stock voices in 10 languages and two
  Chinese dialects, 24 kHz speech; no style instruction.

* qwen/qwen3-tts/qwen3-tts-12hz-1.7b-customvoice: [4.52 GB f32]
  Qwen3-TTS-12Hz-1.7B-CustomVoice: 9 stock voices in 10 languages and two
  Chinese dialects, 24 kHz speech, steered by a style instruction.

* qwen/qwen3-tts/qwen3-tts-12hz-1.7b-voicedesign: [4.52 GB f32]
  Qwen3-TTS-12Hz-1.7B-VoiceDesign: a voice from a description, in 10
  languages, 24 kHz speech; no stock voices.

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

* nanochat-students/base-d20: [2.08 GB bf16; thinking: no-thinking]
  - nanochat-d20-base
  nanochat-students' d20 after pretraining (step 21400): upstream's 2025
  model, 561M, base.

* nanochat-students/mid-d20: [2.08 GB bf16; thinking: no-thinking]
  - nanochat-d20-mid
  nanochat-students' d20 after midtraining (step 765): upstream's 2025
  model, 561M, chat format.

* nanochat-students/rl-d20: [2.08 GB bf16; thinking: no-thinking]
  - nanochat-d20-rl
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

* laya/laya/laya: [846.2 MB f16]
  - laya-en
  Laya, English: ModernBERT-large (28 layers, 1024 wide), 421M parameters,
  prompts of 512 tokens.

* laya/laya/laya-multilingual: [678.2 MB f16]
  Laya, multilingual: mmBERT-base (22 layers, 768 wide, a 256k vocabulary),
  322M parameters, prompts of 1024 tokens.

* laya/laya/laya-typed-decisions: [846.2 MB f16]
  Laya, typed decisions: the English model trained on typed questions,
  prompts of 1024 tokens.

# bunsen-tool classify ...
# Image Classification

# contract: resnet-classifier
# A ResNet image classifier: a PyTorch state dict in the geometry its build
# states, or a timm hub repo's safetensors and `config.json`.
# publisher: torchvision - TorchVision, PyTorch's computer vision library.

* torchvision/resnet18: [47.6 MB f32]
  TorchVision ResNet-18.

* torchvision/resnet34: [88.1 MB f32]
  TorchVision ResNet-34.

* torchvision/resnet50: [103.3 MB f32]
  TorchVision ResNet-50.

* torchvision/resnet101: [179.5 MB f32]
  TorchVision ResNet-101.

* torchvision/resnet152: [242.4 MB f32]
  TorchVision ResNet-152.

# publisher: timm - pytorch-image-models (timm), Ross Wightman's.

* timm/resnet18.a1_in1k: [47.6 MB]
  - resnet18_a1
  timm's `resnet18`, its `a1_in1k` weights.

* timm/resnet18.a2_in1k: [47.6 MB]
  - resnet18_a2
  timm's `resnet18`, its `a2_in1k` weights.

* timm/resnet18.a3_in1k: [47.6 MB]
  - resnet18_a3
  timm's `resnet18`, its `a3_in1k` weights.

* timm/resnet18.fb_ssl_yfcc100m_ft_in1k: [47.6 MB]
  timm's `resnet18`, its `fb_ssl_yfcc100m_ft_in1k` weights.

* timm/resnet18.fb_swsl_ig1b_ft_in1k: [47.6 MB]
  timm's `resnet18`, its `fb_swsl_ig1b_ft_in1k` weights.

* timm/resnet18.gluon_in1k: [47.6 MB]
  timm's `resnet18`, its `gluon_in1k` weights.

* timm/resnet26.bt_in1k: [64.9 MB]
  - resnet26
  timm's `resnet26`, its `bt_in1k` weights.

* timm/resnet34.a1_in1k: [88.0 MB]
  - resnet34_a1
  timm's `resnet34`, its `a1_in1k` weights.

* timm/resnet34.a2_in1k: [88.0 MB]
  - resnet34_a2
  timm's `resnet34`, its `a2_in1k` weights.

* timm/resnet34.a3_in1k: [88.0 MB]
  - resnet34_a3
  timm's `resnet34`, its `a3_in1k` weights.

* timm/resnet34.bt_in1k: [88.0 MB]
  - resnet34
  timm's `resnet34`, its `bt_in1k` weights.

* timm/resnet34.gluon_in1k: [88.0 MB]
  timm's `resnet34`, its `gluon_in1k` weights.

* timm/resnet34.ra4_e3600_r224_in1k: [88.0 MB]
  timm's `resnet34`, its `ra4_e3600_r224_in1k` weights.

* timm/resnet50.a1_in1k: [103.2 MB]
  timm's `resnet50`, its `a1_in1k` weights.

* timm/resnet50.a1h_in1k: [103.2 MB]
  timm's `resnet50`, its `a1h_in1k` weights.

* timm/resnet50.a2_in1k: [103.2 MB]
  timm's `resnet50`, its `a2_in1k` weights.

* timm/resnet50.a3_in1k: [103.2 MB]
  timm's `resnet50`, its `a3_in1k` weights.

* timm/resnet50.am_in1k: [103.2 MB]
  timm's `resnet50`, its `am_in1k` weights.

* timm/resnet50.b1k_in1k: [103.2 MB]
  timm's `resnet50`, its `b1k_in1k` weights.

* timm/resnet50.b2k_in1k: [103.2 MB]
  timm's `resnet50`, its `b2k_in1k` weights.

* timm/resnet50.bt_in1k: [103.2 MB]
  timm's `resnet50`, its `bt_in1k` weights.

* timm/resnet50.c1_in1k: [103.2 MB]
  timm's `resnet50`, its `c1_in1k` weights.

* timm/resnet50.c2_in1k: [103.2 MB]
  timm's `resnet50`, its `c2_in1k` weights.

* timm/resnet50.d_in1k: [103.2 MB]
  timm's `resnet50`, its `d_in1k` weights.

* timm/resnet50.fb_ssl_yfcc100m_ft_in1k: [103.2 MB]
  timm's `resnet50`, its `fb_ssl_yfcc100m_ft_in1k` weights.

* timm/resnet50.fb_swsl_ig1b_ft_in1k: [103.2 MB]
  timm's `resnet50`, its `fb_swsl_ig1b_ft_in1k` weights.

* timm/resnet50.gluon_in1k: [103.2 MB]
  timm's `resnet50`, its `gluon_in1k` weights.

* timm/resnet50.ra_in1k: [103.2 MB]
  timm's `resnet50`, its `ra_in1k` weights.

* timm/resnet50.ram_in1k: [103.2 MB]
  timm's `resnet50`, its `ram_in1k` weights.

* timm/resnet50.tv2_in1k: [103.2 MB]
  timm's `resnet50`, its `tv2_in1k` weights.

* timm/resnet50_gn.a1h_in1k: [103.0 MB]
  timm's `resnet50_gn`, its `a1h_in1k` weights.

* timm/resnet101.a1_in1k: [179.4 MB]
  - resnet101_a1
  timm's `resnet101`, its `a1_in1k` weights.

* timm/resnet101.a1h_in1k: [179.4 MB]
  timm's `resnet101`, its `a1h_in1k` weights.

* timm/resnet101.a2_in1k: [179.4 MB]
  timm's `resnet101`, its `a2_in1k` weights.

* timm/resnet101.a3_in1k: [179.4 MB]
  timm's `resnet101`, its `a3_in1k` weights.

* timm/resnet101.gluon_in1k: [179.4 MB]
  timm's `resnet101`, its `gluon_in1k` weights.

* timm/resnet101.tv2_in1k: [179.4 MB]
  timm's `resnet101`, its `tv2_in1k` weights.

* timm/resnet152.a1_in1k: [242.2 MB]
  timm's `resnet152`, its `a1_in1k` weights.

* timm/resnet152.a1h_in1k: [242.2 MB]
  timm's `resnet152`, its `a1h_in1k` weights.

* timm/resnet152.a2_in1k: [242.2 MB]
  timm's `resnet152`, its `a2_in1k` weights.

* timm/resnet152.a3_in1k: [242.2 MB]
  timm's `resnet152`, its `a3_in1k` weights.

* timm/resnet152.gluon_in1k: [242.2 MB]
  timm's `resnet152`, its `gluon_in1k` weights.

* timm/resnet152.tv2_in1k: [242.2 MB]
  timm's `resnet152`, its `tv2_in1k` weights.

* timm/resnet10t.c3_in1k: [22.5 MB]
  timm's `resnet10t`, its `c3_in1k` weights.

* timm/resnet14t.c3_in1k: [41.2 MB]
  timm's `resnet14t`, its `c3_in1k` weights.

* timm/resnet18d.ra2_in1k: [47.6 MB]
  timm's `resnet18d`, its `ra2_in1k` weights.

* timm/resnet18d.ra4_e3600_r224_in1k: [47.6 MB]
  timm's `resnet18d`, its `ra4_e3600_r224_in1k` weights.

* timm/resnet26d.bt_in1k: [65.0 MB]
  timm's `resnet26d`, its `bt_in1k` weights.

* timm/resnet26t.ra2_in1k: [64.9 MB]
  timm's `resnet26t`, its `ra2_in1k` weights.

* timm/resnet34d.ra2_in1k: [88.1 MB]
  timm's `resnet34d`, its `ra2_in1k` weights.

* timm/resnet50c.gluon_in1k: [103.3 MB]
  timm's `resnet50c`, its `gluon_in1k` weights.

* timm/resnet50d.a1_in1k: [103.3 MB]
  timm's `resnet50d`, its `a1_in1k` weights.

* timm/resnet50d.a2_in1k: [103.3 MB]
  timm's `resnet50d`, its `a2_in1k` weights.

* timm/resnet50d.a3_in1k: [103.3 MB]
  timm's `resnet50d`, its `a3_in1k` weights.

* timm/resnet50d.gluon_in1k: [103.3 MB]
  timm's `resnet50d`, its `gluon_in1k` weights.

* timm/resnet50d.ra2_in1k: [103.3 MB]
  timm's `resnet50d`, its `ra2_in1k` weights.

* timm/resnet50d.ra4_e3600_r224_in1k: [103.3 MB]
  timm's `resnet50d`, its `ra4_e3600_r224_in1k` weights.

* timm/resnet50s.gluon_in1k: [103.7 MB]
  timm's `resnet50s`, its `gluon_in1k` weights.

* timm/resnet101c.gluon_in1k: [179.5 MB]
  timm's `resnet101c`, its `gluon_in1k` weights.

* timm/resnet101d.gluon_in1k: [179.5 MB]
  timm's `resnet101d`, its `gluon_in1k` weights.

* timm/resnet101d.ra2_in1k: [179.5 MB]
  timm's `resnet101d`, its `ra2_in1k` weights.

* timm/resnet101s.gluon_in1k: [179.9 MB]
  timm's `resnet101s`, its `gluon_in1k` weights.

* timm/resnet152c.gluon_in1k: [242.3 MB]
  timm's `resnet152c`, its `gluon_in1k` weights.

* timm/resnet152d.gluon_in1k: [242.3 MB]
  timm's `resnet152d`, its `gluon_in1k` weights.

* timm/resnet152d.ra2_in1k: [242.3 MB]
  timm's `resnet152d`, its `ra2_in1k` weights.

* timm/resnet152s.gluon_in1k: [242.7 MB]
  timm's `resnet152s`, its `gluon_in1k` weights.

* timm/resnet200d.ra2_in1k: [260.3 MB]
  timm's `resnet200d`, its `ra2_in1k` weights.

* timm/resnext50d_32x4d.bt_in1k: [101.2 MB]
  timm's `resnext50d_32x4d`, its `bt_in1k` weights.

* timm/resnext50_32x4d.a1_in1k: [101.2 MB]
  timm's `resnext50_32x4d`, its `a1_in1k` weights.

* timm/resnext50_32x4d.a1h_in1k: [101.2 MB]
  timm's `resnext50_32x4d`, its `a1h_in1k` weights.

* timm/resnext50_32x4d.a2_in1k: [101.2 MB]
  timm's `resnext50_32x4d`, its `a2_in1k` weights.

* timm/resnext50_32x4d.a3_in1k: [101.2 MB]
  timm's `resnext50_32x4d`, its `a3_in1k` weights.

* timm/resnext50_32x4d.fb_ssl_yfcc100m_ft_in1k: [101.2 MB]
  timm's `resnext50_32x4d`, its `fb_ssl_yfcc100m_ft_in1k` weights.

* timm/resnext50_32x4d.fb_swsl_ig1b_ft_in1k: [101.2 MB]
  timm's `resnext50_32x4d`, its `fb_swsl_ig1b_ft_in1k` weights.

* timm/resnext50_32x4d.gluon_in1k: [101.2 MB]
  timm's `resnext50_32x4d`, its `gluon_in1k` weights.

* timm/resnext50_32x4d.ra_in1k: [101.2 MB]
  timm's `resnext50_32x4d`, its `ra_in1k` weights.

* timm/resnext50_32x4d.tv2_in1k: [101.2 MB]
  timm's `resnext50_32x4d`, its `tv2_in1k` weights.

* timm/resnext50_32x4d.tv_in1k: [101.2 MB]
  timm's `resnext50_32x4d`, its `tv_in1k` weights.

* timm/resnext101_32x4d.fb_ssl_yfcc100m_ft_in1k: [178.1 MB]
  timm's `resnext101_32x4d`, its `fb_ssl_yfcc100m_ft_in1k` weights.

* timm/resnext101_32x4d.fb_swsl_ig1b_ft_in1k: [178.1 MB]
  timm's `resnext101_32x4d`, its `fb_swsl_ig1b_ft_in1k` weights.

* timm/resnext101_32x4d.gluon_in1k: [178.1 MB]
  timm's `resnext101_32x4d`, its `gluon_in1k` weights.

* timm/resnext101_32x8d.fb_ssl_yfcc100m_ft_in1k: [356.8 MB]
  timm's `resnext101_32x8d`, its `fb_ssl_yfcc100m_ft_in1k` weights.

* timm/resnext101_32x8d.fb_swsl_ig1b_ft_in1k: [356.8 MB]
  timm's `resnext101_32x8d`, its `fb_swsl_ig1b_ft_in1k` weights.

* timm/resnext101_32x8d.fb_wsl_ig1b_ft_in1k: [356.8 MB]
  timm's `resnext101_32x8d`, its `fb_wsl_ig1b_ft_in1k` weights.

* timm/resnext101_32x8d.tv2_in1k: [356.8 MB]
  timm's `resnext101_32x8d`, its `tv2_in1k` weights.

* timm/resnext101_32x8d.tv_in1k: [356.8 MB]
  timm's `resnext101_32x8d`, its `tv_in1k` weights.

* timm/resnext101_32x16d.fb_ssl_yfcc100m_ft_in1k: [778.2 MB]
  timm's `resnext101_32x16d`, its `fb_ssl_yfcc100m_ft_in1k` weights.

* timm/resnext101_32x16d.fb_swsl_ig1b_ft_in1k: [778.2 MB]
  timm's `resnext101_32x16d`, its `fb_swsl_ig1b_ft_in1k` weights.

* timm/resnext101_32x16d.fb_wsl_ig1b_ft_in1k: [778.2 MB]
  timm's `resnext101_32x16d`, its `fb_wsl_ig1b_ft_in1k` weights.

* timm/resnext101_32x32d.fb_wsl_ig1b_ft_in1k: [1.88 GB]
  timm's `resnext101_32x32d`, its `fb_wsl_ig1b_ft_in1k` weights.

* timm/resnext101_64x4d.c1_in1k: [335.4 MB]
  timm's `resnext101_64x4d`, its `c1_in1k` weights.

* timm/resnext101_64x4d.gluon_in1k: [335.4 MB]
  timm's `resnext101_64x4d`, its `gluon_in1k` weights.

* timm/resnext101_64x4d.tv_in1k: [335.4 MB]
  timm's `resnext101_64x4d`, its `tv_in1k` weights.

* timm/wide_resnet50_2.racm_in1k: [276.6 MB]
  timm's `wide_resnet50_2`, its `racm_in1k` weights.

* timm/wide_resnet50_2.tv2_in1k: [276.6 MB]
  timm's `wide_resnet50_2`, its `tv2_in1k` weights.

* timm/wide_resnet50_2.tv_in1k: [276.6 MB]
  timm's `wide_resnet50_2`, its `tv_in1k` weights.

* timm/wide_resnet101_2.tv2_in1k: [508.9 MB]
  timm's `wide_resnet101_2`, its `tv2_in1k` weights.

* timm/wide_resnet101_2.tv_in1k: [508.9 MB]
  timm's `wide_resnet101_2`, its `tv_in1k` weights.
```

<!-- /card -->
</details>

## What is bunsen?

__bunsen - inventing tomorrow's burn problems today!__

The [burn](https://burn.dev) community is building a stable, long-term foundation for high-performance,
maintainable tensor programming in Rust; work which requires extensive incremental
novel research into Rust type and runtime extensions.

`bunsen` contributes to the community work as a "batteries included" standard library for the frontier, by driving real
models and
applications ahead of the `burn` foundation: exploring the novel problems they raise, unifying their partial
solutions incrementally, and upstreaming stabilized components back into `burn`.

The frontier has two halves: the components model families are built from, and the runtime around
those models (loading, driving and testing them). `bunsen` drives research through a development
pipeline in which each stage increases breadth:

- The frontier is **Expanded** by porting new model families and the components they need, widening
  what is possible.
- The model families are **Consolidated** by building and improving the runtime around those applications (loading,
  driving,
  testing), making them supportable.
- Bespoke solutions are incrementally **Unified** into shared parts, making the next family easier to develop.
- Stabilized foundations are **Upstreamed** into `burn`, making them standard.

Every idea in `bunsen` is headed for `burn`: refined and ported, or supplanted by `burn`'s own
solution.

<table>
  <tr>
    <th colspan="5" align="left">Expansion</th>
  </tr>
  <tr>
    <td colspan="5"><i>The frontier, expanded by porting new model families and the components they need,
      widening what is possible.</i></td>
  </tr>
  <tr>
    <th rowspan="2">Generative language</th>
    <th rowspan="2">Decision</th>
    <th colspan="2">Speech</th>
    <th rowspan="2">Image classification</th>
  </tr>
  <tr>
    <th>ASR</th>
    <th>VAD</th>
  </tr>
  <tr>
    <td>Gemma 1, 2, 3</td>
    <td>Laya</td>
    <td>Whisper</td>
    <td>Silero VAD</td>
    <td>ResNet</td>
  </tr>
  <tr>
    <td>Qwen3, 3.5</td>
    <td></td>
    <td>Qwen3-ASR</td>
    <th>Diarization</th>
    <td></td>
  </tr>
  <tr>
    <td>nanochat</td>
    <td></td>
    <td>Parakeet</td>
    <td>Sortformer</td>
    <td></td>
  </tr>
  <tr>
    <td></td>
    <td></td>
    <td>Voxtral</td>
    <th>TTS</th>
    <td></td>
  </tr>
  <tr>
    <td></td>
    <td></td>
    <td></td>
    <td>Qwen3-TTS</td>
    <td></td>
  </tr>
  <tr>
    <td></td>
    <td></td>
    <td></td>
    <td>ZONOS3</td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5">pretrained public model catalog: 87 models</td>
  </tr>
  <tr>
    <th colspan="5" align="left">Consolidation</th>
  </tr>
  <tr>
    <td colspan="5"><i>
Model family support consolidation (loading, driving, testing):</i></td>
  </tr>
  <tr>
    <td><b>chat driver:</b><br>cached LMs<br>streamed text<br>stop strings</td>
    <td><b>decision models:</b><br>prompt building<br>calibrated answers</td>
    <td><b>ASR stream driver:</b><br>emission policies<br>VAD gate<br>speaker attribution<br>stream clock<br>batched streams</td>
    <td><b>diarization:</b><br>turns, RTTM<br><b>TTS driver:</b><br>request stages<br>take splitting<br>seeding</td>
    <td></td>
  </tr>
  <tr>
    <td><code>bunsen-tool chat</code></td>
    <td><code>bunsen-tool decide</code></td>
    <td><code>bunsen-tool asr</code>:<br>microphone capture<br>transcript publishers</td>
    <td><code>bunsen-tool tts</code>:<br>speaker playback</td>
    <td></td>
  </tr>
  <tr>
    <th>find and fetch</th>
    <th>load</th>
    <th>text in and out</th>
    <th>decoding</th>
    <th>audio</th>
  </tr>
  <tr>
    <td>model catalog<br>pinned resources<br>disk cache<br>hub mirrors<br>offline bundles</td>
    <td>checkpoint readers<br>rekeying<br>dtype placement<br>config interop</td>
    <td>tokenizers<br>text normalizers<br>chat templates<br>stop strings</td>
    <td>searches, sampling<br>rewindable KV caches<br>linear-attention state</td>
    <td>streaming STFT<br>log-mel front ends<br>quiet-point cuts<br>audio decoding</td>
  </tr>
  <tr>
    <th>every command</th>
    <th colspan="2">testing</th>
    <th colspan="2">training</th>
  </tr>
  <tr>
    <td>device selection<br>precision flags</td>
    <td colspan="2">validation against reference implementations<br>audit probes and baselines<br>test
      devices</td>
    <td colspan="2">shard-set datasets<br>Arrow dataloaders<br>parameter groups</td>
  </tr>
  <tr>
    <th colspan="5" align="left">Unification</th>
  </tr>
  <tr>
    <td colspan="5"><i>Bespoke solutions, incrementally unified into shared parts, making the next family
      easier to develop.</i></td>
  </tr>
  <tr>
    <th colspan="2">runtime needs and meta-traits</th>
    <th colspan="3">model isomorphisms</th>
  </tr>
  <tr>
    <td colspan="2"><code>burn</code> reflective meta-descriptions<br>recurrent <code>Config</code> and
      <code>Module</code> traits<br>optimizer extensions<br>shape contracts<br>weight caching<br>common
      format loading and probing</td>
    <td colspan="3">~60 structural combinator <code>Module</code>s in <code>bunsen::blocks</code><br>~14
      model family kits in <code>bunsen::kits</code>, mostly near-trivial combinations of
      <code>bunsen::blocks</code></td>
  </tr>
  <tr>
    <th colspan="5" align="left">Upstreaming</th>
  </tr>
  <tr>
    <td colspan="5"><i>Stabilized foundations, upstreamed into <code>burn</code>, making them
      standard.</i></td>
  </tr>
  <tr>
    <td colspan="5"><a href="https://github.com/tracel-ai/burn/pulls?q=is%3Apr+state%3Aclosed+author%3Acrutcher">&gt;100 merged burn PRs</a></td>
  </tr>
</table>

## Simulations

Simulations written as tensor programs, built the way the kits are, a config and a state you step ([
`bunsen::sims`](https://docs.rs/bunsen/latest/bunsen/sims/index.html)):

- [`sims::conway`](https://docs.rs/bunsen/latest/bunsen/sims/conway/index.html): Conway's Game of Life on
  toroidal 2D and 3D boards.
- [`sims::lbm`](https://docs.rs/bunsen/latest/bunsen/sims/lbm/index.html): a D2Q9 lattice-Boltzmann fluid.

## The workspace

Most users depend on `bunsen` alone. The [bunsen repository](https://github.com/zspacelabs/bunsen) also publishes its companion crates (the firehose data
pipeline, a Parquet-to-tokens data loader, the shape-contract macro, and bundled model assets), and holds the
development, validation and example
crates. [The workspace](https://zspacelabs.ai/bunsen/book/organization/workspace.html)
describes each one.

## Examples

[`examples/`](https://github.com/zspacelabs/bunsen/tree/main/examples) holds runnable programs, one or more per kit or system: speech to text, image-model training
and fine-tuning, GPT training, and live simulations. The book's
[Examples](https://zspacelabs.ai/bunsen/book/organization/examples.html) page says what each one shows.

## Contributing

[CONTRIBUTING.md](https://github.com/zspacelabs/bunsen/blob/main/CONTRIBUTING.md) has the setup, the checks a pull request must pass, and the release process.

## License

`bunsen` is distributed under the terms of both the MIT license and the Apache License (Version 2.0).
See [LICENSE-APACHE](https://github.com/zspacelabs/bunsen/blob/main/LICENSE-APACHE) and [LICENSE-MIT](https://github.com/zspacelabs/bunsen/blob/main/LICENSE-MIT) for details. Opening a pull request is assumed to
signal agreement with these licensing terms.
