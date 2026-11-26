# bunsen-tool

[bunsen](https://crates.io/crates/bunsen)'s models on the command line: speech to text and who spoke when, text to
speech, chat, and typed decisions, each loading its models by name from bunsen's catalog.

This README is a tour. Every command and flag documents itself: `bunsen-tool --help`, `bunsen-tool <command> --help`,
and `-h` for the short form.

## Install

```sh
cargo install bunsen-tool --features wgpu     # or vulkan, metal, cuda; the CPU is always there
bunsen-tool asr transcribe recording.wav
```

On Linux, the microphone and speaker commands link ALSA (`libasound2-dev`). The default features build every command;
`--no-default-features --features speech` builds `asr` and `tts` without audio devices, and `--no-default-features`
alone builds the tool without `asr` and `tts`. `bunsen-tool devices` lists the backends a build has, and what each
device does with each float dtype.

## Models

`bunsen-tool summary` lists every model this binary runs, by command, contract and publisher, with its size, dtypes,
presets (the default starred) and aliases; each command's `list` adds licences and what is cached. As it prints them
(`cargo make cards` rebuilds this card):

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

* timm/ecaresnet26t.ra2_in1k: [64.9 MB]
  timm's `ecaresnet26t`, its `ra2_in1k` weights.

* timm/ecaresnet50d.miil_in1k: [103.3 MB]
  timm's `ecaresnet50d`, its `miil_in1k` weights.

* timm/ecaresnet50t.a1_in1k: [103.3 MB]
  timm's `ecaresnet50t`, its `a1_in1k` weights.

* timm/ecaresnet50t.a2_in1k: [103.3 MB]
  timm's `ecaresnet50t`, its `a2_in1k` weights.

* timm/ecaresnet50t.a3_in1k: [103.3 MB]
  timm's `ecaresnet50t`, its `a3_in1k` weights.

* timm/ecaresnet50t.ra2_in1k: [103.3 MB]
  timm's `ecaresnet50t`, its `ra2_in1k` weights.

* timm/ecaresnetlight.miil_in1k: [121.7 MB]
  timm's `ecaresnetlight`, its `miil_in1k` weights.

* timm/ecaresnet101d.miil_in1k: [179.5 MB]
  timm's `ecaresnet101d`, its `miil_in1k` weights.

* timm/ecaresnet269d.ra2_in1k: [410.3 MB]
  timm's `ecaresnet269d`, its `ra2_in1k` weights.

* timm/seresnet50.a1_in1k: [113.4 MB]
  timm's `seresnet50`, its `a1_in1k` weights.

* timm/seresnet50.a2_in1k: [113.4 MB]
  timm's `seresnet50`, its `a2_in1k` weights.

* timm/seresnet50.a3_in1k: [113.4 MB]
  timm's `seresnet50`, its `a3_in1k` weights.

* timm/seresnet50.ra2_in1k: [113.4 MB]
  timm's `seresnet50`, its `ra2_in1k` weights.

* timm/seresnet152d.ra2_in1k: [268.8 MB]
  timm's `seresnet152d`, its `ra2_in1k` weights.

* timm/seresnext26d_32x4d.bt_in1k: [68.2 MB]
  timm's `seresnext26d_32x4d`, its `bt_in1k` weights.

* timm/seresnext26t_32x4d.bt_in1k: [68.2 MB]
  timm's `seresnext26t_32x4d`, its `bt_in1k` weights.

* timm/seresnext50_32x4d.gluon_in1k: [111.3 MB]
  timm's `seresnext50_32x4d`, its `gluon_in1k` weights.

* timm/seresnext50_32x4d.racm_in1k: [111.3 MB]
  timm's `seresnext50_32x4d`, its `racm_in1k` weights.

* timm/seresnext101_32x4d.gluon_in1k: [197.2 MB]
  timm's `seresnext101_32x4d`, its `gluon_in1k` weights.

* timm/seresnext101_32x8d.ah_in1k: [375.9 MB]
  timm's `seresnext101_32x8d`, its `ah_in1k` weights.

* timm/seresnext101d_32x8d.ah_in1k: [376.0 MB]
  timm's `seresnext101d_32x8d`, its `ah_in1k` weights.

* timm/seresnext101_64x4d.gluon_in1k: [354.6 MB]
  timm's `seresnext101_64x4d`, its `gluon_in1k` weights.

* timm/resnetblur50.bt_in1k: [103.2 MB]
  timm's `resnetblur50`, its `bt_in1k` weights.

* timm/resnetaa50.a1h_in1k: [103.2 MB]
  timm's `resnetaa50`, its `a1h_in1k` weights.

* timm/resnetaa50d.d_in12k: [192.1 MB]
  timm's `resnetaa50d`, its `d_in12k` weights.

* timm/resnetaa50d.sw_in12k: [192.1 MB]
  timm's `resnetaa50d`, its `sw_in12k` weights.

* timm/resnetaa50d.sw_in12k_ft_in1k: [103.3 MB]
  timm's `resnetaa50d`, its `sw_in12k_ft_in1k` weights.

* timm/resnetaa101d.sw_in12k: [268.3 MB]
  timm's `resnetaa101d`, its `sw_in12k` weights.

* timm/resnetaa101d.sw_in12k_ft_in1k: [179.5 MB]
  timm's `resnetaa101d`, its `sw_in12k_ft_in1k` weights.

* timm/seresnextaa101d_32x8d.ah_in1k: [376.0 MB]
  timm's `seresnextaa101d_32x8d`, its `ah_in1k` weights.

* timm/seresnextaa101d_32x8d.sw_in12k: [464.8 MB]
  timm's `seresnextaa101d_32x8d`, its `sw_in12k` weights.

* timm/seresnextaa101d_32x8d.sw_in12k_ft_in1k: [376.0 MB]
  timm's `seresnextaa101d_32x8d`, its `sw_in12k_ft_in1k` weights.

* timm/seresnextaa101d_32x8d.sw_in12k_ft_in1k_288: [376.0 MB]
  timm's `seresnextaa101d_32x8d`, its `sw_in12k_ft_in1k_288` weights.

* timm/seresnextaa201d_32x8d.sw_in12k: [688.7 MB]
  timm's `seresnextaa201d_32x8d`, its `sw_in12k` weights.

* timm/seresnextaa201d_32x8d.sw_in12k_ft_in1k_384: [599.9 MB]
  timm's `seresnextaa201d_32x8d`, its `sw_in12k_ft_in1k_384` weights.

* timm/resnetrs50.tf_in1k: [143.8 MB]
  timm's `resnetrs50`, its `tf_in1k` weights.

* timm/resnetrs101.tf_in1k: [255.7 MB]
  timm's `resnetrs101`, its `tf_in1k` weights.

* timm/resnetrs152.tf_in1k: [348.0 MB]
  timm's `resnetrs152`, its `tf_in1k` weights.

* timm/resnetrs200.tf_in1k: [374.4 MB]
  timm's `resnetrs200`, its `tf_in1k` weights.

* timm/resnetrs270.tf_in1k: [521.4 MB]
  timm's `resnetrs270`, its `tf_in1k` weights.

* timm/resnetrs350.tf_in1k: [658.1 MB]
  timm's `resnetrs350`, its `tf_in1k` weights.

* timm/resnetrs420.tf_in1k: [770.1 MB]
  timm's `resnetrs420`, its `tf_in1k` weights.

# contract: regnet-classifier
# A RegNet image classifier, or headless trunk: a timm hub repo's
# checkpoint (safetensors) and `config.json`.
# publisher: timm - pytorch-image-models (timm), Ross Wightman's.

* timm/regnetx_002.pycls_in1k: [11.6 MB]
  timm's `regnetx_002`, its `pycls_in1k` weights.

* timm/regnetx_004.pycls_in1k: [21.6 MB]
  timm's `regnetx_004`, its `pycls_in1k` weights.

* timm/regnetx_004_tv.tv2_in1k: [22.9 MB]
  timm's `regnetx_004_tv`, its `tv2_in1k` weights.

* timm/regnetx_006.pycls_in1k: [25.7 MB]
  timm's `regnetx_006`, its `pycls_in1k` weights.

* timm/regnetx_008.pycls_in1k: [30.0 MB]
  timm's `regnetx_008`, its `pycls_in1k` weights.

* timm/regnetx_008.tv2_in1k: [30.0 MB]
  timm's `regnetx_008`, its `tv2_in1k` weights.

* timm/regnetx_016.pycls_in1k: [37.7 MB]
  timm's `regnetx_016`, its `pycls_in1k` weights.

* timm/regnetx_016.tv2_in1k: [37.7 MB]
  timm's `regnetx_016`, its `tv2_in1k` weights.

* timm/regnetx_032.pycls_in1k: [62.2 MB]
  timm's `regnetx_032`, its `pycls_in1k` weights.

* timm/regnetx_032.tv2_in1k: [62.2 MB]
  timm's `regnetx_032`, its `tv2_in1k` weights.

* timm/regnetx_040.pycls_in1k: [89.6 MB]
  timm's `regnetx_040`, its `pycls_in1k` weights.

* timm/regnetx_064.pycls_in1k: [105.9 MB]
  timm's `regnetx_064`, its `pycls_in1k` weights.

* timm/regnetx_080.pycls_in1k: [159.4 MB]
  timm's `regnetx_080`, its `pycls_in1k` weights.

* timm/regnetx_080.tv2_in1k: [159.4 MB]
  timm's `regnetx_080`, its `tv2_in1k` weights.

* timm/regnetx_120.pycls_in1k: [185.6 MB]
  timm's `regnetx_120`, its `pycls_in1k` weights.

* timm/regnetx_160.pycls_in1k: [218.4 MB]
  timm's `regnetx_160`, its `pycls_in1k` weights.

* timm/regnetx_160.tv2_in1k: [218.4 MB]
  timm's `regnetx_160`, its `tv2_in1k` weights.

* timm/regnetx_320.pycls_in1k: [432.7 MB]
  timm's `regnetx_320`, its `pycls_in1k` weights.

* timm/regnetx_320.tv2_in1k: [432.7 MB]
  timm's `regnetx_320`, its `tv2_in1k` weights.

* timm/regnety_002.pycls_in1k: [13.5 MB]
  timm's `regnety_002`, its `pycls_in1k` weights.

* timm/regnety_004.pycls_in1k: [18.3 MB]
  timm's `regnety_004`, its `pycls_in1k` weights.

* timm/regnety_004.tv2_in1k: [18.3 MB]
  timm's `regnety_004`, its `tv2_in1k` weights.

* timm/regnety_006.pycls_in1k: [25.1 MB]
  timm's `regnety_006`, its `pycls_in1k` weights.

* timm/regnety_008.pycls_in1k: [26.0 MB]
  timm's `regnety_008`, its `pycls_in1k` weights.

* timm/regnety_008_tv.tv2_in1k: [26.6 MB]
  timm's `regnety_008_tv`, its `tv2_in1k` weights.

* timm/regnety_016.pycls_in1k: [45.8 MB]
  timm's `regnety_016`, its `pycls_in1k` weights.

* timm/regnety_016.tv2_in1k: [45.8 MB]
  timm's `regnety_016`, its `tv2_in1k` weights.

* timm/regnety_032.pycls_in1k: [78.8 MB]
  timm's `regnety_032`, its `pycls_in1k` weights.

* timm/regnety_032.ra_in1k: [78.8 MB]
  timm's `regnety_032`, its `ra_in1k` weights.

* timm/regnety_032.tv2_in1k: [78.8 MB]
  timm's `regnety_032`, its `tv2_in1k` weights.

* timm/regnety_040.pycls_in1k: [83.6 MB]
  timm's `regnety_040`, its `pycls_in1k` weights.

* timm/regnety_040.ra3_in1k: [83.6 MB]
  timm's `regnety_040`, its `ra3_in1k` weights.

* timm/regnety_064.pycls_in1k: [123.5 MB]
  timm's `regnety_064`, its `pycls_in1k` weights.

* timm/regnety_064.ra3_in1k: [123.5 MB]
  timm's `regnety_064`, its `ra3_in1k` weights.

* timm/regnety_080.pycls_in1k: [157.9 MB]
  timm's `regnety_080`, its `pycls_in1k` weights.

* timm/regnety_080.ra3_in1k: [157.9 MB]
  timm's `regnety_080`, its `ra3_in1k` weights.

* timm/regnety_080_tv.tv2_in1k: [158.7 MB]
  timm's `regnety_080_tv`, its `tv2_in1k` weights.

* timm/regnety_120.pycls_in1k: [208.5 MB]
  timm's `regnety_120`, its `pycls_in1k` weights.

* timm/regnety_120.sw_in12k: [305.6 MB]
  timm's `regnety_120`, its `sw_in12k` weights.

* timm/regnety_120.sw_in12k_ft_in1k: [208.5 MB]
  timm's `regnety_120`, its `sw_in12k_ft_in1k` weights.

* timm/regnety_160.deit_in1k: [335.6 MB]
  timm's `regnety_160`, its `deit_in1k` weights.

* timm/regnety_160.lion_in12k_ft_in1k: [335.6 MB]
  timm's `regnety_160`, its `lion_in12k_ft_in1k` weights.

* timm/regnety_160.pycls_in1k: [335.6 MB]
  timm's `regnety_160`, its `pycls_in1k` weights.

* timm/regnety_160.sw_in12k: [466.7 MB]
  timm's `regnety_160`, its `sw_in12k` weights.

* timm/regnety_160.sw_in12k_ft_in1k: [335.6 MB]
  timm's `regnety_160`, its `sw_in12k_ft_in1k` weights.

* timm/regnety_160.swag_ft_in1k: [335.6 MB]
  timm's `regnety_160`, its `swag_ft_in1k` weights.

* timm/regnety_160.swag_lc_in1k: [335.6 MB]
  timm's `regnety_160`, its `swag_lc_in1k` weights.

* timm/regnety_160.tv2_in1k: [335.6 MB]
  timm's `regnety_160`, its `tv2_in1k` weights.

* timm/regnety_320.pycls_in1k: [581.6 MB]
  timm's `regnety_320`, its `pycls_in1k` weights.

* timm/regnety_320.seer: [566.0 MB]
  timm's `regnety_320`, its `seer` weights.

* timm/regnety_320.seer_ft_in1k: [581.6 MB]
  timm's `regnety_320`, its `seer_ft_in1k` weights.

* timm/regnety_320.swag_ft_in1k: [581.6 MB]
  timm's `regnety_320`, its `swag_ft_in1k` weights.

* timm/regnety_320.swag_lc_in1k: [581.6 MB]
  timm's `regnety_320`, its `swag_lc_in1k` weights.

* timm/regnety_320.tv2_in1k: [581.6 MB]
  timm's `regnety_320`, its `tv2_in1k` weights.

* timm/regnety_640.seer: [1.11 GB]
  timm's `regnety_640`, its `seer` weights.

* timm/regnety_640.seer_ft_in1k: [1.13 GB]
  timm's `regnety_640`, its `seer_ft_in1k` weights.

* timm/regnety_1280.seer: [2.55 GB]
  timm's `regnety_1280`, its `seer` weights.

* timm/regnety_1280.seer_ft_in1k: [2.58 GB]
  timm's `regnety_1280`, its `seer_ft_in1k` weights.

* timm/regnety_1280.swag_ft_in1k: [2.58 GB]
  timm's `regnety_1280`, its `swag_ft_in1k` weights.

* timm/regnety_1280.swag_lc_in1k: [2.58 GB]
  timm's `regnety_1280`, its `swag_lc_in1k` weights.

* timm/regnety_2560.seer_ft_in1k: [5.13 GB]
  timm's `regnety_2560`, its `seer_ft_in1k` weights.

* timm/regnetv_040.ra3_in1k: [83.6 MB]
  timm's `regnetv_040`, its `ra3_in1k` weights.

* timm/regnetv_064.ra3_in1k: [123.4 MB]
  timm's `regnetv_064`, its `ra3_in1k` weights.

* timm/regnetz_040.ra3_in1k: [109.8 MB]
  timm's `regnetz_040`, its `ra3_in1k` weights.

* timm/regnetz_040_h.ra3_in1k: [117.1 MB]
  timm's `regnetz_040_h`, its `ra3_in1k` weights.

# contract: vit-classifier
# A ViT image encoder, or classifier: a timm hub repo's checkpoint
# (safetensors, or an older repo's PyTorch state dict) and `config.json`.
# publisher: timm - pytorch-image-models (timm), Ross Wightman's.

* timm/vit_tiny_patch16_224.augreg_in21k: [39.9 MB]
  timm's `vit_tiny_patch16_224`, its `augreg_in21k` weights.

* timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k: [23.6 MB]
  timm's `vit_tiny_patch16_224`, its `augreg_in21k_ft_in1k` weights.

* timm/vit_tiny_patch16_384.augreg_in21k_ft_in1k: [23.9 MB]
  timm's `vit_tiny_patch16_384`, its `augreg_in21k_ft_in1k` weights.

* timm/vit_small_patch32_224.augreg_in21k: [124.6 MB]
  timm's `vit_small_patch32_224`, its `augreg_in21k` weights.

* timm/vit_small_patch32_224.augreg_in21k_ft_in1k: [92.3 MB]
  timm's `vit_small_patch32_224`, its `augreg_in21k_ft_in1k` weights.

* timm/vit_small_patch32_384.augreg_in21k_ft_in1k: [92.4 MB]
  timm's `vit_small_patch32_384`, its `augreg_in21k_ft_in1k` weights.

* timm/vit_small_patch16_224.augreg_in1k: [89.0 MB]
  timm's `vit_small_patch16_224`, its `augreg_in1k` weights.

* timm/vit_small_patch16_224.augreg_in21k: [121.3 MB]
  timm's `vit_small_patch16_224`, its `augreg_in21k` weights.

* timm/vit_small_patch16_224.augreg_in21k_ft_in1k: [89.0 MB]
  timm's `vit_small_patch16_224`, its `augreg_in21k_ft_in1k` weights.

* timm/vit_small_patch16_224.dino: [86.7 MB]
  timm's `vit_small_patch16_224`, its `dino` weights.

* timm/vit_small_patch16_384.augreg_in1k: [89.6 MB]
  timm's `vit_small_patch16_384`, its `augreg_in1k` weights.

* timm/vit_small_patch16_384.augreg_in21k_ft_in1k: [89.6 MB]
  timm's `vit_small_patch16_384`, its `augreg_in21k_ft_in1k` weights.

* timm/vit_small_patch8_224.dino: [86.7 MB]
  timm's `vit_small_patch8_224`, its `dino` weights.

* timm/vit_base_patch32_224.augreg_in1k: [353.7 MB]
  timm's `vit_base_patch32_224`, its `augreg_in1k` weights.

* timm/vit_base_patch32_224.augreg_in21k: [418.0 MB]
  timm's `vit_base_patch32_224`, its `augreg_in21k` weights.

* timm/vit_base_patch32_224.augreg_in21k_ft_in1k: [353.7 MB]
  timm's `vit_base_patch32_224`, its `augreg_in21k_ft_in1k` weights.

* timm/vit_base_patch32_224.orig_in21k: [349.8 MB]
  timm's `vit_base_patch32_224`, its `orig_in21k` weights.

* timm/vit_base_patch32_224.sam_in1k: [353.7 MB]
  timm's `vit_base_patch32_224`, its `sam_in1k` weights.

* timm/vit_base_patch32_384.augreg_in1k: [354.0 MB]
  timm's `vit_base_patch32_384`, its `augreg_in1k` weights.

* timm/vit_base_patch32_384.augreg_in21k_ft_in1k: [354.0 MB]
  timm's `vit_base_patch32_384`, its `augreg_in21k_ft_in1k` weights.

* timm/vit_base_patch16_224.augreg2_in21k_ft_in1k: [347.0 MB]
  timm's `vit_base_patch16_224`, its `augreg2_in21k_ft_in1k` weights.

* timm/vit_base_patch16_224.augreg_in1k: [347.0 MB]
  timm's `vit_base_patch16_224`, its `augreg_in1k` weights.

* timm/vit_base_patch16_224.augreg_in21k: [411.4 MB]
  timm's `vit_base_patch16_224`, its `augreg_in21k` weights.

* timm/vit_base_patch16_224.augreg_in21k_ft_in1k: [347.0 MB]
  timm's `vit_base_patch16_224`, its `augreg_in21k_ft_in1k` weights.

* timm/vit_base_patch16_224.dino: [343.2 MB]
  timm's `vit_base_patch16_224`, its `dino` weights.

* timm/vit_base_patch16_224.mae: [343.2 MB]
  timm's `vit_base_patch16_224`, its `mae` weights.

* timm/vit_base_patch16_224.orig_in21k: [343.2 MB]
  timm's `vit_base_patch16_224`, its `orig_in21k` weights.

* timm/vit_base_patch16_224.orig_in21k_ft_in1k: [347.0 MB]
  timm's `vit_base_patch16_224`, its `orig_in21k_ft_in1k` weights.

* timm/vit_base_patch16_224.sam_in1k: [347.0 MB]
  timm's `vit_base_patch16_224`, its `sam_in1k` weights.

* timm/vit_base_patch16_384.augreg_in1k: [348.2 MB]
  timm's `vit_base_patch16_384`, its `augreg_in1k` weights.

* timm/vit_base_patch16_384.augreg_in21k_ft_in1k: [348.2 MB]
  timm's `vit_base_patch16_384`, its `augreg_in21k_ft_in1k` weights.

* timm/vit_base_patch16_384.orig_in21k_ft_in1k: [348.2 MB]
  timm's `vit_base_patch16_384`, its `orig_in21k_ft_in1k` weights.

* timm/vit_base_patch8_224.augreg2_in21k_ft_in1k: [347.1 MB]
  timm's `vit_base_patch8_224`, its `augreg2_in21k_ft_in1k` weights.

* timm/vit_base_patch8_224.augreg_in21k: [411.4 MB]
  timm's `vit_base_patch8_224`, its `augreg_in21k` weights.

* timm/vit_base_patch8_224.augreg_in21k_ft_in1k: [347.1 MB]
  timm's `vit_base_patch8_224`, its `augreg_in21k_ft_in1k` weights.

* timm/vit_base_patch8_224.dino: [343.2 MB]
  timm's `vit_base_patch8_224`, its `dino` weights.

* timm/vit_large_patch32_224.orig_in21k: [1.22 GB]
  timm's `vit_large_patch32_224`, its `orig_in21k` weights.

* timm/vit_large_patch32_384.orig_in21k_ft_in1k: [1.23 GB]
  timm's `vit_large_patch32_384`, its `orig_in21k_ft_in1k` weights.

* timm/vit_large_patch16_224.augreg_in21k: [1.30 GB]
  timm's `vit_large_patch16_224`, its `augreg_in21k` weights.

* timm/vit_large_patch16_224.augreg_in21k_ft_in1k: [1.22 GB]
  timm's `vit_large_patch16_224`, its `augreg_in21k_ft_in1k` weights.

* timm/vit_large_patch16_224.mae: [1.21 GB]
  timm's `vit_large_patch16_224`, its `mae` weights.

* timm/vit_large_patch16_224.orig_in21k: [1.21 GB]
  timm's `vit_large_patch16_224`, its `orig_in21k` weights.

* timm/vit_large_patch16_384.augreg_in21k_ft_in1k: [1.22 GB]
  timm's `vit_large_patch16_384`, its `augreg_in21k_ft_in1k` weights.

* timm/vit_huge_patch14_224.mae: [2.52 GB]
  timm's `vit_huge_patch14_224`, its `mae` weights.

* timm/vit_huge_patch14_224.orig_in21k: [2.52 GB]
  timm's `vit_huge_patch14_224`, its `orig_in21k` weights.

* timm/vit_base_patch16_224_miil.in21k: [378.5 MB]
  timm's `vit_base_patch16_224_miil`, its `in21k` weights.

* timm/vit_base_patch16_224_miil.in21k_ft_in1k: [346.9 MB]
  timm's `vit_base_patch16_224_miil`, its `in21k_ft_in1k` weights.

* timm/vit_xsmall_patch16_clip_224.tinyclip_yfcc15m: [33.1 MB]
  timm's `vit_xsmall_patch16_clip_224`, its `tinyclip_yfcc15m` weights.

* timm/vit_medium_patch32_clip_224.tinyclip_laion400m: [158.8 MB]
  timm's `vit_medium_patch32_clip_224`, its `tinyclip_laion400m` weights.

* timm/vit_medium_patch16_clip_224.tinyclip_yfcc15m: [154.4 MB]
  timm's `vit_medium_patch16_clip_224`, its `tinyclip_yfcc15m` weights.

* timm/vit_betwixt_patch32_clip_224.tinyclip_laion400m: [245.7 MB]
  timm's `vit_betwixt_patch32_clip_224`, its `tinyclip_laion400m` weights.

* timm/vit_base_patch32_clip_224.datacompxl: [351.4 MB]
  timm's `vit_base_patch32_clip_224`, its `datacompxl` weights.

* timm/vit_base_patch32_clip_224.laion2b: [351.4 MB]
  timm's `vit_base_patch32_clip_224`, its `laion2b` weights.

* timm/vit_base_patch32_clip_224.laion2b_ft_in12k_in1k: [353.7 MB]
  timm's `vit_base_patch32_clip_224`, its `laion2b_ft_in12k_in1k` weights.

* timm/vit_base_patch32_clip_224.laion2b_ft_in1k: [353.7 MB]
  timm's `vit_base_patch32_clip_224`, its `laion2b_ft_in1k` weights.

* timm/vit_base_patch32_clip_224.laion400m_e32: [351.4 MB]
  timm's `vit_base_patch32_clip_224`, its `laion400m_e32` weights.

* timm/vit_base_patch32_clip_224.metaclip_2pt5b: [351.4 MB]
  timm's `vit_base_patch32_clip_224`, its `metaclip_2pt5b` weights.

* timm/vit_base_patch32_clip_224.metaclip_400m: [351.4 MB]
  timm's `vit_base_patch32_clip_224`, its `metaclip_400m` weights.

* timm/vit_base_patch32_clip_224.openai: [605.2 MB]
  timm's `vit_base_patch32_clip_224`, its `openai` weights.

* timm/vit_base_patch32_clip_224.openai_ft_in1k: [353.7 MB]
  timm's `vit_base_patch32_clip_224`, its `openai_ft_in1k` weights.

* timm/vit_base_patch32_clip_256.datacompxl: [351.5 MB]
  timm's `vit_base_patch32_clip_256`, its `datacompxl` weights.

* timm/vit_base_patch32_clip_384.laion2b_ft_in12k_in1k: [354.0 MB]
  timm's `vit_base_patch32_clip_384`, its `laion2b_ft_in12k_in1k` weights.

* timm/vit_base_patch32_clip_384.openai_ft_in12k_in1k: [354.0 MB]
  timm's `vit_base_patch32_clip_384`, its `openai_ft_in12k_in1k` weights.

* timm/vit_base_patch32_clip_448.laion2b_ft_in12k_in1k: [354.1 MB]
  timm's `vit_base_patch32_clip_448`, its `laion2b_ft_in12k_in1k` weights.

* timm/vit_base_patch16_clip_224.datacompxl: [344.8 MB]
  timm's `vit_base_patch16_clip_224`, its `datacompxl` weights.

* timm/vit_base_patch16_clip_224.dfn2b: [344.8 MB]
  timm's `vit_base_patch16_clip_224`, its `dfn2b` weights.

* timm/vit_base_patch16_clip_224.laion2b: [344.8 MB]
  timm's `vit_base_patch16_clip_224`, its `laion2b` weights.

* timm/vit_base_patch16_clip_224.laion2b_ft_in12k: [380.4 MB]
  timm's `vit_base_patch16_clip_224`, its `laion2b_ft_in12k` weights.

* timm/vit_base_patch16_clip_224.laion2b_ft_in12k_in1k: [347.0 MB]
  timm's `vit_base_patch16_clip_224`, its `laion2b_ft_in12k_in1k` weights.

* timm/vit_base_patch16_clip_224.laion2b_ft_in1k: [347.0 MB]
  timm's `vit_base_patch16_clip_224`, its `laion2b_ft_in1k` weights.

* timm/vit_base_patch16_clip_224.laion400m_e32: [344.8 MB]
  timm's `vit_base_patch16_clip_224`, its `laion400m_e32` weights.

* timm/vit_base_patch16_clip_224.metaclip_2pt5b: [344.8 MB]
  timm's `vit_base_patch16_clip_224`, its `metaclip_2pt5b` weights.

* timm/vit_base_patch16_clip_224.metaclip_400m: [344.8 MB]
  timm's `vit_base_patch16_clip_224`, its `metaclip_400m` weights.

* timm/vit_base_patch16_clip_224.openai: [598.6 MB]
  timm's `vit_base_patch16_clip_224`, its `openai` weights.

* timm/vit_base_patch16_clip_224.openai_ft_in12k: [380.4 MB]
  timm's `vit_base_patch16_clip_224`, its `openai_ft_in12k` weights.

* timm/vit_base_patch16_clip_224.openai_ft_in12k_in1k: [347.0 MB]
  timm's `vit_base_patch16_clip_224`, its `openai_ft_in12k_in1k` weights.

* timm/vit_base_patch16_clip_224.openai_ft_in1k: [347.0 MB]
  timm's `vit_base_patch16_clip_224`, its `openai_ft_in1k` weights.

* timm/vit_base_patch16_clip_384.laion2b_ft_in12k_in1k: [348.2 MB]
  timm's `vit_base_patch16_clip_384`, its `laion2b_ft_in12k_in1k` weights.

* timm/vit_base_patch16_clip_384.laion2b_ft_in1k: [348.2 MB]
  timm's `vit_base_patch16_clip_384`, its `laion2b_ft_in1k` weights.

* timm/vit_base_patch16_clip_384.openai_ft_in12k_in1k: [348.2 MB]
  timm's `vit_base_patch16_clip_384`, its `openai_ft_in12k_in1k` weights.

* timm/vit_base_patch16_clip_384.openai_ft_in1k: [348.2 MB]
  timm's `vit_base_patch16_clip_384`, its `openai_ft_in1k` weights.

* timm/vit_base_patch16_plus_clip_240.laion400m_e32: [468.9 MB]
  timm's `vit_base_patch16_plus_clip_240`, its `laion400m_e32` weights.

* timm/vit_large_patch14_clip_224.apple_mclip2_dfndr2b: [1.22 GB]
  timm's `vit_large_patch14_clip_224`, its `apple_mclip2_dfndr2b` weights.

* timm/vit_large_patch14_clip_224.datacompxl: [1.22 GB]
  timm's `vit_large_patch14_clip_224`, its `datacompxl` weights.

* timm/vit_large_patch14_clip_224.dfn2b: [1.22 GB]
  timm's `vit_large_patch14_clip_224`, its `dfn2b` weights.

* timm/vit_large_patch14_clip_224.dfn2b_s39b: [1.22 GB]
  timm's `vit_large_patch14_clip_224`, its `dfn2b_s39b` weights.

* timm/vit_large_patch14_clip_224.laion2b: [1.22 GB]
  timm's `vit_large_patch14_clip_224`, its `laion2b` weights.

* timm/vit_large_patch14_clip_224.laion2b_ft_in12k: [1.26 GB]
  timm's `vit_large_patch14_clip_224`, its `laion2b_ft_in12k` weights.

* timm/vit_large_patch14_clip_224.laion2b_ft_in12k_in1k: [1.22 GB]
  timm's `vit_large_patch14_clip_224`, its `laion2b_ft_in12k_in1k` weights.

* timm/vit_large_patch14_clip_224.laion2b_ft_in1k: [1.22 GB]
  timm's `vit_large_patch14_clip_224`, its `laion2b_ft_in1k` weights.

* timm/vit_large_patch14_clip_224.laion400m_e32: [1.22 GB]
  timm's `vit_large_patch14_clip_224`, its `laion400m_e32` weights.

* timm/vit_large_patch14_clip_224.metaclip_2pt5b: [1.22 GB]
  timm's `vit_large_patch14_clip_224`, its `metaclip_2pt5b` weights.

* timm/vit_large_patch14_clip_224.metaclip_400m: [1.22 GB]
  timm's `vit_large_patch14_clip_224`, its `metaclip_400m` weights.

* timm/vit_large_patch14_clip_224.openai: [1.71 GB]
  timm's `vit_large_patch14_clip_224`, its `openai` weights.

* timm/vit_large_patch14_clip_224.openai_ft_in12k: [1.26 GB]
  timm's `vit_large_patch14_clip_224`, its `openai_ft_in12k` weights.

* timm/vit_large_patch14_clip_224.openai_ft_in12k_in1k: [1.22 GB]
  timm's `vit_large_patch14_clip_224`, its `openai_ft_in12k_in1k` weights.

* timm/vit_large_patch14_clip_224.openai_ft_in1k: [1.22 GB]
  timm's `vit_large_patch14_clip_224`, its `openai_ft_in1k` weights.

* timm/vit_large_patch14_clip_336.datacompxl_ft_augreg_inat21: [1.26 GB]
  timm's `vit_large_patch14_clip_336`, its `datacompxl_ft_augreg_inat21`
  weights.

* timm/vit_large_patch14_clip_336.datacompxl_ft_inat21: [1.26 GB]
  timm's `vit_large_patch14_clip_336`, its `datacompxl_ft_inat21` weights.

* timm/vit_large_patch14_clip_336.laion2b_ft_augreg_inat21: [1.26 GB]
  timm's `vit_large_patch14_clip_336`, its `laion2b_ft_augreg_inat21`
  weights.

* timm/vit_large_patch14_clip_336.laion2b_ft_in12k_in1k: [1.22 GB]
  timm's `vit_large_patch14_clip_336`, its `laion2b_ft_in12k_in1k` weights.

* timm/vit_large_patch14_clip_336.laion2b_ft_in12k_in1k_inat21: [1.26 GB]
  timm's `vit_large_patch14_clip_336`, its `laion2b_ft_in12k_in1k_inat21`
  weights.

* timm/vit_large_patch14_clip_336.laion2b_ft_in1k: [1.22 GB]
  timm's `vit_large_patch14_clip_336`, its `laion2b_ft_in1k` weights.

* timm/vit_large_patch14_clip_336.openai: [1.22 GB]
  timm's `vit_large_patch14_clip_336`, its `openai` weights.

* timm/vit_large_patch14_clip_336.openai_ft_in12k_in1k: [1.22 GB]
  timm's `vit_large_patch14_clip_336`, its `openai_ft_in12k_in1k` weights.

* timm/vit_huge_patch14_clip_224.dfn5b: [2.53 GB]
  timm's `vit_huge_patch14_clip_224`, its `dfn5b` weights.

* timm/vit_huge_patch14_clip_224.laion2b: [2.53 GB]
  timm's `vit_huge_patch14_clip_224`, its `laion2b` weights.

* timm/vit_huge_patch14_clip_224.laion2b_ft_in12k: [2.58 GB]
  timm's `vit_huge_patch14_clip_224`, its `laion2b_ft_in12k` weights.

* timm/vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k: [2.53 GB]
  timm's `vit_huge_patch14_clip_224`, its `laion2b_ft_in12k_in1k` weights.

* timm/vit_huge_patch14_clip_224.laion2b_ft_in1k: [2.53 GB]
  timm's `vit_huge_patch14_clip_224`, its `laion2b_ft_in1k` weights.

* timm/vit_huge_patch14_clip_224.metaclip2_worldwide: [2.53 GB]
  timm's `vit_huge_patch14_clip_224`, its `metaclip2_worldwide` weights.

* timm/vit_huge_patch14_clip_224.metaclip_2pt5b: [2.53 GB]
  timm's `vit_huge_patch14_clip_224`, its `metaclip_2pt5b` weights.

* timm/vit_huge_patch14_clip_224.metaclip_altogether: [2.53 GB]
  timm's `vit_huge_patch14_clip_224`, its `metaclip_altogether` weights.

* timm/vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k: [2.53 GB]
  timm's `vit_huge_patch14_clip_336`, its `laion2b_ft_in12k_in1k` weights.

* timm/vit_huge_patch14_clip_378.dfn5b: [2.53 GB]
  timm's `vit_huge_patch14_clip_378`, its `dfn5b` weights.

* timm/vit_huge_patch14_clip_378.metaclip2_worldwide: [2.53 GB]
  timm's `vit_huge_patch14_clip_378`, its `metaclip2_worldwide` weights.

* timm/vit_giant_patch14_clip_224.laion2b: [4.05 GB]
  timm's `vit_giant_patch14_clip_224`, its `laion2b` weights.

* timm/vit_gigantic_patch14_clip_224.laion2b: [7.38 GB]
  timm's `vit_gigantic_patch14_clip_224`, its `laion2b` weights.

* timm/vit_gigantic_patch14_clip_224.metaclip2_worldwide: [7.38 GB]
  timm's `vit_gigantic_patch14_clip_224`, its `metaclip2_worldwide`
  weights.

* timm/vit_gigantic_patch14_clip_224.metaclip_2pt5b: [7.38 GB]
  timm's `vit_gigantic_patch14_clip_224`, its `metaclip_2pt5b` weights.

* timm/vit_gigantic_patch14_clip_378.metaclip2_worldwide: [7.38 GB]
  timm's `vit_gigantic_patch14_clip_378`, its `metaclip2_worldwide`
  weights.

* timm/aimv2_large_patch14_224.apple_pt: [1.24 GB]
  timm's `aimv2_large_patch14_224`, its `apple_pt` weights.

* timm/aimv2_large_patch14_224.apple_pt_dist: [1.24 GB]
  timm's `aimv2_large_patch14_224`, its `apple_pt_dist` weights.

* timm/aimv2_large_patch14_336.apple_pt: [1.24 GB]
  timm's `aimv2_large_patch14_336`, its `apple_pt` weights.

* timm/aimv2_large_patch14_336.apple_pt_dist: [1.24 GB]
  timm's `aimv2_large_patch14_336`, its `apple_pt_dist` weights.

* timm/aimv2_large_patch14_448.apple_pt: [1.24 GB]
  timm's `aimv2_large_patch14_448`, its `apple_pt` weights.

* timm/aimv2_huge_patch14_224.apple_pt: [2.72 GB]
  timm's `aimv2_huge_patch14_224`, its `apple_pt` weights.

* timm/aimv2_huge_patch14_336.apple_pt: [2.73 GB]
  timm's `aimv2_huge_patch14_336`, its `apple_pt` weights.

* timm/aimv2_huge_patch14_448.apple_pt: [2.73 GB]
  timm's `aimv2_huge_patch14_448`, its `apple_pt` weights.

* timm/aimv2_1b_patch14_224.apple_pt: [4.94 GB]
  timm's `aimv2_1b_patch14_224`, its `apple_pt` weights.

* timm/aimv2_1b_patch14_336.apple_pt: [4.94 GB]
  timm's `aimv2_1b_patch14_336`, its `apple_pt` weights.

* timm/aimv2_1b_patch14_448.apple_pt: [4.95 GB]
  timm's `aimv2_1b_patch14_448`, its `apple_pt` weights.

* timm/aimv2_3b_patch14_224.apple_pt: [10.88 GB]
  timm's `aimv2_3b_patch14_224`, its `apple_pt` weights.

* timm/aimv2_3b_patch14_336.apple_pt: [10.89 GB]
  timm's `aimv2_3b_patch14_336`, its `apple_pt` weights.

* timm/aimv2_3b_patch14_448.apple_pt: [10.89 GB]
  timm's `aimv2_3b_patch14_448`, its `apple_pt` weights.

* timm/beit3_base_patch16_224.in22k_ft_in1k: [347.4 MB]
  timm's `beit3_base_patch16_224`, its `in22k_ft_in1k` weights.

* timm/beit3_base_patch16_224.indomain_in22k_ft_in1k: [347.4 MB]
  timm's `beit3_base_patch16_224`, its `indomain_in22k_ft_in1k` weights.

* timm/beit3_base_patch16_224.indomain_pt: [343.6 MB]
  timm's `beit3_base_patch16_224`, its `indomain_pt` weights.

* timm/beit3_base_patch16_224.pt: [343.6 MB]
  timm's `beit3_base_patch16_224`, its `pt` weights.

* timm/beit3_large_patch16_224.in22k_ft_in1k: [1.22 GB]
  timm's `beit3_large_patch16_224`, its `in22k_ft_in1k` weights.

* timm/beit3_large_patch16_224.indomain_in22k_ft_in1k: [1.22 GB]
  timm's `beit3_large_patch16_224`, its `indomain_in22k_ft_in1k` weights.

* timm/beit3_large_patch16_224.indomain_pt: [1.21 GB]
  timm's `beit3_large_patch16_224`, its `indomain_pt` weights.

* timm/beit3_large_patch16_224.pt: [1.21 GB]
  timm's `beit3_large_patch16_224`, its `pt` weights.

* timm/eva_large_patch14_196.in22k_ft_in1k: [1.22 GB]
  timm's `eva_large_patch14_196`, its `in22k_ft_in1k` weights.

* timm/eva_large_patch14_196.in22k_ft_in22k_in1k: [1.22 GB]
  timm's `eva_large_patch14_196`, its `in22k_ft_in22k_in1k` weights.

* timm/eva_large_patch14_336.in22k_ft_in1k: [1.22 GB]
  timm's `eva_large_patch14_336`, its `in22k_ft_in1k` weights.

* timm/eva_large_patch14_336.in22k_ft_in22k_in1k: [1.22 GB]
  timm's `eva_large_patch14_336`, its `in22k_ft_in22k_in1k` weights.

* timm/flexivit_small.1200ep_in1k: [89.0 MB]
  timm's `flexivit_small`, its `1200ep_in1k` weights.

* timm/flexivit_small.300ep_in1k: [89.0 MB]
  timm's `flexivit_small`, its `300ep_in1k` weights.

* timm/flexivit_small.600ep_in1k: [89.0 MB]
  timm's `flexivit_small`, its `600ep_in1k` weights.

* timm/flexivit_base.1000ep_in21k: [411.4 MB]
  timm's `flexivit_base`, its `1000ep_in21k` weights.

* timm/flexivit_base.1200ep_in1k: [347.1 MB]
  timm's `flexivit_base`, its `1200ep_in1k` weights.

* timm/flexivit_base.300ep_in1k: [347.1 MB]
  timm's `flexivit_base`, its `300ep_in1k` weights.

* timm/flexivit_base.300ep_in21k: [411.4 MB]
  timm's `flexivit_base`, its `300ep_in21k` weights.

* timm/flexivit_base.600ep_in1k: [347.1 MB]
  timm's `flexivit_base`, its `600ep_in1k` weights.

* timm/flexivit_base.patch16_in21k: [411.4 MB]
  timm's `flexivit_base`, its `patch16_in21k` weights.

* timm/flexivit_base.patch30_in21k: [411.4 MB]
  timm's `flexivit_base`, its `patch30_in21k` weights.

* timm/flexivit_large.1200ep_in1k: [1.22 GB]
  timm's `flexivit_large`, its `1200ep_in1k` weights.

* timm/flexivit_large.300ep_in1k: [1.22 GB]
  timm's `flexivit_large`, its `300ep_in1k` weights.

* timm/flexivit_large.600ep_in1k: [1.22 GB]
  timm's `flexivit_large`, its `600ep_in1k` weights.

* timm/vit_base_patch32_clip_quickgelu_224.laion400m_e32: [351.4 MB]
  timm's `vit_base_patch32_clip_quickgelu_224`, its `laion400m_e32`
  weights.

* timm/vit_base_patch32_clip_quickgelu_224.metaclip_2pt5b: [351.4 MB]
  timm's `vit_base_patch32_clip_quickgelu_224`, its `metaclip_2pt5b`
  weights.

* timm/vit_base_patch32_clip_quickgelu_224.metaclip_400m: [351.4 MB]
  timm's `vit_base_patch32_clip_quickgelu_224`, its `metaclip_400m`
  weights.

* timm/vit_base_patch16_clip_quickgelu_224.metaclip_2pt5b: [344.8 MB]
  timm's `vit_base_patch16_clip_quickgelu_224`, its `metaclip_2pt5b`
  weights.

* timm/vit_base_patch16_clip_quickgelu_224.metaclip_400m: [344.8 MB]
  timm's `vit_base_patch16_clip_quickgelu_224`, its `metaclip_400m`
  weights.

* timm/vit_large_patch14_clip_quickgelu_224.dfn2b: [1.22 GB]
  timm's `vit_large_patch14_clip_quickgelu_224`, its `dfn2b` weights.

* timm/vit_large_patch14_clip_quickgelu_224.metaclip_2pt5b: [1.22 GB]
  timm's `vit_large_patch14_clip_quickgelu_224`, its `metaclip_2pt5b`
  weights.

* timm/vit_large_patch14_clip_quickgelu_224.metaclip_400m: [1.22 GB]
  timm's `vit_large_patch14_clip_quickgelu_224`, its `metaclip_400m`
  weights.

* timm/vit_large_patch14_clip_quickgelu_336.openai: [1.22 GB]
  timm's `vit_large_patch14_clip_quickgelu_336`, its `openai` weights.

* timm/vit_huge_patch14_clip_quickgelu_224.dfn5b: [2.53 GB]
  timm's `vit_huge_patch14_clip_quickgelu_224`, its `dfn5b` weights.

* timm/vit_huge_patch14_clip_quickgelu_224.metaclip2_worldwide: [2.53 GB]
  timm's `vit_huge_patch14_clip_quickgelu_224`, its `metaclip2_worldwide`
  weights.

* timm/vit_huge_patch14_clip_quickgelu_224.metaclip_2pt5b: [2.53 GB]
  timm's `vit_huge_patch14_clip_quickgelu_224`, its `metaclip_2pt5b`
  weights.

* timm/vit_huge_patch14_clip_quickgelu_378.dfn5b: [2.53 GB]
  timm's `vit_huge_patch14_clip_quickgelu_378`, its `dfn5b` weights.

* timm/vit_gigantic_patch14_clip_quickgelu_224.metaclip_2pt5b: [7.38 GB]
  timm's `vit_gigantic_patch14_clip_quickgelu_224`, its `metaclip_2pt5b`
  weights.

* timm/vit_small_patch14_dinov2.lvd142m: [88.2 MB]
  timm's `vit_small_patch14_dinov2`, its `lvd142m` weights.

* timm/vit_base_patch14_dinov2.lvd142m: [346.3 MB]
  timm's `vit_base_patch14_dinov2`, its `lvd142m` weights.

* timm/vit_large_patch14_dinov2.lvd142m: [1.22 GB]
  timm's `vit_large_patch14_dinov2`, its `lvd142m` weights.

* timm/vit_giant_patch14_dinov2.lvd142m: [4.55 GB]
  timm's `vit_giant_patch14_dinov2`, its `lvd142m` weights.

* timm/vit_medium_patch16_gap_240.sw_in12k: [178.5 MB]
  timm's `vit_medium_patch16_gap_240`, its `sw_in12k` weights.

* timm/vit_medium_patch16_gap_256.sw_in12k_ft_in1k: [156.2 MB]
  timm's `vit_medium_patch16_gap_256`, its `sw_in12k_ft_in1k` weights.

* timm/vit_medium_patch16_gap_384.sw_in12k_ft_in1k: [156.9 MB]
  timm's `vit_medium_patch16_gap_384`, its `sw_in12k_ft_in1k` weights.

* timm/vit_base_patch16_radio_v3.c_radio: [345.8 MB]
  timm's `vit_base_patch16_radio_v3`, its `c_radio` weights.

* timm/vit_large_patch16_radio_v3.c_radio: [1.22 GB]
  timm's `vit_large_patch16_radio_v3`, its `c_radio` weights.

* timm/vit_huge_patch16_radio_v3.c_radio: [2.53 GB]
  timm's `vit_huge_patch16_radio_v3`, its `c_radio` weights.

* timm/vit_giant_patch16_radio_v3.c_radio: [4.54 GB]
  timm's `vit_giant_patch16_radio_v3`, its `c_radio` weights.

* timm/vit_so400m_patch16_radio_v4.c_radio: [1.65 GB]
  timm's `vit_so400m_patch16_radio_v4`, its `c_radio` weights.

* timm/vit_huge_patch16_radio_v4.c_radio: [2.53 GB]
  timm's `vit_huge_patch16_radio_v4`, its `c_radio` weights.

* timm/vit_wee_patch16_reg1_gap_256.sbb_in1k: [54.5 MB]
  timm's `vit_wee_patch16_reg1_gap_256`, its `sbb_in1k` weights.

* timm/vit_little_patch16_reg1_gap_256.sbb_in12k: [104.8 MB]
  timm's `vit_little_patch16_reg1_gap_256`, its `sbb_in12k` weights.

* timm/vit_little_patch16_reg1_gap_256.sbb_in12k_ft_in1k: [90.8 MB]
  timm's `vit_little_patch16_reg1_gap_256`, its `sbb_in12k_ft_in1k`
  weights.

* timm/vit_medium_patch16_reg1_gap_256.sbb_in1k: [156.3 MB]
  timm's `vit_medium_patch16_reg1_gap_256`, its `sbb_in1k` weights.

* timm/vit_betwixt_patch16_reg1_gap_256.sbb_in1k: [242.4 MB]
  timm's `vit_betwixt_patch16_reg1_gap_256`, its `sbb_in1k` weights.

* timm/vit_so150m2_patch16_reg1_gap_256.sbb_e200_in12k: [581.2 MB]
  timm's `vit_so150m2_patch16_reg1_gap_256`, its `sbb_e200_in12k` weights.

* timm/vit_so150m2_patch16_reg1_gap_256.sbb_e200_in12k_ft_in1k: [545.0 MB]
  timm's `vit_so150m2_patch16_reg1_gap_256`, its `sbb_e200_in12k_ft_in1k`
  weights.

* timm/vit_so150m2_patch16_reg1_gap_384.sbb_e200_in12k_ft_in1k: [546.1 MB]
  timm's `vit_so150m2_patch16_reg1_gap_384`, its `sbb_e200_in12k_ft_in1k`
  weights.

* timm/vit_so150m2_patch16_reg1_gap_448.sbb_e200_in12k_ft_in1k: [546.8 MB]
  timm's `vit_so150m2_patch16_reg1_gap_448`, its `sbb_e200_in12k_ft_in1k`
  weights.

* timm/vit_base_patch14_reg1_tipsv2.webli: [345.3 MB]
  timm's `vit_base_patch14_reg1_tipsv2`, its `webli` weights.

* timm/vit_large_patch14_reg1_tipsv2.webli: [1.22 GB]
  timm's `vit_large_patch14_reg1_tipsv2`, its `webli` weights.

* timm/vit_so400m_patch14_reg1_tipsv2.webli: [1.65 GB]
  timm's `vit_so400m_patch14_reg1_tipsv2`, its `webli` weights.

* timm/vit_giant_patch14_reg1_tipsv2.webli: [4.54 GB]
  timm's `vit_giant_patch14_reg1_tipsv2`, its `webli` weights.

* timm/vit_small_patch14_reg4_dinov2.lvd142m: [88.2 MB]
  timm's `vit_small_patch14_reg4_dinov2`, its `lvd142m` weights.

* timm/vit_base_patch14_reg4_dinov2.lvd142m: [346.3 MB]
  timm's `vit_base_patch14_reg4_dinov2`, its `lvd142m` weights.

* timm/vit_large_patch14_reg4_dinov2.lvd142m: [1.22 GB]
  timm's `vit_large_patch14_reg4_dinov2`, its `lvd142m` weights.

* timm/vit_giant_patch14_reg4_dinov2.lvd142m: [4.55 GB]
  timm's `vit_giant_patch14_reg4_dinov2`, its `lvd142m` weights.

* timm/vit_little_patch16_reg4_gap_256.sbb_in1k: [90.8 MB]
  timm's `vit_little_patch16_reg4_gap_256`, its `sbb_in1k` weights.

* timm/vit_medium_patch16_reg4_gap_256.sbb_in12k: [178.6 MB]
  timm's `vit_medium_patch16_reg4_gap_256`, its `sbb_in12k` weights.

* timm/vit_medium_patch16_reg4_gap_256.sbb_in12k_ft_in1k: [156.3 MB]
  timm's `vit_medium_patch16_reg4_gap_256`, its `sbb_in12k_ft_in1k`
  weights.

* timm/vit_medium_patch16_reg4_gap_256.sbb_in1k: [156.3 MB]
  timm's `vit_medium_patch16_reg4_gap_256`, its `sbb_in1k` weights.

* timm/vit_mediumd_patch16_reg4_gap_256.sbb2_e200_in12k: [279.5 MB]
  timm's `vit_mediumd_patch16_reg4_gap_256`, its `sbb2_e200_in12k` weights.

* timm/vit_mediumd_patch16_reg4_gap_256.sbb2_e200_in12k_ft_in1k: [257.2 MB]
  timm's `vit_mediumd_patch16_reg4_gap_256`, its `sbb2_e200_in12k_ft_in1k`
  weights.

* timm/vit_mediumd_patch16_reg4_gap_256.sbb_in12k: [279.5 MB]
  timm's `vit_mediumd_patch16_reg4_gap_256`, its `sbb_in12k` weights.

* timm/vit_mediumd_patch16_reg4_gap_256.sbb_in12k_ft_in1k: [257.2 MB]
  timm's `vit_mediumd_patch16_reg4_gap_256`, its `sbb_in12k_ft_in1k`
  weights.

* timm/vit_mediumd_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k: [257.9 MB]
  timm's `vit_mediumd_patch16_reg4_gap_384`, its `sbb2_e200_in12k_ft_in1k`
  weights.

* timm/vit_betwixt_patch16_reg4_gap_256.sbb2_e200_in12k: [270.2 MB]
  timm's `vit_betwixt_patch16_reg4_gap_256`, its `sbb2_e200_in12k` weights.

* timm/vit_betwixt_patch16_reg4_gap_256.sbb2_e200_in12k_ft_in1k: [242.4 MB]
  timm's `vit_betwixt_patch16_reg4_gap_256`, its `sbb2_e200_in12k_ft_in1k`
  weights.

* timm/vit_betwixt_patch16_reg4_gap_256.sbb_in12k: [270.2 MB]
  timm's `vit_betwixt_patch16_reg4_gap_256`, its `sbb_in12k` weights.

* timm/vit_betwixt_patch16_reg4_gap_256.sbb_in12k_ft_in1k: [242.4 MB]
  timm's `vit_betwixt_patch16_reg4_gap_256`, its `sbb_in12k_ft_in1k`
  weights.

* timm/vit_betwixt_patch16_reg4_gap_256.sbb_in1k: [242.4 MB]
  timm's `vit_betwixt_patch16_reg4_gap_256`, its `sbb_in1k` weights.

* timm/vit_betwixt_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k: [243.2 MB]
  timm's `vit_betwixt_patch16_reg4_gap_384`, its `sbb2_e200_in12k_ft_in1k`
  weights.

* timm/vit_so150m_patch16_reg4_gap_256.sbb_e250_in12k: [576.2 MB]
  timm's `vit_so150m_patch16_reg4_gap_256`, its `sbb_e250_in12k` weights.

* timm/vit_so150m_patch16_reg4_gap_256.sbb_e250_in12k_ft_in1k: [537.3 MB]
  timm's `vit_so150m_patch16_reg4_gap_256`, its `sbb_e250_in12k_ft_in1k`
  weights.

* timm/vit_so150m_patch16_reg4_gap_384.sbb_e250_in12k_ft_in1k: [538.4 MB]
  timm's `vit_so150m_patch16_reg4_gap_384`, its `sbb_e250_in12k_ft_in1k`
  weights.

* timm/vit_base_patch16_rpn_224.sw_in1k: [346.9 MB]
  timm's `vit_base_patch16_rpn_224`, its `sw_in1k` weights.

* timm/vit_base_patch32_siglip_256.v2_webli: [378.2 MB]
  timm's `vit_base_patch32_siglip_256`, its `v2_webli` weights.

* timm/vit_base_patch16_siglip_224.v2_webli: [371.6 MB]
  timm's `vit_base_patch16_siglip_224`, its `v2_webli` weights.

* timm/vit_base_patch16_siglip_224.webli: [371.6 MB]
  timm's `vit_base_patch16_siglip_224`, its `webli` weights.

* timm/vit_base_patch16_siglip_256.v2_webli: [371.7 MB]
  timm's `vit_base_patch16_siglip_256`, its `v2_webli` weights.

* timm/vit_base_patch16_siglip_256.webli: [371.7 MB]
  timm's `vit_base_patch16_siglip_256`, its `webli` weights.

* timm/vit_base_patch16_siglip_256.webli_i18n: [371.7 MB]
  timm's `vit_base_patch16_siglip_256`, its `webli_i18n` weights.

* timm/vit_base_patch16_siglip_384.v2_webli: [372.7 MB]
  timm's `vit_base_patch16_siglip_384`, its `v2_webli` weights.

* timm/vit_base_patch16_siglip_384.webli: [372.7 MB]
  timm's `vit_base_patch16_siglip_384`, its `webli` weights.

* timm/vit_base_patch16_siglip_512.v2_webli: [374.1 MB]
  timm's `vit_base_patch16_siglip_512`, its `v2_webli` weights.

* timm/vit_base_patch16_siglip_512.webli: [374.1 MB]
  timm's `vit_base_patch16_siglip_512`, its `webli` weights.

* timm/vit_large_patch16_siglip_256.v2_webli: [1.26 GB]
  timm's `vit_large_patch16_siglip_256`, its `v2_webli` weights.

* timm/vit_large_patch16_siglip_256.webli: [1.26 GB]
  timm's `vit_large_patch16_siglip_256`, its `webli` weights.

* timm/vit_large_patch16_siglip_384.v2_webli: [1.27 GB]
  timm's `vit_large_patch16_siglip_384`, its `v2_webli` weights.

* timm/vit_large_patch16_siglip_384.webli: [1.27 GB]
  timm's `vit_large_patch16_siglip_384`, its `webli` weights.

* timm/vit_large_patch16_siglip_512.v2_webli: [1.27 GB]
  timm's `vit_large_patch16_siglip_512`, its `v2_webli` weights.

* timm/vit_so400m_patch16_siglip_256.v2_webli: [1.71 GB]
  timm's `vit_so400m_patch16_siglip_256`, its `v2_webli` weights.

* timm/vit_so400m_patch16_siglip_256.webli_i18n: [1.71 GB]
  timm's `vit_so400m_patch16_siglip_256`, its `webli_i18n` weights.

* timm/vit_so400m_patch16_siglip_384.v2_webli: [1.71 GB]
  timm's `vit_so400m_patch16_siglip_384`, its `v2_webli` weights.

* timm/vit_so400m_patch16_siglip_512.v2_webli: [1.72 GB]
  timm's `vit_so400m_patch16_siglip_512`, its `v2_webli` weights.

* timm/vit_so400m_patch14_siglip_224.v2_webli: [1.71 GB]
  timm's `vit_so400m_patch14_siglip_224`, its `v2_webli` weights.

* timm/vit_so400m_patch14_siglip_224.webli: [1.71 GB]
  timm's `vit_so400m_patch14_siglip_224`, its `webli` weights.

* timm/vit_so400m_patch14_siglip_378.v2_webli: [1.71 GB]
  timm's `vit_so400m_patch14_siglip_378`, its `v2_webli` weights.

* timm/vit_so400m_patch14_siglip_378.webli: [1.71 GB]
  timm's `vit_so400m_patch14_siglip_378`, its `webli` weights.

* timm/vit_so400m_patch14_siglip_378.webli_ft_in1k: [1.72 GB]
  timm's `vit_so400m_patch14_siglip_378`, its `webli_ft_in1k` weights.

* timm/vit_so400m_patch14_siglip_384.webli: [1.71 GB]
  timm's `vit_so400m_patch14_siglip_384`, its `webli` weights.

* timm/vit_giantopt_patch16_siglip_256.v2_webli: [4.65 GB]
  timm's `vit_giantopt_patch16_siglip_256`, its `v2_webli` weights.

* timm/vit_giantopt_patch16_siglip_384.v2_webli: [4.65 GB]
  timm's `vit_giantopt_patch16_siglip_384`, its `v2_webli` weights.

* timm/vit_base_patch32_siglip_gap_256.v2_webli: [349.9 MB]
  timm's `vit_base_patch32_siglip_gap_256`, its `v2_webli` weights.

* timm/vit_base_patch16_siglip_gap_224.v2_webli: [343.2 MB]
  timm's `vit_base_patch16_siglip_gap_224`, its `v2_webli` weights.

* timm/vit_base_patch16_siglip_gap_224.webli: [343.2 MB]
  timm's `vit_base_patch16_siglip_gap_224`, its `webli` weights.

* timm/vit_base_patch16_siglip_gap_256.v2_webli: [343.4 MB]
  timm's `vit_base_patch16_siglip_gap_256`, its `v2_webli` weights.

* timm/vit_base_patch16_siglip_gap_256.webli: [343.4 MB]
  timm's `vit_base_patch16_siglip_gap_256`, its `webli` weights.

* timm/vit_base_patch16_siglip_gap_256.webli_i18n: [343.4 MB]
  timm's `vit_base_patch16_siglip_gap_256`, its `webli_i18n` weights.

* timm/vit_base_patch16_siglip_gap_384.v2_webli: [344.4 MB]
  timm's `vit_base_patch16_siglip_gap_384`, its `v2_webli` weights.

* timm/vit_base_patch16_siglip_gap_384.webli: [344.4 MB]
  timm's `vit_base_patch16_siglip_gap_384`, its `webli` weights.

* timm/vit_base_patch16_siglip_gap_512.v2_webli: [345.7 MB]
  timm's `vit_base_patch16_siglip_gap_512`, its `v2_webli` weights.

* timm/vit_base_patch16_siglip_gap_512.webli: [345.7 MB]
  timm's `vit_base_patch16_siglip_gap_512`, its `webli` weights.

* timm/vit_large_patch16_siglip_gap_256.v2_webli: [1.21 GB]
  timm's `vit_large_patch16_siglip_gap_256`, its `v2_webli` weights.

* timm/vit_large_patch16_siglip_gap_256.webli: [1.21 GB]
  timm's `vit_large_patch16_siglip_gap_256`, its `webli` weights.

* timm/vit_large_patch16_siglip_gap_384.v2_webli: [1.21 GB]
  timm's `vit_large_patch16_siglip_gap_384`, its `v2_webli` weights.

* timm/vit_large_patch16_siglip_gap_384.webli: [1.21 GB]
  timm's `vit_large_patch16_siglip_gap_384`, its `webli` weights.

* timm/vit_large_patch16_siglip_gap_512.v2_webli: [1.22 GB]
  timm's `vit_large_patch16_siglip_gap_512`, its `v2_webli` weights.

* timm/vit_so400m_patch16_siglip_gap_256.v2_webli: [1.65 GB]
  timm's `vit_so400m_patch16_siglip_gap_256`, its `v2_webli` weights.

* timm/vit_so400m_patch16_siglip_gap_256.webli_i18n: [1.65 GB]
  timm's `vit_so400m_patch16_siglip_gap_256`, its `webli_i18n` weights.

* timm/vit_so400m_patch16_siglip_gap_384.v2_webli: [1.65 GB]
  timm's `vit_so400m_patch16_siglip_gap_384`, its `v2_webli` weights.

* timm/vit_so400m_patch16_siglip_gap_512.v2_webli: [1.65 GB]
  timm's `vit_so400m_patch16_siglip_gap_512`, its `v2_webli` weights.

* timm/vit_so400m_patch14_siglip_gap_224.pali2_10b_pt: [1.65 GB]
  timm's `vit_so400m_patch14_siglip_gap_224`, its `pali2_10b_pt` weights.

* timm/vit_so400m_patch14_siglip_gap_224.pali2_3b_pt: [1.65 GB]
  timm's `vit_so400m_patch14_siglip_gap_224`, its `pali2_3b_pt` weights.

* timm/vit_so400m_patch14_siglip_gap_224.pali_mix: [1.65 GB]
  timm's `vit_so400m_patch14_siglip_gap_224`, its `pali_mix` weights.

* timm/vit_so400m_patch14_siglip_gap_224.pali_pt: [1.65 GB]
  timm's `vit_so400m_patch14_siglip_gap_224`, its `pali_pt` weights.

* timm/vit_so400m_patch14_siglip_gap_224.v2_webli: [1.65 GB]
  timm's `vit_so400m_patch14_siglip_gap_224`, its `v2_webli` weights.

* timm/vit_so400m_patch14_siglip_gap_224.webli: [1.65 GB]
  timm's `vit_so400m_patch14_siglip_gap_224`, its `webli` weights.

* timm/vit_so400m_patch14_siglip_gap_378.v2_webli: [1.65 GB]
  timm's `vit_so400m_patch14_siglip_gap_378`, its `v2_webli` weights.

* timm/vit_so400m_patch14_siglip_gap_378.webli: [1.65 GB]
  timm's `vit_so400m_patch14_siglip_gap_378`, its `webli` weights.

* timm/vit_so400m_patch14_siglip_gap_378.webli_ft_in1k: [1.66 GB]
  timm's `vit_so400m_patch14_siglip_gap_378`, its `webli_ft_in1k` weights.

* timm/vit_so400m_patch14_siglip_gap_384.webli: [1.65 GB]
  timm's `vit_so400m_patch14_siglip_gap_384`, its `webli` weights.

* timm/vit_so400m_patch14_siglip_gap_448.pali2_10b_docci: [1.65 GB]
  timm's `vit_so400m_patch14_siglip_gap_448`, its `pali2_10b_docci`
  weights.

* timm/vit_so400m_patch14_siglip_gap_448.pali2_10b_pt: [1.65 GB]
  timm's `vit_so400m_patch14_siglip_gap_448`, its `pali2_10b_pt` weights.

* timm/vit_so400m_patch14_siglip_gap_448.pali2_3b_docci: [1.65 GB]
  timm's `vit_so400m_patch14_siglip_gap_448`, its `pali2_3b_docci` weights.

* timm/vit_so400m_patch14_siglip_gap_448.pali2_3b_pt: [1.65 GB]
  timm's `vit_so400m_patch14_siglip_gap_448`, its `pali2_3b_pt` weights.

* timm/vit_so400m_patch14_siglip_gap_448.pali_mix: [1.65 GB]
  timm's `vit_so400m_patch14_siglip_gap_448`, its `pali_mix` weights.

* timm/vit_so400m_patch14_siglip_gap_448.pali_ocrvqa: [1.65 GB]
  timm's `vit_so400m_patch14_siglip_gap_448`, its `pali_ocrvqa` weights.

* timm/vit_so400m_patch14_siglip_gap_448.pali_pt: [1.65 GB]
  timm's `vit_so400m_patch14_siglip_gap_448`, its `pali_pt` weights.

* timm/vit_so400m_patch14_siglip_gap_448.pali_refcoco_seg: [1.65 GB]
  timm's `vit_so400m_patch14_siglip_gap_448`, its `pali_refcoco_seg`
  weights.

* timm/vit_so400m_patch14_siglip_gap_896.pali2_10b_pt: [1.67 GB]
  timm's `vit_so400m_patch14_siglip_gap_896`, its `pali2_10b_pt` weights.

* timm/vit_so400m_patch14_siglip_gap_896.pali2_3b_pt: [1.67 GB]
  timm's `vit_so400m_patch14_siglip_gap_896`, its `pali2_3b_pt` weights.

* timm/vit_so400m_patch14_siglip_gap_896.pali_ocrvqa: [1.67 GB]
  timm's `vit_so400m_patch14_siglip_gap_896`, its `pali_ocrvqa` weights.

* timm/vit_so400m_patch14_siglip_gap_896.pali_pt: [1.67 GB]
  timm's `vit_so400m_patch14_siglip_gap_896`, its `pali_pt` weights.

* timm/vit_so400m_patch14_siglip_gap_896.pali_refcoco_seg: [1.67 GB]
  timm's `vit_so400m_patch14_siglip_gap_896`, its `pali_refcoco_seg`
  weights.

* timm/vit_giantopt_patch16_siglip_gap_256.v2_webli: [4.54 GB]
  timm's `vit_giantopt_patch16_siglip_gap_256`, its `v2_webli` weights.

* timm/vit_giantopt_patch16_siglip_gap_384.v2_webli: [4.54 GB]
  timm's `vit_giantopt_patch16_siglip_gap_384`, its `v2_webli` weights.

# contract: deit-classifier
# A DeiT image classifier: a timm hub repo's checkpoint (safetensors, or an
# older repo's PyTorch state dict) and `config.json`.
# publisher: timm - pytorch-image-models (timm), Ross Wightman's.

* timm/deit_tiny_patch16_224.fb_in1k: [23.6 MB]
  timm's `deit_tiny_patch16_224`, its `fb_in1k` weights.

* timm/deit_small_patch16_224.fb_in1k: [89.0 MB]
  timm's `deit_small_patch16_224`, its `fb_in1k` weights.

* timm/deit_base_patch16_224.fb_in1k: [347.0 MB]
  timm's `deit_base_patch16_224`, its `fb_in1k` weights.

* timm/deit_base_patch16_384.fb_in1k: [348.2 MB]
  timm's `deit_base_patch16_384`, its `fb_in1k` weights.

* timm/deit_tiny_distilled_patch16_224.fb_in1k: [24.4 MB]
  timm's `deit_tiny_distilled_patch16_224`, its `fb_in1k` weights.

* timm/deit_small_distilled_patch16_224.fb_in1k: [90.5 MB]
  timm's `deit_small_distilled_patch16_224`, its `fb_in1k` weights.

* timm/deit_base_distilled_patch16_224.fb_in1k: [350.1 MB]
  timm's `deit_base_distilled_patch16_224`, its `fb_in1k` weights.

* timm/deit_base_distilled_patch16_384.fb_in1k: [351.3 MB]
  timm's `deit_base_distilled_patch16_384`, its `fb_in1k` weights.

* timm/deit3_small_patch16_224.fb_in1k: [89.0 MB]
  timm's `deit3_small_patch16_224`, its `fb_in1k` weights.

* timm/deit3_small_patch16_224.fb_in22k_ft_in1k: [89.0 MB]
  timm's `deit3_small_patch16_224`, its `fb_in22k_ft_in1k` weights.

* timm/deit3_small_patch16_384.fb_in1k: [89.6 MB]
  timm's `deit3_small_patch16_384`, its `fb_in1k` weights.

* timm/deit3_small_patch16_384.fb_in22k_ft_in1k: [89.6 MB]
  timm's `deit3_small_patch16_384`, its `fb_in22k_ft_in1k` weights.

* timm/deit3_medium_patch16_224.fb_in1k: [156.2 MB]
  timm's `deit3_medium_patch16_224`, its `fb_in1k` weights.

* timm/deit3_medium_patch16_224.fb_in22k_ft_in1k: [156.2 MB]
  timm's `deit3_medium_patch16_224`, its `fb_in22k_ft_in1k` weights.

* timm/deit3_base_patch16_224.fb_in1k: [347.1 MB]
  timm's `deit3_base_patch16_224`, its `fb_in1k` weights.

* timm/deit3_base_patch16_224.fb_in22k_ft_in1k: [347.1 MB]
  timm's `deit3_base_patch16_224`, its `fb_in22k_ft_in1k` weights.

* timm/deit3_base_patch16_384.fb_in1k: [348.3 MB]
  timm's `deit3_base_patch16_384`, its `fb_in1k` weights.

* timm/deit3_base_patch16_384.fb_in22k_ft_in1k: [348.3 MB]
  timm's `deit3_base_patch16_384`, its `fb_in22k_ft_in1k` weights.

* timm/deit3_large_patch16_224.fb_in1k: [1.22 GB]
  timm's `deit3_large_patch16_224`, its `fb_in1k` weights.

* timm/deit3_large_patch16_224.fb_in22k_ft_in1k: [1.22 GB]
  timm's `deit3_large_patch16_224`, its `fb_in22k_ft_in1k` weights.

* timm/deit3_large_patch16_384.fb_in1k: [1.22 GB]
  timm's `deit3_large_patch16_384`, its `fb_in1k` weights.

* timm/deit3_large_patch16_384.fb_in22k_ft_in1k: [1.22 GB]
  timm's `deit3_large_patch16_384`, its `fb_in22k_ft_in1k` weights.

* timm/deit3_huge_patch14_224.fb_in1k: [2.53 GB]
  timm's `deit3_huge_patch14_224`, its `fb_in1k` weights.

* timm/deit3_huge_patch14_224.fb_in22k_ft_in1k: [2.53 GB]
  timm's `deit3_huge_patch14_224`, its `fb_in22k_ft_in1k` weights.

# contract: swin-v1-classifier
# A Swin V1 image classifier: a timm hub repo's checkpoint (safetensors)
# and `config.json`.
# publisher: timm - pytorch-image-models (timm), Ross Wightman's.

* timm/swin_tiny_patch4_window7_224.ms_in1k: [115.0 MB]
  timm's `swin_tiny_patch4_window7_224`, its `ms_in1k` weights.

* timm/swin_tiny_patch4_window7_224.ms_in22k: [178.4 MB]
  timm's `swin_tiny_patch4_window7_224`, its `ms_in22k` weights.

* timm/swin_tiny_patch4_window7_224.ms_in22k_ft_in1k: [115.0 MB]
  timm's `swin_tiny_patch4_window7_224`, its `ms_in22k_ft_in1k` weights.

* timm/swin_small_patch4_window7_224.ms_in1k: [200.8 MB]
  timm's `swin_small_patch4_window7_224`, its `ms_in1k` weights.

* timm/swin_small_patch4_window7_224.ms_in22k: [264.1 MB]
  timm's `swin_small_patch4_window7_224`, its `ms_in22k` weights.

* timm/swin_small_patch4_window7_224.ms_in22k_ft_in1k: [200.8 MB]
  timm's `swin_small_patch4_window7_224`, its `ms_in22k_ft_in1k` weights.

* timm/swin_base_patch4_window7_224.ms_in1k: [353.4 MB]
  timm's `swin_base_patch4_window7_224`, its `ms_in1k` weights.

* timm/swin_base_patch4_window7_224.ms_in22k: [438.1 MB]
  timm's `swin_base_patch4_window7_224`, its `ms_in22k` weights.

* timm/swin_base_patch4_window7_224.ms_in22k_ft_in1k: [353.4 MB]
  timm's `swin_base_patch4_window7_224`, its `ms_in22k_ft_in1k` weights.

* timm/swin_base_patch4_window12_384.ms_in1k: [366.0 MB]
  timm's `swin_base_patch4_window12_384`, its `ms_in1k` weights.

* timm/swin_base_patch4_window12_384.ms_in22k: [450.7 MB]
  timm's `swin_base_patch4_window12_384`, its `ms_in22k` weights.

* timm/swin_base_patch4_window12_384.ms_in22k_ft_in1k: [366.0 MB]
  timm's `swin_base_patch4_window12_384`, its `ms_in22k_ft_in1k` weights.

* timm/swin_large_patch4_window7_224.ms_in22k: [915.9 MB]
  timm's `swin_large_patch4_window7_224`, its `ms_in22k` weights.

* timm/swin_large_patch4_window7_224.ms_in22k_ft_in1k: [788.5 MB]
  timm's `swin_large_patch4_window7_224`, its `ms_in22k_ft_in1k` weights.

* timm/swin_large_patch4_window12_384.ms_in22k: [928.7 MB]
  timm's `swin_large_patch4_window12_384`, its `ms_in22k` weights.

* timm/swin_large_patch4_window12_384.ms_in22k_ft_in1k: [801.3 MB]
  timm's `swin_large_patch4_window12_384`, its `ms_in22k_ft_in1k` weights.

* timm/swin_s3_tiny_224.ms_in1k: [116.8 MB]
  timm's `swin_s3_tiny_224`, its `ms_in1k` weights.

* timm/swin_s3_small_224.ms_in1k: [209.6 MB]
  timm's `swin_s3_small_224`, its `ms_in1k` weights.

* timm/swin_s3_base_224.ms_in1k: [295.4 MB]
  timm's `swin_s3_base_224`, its `ms_in1k` weights.

# contract: swin-v2-classifier
# A Swin V2 image classifier: a timm hub repo's checkpoint (safetensors)
# and `config.json`.
# publisher: timm - pytorch-image-models (timm), Ross Wightman's.

* timm/swinv2_tiny_window16_256.ms_in1k: [119.4 MB]
  timm's `swinv2_tiny_window16_256`, its `ms_in1k` weights.

* timm/swinv2_tiny_window8_256.ms_in1k: [115.7 MB]
  timm's `swinv2_tiny_window8_256`, its `ms_in1k` weights.

* timm/swinv2_small_window16_256.ms_in1k: [205.0 MB]
  timm's `swinv2_small_window16_256`, its `ms_in1k` weights.

* timm/swinv2_small_window8_256.ms_in1k: [201.6 MB]
  timm's `swinv2_small_window8_256`, its `ms_in1k` weights.

* timm/swinv2_base_window16_256.ms_in1k: [357.7 MB]
  timm's `swinv2_base_window16_256`, its `ms_in1k` weights.

* timm/swinv2_base_window8_256.ms_in1k: [354.4 MB]
  timm's `swinv2_base_window8_256`, its `ms_in1k` weights.

* timm/swinv2_base_window12_192.ms_in22k: [438.8 MB]
  timm's `swinv2_base_window12_192`, its `ms_in22k` weights.

* timm/swinv2_base_window12to16_192to256.ms_in22k_ft_in1k: [357.7 MB]
  timm's `swinv2_base_window12to16_192to256`, its `ms_in22k_ft_in1k`
  weights.

* timm/swinv2_base_window12to24_192to384.ms_in22k_ft_in1k: [379.0 MB]
  timm's `swinv2_base_window12to24_192to384`, its `ms_in22k_ft_in1k`
  weights.

* timm/swinv2_large_window12_192.ms_in22k: [916.8 MB]
  timm's `swinv2_large_window12_192`, its `ms_in22k` weights.

* timm/swinv2_large_window12to16_192to256.ms_in22k_ft_in1k: [793.0 MB]
  timm's `swinv2_large_window12to16_192to256`, its `ms_in22k_ft_in1k`
  weights.

* timm/swinv2_large_window12to24_192to384.ms_in22k_ft_in1k: [814.3 MB]
  timm's `swinv2_large_window12to24_192to384`, its `ms_in22k_ft_in1k`
  weights.
```

<!-- /card -->
</details>

Models are fetched into the weights cache on first use. `fetch NAME...` brings them in ahead of time, for a deployment
that runs `--offline`.

## asr: speech to text

Name a **preset**, a model chosen by what it is for, and `asr` picks the family, the model and its settings, for files
(`transcribe`) or the microphone (`live`):

```sh
bunsen-tool asr transcribe recording.wav                       # English, the default preset
bunsen-tool asr transcribe --preset multi --json recording.wav # any language, as JSON lines
bunsen-tool asr live --preset translate-en                     # the microphone, translated to English
bunsen-tool asr transcribe --language de --quality fast recording.wav
bunsen-tool asr presets list                                   # what each preset is for, and runs
```

| preset | for | `transcribe` runs | `live` runs |
|---|---|---|---|
| `en` (`default`) | English | Parakeet, `fermion/phonon-2` | the same |
| `multi` | many languages, found from the speech | Qwen3-ASR, `qwen/qwen3-asr-1.7b` | `qwen/qwen3-asr-0.6b` |
| `translate-en` | other speech, as English text | Whisper, `large-v3[translate]` | `tiny[translate]` |

A preset chooses its model by the catalog's tags, so a better model, when it arrives, is chosen without the command line
changing. `--speech`, `--task`, `--quality` and `--language` choose a preset, or adjust one, by what is needed:

| command line | runs | task |
|---|---|---|
| (nothing) | `en`: Parakeet | transcribe |
| `--language de` | `multi`: Qwen3-ASR | transcribe |
| `--task translate` | `translate-en`: Whisper | translate |
| `--speech en --task translate` | `en`: Parakeet (English is never translated to English) | transcribe |
| `--preset multi --task translate` | `multi`, translating: Whisper | translate |
| `--preset en --language de` | an error: `en` hears only English | |

`presets show` prints what a preset runs as the `asr family` command line, and `-v` logs the choice.

### Model families, directly

`asr family <family>` runs one family, with only the flags that apply to it: `parakeet`, `qwen3-asr`, `voxtral` and
`whisper`, each with `list`, `fetch`, `transcribe` and `live`.

```sh
bunsen-tool asr family qwen3-asr list
bunsen-tool asr family whisper transcribe --model 'openai/small[translate]{beam_size: 1}' recording.wav
bunsen-tool asr family parakeet live --emission drafting
```

`--model` takes a lookup expression: a name, its presets in brackets, and a YAML patch of its settings in braces (quote
it: `[` and `]` glob). `list` shows each model's presets. The settings a patch can set:

| family | settings |
|---|---|
| `parakeet` | `window_seconds` (60) |
| `qwen3-asr` | `window_seconds` (30) |
| `voxtral` | `window_seconds` (30); `delay_ms` (480): how far the transcript runs behind the audio |
| `whisper` | `language`, `task`, `beam_size` (5), `timestamps`, `max_tokens`, `condition_on_previous_text`, `fallback` |

Every family also takes `hf:org/repo`, a Hugging Face repo in `transformers`' layout. `--emission` says when to decode:
`offline` (whole windows), `conservative` (each speech region as it ends, found by a VAD) or `drafting` (conservative,
plus a draft of the speech so far, which `live` shows).

### Who spoke when

```sh
bunsen-tool asr diarize run --rttm meeting.wav > meeting.rttm            # speaker turns alone
bunsen-tool asr transcribe --preset multi --diarize --json meeting.wav   # each statement with its speaker
bunsen-tool asr live --diarize
```

`diarize` runs NVIDIA's Sortformer diarizers: `nvidia/nemotron-3-diarization` (up to 8 speakers) by default, or
Sortformer v2 and v2.1 (up to 4). Their presets choose the latency (`ultra-low` to `offline`) and the post-processing.

### Where the transcript goes

Without `--publish`, to stdout, as text, `--json` lines or `--rttm`. `--publish CONFIG` sends it to every target a
config names at once, inline or from a file, in YAML or JSON:

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

The records are `statement` (committed text, with its speaker under `--diarize`), `draft` (`live`'s latest draft) and
`transcript` (a file's whole transcript); `times`, `speakers` and `events` may be set for every target or per target. A
network target delivers from its own thread, so a slow one never holds the audio up.

## tts: text to speech

```sh
bunsen-tool tts run -o hello.wav "Hello, this is ZONOS3."
bunsen-tool tts run --play "Spoken, not saved."
bunsen-tool tts run --model qwen3-tts --voice ryan -o ryan.wav "Hello from Qwen."
bunsen-tool tts voices                                   # a model's stock voices
```

ZONOS3-Mini is the default; Qwen3-TTS's checkpoints add a style instruction (`--instruct`) and a voice described in
words (`--describe`). The text is normalized as the model's release normalizes it, so `$12.50` is spoken as "twelve
dollars fifty cents", and long text is spoken as several takes. A take is sampled; `--seed` repeats one.

### Streaming

```sh
bunsen-tool tts run --stream --play "Spoken while the rest is still being made."
bunsen-tool tts run --stream -o out.wav "..."      # the same WAV as without --stream
```

`--stream` speaks each take as it is generated: its frames are decoded as they become final, so `--play` starts once the
first frames are decoded and `--prebuffer-ms` (200 by default) is queued, rather than after the whole text. A fresh
process compiles its GPU kernels first, which can take seconds. The audio is the same as without it. When synthesis is
slower than real time, playback runs dry and fills again; the summary counts the underruns, and a larger
`--prebuffer-ms` waits longer before it starts. `--first-chunk` and `--max-chunk` set how many frames each decode takes
(1, then up to 15).

### Cloning a voice

ZONOS3-Mini speaks in a voice cloned from recordings of a speaker. Qwen3-TTS's checkpoints clone none.

```sh
bunsen-tool tts clone ada.voice-ref ada-1.wav ada-2.mp3 --voice-dir voices
bunsen-tool tts run --voice ada --voice-dir voices -o ada.wav "Hello, in Ada's voice."
bunsen-tool tts voices --voice-dir voices           # stock voices, then the voice files
```

`clone` takes a reference directory to make, new or empty, and the clips: WAV, mp3, ..., at any rate and channels. Up
to a minute of one speaker, with no music or other voices, is used. It does two steps, which also run alone:

- `tts collect-voice ada.voice-ref CLIPS...` copies the clips into the reference directory as they are, with a
  `reference.json` that records them and what you give of the voice: `--name` (the reference directory's name, less
  `.voice-ref`, when omitted), `--language`, `--gender`, `--style` and `--transcript`. With the `live` feature,
  `--record` adds a clip from the microphone, until Ctrl-C or `--seconds`.
- `tts build-voice ada.voice-ref` builds the voice file from the reference for one checkpoint: `-o FILE`, or
  `--voice-dir DIR` for `DIR/NAME.voice.safetensors`. A voice file is never written over.

A voice file holds the speaker embedding, and is spoken only by checkpoints with the speaker encoder it was built with,
so keep the reference directory: `build-voice` makes the voice again for another checkpoint. `tts run` speaks in a
cloned voice three ways:

```sh
bunsen-tool tts run --voice ada --voice-dir voices -o out.wav "..."          # by name, from a voice directory
bunsen-tool tts run --voice-file ada.voice.safetensors -o out.wav "..."      # a voice file
bunsen-tool tts run --voice-reference ada.voice-ref -o out.wav "..."         # built for this run, not written
```

A stock voice's name wins over a voice file's: name a cloned voice so that it differs from the model's (`tts voices`).

## chat: Qwen3, Qwen3.5, Gemma and nanochat

```sh
bunsen-tool chat run --no-think "What is the capital of France?"
bunsen-tool chat repl --model qwen3-4b-instruct-2507 --stats
bunsen-tool chat run --model gemma-3-270m-it "What is the capital of France?"
bunsen-tool chat family nanochat run "What is the capital of France?"
```

`run` answers one prompt; `repl` holds a conversation, prefilling only what the KV cache does not already hold. The
chat driver is also a library, [`bunsen_tool::chat`](https://docs.rs/bunsen-tool/latest/bunsen_tool/chat/).

## decide: typed questions answered by Laya

`decide run` answers questions about a context with calibrated probabilities over each question's options, in one
forward pass of a Laya decision model. The request is YAML or JSON:

```yaml
context: "Customer: my kettle arrived with a cracked lid and I asked for a refund twice with no answer."
choices:
  - type: choice                    # a pick among named options
    question: "What should the agent do next?"
    options: [refund, replace, {escalate: "hand the case to a supervisor"}, close]
  - type: noul                      # a yes/no probability; `score` is a level on a scale
    question: "Is the customer satisfied?"
```

```sh
bunsen-tool decide run request.yml
```

Each result has each option's probability, the most probable (`answer`) and a `confidence`.

## classify and embed: images with timm and torchvision checkpoints

`classify run` gives each image file its top classes, most probable first, through the checkpoint's own eval
transform (timm's `pretrained_cfg`: its size, resize filter, crop and normalization), named by its labels:

```sh
bunsen-tool classify run --model timm/resnet50.a1_in1k -k 3 cat.jpg dog.png
```

`--model` takes a catalog name (`classify list` shows them, by family: the ResNets, torchvision's and timm's hub
repos; the ViTs, the DeiTs and the Swins, V1 and V2, timm's) or a timm hub repo, `hf:timm/<repo>`, of any family's
architectures; the default is `timm/resnet50.a1_in1k`. `--format json` prints JSON. `embed run` writes the images'
pooled features (the model's pre-logits) as one safetensors file; it also takes the headless towers, which `classify`
refuses (DINO's, MAE's):

```sh
bunsen-tool embed run -o features.safetensors cat.jpg dog.png
bunsen-tool embed run --model timm/vit_small_patch16_224.dino -o dino.safetensors cat.jpg
```

## text: normalization on its own

```sh
$ printf 'It costs $5.32\non 3/4/2025.\n' | bunsen-tool text normalize --normalize zyphra --normalize transcript --language en
it costs five dollars thirty two cents on march fourth twenty twenty five
```

The normalizers TTS speaks through and word error rates compare through; `text normalizers` lists them.

## Choosing a model

`chat`, `decide` and `tts` choose their model by a pipeline over the catalog's builds, by their tags:

```sh
bunsen-tool chat run --model qwen --le parameters 2e9 "Hi"                       # a group, filtered
bunsen-tool chat run --model qwen/qwen3 --eq thinking never --asc parameters "Hi"  # stages, in order
```

`--model` names a build, an alias or a group; stage flags (`--keep`, `--le`, `--weigh`, `--asc`, ...) keep and order
within it; presets (`--preset`, `--on`, `--off`) apply to the one chosen. `-v` logs the choice and why.
`bunsen-tool catalog` browses the catalog itself: `tree`, `show`, `resolve`, and `select`, the same pipeline.

## The weights cache

```sh
bunsen-tool cache path                          # where it is
bunsen-tool cache list [NAME]                   # what it holds, and when each file was last used
bunsen-tool cache clear [NAME] --age 30d --yes  # what has not been used for 30 days
```

A language model's first load also keeps a converted copy of its weights, as the load placed them,
under the build's `converted` name (`qwen/qwen3/qwen3-4b/converted`); later loads read it instead of
the checkpoint (Qwen3-4B on CUDA: 18 s the first time, under 5 s after), and still do with the
checkpoint cleared (`cache clear qwen/qwen3/qwen3-4b/checkpoint`). `--converted-weights false` keeps
and uses no copy.

Every fetched file is pinned to a SHA-256 and verified as it lands. `--cache-dir` moves the cache, and `--offline` never
reaches the network. Every command also takes `--device`, and the commands that load a model `--min-precision f32`, a
floor under the dtypes its weights are placed in.
