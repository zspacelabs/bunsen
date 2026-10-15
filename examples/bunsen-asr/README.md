# bunsen-asr

`bunsen-asr`: speech to text on [bunsen](../../crates/public/bunsen), one subcommand per model family, each with
its own `list` and `transcribe` and only the flags that apply to it:

```sh
bunsen-asr parakeet  (list|transcribe) [FLAGS]   # NVIDIA's Parakeet TDT; FermionResearch's Phonon-2 by default
bunsen-asr qwen3-asr (list|transcribe) [FLAGS]   # Qwen's Qwen3-ASR; FermionResearch's Phonon-1 by default
bunsen-asr whisper   (list|transcribe) [FLAGS]   # OpenAI's Whisper; openai/base by default
```

```sh
cargo run --release -p bunsen-asr --features wgpu -- parakeet transcribe recording.wav
cargo run --release -p bunsen-asr --features wgpu -- qwen3-asr transcribe --model qwen/qwen3-asr-1.7b --json recording.wav
cargo run --release -p bunsen-asr --features wgpu -- whisper transcribe --preset conservative --language en recording.wav
cargo run --release -p bunsen-asr -- qwen3-asr list
```

`list` shows the family's named models, with each one's licence, its files, and whether each is cached; every family
also takes `hf:org/repo`, a Hugging Face repo in `transformers`' layout, whose `config.json` gives the geometry.

`transcribe` loads `--model` (the family's default unless named) through bunsen's pretrained cache, fetching it on
first use, logs its info card (name, licence, origin, every file and where it came from) at `INFO` (`-v`), and prints
each file's text, or with `--json` one object per file: the text, the language when the model names one (Qwen3-ASR,
Whisper), each word's start and end in seconds (Parakeet), and each segment's (Whisper). Audio is read at the model's
rate, 16 kHz mono.

| Family | Flags of its own |
|---|---|
| `parakeet` | `--window-seconds` (60): longer recordings are cut at quiet points; `--batch` (4): windows at once |
| `qwen3-asr` | `--window-seconds` (30, the most Qwen3-ASR was trained on) |
| `whisper` | `--language` (detected when omitted), `--task` (`transcribe`, `translate`), `--preset` (`offline`, `conservative`: speech regions found by Silero's VAD, `silero/vad-16k`), `--beam-size` (5) |

Every command takes `--cache-dir`, `--offline`, `--device` (from the backends compiled in: `--features wgpu`, `vulkan`,
`metal`, `cuda`; the CPU is always there) and `-v`/`-q`.

| model | | licence |
|---|---|---|
| `parakeet`: `fermion/phonon-2` | 164 MB, `parakeet-tdt-0.6b-v3` with a five-value encoder, English | CC-BY-4.0 |
| `parakeet`: `nvidia/parakeet-tdt-0.6b-v3` | 2.5 GB, NVIDIA's base model, 25 European languages | CC-BY-4.0 |
| `qwen3-asr`: `fermion/phonon-1` | 415 MB, `Qwen3-ASR-0.6B` re-packed, 30 languages; `-micro` (285 MB) and `-big` (581 MB) builds | Apache-2.0 |
| `qwen3-asr`: `qwen/qwen3-asr-0.6b`, `qwen/qwen3-asr-1.7b` | 1.6 and 4.1 GB, Qwen's base models | Apache-2.0 |
| `whisper`: `openai/*` | every `openai-whisper` checkpoint, `tiny` to `large-v3-turbo` | MIT |
