# whisper-dev example

A development tool for running an OpenAI Whisper checkpoint end to end in Burn. It loads a PyTorch `.pt`/`.pth` state
dict (`model_state_dict`, or a caller-specified top-level key) through bunsen's Whisper scanner, prints the inferred
configuration, converts an audio file to log-mels with the streaming front end, and greedily decodes each 30 s window,
printing the ids and, given a vocabulary, the text.

## Bunsen features exercised

- `bunsen::kits::speech::whisper::pretrained::PytorchWhisperScanner` — scans a PyTorch state dict, infers the Whisper
  architecture and configuration, and materializes a Burn module with the loaded weights (`with_top_level_key`,
  `load`), on top of `burn-store`'s PyTorch loader.
- `bunsen::kits::speech::whisper::mel` and `bunsen::ops::signal::mels` — the streaming mel front end, fed in chunks as a
  live transcription loop would be; feeding the whole clip at once gives the same result.
- `bunsen::kits::speech::whisper::decode` and `WhisperTokenLayout` — greedy decoding per window, with the prompt and
  stop token derived from the checkpoint's own vocabulary size, so an English-only and a multilingual model each get the
  ids they were trained on.
- `bunsen::kits::speech::whisper::text` — a `wordchipper` detokenizer over a
  `.tiktoken` rank file, for text output.

## The backend

The example computes on `bunsen::support::testing::PerformanceBackend`: the backend bunsen's own compute-heavy tests run
on. This crate does not choose it. **bunsen's backend feature** does, at build time, so the flag that picks the backend
for `cargo test` picks it here too, and what the example runs on is what the tests ran on.

| build with                           | backend                     |
|--------------------------------------|-----------------------------|
| `--features bunsen/cuda`             | `burn::backend::Cuda`       |
| `--features bunsen/metal`            | `burn::backend::Metal`      |
| `--features bunsen/wgpu`             | `burn::backend::Wgpu`       |
| `--features bunsen/flex`, or nothing | `burn::backend::Flex` (CPU) |

When several are on, the first of `cuda`, `metal`, `wgpu`, `flex` wins. bunsen's default `testing` feature enables
`flex`, which is why a bare build runs on the CPU. The `dependency/feature` form of `--features` works from any package
in the workspace, so `cargo run -p whisper-dev --features bunsen/wgpu`
from the root and `cargo run --features bunsen/wgpu` from this directory are the same build.

## Running the example

```bash
$ cargo run --release -p whisper-dev --features bunsen/wgpu -- \
  --source /path/to/whisper.pt \
  --audio /path/to/clip.wav \
  --vocab /path/to/multilingual.tiktoken
```

Options:

- `--top-level-key` — the state dict's key in the checkpoint (default
  `model_state_dict`).
- `--sample-rate` — the rate the checkpoint is declared at, and the audio file is decoded at (default `16000`; must be a
  multiple of 200 Hz).
- `--chunk-ms` — milliseconds of audio per streaming chunk, a whole number of 10 ms hops (default `1000`).
- `--max-tokens` — cap on generated tokens per 30 s window (default `32`).
- `--language` — a Whisper language code (default `en`) and `--task` —
  `transcribe` or `translate` (default `transcribe`); both ignored by an English-only checkpoint, which takes no such
  tokens.
- `--timestamps` — let the model emit timestamp tokens.
- `--vocab` — a `.tiktoken` rank file; without it only ids are printed.
