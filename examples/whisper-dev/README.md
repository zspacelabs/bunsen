# whisper-dev example

A small development utility for importing OpenAI Whisper speech-to-text models
from PyTorch checkpoints into Burn. It loads a `model_state_dict` (or a
caller-specified top-level key) from a `.pt`/`.pth` file, reconstructs the
matching Whisper module and its configuration, and prints the inferred config.

This is primarily a scaffold for exercising and debugging the PyTorch weight
scanner against real Whisper checkpoints.

## Bunsen features exercised

- `bunsen::kits::speech::whisper::pretrained::PytorchWhisperScanner` — scans a
  PyTorch state dict, infers the Whisper architecture/config, and materializes a
  Burn module with the loaded weights (`with_top_level_key`, `load`).

It demonstrates bunsen's pretrained-weight import path for speech models layered
on top of `burn-store`'s PyTorch loader.

## Running the Example

Select `BACKEND` from `cuda`, `metal`, `wgpu`, or `flex` (default: `flex`):

```bash
$ cargo run --release -p whisper-dev --features BACKEND -- \
  --source /path/to/whisper.pt
```
