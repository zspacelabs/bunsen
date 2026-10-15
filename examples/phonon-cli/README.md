# phonon-cli

`phonon`: speech to text with `FermionResearch`'s Phonon models on
[bunsen](../../crates/public/bunsen): Phonon-2 (NVIDIA's Parakeet TDT) by
default, and Phonon-1 (Qwen's Qwen3-ASR).

```sh
cargo run --release -p phonon-cli --features wgpu -- \
  transcribe --device wgpu recording.wav
cargo run --release -p phonon-cli --features wgpu -- \
  transcribe --device wgpu --model fermion/phonon-1 --json recording.wav
cargo run --release -p phonon-cli -- models list
```

`transcribe` loads `--model` (`fermion/phonon-2` unless named) through bunsen's
pretrained cache, fetching it on first use, and prints each file's text, or with
`--json` one object per file: the text, every word's start and end in seconds
(Phonon-2), and the language the model heard (Phonon-1). Audio is read at
16 kHz mono. Recordings longer than `--window-seconds` (60 for Phonon-2, 30 for
Phonon-1, the longest it was trained on) are transcribed in windows cut at quiet
points.

| model | | |
|---|---|---|
| `fermion/phonon-2` | 164 MB | `FermionResearch/Phonon-2`: `parakeet-tdt-0.6b-v3` with a five-value encoder, English |
| `nvidia/parakeet-tdt-0.6b-v3` | 2.5 GB | NVIDIA's base model, as `transformers` ships it |
| `fermion/phonon-1` | 415 MB | `FermionResearch/Phonon-1`: `Qwen3-ASR-0.6B` re-packed, 30 languages |
| `fermion/phonon-1-micro` | 285 MB | Phonon-1, its audio tower and embedding 4-bit |
| `fermion/phonon-1-big` | 581 MB | Phonon-1, its audio tower and embedding `bf16` |
| `qwen/qwen3-asr-0.6b` | 1.6 GB | Qwen's base model, as `transformers` ships it |

The Parakeet models are CC-BY-4.0; the Qwen3-ASR models are Apache-2.0.
