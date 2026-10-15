# phonon-cli

`phonon`: speech to text with NVIDIA's Parakeet TDT models on
[bunsen](../../crates/public/bunsen), `FermionResearch`'s Phonon-2 by default.

```sh
cargo run --release -p phonon-cli --features wgpu -- \
  transcribe --device wgpu recording.wav
cargo run --release -p phonon-cli --features wgpu -- \
  transcribe --device wgpu --json recording.wav
cargo run --release -p phonon-cli -- models list
```

`transcribe` loads `--model` (`fermion/phonon-2` unless named) through bunsen's
pretrained cache, fetching it on first use, and prints each file's text, or with
`--json` one object per file with every word's start and end in seconds.
Audio is read at 16 kHz mono; bunsen does not resample. Recordings longer than
`--window-seconds` (60) are transcribed in windows cut at quiet points.

| model | | |
|---|---|---|
| `fermion/phonon-2` | 164 MB | `FermionResearch/Phonon-2`: `parakeet-tdt-0.6b-v3` with a five-value encoder |
| `nvidia/parakeet-tdt-0.6b-v3` | 2.5 GB | NVIDIA's base model, as `transformers` ships it |

Both are CC-BY-4.0.
