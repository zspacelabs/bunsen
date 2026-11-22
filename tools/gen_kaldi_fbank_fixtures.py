#!/usr/bin/env python3
"""Generate the Kaldi fbank fixtures: `torchaudio.compliance.kaldi.fbank` on fixed signals.

Run once; commit the output. Regenerating is not part of CI.

    python3.12 -m venv venv
    venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1 torchaudio==2.9.1
    venv/bin/pip install safetensors numpy
    venv/bin/python -I tools/gen_kaldi_fbank_fixtures.py

Pinned versions used to produce the committed fixture:
    torch 2.9.1+cpu, torchaudio 2.9.1+cpu, safetensors 0.8.0, Python 3.12.

The oracle is torchaudio's own Kaldi port with dither 0 (so the features are
deterministic), run in float64 and stored as float32: the truth a float32 run is measured
against. (torchaudio's own float32 run is off it by up to 6.6e-4 on the `zonos3` log
features, in the quietest bins, about 23 nats below a frame's loudest: the float32
spectrum's rounding, relative to the frame's peak energy.) Each case is one config over two
signals of the same length (a batch of two), each signal computed by its own `fbank` call.

Output, `crates/public/bunsen/testdata/kaldi_fbank/fbank.safetensors`:

- `signal_16k` `[2, 9600]` and `signal_8k` `[2, 4800]`: 0.6 s each, a chirp plus seeded noise
  plus a DC offset (so the DC removal shows), at int16 scale (about +-10^4, as a speaker
  encoder scales its waveform before the fbank);
- `zonos3` `[2, 58, 80]`: ZONOS3's speaker encoder's call (16 kHz, 80 bins, 25 / 10 ms, a
  Hamming window, pre-emphasis 0.97, the DC removed, power, log, 20 Hz to Nyquist);
- `zonos3_mean` `[2, 58, 80]`: the same with `subtract_mean=True`;
- `variant` `[2, 73, 40]`: 8 kHz, 40 bins, 20 / 8 ms (160-sample frames padded to 256),
  64 Hz to Nyquist less 200 Hz, a Hanning window, pre-emphasis 0.5, the DC kept, magnitude
  (not power), log;
- `linear` `[2, 73, 23]`: 8 kHz, 23 bins, 20 / 8 ms, a Blackman window, no pre-emphasis,
  power, no log.
"""

import pathlib

import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi
from safetensors.torch import save_file

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "crates/public/bunsen/testdata/kaldi_fbank/fbank.safetensors"


def signals(rate: int, seed: int) -> torch.Tensor:
    """Two 0.6 s signals at `rate`: a chirp, seeded noise and a DC offset, at int16 scale."""
    n = int(0.6 * rate)
    t = np.arange(n) / rate
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(2):
        f0, f1 = 120.0 * (k + 1), 0.45 * rate
        phase = 2 * np.pi * (f0 * t + (f1 - f0) * t * t / (2 * 0.6))
        x = 0.4 * np.sin(phase) + 0.05 * rng.standard_normal(n) + 0.02 * (k + 1)
        rows.append(x * 32768.0)
    return torch.tensor(np.stack(rows), dtype=torch.float32)


def fbank(signal: torch.Tensor, **options) -> torch.Tensor:
    """`kaldi.fbank` over each row of `signal`, stacked."""
    rows = [kaldi.fbank(row[None].double(), dither=0.0, **options) for row in signal]
    return torch.stack(rows).float()


def main() -> None:
    s16, s8 = signals(16000, 1234), signals(8000, 4321)
    zonos3 = dict(
        sample_frequency=16000.0,
        num_mel_bins=80,
        frame_length=25.0,
        frame_shift=10.0,
        window_type="hamming",
        energy_floor=0.0,
    )
    tensors = {
        "signal_16k": s16,
        "signal_8k": s8,
        "zonos3": fbank(s16, **zonos3),
        "zonos3_mean": fbank(s16, subtract_mean=True, **zonos3),
        "variant": fbank(
            s8,
            sample_frequency=8000.0,
            num_mel_bins=40,
            frame_length=20.0,
            frame_shift=8.0,
            low_freq=64.0,
            high_freq=-200.0,
            window_type="hanning",
            preemphasis_coefficient=0.5,
            remove_dc_offset=False,
            use_power=False,
        ),
        "linear": fbank(
            s8,
            sample_frequency=8000.0,
            num_mel_bins=23,
            frame_length=20.0,
            frame_shift=8.0,
            window_type="blackman",
            preemphasis_coefficient=0.0,
            use_log_fbank=False,
        ),
    }
    tensors = {k: v.contiguous() for k, v in tensors.items()}
    for k, v in tensors.items():
        print(k, tuple(v.shape))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, OUT)


if __name__ == "__main__":
    main()
