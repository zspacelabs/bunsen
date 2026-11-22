#!/usr/bin/env python3
"""Generate the Hann-windowed sinc resample fixtures: `torchaudio.transforms.Resample`.

Run once; commit the output. Regenerating is not part of CI.

    python3.12 -m venv venv
    venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1 torchaudio==2.9.1
    venv/bin/pip install safetensors numpy
    venv/bin/python -I tools/gen_resample_fixtures.py

Pinned versions used to produce the committed fixture:
    torch 2.9.1+cpu, torchaudio 2.9.1+cpu, safetensors 0.8.0, Python 3.12.

The oracle is `torchaudio.transforms.Resample` with its defaults (`sinc_interp_hann`,
`lowpass_filter_width` 6, `rolloff` 0.99): its kernel, which it computes in float64 and
stores in float32, applied by torchaudio's own `_apply_sinc_resample_kernel` in float64. So
the reference is the float32 kernel applied exactly; torchaudio's float32 convolution is
off it by float32's accumulation.

Output, `crates/public/bunsen/testdata/resample/hann_sinc.safetensors`, for each case
`{in}_{out}` of 24000_16000, 44100_16000, 22050_16000, 16000_24000 and 16000_44100:

- `{in}_{out}.input` `[2, n]`: 0.25 s at the input rate, two rows: a sum of tones (one
  above the output's Nyquist when downsampling) plus seeded noise, within `[-1, 1]`;
- `{in}_{out}.output` `[2, ceil(n * out / in)]`: the resampled rows.
"""

import math
import pathlib

import numpy as np
import torch
import torchaudio.functional.functional as F
import torchaudio.transforms as T
from safetensors.torch import save_file

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "crates/public/bunsen/testdata/resample/hann_sinc.safetensors"

CASES = [(24000, 16000), (44100, 16000), (22050, 16000), (16000, 24000), (16000, 44100)]


def signal(rate: int, seed: int) -> torch.Tensor:
    """Two 0.25 s rows at `rate`: tones at 300, 2500 and 9000 Hz plus seeded noise."""
    n = rate // 4
    t = np.arange(n) / rate
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(2):
        x = sum(0.2 * np.sin(2 * np.pi * f * (k + 1) * t + k) for f in (300.0, 2500.0, 9000.0))
        rows.append(x + 0.05 * rng.standard_normal(n))
    return torch.tensor(np.stack(rows), dtype=torch.float32)


def main() -> None:
    tensors = {}
    for i, (orig, new) in enumerate(CASES):
        x = signal(orig, 100 + i)
        resample = T.Resample(orig, new)
        gcd = math.gcd(orig, new)
        y = F._apply_sinc_resample_kernel(
            x.double(), orig, new, gcd, resample.kernel.double(), resample.width
        )
        assert y.shape[-1] == math.ceil(x.shape[-1] * new / orig)
        tensors[f"{orig}_{new}.input"] = x.contiguous()
        tensors[f"{orig}_{new}.output"] = y.float().contiguous()
        print(orig, new, tuple(x.shape), tuple(y.shape))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, OUT)


if __name__ == "__main__":
    main()
