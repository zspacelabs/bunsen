#!/usr/bin/env python3
"""Generate the NeMo mel front-end fixture for Nemotron-3-Diarization.

Run once; commit the output. Regenerating is not part of CI.

    uv venv --python 3.12 /tmp/tfvenv
    uv pip install --python /tmp/tfvenv/bin/python \
        --index-url https://download.pytorch.org/whl/cpu torch
    uv pip install --python /tmp/tfvenv/bin/python safetensors numpy librosa \
        "transformers @ git+https://github.com/huggingface/transformers@4321355eeae7934bba83fce43269345c5f17feda"
    /tmp/tfvenv/bin/python tools/gen_nemo_front_end_fixtures.py MODEL_WEIGHTS_CKPT

`MODEL_WEIGHTS_CKPT` is `model_weights.ckpt` from inside
`nvidia/Nemotron-3-Diarization`'s `Nemotron-3-Diarization.nemo` (a plain tar)
at revision f667ed73aee57d40cc39428eb768b4fd87a0a29e. It is read with
`torch.load(weights_only=True)`, and only its two featurizer tables are kept.

Pinned versions used to produce the committed fixture:
    transformers 5.21.0.dev0 (4321355e), torch 2.14.1+cpu, librosa 1.0.0,
    safetensors 0.8.0, numpy 2.5.4

Output, `crates/public/bunsen/testdata/nemo/front_end.safetensors`:

`waveforms [2, N]`, `lengths [2]`, `features [2, T, 128]`, `mask [2, T]`
    `NemotronAsrStreamingFeatureExtractor` (128 mels, no normalization, its
    tables computed in float32) on two synthetic waveforms of different
    lengths, padded as one batch.
`window [400]`, `filterbank [128, 257]`
    The checkpoint's own `preprocessor.featurizer.window` and
    `preprocessor.featurizer.fb`: the bf16 tables NeMo computes with, widened
    to float32 exactly.
"""

import pathlib
import sys

import numpy as np
import torch
from safetensors.torch import save_file
from transformers.models.nemotron_asr_streaming.feature_extraction_nemotron_asr_streaming import (
    NemotronAsrStreamingFeatureExtractor,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "crates/public/bunsen/testdata/nemo"


def main(ckpt: pathlib.Path) -> None:
    fe = NemotronAsrStreamingFeatureExtractor(feature_size=128)
    n = [16000 + 333, 9000]
    waves = []
    for i, length in enumerate(n):
        t = np.arange(length, dtype=np.float64)
        w = (
            0.3 * np.sin(2 * np.pi * (300.0 + 200.0 * i) * t / 16000.0)
            + 0.1 * np.sin(2 * np.pi * 2500.0 * t / 16000.0)
            + 0.02 * (((t * 7919 + 13 * i) % 1009) / 504.5 - 1.0)
        )
        waves.append(w.astype(np.float32))
    out = fe(waves, sampling_rate=16000, return_tensors="pt", padding="longest")
    padded = np.zeros((len(n), max(n)), np.float32)
    for i, w in enumerate(waves):
        padded[i, : len(w)] = w

    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    window = state["preprocessor.featurizer.window"]
    fb = state["preprocessor.featurizer.fb"]
    assert window.dtype == torch.bfloat16 and fb.dtype == torch.bfloat16
    assert tuple(window.shape) == (400,) and tuple(fb.shape) == (1, 128, 257)

    OUT.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "waveforms": torch.from_numpy(padded),
            "lengths": torch.tensor(n, dtype=torch.float32),
            "features": out["input_features"].float().contiguous(),
            "mask": out["attention_mask"].float().contiguous(),
            "window": window.float().contiguous(),
            "filterbank": fb[0].float().contiguous(),
        },
        OUT / "front_end.safetensors",
    )
    print("wrote", OUT / "front_end.safetensors")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
