#!/usr/bin/env python3
"""Generate the Nemotron-3-Diarization toy-model fixture.

Run once; commit the outputs. Regenerating is not part of CI.

    uv venv --python 3.12 /tmp/tfvenv
    uv pip install --python /tmp/tfvenv/bin/python \
        --index-url https://download.pytorch.org/whl/cpu torch
    uv pip install --python /tmp/tfvenv/bin/python safetensors numpy librosa \
        "transformers @ git+https://github.com/huggingface/transformers@4321355eeae7934bba83fce43269345c5f17feda"
    /tmp/tfvenv/bin/python tools/gen_nemotron_diar_toy_fixtures.py

Pinned versions used to produce the committed fixtures:
    transformers 5.21.0.dev0 (4321355e), torch 2.14.1+cpu, safetensors 0.8.0,
    numpy 2.5.4

A tiny seeded `Nemotron3DiarizationForAudioFrameClassification` --
transformers' own port of NVIDIA's Nemotron-3-Diarization, whose real
checkpoint bunsen loads under the same names -- run on the CPU in float64
(`sdpa` attention) and stored as float32, so every recorded value is the
reference's answer to float32 rounding.

The model is run stage by stage on whole inputs of one item each (no chunking,
no speaker cache), at two lengths: one a multiple of the stacking factor and
one not, which the stacking pads.

Outputs, under `crates/public/bunsen/testdata/sortformer/`:

`nemotron_toy.safetensors`
    `weights.*`: the toy's state dict under transformers' names.
    `io.{case}.*`: per case, the mels in and every stage out (see `STAGES`).
`nemotron_toy.json`
    The toy's config, and the cases' lengths.
"""

import json
import pathlib

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import Nemotron3DiarizationConfig, Nemotron3DiarizationForAudioFrameClassification

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "crates/public/bunsen/testdata/sortformer"

N_MELS = 24
CASES = {"even": 40, "ragged": 37}

CONFIG = dict(
    audio_config=dict(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        hidden_act="gelu",
        num_mel_bins=N_MELS,
        subsampling_factor=8,
        max_position_embeddings=5000,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    ),
    head_config=dict(hidden_size=8, num_speakers=3, audio_hidden_size=16, subsampling_factor=8),
    streaming_config=dict(num_speakers=3, subsampling_factor=8),
)

# Stage outputs recorded per case:
#   stacked  [1, ceil(T/8), 16]  the feature stacking
#   encoded  [1, ceil(T/8), 16]  the encoder (input norm, layers, final norm)
#   projected [1, ceil(T/8), 8]  the projection to the head's width
#   upsampled [1, 8*ceil(T/8), 8]
#   probs    [1, 8*ceil(T/8), 3] sigmoid of the classifier, 10 ms
#   pooled   [1, ceil(T/8), 3]   the 8-frame mean of `probs`, 80 ms
STAGES = ["stacked", "encoded", "projected", "upsampled", "probs", "pooled"]


def build_model() -> Nemotron3DiarizationForAudioFrameClassification:
    torch.manual_seed(1234)
    model = Nemotron3DiarizationForAudioFrameClassification(Nemotron3DiarizationConfig(**CONFIG))
    model = model.double().eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name and name.endswith(".weight"):
                p.copy_(1.0 + 0.2 * torch.randn_like(p))
            elif "norm" in name and name.endswith(".bias"):
                p.copy_(0.1 * torch.randn_like(p))
            else:
                fan_in = p.shape[1] if p.ndim >= 2 else 4
                for d in p.shape[2:]:
                    fan_in *= d
                p.copy_(torch.randn_like(p) / np.sqrt(fan_in))
        # A larger head output, so the probabilities straddle 0.5.
        model.classifier.out_proj.weight.mul_(3.0)
    return model


def mels(frames: int, seed: int) -> torch.Tensor:
    """`[1, frames, N_MELS]`, log-mel-like values."""
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(1, frames, N_MELS, generator=g, dtype=torch.float64) * 2.0 - 4.0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = build_model()
    tower = model.model.audio_tower

    tensors = {f"weights.{k}": v.float().contiguous() for k, v in model.state_dict().items()}
    with torch.no_grad():
        for i, (case, frames) in enumerate(CASES.items()):
            x = mels(frames, seed=10 + i)
            stacked = tower.embedder(x)
            encoded = tower(inputs_embeds=stacked).last_hidden_state
            projected = model.model.proj(encoded)
            upsampled = model.model.upsampler(projected)
            probs = model.classifier(upsampled).sigmoid()
            pooled = torch.nn.functional.avg_pool1d(probs.transpose(1, 2), 8, 8).transpose(1, 2)
            stages = dict(
                mels=x,
                stacked=stacked,
                encoded=encoded,
                projected=projected,
                upsampled=upsampled,
                probs=probs,
                pooled=pooled,
            )
            for name, t in stages.items():
                tensors[f"io.{case}.{name}"] = t.float().contiguous()

    save_file(tensors, OUT / "nemotron_toy.safetensors")
    (OUT / "nemotron_toy.json").write_text(
        json.dumps({"config": CONFIG, "cases": CASES}, indent=1) + "\n"
    )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
