#!/usr/bin/env python3
"""Generate the Parakeet toy-model fixture.

Run once; commit the outputs. Regenerating is not part of CI.

    python3 -m venv /tmp/pkvenv
    /tmp/pkvenv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
    /tmp/pkvenv/bin/pip install transformers safetensors librosa
    /tmp/pkvenv/bin/python tools/gen_parakeet_toy_fixtures.py

Pinned versions used to produce the committed fixtures:
    transformers 5.19.0, torch 2.14.1+cpu, safetensors 0.8.0, librosa 1.0.0

A tiny `ParakeetForTDT` -- transformers' own implementation, the one
`FermionResearch/Phonon-2`'s reference runs -- with seeded random weights, run
on the CPU in float64 and stored as float32, so every recorded value is the
reference's answer to float32 rounding.

Outputs, under `crates/public/bunsen/testdata/parakeet/`:

`toy.safetensors`
    `model.*`: the toy's weights under their `ParakeetForTDT` state-dict
    names, as a checkpoint would carry them.
    `io.*`: inputs and the intermediate and final outputs bunsen is pinned
    against (see `IO` below).
`toy.json`
    The toy's `ParakeetTDTConfig`, and the input lengths.
`front_end.safetensors`
    `ParakeetFeatureExtractor` on two synthetic waveforms of different
    lengths, padded as one batch: `waveforms [2, N]`, `lengths [2]`,
    `features [2, T, 128]`, `mask [2, T]`.
"""

import json
import pathlib

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import ParakeetFeatureExtractor, ParakeetForTDT, ParakeetTDTConfig

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "crates/public/bunsen/testdata/parakeet"

# Two utterances of different lengths, so masking is exercised.
N_FRAMES = 53
LENGTHS = [53, 37]
TOKENS = [[3, 7, 1, 9], [10, 2, 2, 5]]

CONFIG = dict(
    encoder_config=dict(
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=32,
        num_hidden_layers=2,
        num_mel_bins=24,
        subsampling_conv_channels=4,
        subsampling_conv_kernel_size=3,
        subsampling_conv_stride=2,
        subsampling_factor=8,
        conv_kernel_size=5,
        attention_bias=False,
        convolution_bias=False,
        scale_input=False,
        max_position_embeddings=5000,
        dropout=0.0,
        dropout_positions=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        layerdrop=0.0,
    ),
    decoder_hidden_size=12,
    num_decoder_layers=2,
    vocab_size=11,
    blank_token_id=10,
    pad_token_id=2,
    durations=[0, 1, 2, 3, 4],
    hidden_act="relu",
    max_symbols_per_step=10,
)


def build_model() -> ParakeetForTDT:
    torch.manual_seed(1234)
    model = ParakeetForTDT(ParakeetTDTConfig(**CONFIG)).double().eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name and name.endswith(".weight"):
                p.copy_(1.0 + 0.2 * torch.randn_like(p))
            else:
                fan_in = p.shape[1] if p.ndim >= 2 else 4
                for d in p.shape[2:]:
                    fan_in *= d
                p.copy_(torch.randn_like(p) / np.sqrt(fan_in))
        for name, b in model.named_buffers():
            if name.endswith("running_mean"):
                b.copy_(0.3 * torch.randn_like(b))
            elif name.endswith("running_var"):
                b.copy_(0.5 + torch.rand_like(b))
    return model


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = build_model()
    enc = model.encoder

    torch.manual_seed(99)
    n_mels = CONFIG["encoder_config"]["num_mel_bins"]
    features = torch.randn(len(LENGTHS), N_FRAMES, n_mels, dtype=torch.float64)
    mask = torch.arange(N_FRAMES)[None, :] < torch.tensor(LENGTHS)[:, None]
    features = features * mask[:, :, None]

    io = {}
    captured = {}

    def keep(name):
        def hook(_module, _inputs, output):
            captured[name] = output[0] if isinstance(output, tuple) else output

        return hook

    enc.subsampling.register_forward_hook(keep("subsampling"))
    enc.encode_positions.register_forward_hook(keep("positions"))
    for i, layer in enumerate(enc.layers):
        layer.register_forward_hook(keep(f"layer{i}"))

    with torch.no_grad():
        out = enc(input_features=features, attention_mask=mask.long())
        projected = model.encoder_projector(out.last_hidden_state)
        ids = torch.tensor(TOKENS)
        decoded = model.decoder(ids)
        logits = model.joint(
            encoder_hidden_states=projected[:, :, None, :],
            decoder_hidden_states=decoded[:, None, :, :],
        )

        # One layer's pieces on the first layer's input, unmasked, so each
        # sub-module is pinned on its own.
        layer = enc.layers[0]
        x = captured["subsampling"]
        pos = captured["positions"]
        io["block.ff1"] = layer.feed_forward1(layer.norm_feed_forward1(x))
        io["block.attn"] = layer.self_attn(
            hidden_states=layer.norm_self_att(x), position_embeddings=pos
        )[0]
        io["block.conv"] = layer.conv(layer.norm_conv(x))

    io["features"] = features
    io["mask"] = mask.to(torch.float64)
    io["subsampling"] = captured["subsampling"]
    io["positions"] = captured["positions"]
    for i in range(len(enc.layers)):
        io[f"layer{i}"] = captured[f"layer{i}"]
    io["encoder"] = out.last_hidden_state
    io["encoder_mask"] = out.attention_mask.to(torch.float64)
    io["projected"] = projected
    io["tokens"] = ids.to(torch.float64)
    io["decoder"] = decoded
    io["logits"] = logits

    tensors = {f"model.{k}": v.float().contiguous() for k, v in model.state_dict().items()}
    tensors.update({f"io.{k}": v.float().contiguous() for k, v in io.items()})
    save_file(tensors, OUT / "toy.safetensors")
    (OUT / "toy.json").write_text(
        json.dumps({"config": CONFIG, "n_frames": N_FRAMES, "lengths": LENGTHS}, indent=1)
        + "\n"
    )

    front_end()
    print("wrote", OUT)


def front_end() -> None:
    fe = ParakeetFeatureExtractor(feature_size=128)
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
    save_file(
        {
            "waveforms": torch.from_numpy(padded),
            "lengths": torch.tensor(n, dtype=torch.float32),
            "features": out["input_features"].float().contiguous(),
            "mask": out["attention_mask"].float().contiguous(),
        },
        OUT / "front_end.safetensors",
    )


if __name__ == "__main__":
    main()
