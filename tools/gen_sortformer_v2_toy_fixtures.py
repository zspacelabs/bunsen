#!/usr/bin/env python3
"""Generate the Sortformer v2 toy-model fixture: NeMo's own forward, stage by stage.

Run once; commit the outputs. Regenerating is not part of CI.

    uv venv --python 3.12 /tmp/nemovenv
    uv pip install --python /tmp/nemovenv/bin/python \
        --index-url https://download.pytorch.org/whl/cpu torch torchaudio
    uv pip install --python /tmp/nemovenv/bin/python \
        "nemo_toolkit[asr] @ git+https://github.com/NVIDIA-NeMo/Speech@6f558408dbe1c2a7e6b730184e8d46e1474d6dca"
    /tmp/nemovenv/bin/python -I tools/gen_sortformer_v2_toy_fixtures.py

Pinned versions used to produce the committed fixtures:
    nemo_toolkit 3.1.0+6f558408d, torch 2.14.1+cpu, safetensors 0.8.0, numpy 2.5.4

A tiny seeded `SortformerEncLabelModel` in Sortformer v2's layout (the
geometry of `tools/gen_nemo_fixtures.py`'s toy: a 2-layer FastConformer of
width 16, dw_striding x8 with 4 channels, conv kernel 3, xscaling; a
projection to 8; a 2-layer post-LN Transformer of width 8, 2 heads, inner 16,
ReLU; 3 speakers; 24 mels). Its weights are drawn in float32 and the model is
run on the CPU in float64, so every recorded value is the reference's answer to
weights bunsen holds exactly.

Each case runs NeMo's offline path stage by stage, on mels given directly (the
front end is held to NeMo on its own): `encoder.pre_encode`, the encoder past it
(`bypass_pre_encode`), `encoder_proj`, the Transformer under its length mask,
and the head's sigmoid, times the mask.

Outputs, under `crates/public/bunsen/testdata/sortformer/`:

`v2_toy.safetensors`
    `weights.*`: the toy's state dict under NeMo's names, float32, the entries
    the loader drops included (`num_batches_tracked`, the featurizer's tables,
    `hidden_to_spks`).
    `io.{case}.*`: per case, `mels [B, T, 24]` in, and `pre_encoded`,
    `encoded`, `projected`, `transformed`, `probs` out, each `[B, T', *]`.
`v2_toy.json`
    The cases' lengths, in mel frames and in encoder frames.
"""

import json
import pathlib

import numpy as np
import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

from nemo.collections.asr.models import SortformerEncLabelModel

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "crates/public/bunsen/testdata/sortformer"

N_MELS, N_SPK = 24, 3

# Mel lengths per item: one a multiple of the subsampling, one not, and a
# padded batch of two.
CASES = {"even": [40], "ragged": [37], "batch": [40, 29]}

CONFIG = dict(
    sample_rate=16000,
    pil_weight=0.5,
    ats_weight=0.5,
    max_num_of_spks=N_SPK,
    loss=dict(_target_="nemo.collections.asr.losses.bce_loss.BCELoss", weight=None, reduction="mean"),
    preprocessor=dict(
        _target_="nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
        normalize="NA", window_size=0.025, sample_rate=16000, window_stride=0.01, window="hann",
        features=N_MELS, n_fft=512, frame_splicing=1, dither=0.0,
    ),
    streaming_mode=False,
    model_defaults=dict(fc_d_model=16, tf_d_model=8),
    sortformer_modules=dict(
        _target_="nemo.collections.asr.modules.sortformer_modules.SortformerModules",
        num_spks=N_SPK, dropout_rate=0.0, fc_d_model=16, tf_d_model=8, chunk_left_context=1,
        chunk_right_context=1, causal_attn_rate=0, spkcache_len=12, fifo_len=6, chunk_len=3,
        spkcache_update_period=4, spkcache_sil_frames_per_spk=1,
    ),
    encoder=dict(
        _target_="nemo.collections.asr.modules.ConformerEncoder",
        feat_in=N_MELS, feat_out=-1, n_layers=2, d_model=16, subsampling="dw_striding", subsampling_factor=8,
        subsampling_conv_channels=4, causal_downsampling=False, ff_expansion_factor=2,
        self_attention_model="rel_pos", n_heads=2, att_context_size=[-1, -1], att_context_style="regular",
        xscaling=True, untie_biases=True, pos_emb_max_len=5000, conv_kernel_size=3, conv_norm_type="batch_norm",
        conv_context_size=None, dropout=0.0, dropout_pre_encoder=0.0, dropout_emb=0.0, dropout_att=0.0,
    ),
    transformer_encoder=dict(
        _target_="nemo.collections.asr.modules.transformer.transformer_encoders.TransformerEncoder",
        num_layers=2, hidden_size=8, inner_size=16, num_attention_heads=2, attn_score_dropout=0.0,
        attn_layer_dropout=0.0, ffn_dropout=0.0, hidden_act="relu", pre_ln=False, pre_ln_final_layer_norm=True,
    ),
)


def build(seed: int) -> SortformerEncLabelModel:
    """The toy, its weights drawn in float32, as float64."""
    torch.manual_seed(seed)
    model = SortformerEncLabelModel(cfg=OmegaConf.create(CONFIG)).float().eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if ("norm" in name or "layer_norm" in name) and name.endswith(".weight") and p.ndim == 1:
                p.copy_(1.0 + 0.2 * torch.randn_like(p))
            elif p.ndim == 1 and name.endswith(".bias"):
                p.copy_(0.1 * torch.randn_like(p))
            else:
                fan_in = p.shape[1] if p.ndim >= 2 else 4
                for d in p.shape[2:]:
                    fan_in *= d
                p.copy_(torch.randn_like(p) / np.sqrt(fan_in))
        model.sortformer_modules.single_hidden_to_spks.weight.mul_(3.0)
        model.sortformer_modules.single_hidden_to_spks.bias.add_(0.7)
        for name, b in model.named_buffers():
            if name.endswith("running_mean"):
                b.copy_(0.3 * torch.randn_like(b))
            elif name.endswith("running_var"):
                b.copy_(0.5 + torch.rand_like(b))
    return model.double()


def mels(lengths: list, seed: int) -> torch.Tensor:
    """`[B, max(lengths), N_MELS]` log-mel-sized values, zero past each item's length."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(len(lengths), max(lengths), N_MELS, generator=g, dtype=torch.float64) * 2.0 - 4.0
    for b, n in enumerate(lengths):
        x[b, n:] = 0.0
    return x


def stages(model: SortformerEncLabelModel, x: torch.Tensor, lengths: torch.Tensor) -> dict:
    out = {}
    with torch.no_grad():
        pre, pre_lengths = model.encoder.pre_encode(x=x, lengths=lengths)
        out["pre_encoded"] = pre
        encoded, enc_lengths = model.encoder(audio_signal=pre, length=pre_lengths, bypass_pre_encode=True)
        out["encoded"] = encoded.transpose(1, 2)
        projected = model.sortformer_modules.encoder_proj(out["encoded"])
        out["projected"] = projected
        mask = model.sortformer_modules.length_to_mask(enc_lengths, projected.shape[1])
        out["transformed"] = model.transformer_encoder(encoder_states=projected, encoder_mask=mask)
        out["probs"] = model.forward_infer(projected, enc_lengths)
    return out, [int(v) for v in enc_lengths]


def main() -> None:
    model = build(seed=11)
    tensors = {f"weights.{k}": v.float().contiguous() for k, v in model.state_dict().items()}
    for name, v in model.state_dict().items():
        if v.dtype == torch.int64:
            tensors[f"weights.{name}"] = v.contiguous()
    meta = {"cases": {}}
    for case, lengths in CASES.items():
        x = mels(lengths, seed=len(case) + sum(lengths))
        out, enc_lengths = stages(model, x, torch.tensor(lengths))
        tensors[f"io.{case}.mels"] = x.float().contiguous()
        for name, t in out.items():
            tensors[f"io.{case}.{name}"] = t.float().contiguous()
        meta["cases"][case] = {"mel_lengths": lengths, "encoder_lengths": enc_lengths}
        print(f"{case}: mels {list(x.shape)} -> probs {list(out['probs'].shape)}, lengths {enc_lengths}")

    OUT.mkdir(parents=True, exist_ok=True)
    save_file(tensors, OUT / "v2_toy.safetensors")
    meta["source"] = "tools/gen_sortformer_v2_toy_fixtures.py: SortformerEncLabelModel(seed 11), float64"
    meta["runtime"] = "nemo_toolkit 3.1.0+6f558408d, torch 2.14.1+cpu"
    (OUT / "v2_toy.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT / 'v2_toy.safetensors'}")


if __name__ == "__main__":
    main()
