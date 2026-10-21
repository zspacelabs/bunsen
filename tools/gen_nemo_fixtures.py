#!/usr/bin/env python3
"""Generate the `.nemo` archive fixture: a tiny Sortformer v2, saved by NeMo itself.

Run once; commit the output. Regenerating is not part of CI.

    uv venv --python 3.12 /tmp/nemovenv
    uv pip install --python /tmp/nemovenv/bin/python \
        --index-url https://download.pytorch.org/whl/cpu torch torchaudio
    uv pip install --python /tmp/nemovenv/bin/python \
        "nemo_toolkit[asr] @ git+https://github.com/NVIDIA-NeMo/Speech@6f558408dbe1c2a7e6b730184e8d46e1474d6dca"
    /tmp/nemovenv/bin/python -I tools/gen_nemo_fixtures.py

Pinned versions used to produce the committed fixture:
    nemo_toolkit 3.1.0+6f558408d, torch 2.14.1+cpu, numpy 2.5.4

A seeded `SortformerEncLabelModel` in Sortformer v2's layout, at a toy geometry:
a FastConformer (dw_striding x8 with 4 channels, 2 layers of width 16, 2 heads,
conv kernel 3, xscaling), a 2-layer post-LN Transformer of width 8 (2 heads,
inner 16, ReLU), 3 speakers, 24 mels. Its weights are drawn as the toy parity
fixtures draw them (unit-ish norms, small biases, fan-in scaled matrices, batch
norm statistics away from 0 and 1), in float32, as NeMo's own checkpoints are.

Output, in `crates/public/bunsen/testdata/nemo/`:

    toy_sortformer_v2.nemo   `model.save_to(...)`: NeMo's tar of
                             `model_config.yaml` and `model_weights.ckpt`
    toy_sortformer_v2.json   every state-dict entry: dtype, shape, and the sum
                             of its values (float64), to hold a reader to
"""

import json
import pathlib

import numpy as np
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.models import SortformerEncLabelModel

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "crates/public/bunsen/testdata/nemo"

N_MELS, N_SPK = 24, 3

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
    streaming_mode=True,
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
        for name, b in model.named_buffers():
            if name.endswith("running_mean"):
                b.copy_(0.3 * torch.randn_like(b))
            elif name.endswith("running_var"):
                b.copy_(0.5 + torch.rand_like(b))
    return model


def main() -> None:
    model = build(seed=7)
    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / "toy_sortformer_v2.nemo"
    model.save_to(str(archive))

    tensors = {}
    for name, value in model.state_dict().items():
        tensors[name] = {
            "dtype": str(value.dtype).removeprefix("torch."),
            "shape": list(value.shape),
            "sum": float(value.double().sum()),
        }
    manifest = {
        "source": "tools/gen_nemo_fixtures.py: SortformerEncLabelModel(seed 7).save_to",
        "runtime": "nemo_toolkit 3.1.0+6f558408d, torch 2.14.1+cpu",
        "tensors": tensors,
    }
    (OUT / "toy_sortformer_v2.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(f"wrote {archive} ({archive.stat().st_size} bytes), {len(tensors)} tensors")


if __name__ == "__main__":
    main()
