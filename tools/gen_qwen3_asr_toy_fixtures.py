#!/usr/bin/env python3
"""Generate the Qwen3-ASR toy-model fixture.

Run once; commit the outputs. Regenerating is not part of CI.

    python3 -m venv /tmp/pkvenv
    /tmp/pkvenv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
    /tmp/pkvenv/bin/pip install transformers safetensors
    /tmp/pkvenv/bin/python tools/gen_qwen3_asr_toy_fixtures.py

Pinned versions used to produce the committed fixture:
    transformers 5.19.0, torch 2.14.1+cpu, safetensors 0.8.0

A tiny `Qwen3ASRForConditionalGeneration` -- `transformers`' own implementation -- with
seeded random weights, run on the CPU in float64 and stored as float32. The toy keeps the
shape rules that matter and nothing else: chunks of `2 * n_window` mel frames, attention
windows of two chunks, a partial last chunk, a text head dimension that is not
`hidden_size / num_attention_heads`, grouped-query attention, and a tied head.

Output, `crates/public/bunsen/testdata/qwen3_asr/toy.safetensors`:

- `model.*`: the toy's weights under their state-dict names (the head is tied to
  `model.language_model.embed_tokens.weight` and not stored).
- `io.*`: the inputs and the outputs bunsen is pinned against: `features` `[1, mels, 300]`
  with `feature_len` valid frames; the audio tower's `conv` (the chunked convolution stem,
  positions added, valid frames kept), `audio_layer0` and `audio` (its final output);
  `audio_projected`; then the text model on `input_ids` (the audio placeholders replaced by
  `audio_projected`): `text_layer0`, `text_hidden` (after the final norm) and `logits`.
"""

import json
import pathlib

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import Qwen3ASRConfig, Qwen3ASRForConditionalGeneration

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "crates/public/bunsen/testdata/qwen3_asr"

AUDIO_TOKEN = 7
FEATURE_FRAMES = 300
FEATURE_LEN = 250

CONFIG = dict(
    audio_config=dict(
        d_model=32,
        encoder_attention_heads=4,
        encoder_ffn_dim=64,
        encoder_layers=2,
        num_mel_bins=16,
        downsample_hidden_size=8,
        output_dim=24,
        n_window=50,
        n_window_infer=200,
        max_position_embeddings=13,
        activation_function="gelu",
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        scale_embedding=False,
    ),
    text_config=dict(
        hidden_size=24,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=48,
        num_hidden_layers=2,
        vocab_size=50,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        hidden_act="silu",
        attention_bias=False,
        tie_word_embeddings=True,
        max_position_embeddings=512,
    ),
    audio_token_id=AUDIO_TOKEN,
    tie_word_embeddings=True,
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(4321)
    model = Qwen3ASRForConditionalGeneration(Qwen3ASRConfig(**CONFIG)).double().eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name and name.endswith(".weight"):
                p.copy_(1.0 + 0.2 * torch.randn_like(p))
            else:
                fan_in = p.shape[1] if p.ndim >= 2 else 4
                for d in p.shape[2:]:
                    fan_in *= d
                p.copy_(torch.randn_like(p) / np.sqrt(fan_in))
    model.tie_weights()

    torch.manual_seed(77)
    n_mels = CONFIG["audio_config"]["num_mel_bins"]
    features = torch.randn(1, n_mels, FEATURE_FRAMES, dtype=torch.float64)
    mask = (torch.arange(FEATURE_FRAMES) < FEATURE_LEN)[None].long()
    features = features * mask[:, None, :]

    tower = model.model.audio_tower
    captured = {}

    def keep(name):
        def hook(_m, _i, out):
            captured[name] = out[0] if isinstance(out, tuple) else out
        return hook

    tower.layers[0].register_forward_hook(keep("audio_layer0"))

    def conv_pre(_m, args, kwargs=None):
        captured["conv"] = args[0]
    tower.layers[0].register_forward_pre_hook(conv_pre)

    with torch.no_grad():
        audio = tower(input_features=features, input_features_mask=mask).last_hidden_state
        projected = model.model.multi_modal_projector(audio)
        n_audio = projected.shape[0]
        ids = [1, 2, 3] + [AUDIO_TOKEN] * n_audio + [4, 5, 6, 9]
        input_ids = torch.tensor([ids])
        lm = model.model.language_model
        lm.layers[0].register_forward_hook(keep("text_layer0"))
        out = model(input_ids=input_ids, input_features=features, input_features_mask=mask)
        hidden = model.model(input_ids=input_ids, input_features=features,
                             input_features_mask=mask).last_hidden_state

    io = {
        "features": features,
        "feature_len": torch.tensor([FEATURE_LEN], dtype=torch.float64),
        "conv": captured["conv"],
        "audio_layer0": captured["audio_layer0"],
        "audio": audio,
        "audio_projected": projected,
        "input_ids": input_ids.to(torch.float64),
        "text_layer0": captured["text_layer0"],
        "text_hidden": hidden,
        "logits": out.logits,
    }
    # Every weight name starts `model.`; the head is tied, so it is not stored.
    tensors = {k: v.float().contiguous()
               for k, v in model.state_dict().items() if k != "lm_head.weight"}
    tensors.update({f"io.{k}": v.float().contiguous() for k, v in io.items()})
    save_file(tensors, OUT / "toy.safetensors")
    (OUT / "toy.json").write_text(json.dumps({"config": CONFIG, "feature_len": FEATURE_LEN,
                                              "audio_frames": n_audio}, indent=1) + "\n")
    print("wrote", OUT, "audio frames", n_audio, {k: tuple(v.shape) for k, v in io.items()})


if __name__ == "__main__":
    main()
