#!/usr/bin/env python3
"""Generate the dense Qwen3.5 toy-model fixtures.

Run once; commit the outputs. Regenerating is not part of CI.

    python3 -m venv /tmp/pkvenv
    /tmp/pkvenv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
    /tmp/pkvenv/bin/pip install transformers safetensors
    /tmp/pkvenv/bin/python -I tools/gen_qwen3_5_toy_fixtures.py

Pinned versions used to produce the committed fixture:
    transformers 5.19.0, torch 2.14.1+cpu, safetensors 0.8.0

Two tiny `Qwen3_5ForConditionalGeneration`s -- `transformers`' own implementation, with
eager attention -- with seeded random weights, run on the CPU in float64 and stored as
float32. They are saved as the published checkpoints are: a `config.json` with
`text_config` and `vision_config` sections, and a state dict holding the text model
under `model.language_model.`, a tiny vision tower under `model.visual.` and a
multi-token-prediction head under `mtp.` beside it (`transformers` builds no `mtp`
module and ignores those tensors on load; they are random here, so a loader that read
them would be caught). Each makes the hybrid's features bind:

- linear-attention (gated DeltaNet) and full-attention layers, each type occurring at
  least once in four layers: the 3:1 pattern (`toy`) and 1:1 (`toy_untied`);
- the full layers' gated attention (`q_proj` of twice the query width), QK-norm, a head
  dimension that is not `hidden_size / num_attention_heads`, grouped-query attention;
- partial rotary, a quarter of the head dimension, through `transformers`' M-RoPE on
  text positions;
- linear key and value heads of different widths, equal in number (`toy`, as the 0.8B
  and 2B) and with the value heads twice the key heads (`toy_untied`, as the 4B and 9B);
- 72 input tokens, more than the chunked delta rule's 64, so the reference crosses a
  chunk boundary;
- random norm weights: about zero for the `1 + w` norms (`transformers` initializes them
  to zero, which would leave the `1 + w` untested), about one for the DeltaNet's gated
  norm, which is applied as stored.

`toy` has a tied head, `toy_untied` a head of its own (as `Qwen3.5-9B` has).

`transformers` computes some parts in float32 whatever the model's dtype: the norms, the
delta rule (both forms), the decay `g` and the rotary angles. The outputs are float64
everywhere else, so they hold a float32 implementation to float32's precision.

Outputs, under `crates/public/bunsen/testdata/qwen3_5/`:

`toy.safetensors`
    - `model.language_model.*`, `model.visual.*`, `mtp.*`: the toy's weights.
    - `lm_head.weight`: NOT the tied head but random values, as a checkpoint may carry a
      copy of it. `transformers` does not read it under `tie_word_embeddings`; a loader
      that did would not reproduce `io.logits`.
    - `io.*`: `input_ids` `[1, 72]`; the outputs of `transformers`' forward over them:
      - `layer0`, the first decoder layer's output (a linear layer);
      - `layer0_core`, that layer's delta-rule output before its gated norm,
        `[1, 72, H_v, d_v]`;
      - `full0_gated`, the first full layer's attention output after its sigmoid gate
        and before `o_proj`, `[1, 72, H * head_dim]`;
      - `full0`, that full layer's output;
      - `hidden` (after the final norm) and `logits`.
`toy.config.json`
    The toy's `config.json`, as `save_pretrained` writes it.
`toy_untied.safetensors`, `toy_untied.config.json`
    The same for the untied toy: `lm_head.weight` is its head, which `io.logits` are
    computed through.
"""

import pathlib
import sys

import numpy as np
import torch
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from toy_fixture import (  # noqa: E402
    TESTDATA, Capture, build, fan_in, io_tensors, shapes, weights, write, write_config_json
)

OUT = TESTDATA / "qwen3_5"

L = "linear_attention"
F = "full_attention"

TEXT = dict(
    hidden_size=32,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    intermediate_size=48,
    num_hidden_layers=4,
    vocab_size=64,
    rms_norm_eps=1e-6,
    hidden_act="silu",
    attention_bias=False,
    attn_output_gate=True,
    linear_conv_kernel_dim=4,
    linear_key_head_dim=8,
    linear_value_head_dim=12,
    linear_num_key_heads=2,
    linear_num_value_heads=2,
    layer_types=[L, L, L, F],
    max_position_embeddings=512,
    rope_parameters={
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.25,
        "mrope_section": [1, 1, 0],
        "mrope_interleaved": True,
    },
    tie_word_embeddings=True,
    eos_token_id=1,
)

VISION = dict(
    depth=1,
    hidden_size=16,
    intermediate_size=32,
    num_heads=2,
    in_channels=3,
    patch_size=2,
    temporal_patch_size=2,
    spatial_merge_size=2,
    num_position_embeddings=4,
    out_hidden_size=32,
)

TOYS = [
    ("toy", dict(TEXT)),
    (
        "toy_untied",
        dict(
            TEXT,
            num_key_value_heads=1,
            linear_num_value_heads=4,
            layer_types=[L, F, L, F],
            tie_word_embeddings=False,
        ),
    ),
]

INPUT_IDS = [
    (7 * i + 3) % 64 for i in range(72)
]

ZERO_CENTERED = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "q_norm.weight",
    "k_norm.weight",
)


def is_zero_centered(name: str) -> bool:
    """A `1 + w` norm's gain: the layers' and the text model's final norm."""
    return name.endswith(ZERO_CENTERED) or name == "model.language_model.norm.weight"


@torch.no_grad()
def init(model: torch.nn.Module) -> None:
    """Seeded random parameters, in `named_parameters` order, one draw each: the `1 + w`
    gains `0.2 * N(0, 1)`, the gated norm's `1 + 0.2 * N(0, 1)`, the rest
    `N(0, 1) / sqrt(fan_in)`."""
    for name, p in model.named_parameters():
        if is_zero_centered(name):
            p.copy_(0.2 * torch.randn_like(p))
        elif name.endswith("linear_attn.norm.weight"):
            p.copy_(1.0 + 0.2 * torch.randn_like(p))
        else:
            p.copy_(torch.randn_like(p) / np.sqrt(fan_in(p)))


def mtp_tensors(hidden: int, generator: torch.Generator) -> dict:
    """A stand-in for the checkpoints' multi-token-prediction head: two of its tensors,
    random."""
    return {
        "mtp.fc.weight": torch.randn(hidden, 2 * hidden, generator=generator),
        "mtp.norm.weight": torch.randn(hidden, generator=generator),
    }


def write_toy(name: str, text: dict) -> None:
    config = Qwen3_5Config(text_config=text, vision_config=VISION, tie_word_embeddings=text["tie_word_embeddings"])
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    model = build(lambda: Qwen3_5ForConditionalGeneration(config), seed=3535)
    init(model)
    tied = text["tie_word_embeddings"]
    lm = model.model.language_model
    if tied:
        model.tie_weights()
        assert model.lm_head.weight.data_ptr() == lm.embed_tokens.weight.data_ptr()
    else:
        assert model.lm_head.weight.data_ptr() != lm.embed_tokens.weight.data_ptr()

    layer_types = text["layer_types"]
    first_linear = layer_types.index(L)
    first_full = layer_types.index(F)
    assert first_linear == 0

    capture = Capture()
    capture.output("layer0", lm.layers[0])
    capture.input("layer0_core", lm.layers[0].linear_attn.norm)
    capture.input("full0_gated", lm.layers[first_full].self_attn.o_proj)
    capture.output("full0", lm.layers[first_full])

    input_ids = torch.tensor([INPUT_IDS])
    with torch.no_grad():
        hidden = lm(input_ids=input_ids).last_hidden_state
        logits = model(input_ids=input_ids).logits

    t = len(INPUT_IDS)
    core = capture.values["layer0_core"].reshape(1, t, text["linear_num_value_heads"], text["linear_value_head_dim"])
    io = {
        "input_ids": input_ids.to(torch.float64),
        "layer0": capture.values["layer0"],
        "layer0_core": core,
        "full0_gated": capture.values["full0_gated"],
        "full0": capture.values["full0"],
        "hidden": hidden,
        "logits": logits,
    }

    generator = torch.Generator().manual_seed(53)
    extra = mtp_tensors(text["hidden_size"], generator)
    tensors = weights(model, drop=["lm_head.weight"] if tied else [])
    if tied:
        extra["lm_head.weight"] = torch.randn(text["vocab_size"], text["hidden_size"], generator=generator)
    write(OUT / f"{name}.safetensors", tensors, extra, io_tensors(io))
    write_config_json(OUT / f"{name}.config.json", config, ["Qwen3_5ForConditionalGeneration"])
    print("wrote", name, shapes(io))


def main() -> None:
    for toy in TOYS:
        write_toy(*toy)


if __name__ == "__main__":
    main()
