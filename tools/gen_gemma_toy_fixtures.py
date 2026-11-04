#!/usr/bin/env python3
"""Generate the Gemma toy-model fixtures.

Run once; commit the outputs. Regenerating is not part of CI.

    python3 -m venv /tmp/pkvenv
    /tmp/pkvenv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
    /tmp/pkvenv/bin/pip install transformers safetensors
    /tmp/pkvenv/bin/python -I tools/gen_gemma_toy_fixtures.py

Pinned versions used to produce the committed fixture:
    transformers 5.19.0, torch 2.14.1+cpu, safetensors 0.8.0

Tiny `Gemma3ForCausalLM`s, a `Gemma2ForCausalLM` and a `GemmaForCausalLM` --
`transformers`' own implementations, with eager attention (sdpa drops Gemma 2's
attention soft-cap) -- with seeded random weights, run on the CPU in float64 and stored
as float32. Each makes the features of its generation bind:

- a head dimension that is not `hidden_size / num_attention_heads`;
- a `query_pre_attn_scalar` that is not the head dimension (Gemma 2 and 3);
- a sliding window of 4 against 16 tokens, on a `layer_types` that is not the default
  (Gemma 2 and 3);
- two rotary tables, with a `linear` factor on the full layers' table only (Gemma 3);
- grouped-query attention (`toy`, Gemma 2) and multi-query attention (`toy_mqa`,
  Gemma 1);
- attention and final-logit soft-caps small enough to bind, 2 and 3 (Gemma 2);
- random norm weights, about zero: `transformers` initializes every Gemma norm's `w` to
  zero, which would leave its `1 + w` untested.

Every head is tied, as every Gemma checkpoint's is; the files carry no `lm_head.weight`.

Outputs, under `crates/public/bunsen/testdata/gemma/`, for each of `gemma3_toy`,
`gemma3_toy_mqa`, `gemma2_toy` and `gemma_toy`:

`{name}.safetensors`
    - `model.*`: the toy's weights under their state-dict names.
    - `io.*`: `input_ids` `[1, 16]`; the outputs of `transformers`' forward over them:
      `layer0` (the first decoder layer's output), `hidden` (after the final norm) and
      `logits` (after any final soft-cap).
`{name}.config.json`
    The toy's `config.json`, as `save_pretrained` writes it.
"""

import pathlib
import sys

import torch
from transformers import (
    Gemma2Config, Gemma2ForCausalLM, Gemma3ForCausalLM, Gemma3TextConfig, GemmaConfig,
    GemmaForCausalLM
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from toy_fixture import (  # noqa: E402
    TESTDATA, Capture, build, init_weights, io_tensors, shapes, weights, write,
    write_config_json
)

OUT = TESTDATA / "gemma"

S = "sliding_attention"
F = "full_attention"

COMMON = dict(
    hidden_size=24,
    num_attention_heads=4,
    head_dim=8,
    intermediate_size=48,
    vocab_size=50,
    rms_norm_eps=1e-6,
    hidden_activation="gelu_pytorch_tanh",
    attention_bias=False,
    tie_word_embeddings=True,
    max_position_embeddings=512,
    pad_token_id=0,
    eos_token_id=1,
    bos_token_id=2,
)

GEMMA3 = dict(
    COMMON,
    num_key_value_heads=2,
    num_hidden_layers=3,
    query_pre_attn_scalar=12,
    sliding_window=4,
    layer_types=[S, F, S],
    rope_parameters={
        S: {"rope_type": "default", "rope_theta": 10000.0},
        F: {"rope_type": "linear", "factor": 2.0, "rope_theta": 1000000.0},
    },
)

GEMMA3_MQA = dict(GEMMA3, num_key_value_heads=1, layer_types=[F, S, S])

GEMMA2 = dict(
    COMMON,
    num_key_value_heads=2,
    num_hidden_layers=3,
    query_pre_attn_scalar=12,
    sliding_window=4,
    layer_types=[F, S, S],
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    attn_logit_softcapping=2.0,
    final_logit_softcapping=3.0,
)

GEMMA = dict(
    COMMON,
    num_key_value_heads=1,
    num_hidden_layers=2,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
)

TOYS = [
    ("gemma3_toy", Gemma3TextConfig, Gemma3ForCausalLM, GEMMA3, "Gemma3ForCausalLM"),
    ("gemma3_toy_mqa", Gemma3TextConfig, Gemma3ForCausalLM, GEMMA3_MQA, "Gemma3ForCausalLM"),
    ("gemma2_toy", Gemma2Config, Gemma2ForCausalLM, GEMMA2, "Gemma2ForCausalLM"),
    ("gemma_toy", GemmaConfig, GemmaForCausalLM, GEMMA, "GemmaForCausalLM"),
]

INPUT_IDS = [2, 5, 7, 3, 9, 49, 0, 12, 33, 7, 18, 4, 41, 26, 1, 30]


def write_toy(name: str, config_class, model_class, fields: dict, arch: str) -> None:
    config = config_class(**fields)
    config._attn_implementation = "eager"
    model = build(lambda: model_class(config), seed=4321)
    init_weights(model, norm_center=0.0)
    model.tie_weights()
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()

    capture = Capture()
    capture.output("layer0", model.model.layers[0])
    capture.output("head", model.lm_head)
    input_ids = torch.tensor([INPUT_IDS])
    with torch.no_grad():
        hidden = model.model(input_ids=input_ids).last_hidden_state
        logits = model(input_ids=input_ids).logits

    cap = fields.get("final_logit_softcapping")
    if cap is not None:
        head = capture.values["head"]
        assert head.abs().max() > cap, f"{name}: the final cap {cap} does not bind"
        assert torch.allclose(logits, cap * torch.tanh(head / cap))

    io = {
        "input_ids": input_ids.to(torch.float64),
        "layer0": capture.values["layer0"],
        "hidden": hidden,
        "logits": logits,
    }
    write(OUT / f"{name}.safetensors", weights(model, drop=["lm_head.weight"]), io_tensors(io))
    write_config_json(OUT / f"{name}.config.json", config, [arch])
    print("wrote", name, shapes(io))


def main() -> None:
    for toy in TOYS:
        write_toy(*toy)


if __name__ == "__main__":
    main()
