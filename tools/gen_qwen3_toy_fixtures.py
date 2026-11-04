#!/usr/bin/env python3
"""Generate the Qwen3 toy-model fixtures.

Run once; commit the outputs. Regenerating is not part of CI.

    python3 -m venv /tmp/pkvenv
    /tmp/pkvenv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
    /tmp/pkvenv/bin/pip install transformers safetensors
    /tmp/pkvenv/bin/python -I tools/gen_qwen3_toy_fixtures.py

Pinned versions used to produce the committed fixture:
    transformers 5.19.0, torch 2.14.1+cpu, safetensors 0.8.0

Two tiny `Qwen3ForCausalLM`s -- `transformers`' own implementation -- with seeded random
weights, run on the CPU in float64 and stored as float32. Each keeps the shape rules that
matter and nothing else: a head dimension that is not `hidden_size / num_attention_heads`,
and grouped-query attention. `toy` has a tied head, `toy_untied` a head of its own (as
`Qwen3-8B` has).

Outputs, under `crates/public/bunsen/testdata/qwen3/`:

`toy.safetensors`
    - `model.*`: the toy's weights under their state-dict names.
    - `lm_head.weight`: NOT the tied head but random values, as a checkpoint may carry a
      copy of it. `transformers` does not read it under `tie_word_embeddings`; a
      loader that did would not reproduce `io.logits`.
    - `io.*`: `input_ids` `[1, 7]`; the outputs of `transformers`' forward over them:
      `layer0` (the first decoder layer's output), `hidden` (after the final norm) and
      `logits`.
`toy.config.json`
    The toy's `config.json`, as `save_pretrained` writes it.
`toy_untied.safetensors`, `toy_untied.config.json`
    The same for the untied toy: `lm_head.weight` is its head, which `io.logits` are
    computed through.
"""

import pathlib
import sys

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from toy_fixture import (  # noqa: E402
    TESTDATA, Capture, build, init_weights, io_tensors, shapes, weights, write,
    write_config_json
)

OUT = TESTDATA / "qwen3"

CONFIG = dict(
    hidden_size=24,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    intermediate_size=48,
    num_hidden_layers=2,
    vocab_size=50,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    hidden_act="silu",
    attention_bias=False,
    tie_word_embeddings=True,
    max_position_embeddings=512,
    use_sliding_window=False,
)

INPUT_IDS = [1, 5, 7, 2, 9, 49, 0]


def write_toy(name: str, tied: bool) -> None:
    config = Qwen3Config(**{**CONFIG, "tie_word_embeddings": tied})
    model = build(lambda: Qwen3ForCausalLM(config), seed=1234)
    init_weights(model)
    if tied:
        model.tie_weights()
        assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    else:
        assert model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()

    capture = Capture()
    capture.output("layer0", model.model.layers[0])
    input_ids = torch.tensor([INPUT_IDS])
    with torch.no_grad():
        hidden = model.model(input_ids=input_ids).last_hidden_state
        logits = model(input_ids=input_ids).logits

    io = {
        "input_ids": input_ids.to(torch.float64),
        "layer0": capture.values["layer0"],
        "hidden": hidden,
        "logits": logits,
    }
    if tied:
        tensors = weights(model, drop=["lm_head.weight"])
        torch.manual_seed(99)
        tensors["lm_head.weight"] = torch.randn(CONFIG["vocab_size"], CONFIG["hidden_size"])
    else:
        tensors = weights(model)
    write(OUT / f"{name}.safetensors", tensors, io_tensors(io))
    write_config_json(OUT / f"{name}.config.json", config, ["Qwen3ForCausalLM"])
    print("wrote", name, shapes(io))


def main() -> None:
    write_toy("toy", tied=True)
    write_toy("toy_untied", tied=False)


if __name__ == "__main__":
    main()
