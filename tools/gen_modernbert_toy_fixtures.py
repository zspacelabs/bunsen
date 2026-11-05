#!/usr/bin/env python3
"""Generate the ModernBERT toy-model fixture.

Run once; commit the outputs. Regenerating is not part of CI.

    python3 -m venv /tmp/pkvenv
    /tmp/pkvenv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
    /tmp/pkvenv/bin/pip install transformers safetensors
    /tmp/pkvenv/bin/python -I tools/gen_modernbert_toy_fixtures.py

Pinned versions used to produce the committed fixture:
    transformers 5.19.0, torch 2.14.1+cpu, safetensors 0.8.0

A tiny `ModernBertModel` -- `transformers`' own implementation, with eager attention --
with seeded random weights, run on the CPU in float64 and stored as float32. Its
features are set to bind:

- local layers among global ones, on the default pattern (every third global, from the
  first), with a two-sided window of +-2 positions (`local_attention` 4) against 12
  tokens;
- two rotary bases, different enough to tell apart (1000 global, 100 local);
- random norm gains about one (`transformers` initializes them to one), so a misplaced
  norm shows; no norm or projection has a bias;
- a padded batch: item 1 is 9 tokens of 12, its last 3 padding, which no query may
  attend to.

The references are each item run alone, unpadded: what padding must not change.
`transformers`' eager path over the padded batch is no reference for item 1. Its last
position's local window holds only padding, a softmax over no keys, which is NaN; the
next global layer's zero weights carry the NaN into every position of the item
(`0 * NaN`). The unpadded paths (`ModernBERT`'s FlashAttention) do not see it.

Outputs, under `crates/public/bunsen/testdata/modernbert/`:

`modernbert_toy.safetensors`
    - the toy's weights under their `ModernBertModel` state-dict names (`Wqkv` and `Wi`
      fused, as upstream stores them);
    - `io.input_ids` `[2, 12]` (item 1 padded with `pad_token_id`), `io.lengths` `[2]`
      (12 and 9); the outputs of `transformers`' forward over each item alone, zero at
      the padding: `layer0` (the first layer's output), `layer1` (the second's, a local
      layer) and `hidden` (after the final norm).
`modernbert_toy.config.json`
    The toy's `config.json`, as `save_pretrained` writes it.
"""

import pathlib
import sys

import torch
from transformers import ModernBertConfig, ModernBertModel

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from toy_fixture import (  # noqa: E402
    TESTDATA, Capture, build, init_weights, io_tensors, shapes, weights, write,
    write_config_json
)

OUT = TESTDATA / "modernbert"

CONFIG = dict(
    vocab_size=40,
    hidden_size=16,
    intermediate_size=24,
    num_hidden_layers=4,
    num_attention_heads=2,
    hidden_activation="gelu",
    max_position_embeddings=64,
    norm_eps=1e-5,
    norm_bias=False,
    attention_bias=False,
    mlp_bias=False,
    local_attention=4,
    global_attn_every_n_layers=3,
    global_rope_theta=1000.0,
    local_rope_theta=100.0,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
    cls_token_id=1,
    sep_token_id=2,
)

SEQ = 12
LENGTHS = [12, 9]


def main():
    config = ModernBertConfig(**CONFIG)
    config._attn_implementation = "eager"
    model = build(lambda: ModernBertModel(config), seed=4321)
    init_weights(model)

    generator = torch.Generator().manual_seed(7)
    ids = torch.randint(3, CONFIG["vocab_size"], (len(LENGTHS), SEQ), generator=generator)
    for row, n in enumerate(LENGTHS):
        ids[row, n:] = CONFIG["pad_token_id"]

    capture = Capture()
    capture.output("layer0", model.layers[0])
    capture.output("layer1", model.layers[1])
    outputs = {name: torch.zeros(len(LENGTHS), SEQ, CONFIG["hidden_size"], dtype=torch.float64)
               for name in ("layer0", "layer1", "hidden")}
    for row, n in enumerate(LENGTHS):
        with torch.no_grad():
            hidden = model(input_ids=ids[row:row + 1, :n]).last_hidden_state
        for name, value in {**capture.values, "hidden": hidden}.items():
            outputs[name][row, :n] = value[0]

    io = io_tensors({
        **outputs,
        "lengths": torch.tensor(LENGTHS, dtype=torch.float64),
    })
    io["io.input_ids"] = ids.to(torch.int64).contiguous()
    tensors = weights(model)
    write(OUT / "modernbert_toy.safetensors", tensors, io)
    write_config_json(OUT / "modernbert_toy.config.json", config, ["ModernBertModel"])
    print(shapes({**tensors, **io}))


if __name__ == "__main__":
    main()
