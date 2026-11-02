#!/usr/bin/env python3
"""Generate the Qwen3 toy-model fixtures.

Run once; commit the outputs. Regenerating is not part of CI.

    python3 -m venv /tmp/pkvenv
    /tmp/pkvenv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
    /tmp/pkvenv/bin/pip install transformers safetensors
    /tmp/pkvenv/bin/python tools/gen_qwen3_toy_fixtures.py

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

import json
import pathlib

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import Qwen3Config, Qwen3ForCausalLM

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "crates/public/bunsen/testdata/qwen3"

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
    torch.manual_seed(1234)
    config = Qwen3Config(**{**CONFIG, "tie_word_embeddings": tied})
    model = Qwen3ForCausalLM(config).double().eval()
    with torch.no_grad():
        for pname, p in model.named_parameters():
            if "norm" in pname and pname.endswith(".weight"):
                p.copy_(1.0 + 0.2 * torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) / np.sqrt(p.shape[1] if p.ndim >= 2 else 4))
    if tied:
        model.tie_weights()
        assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    else:
        assert model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()

    captured = {}
    model.model.layers[0].register_forward_hook(
        lambda _m, _i, out: captured.__setitem__("layer0", out[0] if isinstance(out, tuple) else out)
    )
    input_ids = torch.tensor([INPUT_IDS])
    with torch.no_grad():
        hidden = model.model(input_ids=input_ids).last_hidden_state
        logits = model(input_ids=input_ids).logits

    io = {
        "input_ids": input_ids.to(torch.float64),
        "layer0": captured["layer0"],
        "hidden": hidden,
        "logits": logits,
    }
    if tied:
        tensors = {k: v.float().contiguous()
                   for k, v in model.state_dict().items() if k != "lm_head.weight"}
        torch.manual_seed(99)
        tensors["lm_head.weight"] = torch.randn(CONFIG["vocab_size"], CONFIG["hidden_size"])
    else:
        tensors = {k: v.float().contiguous() for k, v in model.state_dict().items()}
    tensors.update({f"io.{k}": v.float().contiguous() for k, v in io.items()})
    save_file(tensors, OUT / f"{name}.safetensors")

    saved = config.to_dict()
    for k in ("transformers_version", "dtype", "torch_dtype"):
        saved.pop(k, None)
    saved["architectures"] = ["Qwen3ForCausalLM"]
    (OUT / f"{name}.config.json").write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n")
    print("wrote", name, {k: tuple(v.shape) for k, v in io.items()})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_toy("toy", tied=True)
    write_toy("toy_untied", tied=False)


if __name__ == "__main__":
    main()
