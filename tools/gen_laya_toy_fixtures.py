#!/usr/bin/env python3
"""Generate the Laya toy-model fixture.

Run once; commit the outputs. Regenerating is not part of CI.

    python3 -m venv /tmp/pkvenv
    /tmp/pkvenv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
    /tmp/pkvenv/bin/pip install transformers safetensors
    /tmp/pkvenv/bin/python -I tools/gen_laya_toy_fixtures.py

Pinned versions used to produce the committed fixture:
    transformers 5.19.0, torch 2.14.1+cpu, safetensors 0.8.0

A tiny Laya `DecisionModel` with seeded random weights, run on the CPU in float64 and
stored as float32. `DecisionModel` below is upstream's (`rl_common.py`,
`convaiinnovations/laya`, Apache-2.0), its forward kept as written, without the
training-only paths (checkpointing, a detached encoder) and with its `.float()` casts in
the run's dtype, float64; its encoder is a tiny
`transformers` `ModernBertModel` with eager attention, local and global layers.

Two prompts, of 12 and 9 tokens: a 3-option `choice` and a `noul` (2 options, a third
marker slot unused). The references are each prompt run alone, unpadded: what padding
must not change (`transformers`' eager `ModernBERT` over a padded batch is NaN for a
padded item whose padding fills a local window).

Outputs, under `crates/public/bunsen/testdata/laya/`:

`laya_toy.safetensors`
    - the model's state dict under upstream's names (`encoder.*` with `Wqkv` and `Wi`
      fused, `head.layers.N.*` with `in_proj_*` fused, `type_emb`, `scorer.{0,1,3}`,
      `act_head.{0,2}`, `temperature`);
    - `io.input_ids` `[2, 12]` (padded with 0), `io.lengths` `[2]`, `io.kinds` `[2]`,
      `io.markers` `[2, 3]`, `io.options` `[2, 3]` (1 at a question's options); the
      outputs of each prompt alone: `head` (the head's states, zero at the padding),
      `logits` `[2, 3]` (`-1e4` past a question's options) and `act_logits` `[2, 2]`.
`laya_toy.encoder.config.json`
    The encoder's `config.json`, as `save_pretrained` writes it.
"""

import pathlib
import sys

import torch
import torch.nn as nn
from transformers import ModernBertConfig, ModernBertModel

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from toy_fixture import (  # noqa: E402
    TESTDATA, build, init_weights, io_tensors, shapes, weights, write, write_config_json
)

OUT = TESTDATA / "laya"

ENCODER = dict(
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
KINDS = [0, 2]
MARKERS = [[2, 5, 8], [3, 6, 0]]
OPTIONS = [[1, 1, 1], [1, 1, 0]]


class DecisionModel(nn.Module):
    """Upstream's `DecisionModel` (`rl_common.py`), as inference runs it."""

    def __init__(self, encoder: nn.Module, head_layers: int = 2, n_act: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        nhead = max(1, d // 64)
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False) if head_layers > 0 else None
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
        self.register_buffer("temperature", torch.ones(3))

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = h + self.type_emb(qtype)[:, None, :]
        pad = ~attention_mask.bool()
        for layer in self.head.layers:
            h = layer(h, src_key_padding_mask=pad)
        head = h
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).to(h.dtype)
        logits = logits.masked_fill(~marker_mask, -1e4)
        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).to(h.dtype)
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        top2 = p.topk(2, -1).values
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        pooled = h[:, 0]
        act_logits = self.act_head(torch.cat([pooled, feats], -1))
        return head, logits, act_logits


def main():
    config = ModernBertConfig(**ENCODER)
    config._attn_implementation = "eager"
    model = build(lambda: DecisionModel(ModernBertModel(config)), seed=2468)
    init_weights(model, norm_bias=0.1)

    generator = torch.Generator().manual_seed(9)
    ids = torch.randint(3, ENCODER["vocab_size"], (len(LENGTHS), SEQ), generator=generator)
    for row, n in enumerate(LENGTHS):
        ids[row, n:] = ENCODER["pad_token_id"]
    markers = torch.tensor(MARKERS)
    options = torch.tensor(OPTIONS).bool()
    kinds = torch.tensor(KINDS)

    head = torch.zeros(len(LENGTHS), SEQ, ENCODER["hidden_size"], dtype=torch.float64)
    logits, act_logits = [], []
    for row, n in enumerate(LENGTHS):
        with torch.no_grad():
            h, lg, act = model(
                ids[row:row + 1, :n],
                torch.ones(1, n, dtype=torch.long),
                markers[row:row + 1],
                options[row:row + 1],
                kinds[row:row + 1],
            )
        head[row, :n] = h[0]
        logits.append(lg[0])
        act_logits.append(act[0])

    io = io_tensors({
        "head": head,
        "logits": torch.stack(logits),
        "act_logits": torch.stack(act_logits),
        "lengths": torch.tensor(LENGTHS, dtype=torch.float64),
        "options": options.double(),
    })
    io["io.input_ids"] = ids.to(torch.int64).contiguous()
    io["io.kinds"] = kinds.to(torch.int64).contiguous()
    io["io.markers"] = markers.to(torch.int64).contiguous()
    tensors = weights(model)
    write(OUT / "laya_toy.safetensors", tensors, io)
    write_config_json(OUT / "laya_toy.encoder.config.json", config, ["ModernBertModel"])
    print(shapes({**tensors, **io}))


if __name__ == "__main__":
    main()
