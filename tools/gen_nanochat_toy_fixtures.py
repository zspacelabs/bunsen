#!/usr/bin/env python3
"""Generate the nanochat toy-model fixtures, one per upstream model generation.

Run once; commit the outputs. Regenerating is not part of CI.

    git clone https://github.com/karpathy/nanochat.git /tmp/nanochat
    git -C /tmp/nanochat worktree add /tmp/nanochat-2025 a088b7a6
    git -C /tmp/nanochat worktree add /tmp/nanochat-2026 a3ca42a6
    python3 -m venv /tmp/ncvenv
    /tmp/ncvenv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
    /tmp/ncvenv/bin/pip install numpy filelock tiktoken tokenizers rustbpe safetensors
    /tmp/ncvenv/bin/python tools/gen_nanochat_toy_fixtures.py /tmp/nanochat-2025 /tmp/nanochat-2026

Pinned versions used to produce the committed fixtures:
    torch 2.15.0+cpu, safetensors 0.8.0; karpathy/nanochat at a088b7a6 (2025) and a3ca42a6 (2026)

For each generation, a tiny `nanochat.gpt.GPT` -- upstream's own model, imported from the
checkout at the pinned commit -- with seeded random weights (every tensor, so none of upstream's
zero or neutral initial values hides a misplaced one), run on the CPU in fp32. The 2025 toy has
grouped-query heads; the 2026 toy has every 2026 part (the smear, the residual lambdas, value
embeddings, the backout, the 1.2 query/key gain), a window pattern ("SL") whose long window is
shorter than the sequence it runs on, and a vocabulary (60) its embedding pads to 64.

Outputs, under `crates/public/bunsen/testdata/nanochat/toy_<generation>/`, named as upstream's
checkpoint manager names a checkpoint:

- `model_000001.pt`: the toy's state dict, `torch.save`d as upstream saves one;
- `meta_000001.json`: its `meta`, `model_config` as upstream writes it;
- `io.safetensors`: the inputs and answers bunsen is pinned against: `input_ids` `[1, 12]`; the
  first block's output `block0`; `hidden`, the head's input; `logits`, the whole sequence's; and
  `logits_cached`, the same sequence through upstream's `engine.KVCache`, a 3-token prefill and
  then one token per step.
"""

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "crates/public/bunsen/testdata/nanochat"

INPUT_IDS = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8]
PREFILL = 3

TOYS = {
    "2025": dict(sequence_len=8, vocab_size=48, n_layer=3, n_head=4, n_kv_head=2, n_embd=32),
    "2026": dict(sequence_len=4, vocab_size=60, n_layer=3, n_head=4, n_kv_head=2, n_embd=32,
                 window_pattern="SL"),
}


def child(generation: str) -> None:
    """Build, run and save one toy; run with the generation's checkout on `sys.path`."""
    import torch
    from safetensors.torch import save_file
    from nanochat.engine import KVCache
    from nanochat.gpt import GPT, GPTConfig

    torch.manual_seed({"2025": 2025, "2026": 2026}[generation])
    config = GPTConfig(**TOYS[generation])
    model = GPT(config)
    model.init_weights()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 1:
                # Per-layer and scalar lambdas: around their usual values, never neutral.
                p.copy_(0.5 + 0.5 * torch.rand_like(p))
            else:
                p.copy_(torch.randn_like(p) * (1.0 if "wte" in name or "value_embeds" in name
                                               else 0.3))
    model.eval()

    captured = {}
    model.transformer.h[0].register_forward_hook(
        lambda _m, _i, out: captured.__setitem__("block0", out))
    model.lm_head.register_forward_pre_hook(
        lambda _m, args: captured.__setitem__("hidden", args[0]))

    idx = torch.tensor([INPUT_IDS])
    with torch.no_grad():
        logits = model(idx)
        block0, hidden = captured["block0"], captured["hidden"]

        head_dim = config.n_embd // config.n_head
        cache_args = (1, config.n_kv_head, len(INPUT_IDS), head_dim, config.n_layer)
        if generation == "2026":
            cache_args += (torch.device("cpu"), torch.float32)
        cache = KVCache(*cache_args)
        steps = [model(idx[:, :PREFILL], kv_cache=cache)]
        for i in range(PREFILL, len(INPUT_IDS)):
            steps.append(model(idx[:, i:i + 1], kv_cache=cache))
        logits_cached = torch.cat(steps, dim=1)

    out = OUT / f"toy_{generation}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model_000001.pt")
    meta = {"step": 1, "model_config": TOYS[generation]}
    (out / "meta_000001.json").write_text(json.dumps(meta, indent=2) + "\n")
    io = {
        "input_ids": idx.to(torch.float32),
        "block0": block0,
        "hidden": hidden,
        "logits": logits,
        "logits_cached": logits_cached,
    }
    save_file({k: v.float().contiguous() for k, v in io.items()}, out / "io.safetensors")
    gap = (logits_cached - logits).abs().max().item()
    print(f"wrote {out}: {len(model.state_dict())} tensors, cached vs full {gap:.2e}",
          {k: tuple(v.shape) for k, v in io.items()})


def main() -> None:
    if sys.argv[1:2] == ["--child"]:
        child(sys.argv[2])
        return
    checkouts = dict(zip(TOYS, sys.argv[1:3]))
    if len(checkouts) != 2:
        sys.exit(__doc__)
    for generation, checkout in checkouts.items():
        # Each generation imports its own `nanochat` package: one process each.
        subprocess.run([sys.executable, __file__, "--child", generation], cwd=checkout,
                       env={"PYTHONPATH": checkout}, check=True)


if __name__ == "__main__":
    main()
