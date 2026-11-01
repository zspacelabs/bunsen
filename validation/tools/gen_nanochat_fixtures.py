#!/usr/bin/env python3
"""Generate `nanochat-model-validation`'s fixtures: upstream nanochat's own decodes of the rows.

Run once; commit the outputs. Regenerating is not part of CI.

    git clone https://github.com/karpathy/nanochat.git /tmp/nanochat
    git -C /tmp/nanochat worktree add /tmp/nanochat-2025 a088b7a6
    git -C /tmp/nanochat worktree add /tmp/nanochat-2026 a3ca42a6
    python3 -m venv /tmp/ncvenv
    /tmp/ncvenv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
    /tmp/ncvenv/bin/pip install numpy filelock tiktoken tokenizers rustbpe
    /tmp/ncvenv/bin/python validation/tools/gen_nanochat_fixtures.py fp32 \\
        /tmp/nanochat-2025 /tmp/nanochat-2026 CACHE_ROOT

    # upstream's greedy chat replies, multi-turn, in float32 on the CPU:
    /tmp/ncvenv/bin/python validation/tools/gen_nanochat_fixtures.py replies \\
        /tmp/nanochat-2025 /tmp/nanochat-2026 CACHE_ROOT

    # bf16, on a CUDA device, from a venv with a CUDA torch wheel:
    /tmp/ncvenv-cuda/bin/python validation/tools/gen_nanochat_fixtures.py bf16 \\
        /tmp/nanochat-2025 /tmp/nanochat-2026 CACHE_ROOT

`CACHE_ROOT` is a bunsen weights cache holding the rows (`BUNSEN_CACHE_DIR`, or the user's
cache directory, after the validation tests or `bunsen-tool chat fetch` fetched them): the
rows' files under `pretrained/nanochat/<namespace>/<sha256>/<file>`.

Pinned versions used to produce the committed fixtures:
    fp32: torch 2.15.0+cpu, numpy 2.5.4, tiktoken 0.14.0, rustbpe 0.1.0
    bf16: recorded in `upstream_bf16.json`'s `torch`

For each row, upstream's own code at the row's generation's pinned commit (`a088b7a6` for the
2025 model, `a3ca42a6` for the 2026 one; each in a process of its own, since both are the
`nanochat` package): the model built from the row's `meta_<step>.json` and `model_<step>.pt` as
upstream's `checkpoint_manager.build_model` builds it (the meta's `model_config` keys
`GPTConfig` has; some uploads add others), and the tokenizer from the row's `tokenizer.pkl`.
`PROMPT` is encoded after `<|bos|>`, then `NEW_TOKENS` greedy tokens are decoded through
upstream's `engine.KVCache`, one token per step after the prompt's prefill.

`fp32` runs on the CPU, where both generations compute in float32 (the 2025 model's rotary
tables in bf16, as it keeps them). `bf16` runs on CUDA as upstream serves there: the 2025 model
under `torch.amp.autocast(bfloat16)`, as its chat scripts run it, the 2026 one in its
`COMPUTE_DTYPE`, bf16.

`replies` holds a chat on each chat-tuned row (`REPLY_ROWS`) for each of `SESSIONS`: each user
turn appended to the conversation, rendered by upstream's `render_for_completion`, and answered
greedily through upstream's KV cache until `<|assistant_end|>` or `<|bos|>` (upstream's engine's
stops) or `REPLY_TOKENS`; the reply's decoded text joins the conversation for the next turn.
Output: `chat_replies.json`, by row, a list of sessions, each a list of turns: the `user` text,
the `prompt_ids`, the reply's `ids` (its stop token last, when one ended it), its `text`, and each
step's `margins`.

Outputs, under `validation/models/nanochat-model-validation/testdata/`, `upstream.json` (fp32)
or `upstream_bf16.json` (bf16): for each row, the `source`, the `prompt`, its `prompt_ids`, the
generated `ids` and `text`, each step's `margins` (top-1 minus top-2 logit, so a reader can tell
a near-tie from a disagreement), and the prefill's last-position logits, `last_logits`, recorded
as their `shape`, `sum`, `abs_sum` and a strided sample.
"""

import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "validation/models/nanochat-model-validation/testdata"

PROMPT = "The lighthouse keeper climbed the stairs every evening, and"
NEW_TOKENS = 32
N_RECORDED = 512

# Each row: its generation, cache namespace, and (sha256, path in the repository) of its meta,
# state dict and tokenizer, as bunsen's rows pin them.
ROWS = {
    "nanochat-students/base-d20": ("2025", "nanochat-students", {
        "meta": ("a34a9b144a30d2a8066e64a91b7becc8579ea355c075e20427c28afb83ef106c", "meta_021400.json"),
        "model": ("574670de45c667a98fe72c2d145f24007ecc1778721e2481e41500842dd3ba13", "model_021400.pt"),
        "tokenizer": ("8467414b90511a50c4dac438af25c075817e9d62d799a5ef613b186c977f5d1b", "tokenizer.pkl"),
    }),
    "nanochat-students/mid-d20": ("2025", "nanochat-students", {
        "meta": ("1282220593eaf3e120b3e6be0eb3894e054638d46021511ede8d60b074ece58d", "meta_000765.json"),
        "model": ("cc0560b6adf23e663191ba3e73707381b648e94a6489493c2589baf87eb1863a", "model_000765.pt"),
        "tokenizer": ("8467414b90511a50c4dac438af25c075817e9d62d799a5ef613b186c977f5d1b", "tokenizer.pkl"),
    }),
    "nanochat-students/rl-d20": ("2025", "nanochat-students", {
        "meta": ("353d774b8548a317a5e71762b775ee8ea506332f9b2acf8dff4fbfd9e813131a", "meta_000650.json"),
        "model": ("ff2eee182e2aa396615d3b481ddf17884a7dbabb3caa47f66eede343135accff", "model_000650.pt"),
        "tokenizer": ("8467414b90511a50c4dac438af25c075817e9d62d799a5ef613b186c977f5d1b", "tokenizer.pkl"),
    }),
    "nitishpandey04/nanochat-d18-base": ("2026", "nitishpandey04", {
        "meta": ("7ecc0e6f4f0315d106d4fb213fec593879456475a21de34b71c329a4a8a4e7b4",
                 "base_checkpoints/d18/meta_003712.json"),
        "model": ("979a5ac9ea785da98d38dd9b58bd07d39d06230a6644d1f4e157f045499989b3",
                  "base_checkpoints/d18/model_003712.pt"),
        "tokenizer": ("ae73c5f7a960edc56022dfd46a653df2b9b38d84456e5e9f48eb5e02a60c21c2",
                      "tokenizer/tokenizer.pkl"),
    }),
    "nitishpandey04/nanochat-d18-sft": ("2026", "nitishpandey04", {
        "meta": ("e0acf64f9e195a42d5de9dc284f5d9a814a0b29d7a48e89f5f96ebb891967028",
                 "chatsft_checkpoints/d18/meta_000478.json"),
        "model": ("951589ed950a12358d69ab3180c39a237cf24fea80f01a4b6691f90e3bacf17a",
                  "chatsft_checkpoints/d18/model_000478.pt"),
        "tokenizer": ("ae73c5f7a960edc56022dfd46a653df2b9b38d84456e5e9f48eb5e02a60c21c2",
                      "tokenizer/tokenizer.pkl"),
    }),
}
COMMITS = {"2025": "a088b7a6", "2026": "a3ca42a6"}

REPLY_ROWS = ["nanochat-students/rl-d20", "nitishpandey04/nanochat-d18-sft"]
REPLY_TOKENS = 48
SESSIONS = [
    ["What is the capital of France?", "And what river runs through it?"],
    ["Name three primary colors.", "Which of them is the warmest?"],
]


def record(x) -> dict:
    import numpy as np
    flat = np.asarray(x, dtype=np.float64).reshape(-1)
    stride = max(1, flat.size // N_RECORDED)
    return {
        "shape": list(np.shape(x)),
        "sum": float(flat.sum()),
        "abs_sum": float(np.abs(flat).sum()),
        "stride": stride,
        "samples": [float(v) for v in flat[::stride]],
    }


def child(mode: str, generation: str, cache_root: str) -> None:
    """Decode every row of `generation`; print one JSON object, by row."""
    import contextlib
    import dataclasses

    import torch
    from nanochat.engine import KVCache
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.tokenizer import RustBPETokenizer

    device = torch.device("cuda" if mode == "bf16" else "cpu")
    rows = REPLY_ROWS if mode == "replies" else list(ROWS)
    autocast = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
                if mode == "bf16" and generation == "2025" else contextlib.nullcontext())
    fields = {f.name for f in dataclasses.fields(GPTConfig)}
    out = {}
    for row, (gen, namespace, files) in ROWS.items():
        if gen != generation or row not in rows:
            continue
        path = lambda key: (pathlib.Path(cache_root) / "pretrained/nanochat" / namespace
                            / files[key][0] / files[key][1])
        meta = json.loads(path("meta").read_text())
        config = GPTConfig(**{k: v for k, v in meta["model_config"].items() if k in fields})
        state = torch.load(path("model"), map_location=device, weights_only=True)
        if device.type == "cpu":
            # As upstream's `build_model` on a CPU.
            state = {k: v.float() if v.dtype == torch.bfloat16 else v for k, v in state.items()}
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
        with torch.device("meta"):
            model = GPT(config)
        model.to_empty(device=device)
        model.init_weights()
        model.load_state_dict(state, strict=True, assign=True)
        model.eval()
        tok = RustBPETokenizer.from_directory(str(path("tokenizer").parent))
        if mode == "replies":
            out[row] = replies(model, tok, config, generation, device)
            print(f"{row}: {[[t['text'] for t in s] for s in out[row]]!r}", file=sys.stderr)
            continue
        prompt_ids = tok.encode(PROMPT, prepend="<|bos|>")

        head_dim = config.n_embd // config.n_head
        cache_args = (1, config.n_kv_head, len(prompt_ids) + NEW_TOKENS, head_dim, config.n_layer)
        if generation == "2026":
            from nanochat.common import COMPUTE_DTYPE
            cache_args += (device, COMPUTE_DTYPE)
        cache = KVCache(*cache_args)
        ids, margins = [], []
        with torch.no_grad(), autocast:
            logits = model(torch.tensor([prompt_ids], device=device), kv_cache=cache)[:, -1]
            last_logits = logits.float().cpu()
            for _ in range(NEW_TOKENS):
                top = torch.topk(logits.float(), 2, dim=-1)
                margins.append(float(top.values[0, 0] - top.values[0, 1]))
                nxt = int(top.indices[0, 0])
                ids.append(nxt)
                logits = model(torch.tensor([[nxt]], device=device), kv_cache=cache)[:, -1]
        out[row] = {
            "source": f"{row} on karpathy/nanochat@{COMMITS[generation]}",
            "prompt": PROMPT,
            "prompt_ids": prompt_ids,
            "ids": ids,
            "text": tok.decode(ids),
            "margins": margins,
            "last_logits": record(last_logits.numpy()),
        }
        print(f"{row}: {tok.decode(ids)!r}", file=sys.stderr)
    out["_torch"] = torch.__version__
    print(json.dumps(out))


def replies(model, tok, config, generation, device) -> list:
    """Upstream's greedy multi-turn chats on one row: `SESSIONS`, turn by turn."""
    import torch
    from nanochat.engine import KVCache

    stops = {tok.encode_special("<|assistant_end|>"), tok.get_bos_token_id()}
    head_dim = config.n_embd // config.n_head
    sessions = []
    for users in SESSIONS:
        messages, turns = [], []
        for user in users:
            messages.append({"role": "user", "content": user})
            prompt_ids = tok.render_for_completion(
                {"messages": messages + [{"role": "assistant", "content": ""}]})
            cache_args = (1, config.n_kv_head, len(prompt_ids) + REPLY_TOKENS, head_dim,
                          config.n_layer)
            if generation == "2026":
                cache_args += (device, torch.float32)
            cache = KVCache(*cache_args)
            ids, margins = [], []
            with torch.no_grad():
                logits = model(torch.tensor([prompt_ids], device=device), kv_cache=cache)[:, -1]
                for _ in range(REPLY_TOKENS):
                    top = torch.topk(logits.float(), 2, dim=-1)
                    margins.append(float(top.values[0, 0] - top.values[0, 1]))
                    nxt = int(top.indices[0, 0])
                    ids.append(nxt)
                    if nxt in stops:
                        break
                    logits = model(torch.tensor([[nxt]], device=device), kv_cache=cache)[:, -1]
            text = tok.decode([i for i in ids if i not in stops])
            turns.append({"user": user, "prompt_ids": prompt_ids, "ids": ids, "text": text,
                          "margins": margins})
            messages.append({"role": "assistant", "content": text})
        sessions.append(turns)
    return sessions


def main() -> None:
    if sys.argv[1:2] == ["--child"]:
        child(*sys.argv[2:5])
        return
    if len(sys.argv) != 5 or sys.argv[1] not in ("fp32", "bf16", "replies"):
        sys.exit(__doc__)
    mode, checkouts, cache_root = sys.argv[1], dict(zip(COMMITS, sys.argv[2:4])), sys.argv[4]
    fixture = {}
    for generation, checkout in checkouts.items():
        # Each generation imports its own `nanochat` package: one process each.
        run = subprocess.run([sys.executable, __file__, "--child", mode, generation, cache_root],
                             cwd=checkout, env={"PYTHONPATH": checkout, "PATH": "/usr/bin:/bin",
                                  **{k: v for k, v in os.environ.items() if k.startswith("CUDA")}},
                             check=True, capture_output=True, text=True)
        sys.stderr.write("\n".join(l for l in run.stderr.splitlines() if not "Warning" in l))
        fixture.update(json.loads(run.stdout.strip().splitlines()[-1]))
    torch_version = fixture.pop("_torch")
    if mode == "bf16":
        fixture = {"torch": torch_version, "rows": fixture}
    name = {"fp32": "upstream.json", "bf16": "upstream_bf16.json",
            "replies": "chat_replies.json"}[mode]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(fixture, indent=1) + "\n")
    print(f"wrote {OUT / name}")


if __name__ == "__main__":
    main()
