#!/usr/bin/env python3
"""Generate the Qwen3 reference fixtures.

Run once; commit the outputs. Regenerating is not part of CI.

    python3 -m venv /tmp/pkvenv
    /tmp/pkvenv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
    /tmp/pkvenv/bin/pip install transformers onnxruntime safetensors tokenizers numpy
    /tmp/pkvenv/bin/python validation/tools/gen_qwen3_fixtures.py ONNX_DIR ROW=HF_DIR...
    /tmp/pkvenv/bin/python validation/tools/gen_qwen3_fixtures.py --bf16 ROW=HF_DIR...
    /tmp/pkvenv/bin/python validation/tools/gen_qwen3_fixtures.py --long ROW=HF_DIR...

- `ONNX_DIR`: `onnx-community/Qwen3-0.6B-ONNX` at 58ca03a5e7b354b480b1a82b34bde8dc7b854cc0
  (`model.onnx` + `model.onnx_data` from its `onnx/`): the published graph, not the one
  `bind-qwen3-onnx` rewrites.
- `ROW=HF_DIR`: a bunsen row name and a directory holding that row's pinned files under their
  repo names (`config.json`, the safetensors, `tokenizer.json`, ...), for each of
  `qwen3-0.6b`, `qwen3-1.7b`, `qwen3-4b`, `qwen3-4b-instruct-2507` (revisions as bunsen's rows
  pin them).

Pinned versions used to produce the committed fixtures:
    transformers 5.19.0, torch 2.14.1+cpu, onnxruntime 1.31.0, numpy 2.5.4, tokenizers 0.23.3

Outputs, under `validation/models/qwen3-model-validation/testdata/`:

`onnx_reference.json`
    The published graph run once by onnxruntime (CPU, graph optimizations off) on fixed ids:
    a prefill of `ids` from an empty cache, then one step on the prefill's greedy next token
    `step_id`. Each output is recorded as its `shape`, `sum`, `abs_sum` and a strided sample.
`transformers.json`
    For each row: `transformers`' `Qwen3ForCausalLM` on the CPU in float32, the row's
    tokenizer encoding `PROMPT` (no chat template) to `prompt_ids`, then `NEW_TOKENS` greedy
    tokens through its own KV cache: the generated `ids`, their `text`, each step's `margins`
    (top-1 minus top-2 logit, so a reader can tell a near-tie from a disagreement), and the
    prefill's last-position logits, `last_logits`. For `qwen3-0.6b`, also how far the
    `lm_head.weight` its checkpoint carries is from `embed_tokens` (`transformers` ties it).

With `--bf16`, for each of `BF16_ROWS` (`qwen3-8b`, which does not fit a 24 GB device in
float32), one output instead:

`transformers_bf16.json`
    For each row: the same greedy decode, but in bfloat16 on the CPU (weights and activations
    as the checkpoint ships them), its `last_logits` read as float32; and
    `f32_last_logits_max_abs`, how far the float32 model's prefill logits are from those, a
    bf16 run's own distance from float32 to measure bunsen's bf16 run against.

With `--long`, for each of `LONG_ROWS` (`qwen3-0.6b`, `qwen3-4b`, `qwen3-8b`), one output:

`transformers_long.json`
    `conversation`: a long multi-turn conversation over a generated warehouse ledger (no
    third-party text), its last turn a question whose answer is in the ledger, far behind it;
    rendered by each row's chat template (generation prompt on, `enable_thinking` false) to
    at least `LONG_MIN_POSITIONS` ids. For each row: its `prompt_ids`; the greedy decode of
    `LONG_NEW_TOKENS` tokens through `transformers`' KV cache on the CPU in float32 (`f32`)
    and in bfloat16 (`bf16`), each its `ids`, `text`, `margins` and prefill `last_logits`
    (read as float32); how far the two prefills' logits are apart,
    `bf16_vs_f32_last_logits_max_abs`; and how many leading tokens the two decodes agree on,
    `bf16_f32_agree`. bf16's distance from float32 is what bunsen's bf16 run is measured
    against.
"""

import json
import random
import pathlib
import sys

import numpy as np
import onnxruntime as ort
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "validation/models/qwen3-model-validation/testdata"

N_RECORDED = 2048
N_LAYERS = 28
ONNX_IDS = [9707, 11, 1879, 0, 151643, 13, 576, 6722, 315, 9625, 374]
PROMPT = "The lighthouse keeper climbed the stairs every evening, and"
NEW_TOKENS = 32
ROWS = ["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-4b-instruct-2507"]
BF16_ROWS = ["qwen3-8b"]
LONG_ROWS = ["qwen3-0.6b", "qwen3-4b", "qwen3-8b"]
LONG_NEW_TOKENS = 48
LONG_MIN_POSITIONS = 2048
LONG_SEED = 20261103


def record(x) -> dict:
    flat = np.asarray(x, dtype=np.float64).reshape(-1)
    stride = max(1, flat.size // N_RECORDED)
    return {
        "shape": list(np.shape(x)),
        "sum": float(flat.sum()),
        "abs_sum": float(np.abs(flat).sum()),
        "stride": stride,
        "samples": [float(v) for v in flat[::stride]],
    }


def onnx_reference(onnx_dir: pathlib.Path) -> dict:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(str(onnx_dir / "model.onnx"), so,
                                providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]

    def run(ids, past, past_len):
        s = len(ids)
        feed = {
            "input_ids": np.array([ids], np.int64),
            "attention_mask": np.ones((1, past_len + s), np.int64),
            "position_ids": np.arange(past_len, past_len + s, dtype=np.int64)[None],
        }
        for i in range(N_LAYERS):
            feed[f"past_key_values.{i}.key"], feed[f"past_key_values.{i}.value"] = past[i]
        d = dict(zip(names, sess.run(None, feed)))
        return d["logits"], [(d[f"present.{i}.key"], d[f"present.{i}.value"])
                             for i in range(N_LAYERS)]

    empty = [(np.zeros((1, 8, 0, 128), np.float32),) * 2 for _ in range(N_LAYERS)]
    logits, cache = run(ONNX_IDS, empty, 0)
    step_id = int(logits[0, -1].argmax())
    step_logits, step_cache = run([step_id], cache, len(ONNX_IDS))
    return {
        "source": "onnx-community/Qwen3-0.6B-ONNX@58ca03a5e7b354b480b1a82b34bde8dc7b854cc0 onnx/model.onnx",
        "ids": ONNX_IDS,
        "prefill_logits": record(logits),
        "prefill_key_0": record(cache[0][0]),
        "prefill_value_27": record(cache[27][1]),
        "step_id": step_id,
        "step_logits": record(step_logits),
        "step_key_27": record(step_cache[27][0]),
    }


def head_distance(hf_dir: pathlib.Path) -> float:
    with safe_open(str(hf_dir / "model.safetensors"), "pt") as f:
        if "lm_head.weight" not in f.keys():
            return 0.0
        head = f.get_tensor("lm_head.weight").float()
        embed = f.get_tensor("model.embed_tokens.weight").float()
    return float((head - embed).abs().max())


def greedy(row: str, hf_dir: pathlib.Path, dtype=torch.float32) -> dict:
    tok = AutoTokenizer.from_pretrained(hf_dir)
    model = AutoModelForCausalLM.from_pretrained(hf_dir, dtype=dtype).eval()
    prompt_ids = tok(PROMPT, add_special_tokens=False)["input_ids"]
    ids, margins = [], []
    with torch.no_grad():
        out = model(input_ids=torch.tensor([prompt_ids]), use_cache=True)
        last = out.logits[0, -1].float()
        last_logits = last.numpy()[None]
        past = out.past_key_values
        for _ in range(NEW_TOKENS):
            top = torch.topk(last, 2)
            ids.append(int(top.indices[0]))
            margins.append(float(top.values[0] - top.values[1]))
            out = model(input_ids=torch.tensor([[ids[-1]]]), past_key_values=past, use_cache=True)
            last, past = out.logits[0, -1].float(), out.past_key_values
    entry = {
        "prompt": PROMPT,
        "prompt_ids": prompt_ids,
        "ids": ids,
        "text": tok.decode(ids),
        "margins": margins,
        "last_logits": record(last_logits),
    }
    print(row, entry["text"][:80].replace("\n", " "), "min margin", min(margins))
    return entry


def ledger_conversation(items: int) -> list[dict]:
    """A conversation over a generated ledger of `items` lines: the ledger, two questions
    answered from it, then a last question for the model, about lines far behind it."""
    rng = random.Random(LONG_SEED)
    materials = ["copper", "steel", "oak", "brass", "nylon", "cedar", "glass", "rubber",
                 "cotton", "granite", "aluminium", "linen"]
    goods = ["wire", "bolts", "planks", "hinges", "rope", "beams", "panes", "gaskets",
             "thread", "tiles", "brackets", "canvas", "nails", "pipes", "springs", "valves"]
    ledger = []
    for i in range(1, items + 1):
        ledger.append({
            "item": i,
            "what": f"{rng.choice(materials)} {rng.choice(goods)}",
            "units": rng.randint(3, 480),
            "price": rng.randint(25, 9999) / 100,
            "aisle": rng.choice("ABCDEFGH"),
            "shelf": rng.randint(1, 6),
        })
    lines = [f"Item {e['item']:03d}: {e['what']}, {e['units']} units at {e['price']:.2f} each, "
             f"aisle {e['aisle']}, shelf {e['shelf']}." for e in ledger]
    a, b = ledger[items // 5], ledger[items // 2]
    last = ledger[3]
    return [
        {"role": "user", "content": "Here is the warehouse ledger. Read it carefully: I will "
                                    "ask about it.\n\n" + "\n".join(lines)},
        {"role": "assistant", "content": f"I have read the ledger of {items} items, and I am "
                                         "ready for your questions."},
        {"role": "user", "content": f"Where is item {a['item']:03d} kept?"},
        {"role": "assistant", "content": f"Item {a['item']:03d}, the {a['what']}, is kept in "
                                         f"aisle {a['aisle']}, shelf {a['shelf']}."},
        {"role": "user", "content": f"How many units of item {b['item']:03d} are there, and "
                                    "at what price?"},
        {"role": "assistant", "content": f"There are {b['units']} units of item "
                                         f"{b['item']:03d}, the {b['what']}, at "
                                         f"{b['price']:.2f} each."},
        {"role": "user", "content": f"What is item {last['item']:03d}, and where is it kept? "
                                    "Then list every other item kept in the same aisle."},
    ]


def long_prompt(tok) -> tuple[list[dict], list[int]]:
    """The fewest ledger lines (in steps of 10) whose conversation renders to at least
    `LONG_MIN_POSITIONS` ids."""
    for items in range(60, 400, 10):
        conversation = ledger_conversation(items)
        ids = tok.apply_chat_template(conversation, add_generation_prompt=True,
                                      enable_thinking=False, tokenize=True,
                                      return_dict=False)
        if len(ids) >= LONG_MIN_POSITIONS:
            return conversation, list(ids)
    raise AssertionError("no ledger is long enough")


def decode(model, tok, prompt_ids: list[int], new_tokens: int) -> tuple[dict, np.ndarray]:
    """A greedy decode of `new_tokens` from `prompt_ids` through `model`'s own KV cache, and
    the prefill's whole last-position logits, as float32."""
    ids, margins = [], []
    with torch.no_grad():
        out = model(input_ids=torch.tensor([prompt_ids]), use_cache=True)
        last = out.logits[0, -1].float()
        last_logits = last.numpy()[None]
        past = out.past_key_values
        for _ in range(new_tokens):
            top = torch.topk(last, 2)
            ids.append(int(top.indices[0]))
            margins.append(float(top.values[0] - top.values[1]))
            out = model(input_ids=torch.tensor([[ids[-1]]]), past_key_values=past, use_cache=True)
            last, past = out.logits[0, -1].float(), out.past_key_values
    return {"ids": ids, "text": tok.decode(ids), "margins": margins,
            "last_logits": record(last_logits)}, last_logits


def long_main(args: list[str]) -> None:
    dirs = dict(arg.split("=", 1) for arg in args)
    assert sorted(dirs) == sorted(LONG_ROWS), f"need {LONG_ROWS}, got {sorted(dirs)}"
    fixture = {"enable_thinking": False, "new_tokens": LONG_NEW_TOKENS, "rows": {}}
    for row in LONG_ROWS:
        hf_dir = pathlib.Path(dirs[row])
        tok = AutoTokenizer.from_pretrained(hf_dir)
        conversation, prompt_ids = long_prompt(tok)
        if "conversation" in fixture:
            assert fixture["conversation"] == conversation, f"{row}: another conversation"
        fixture["conversation"] = conversation
        entry = {"prompt_ids": prompt_ids}
        logits = {}
        for name, dtype in [("f32", torch.float32), ("bf16", torch.bfloat16)]:
            model = AutoModelForCausalLM.from_pretrained(hf_dir, dtype=dtype).eval()
            entry[name], logits[name] = decode(model, tok, prompt_ids, LONG_NEW_TOKENS)
            del model
            print(row, name, len(prompt_ids), "positions:",
                  entry[name]["text"][:80].replace("\n", " "),
                  "| min margin", min(entry[name]["margins"]))
        entry["bf16_vs_f32_last_logits_max_abs"] = float(np.abs(logits["f32"] - logits["bf16"]).max())
        agree = 0
        for x, y in zip(entry["f32"]["ids"], entry["bf16"]["ids"]):
            if x != y:
                break
            agree += 1
        entry["bf16_f32_agree"] = agree
        print(row, "bf16 vs f32: prefill logits max abs",
              entry["bf16_vs_f32_last_logits_max_abs"], "| agree on", agree, "tokens")
        fixture["rows"][row] = entry
    (OUT / "transformers_long.json").write_text(json.dumps(fixture) + "\n")


def prefill_logits(hf_dir: pathlib.Path, prompt_ids: list[int], dtype) -> np.ndarray:
    model = AutoModelForCausalLM.from_pretrained(hf_dir, dtype=dtype).eval()
    with torch.no_grad():
        return model(input_ids=torch.tensor([prompt_ids])).logits[0, -1].float().numpy()


def bf16_main(args: list[str]) -> None:
    dirs = dict(arg.split("=", 1) for arg in args)
    assert sorted(dirs) == sorted(BF16_ROWS), f"need {BF16_ROWS}, got {sorted(dirs)}"
    fixture = {}
    for row in BF16_ROWS:
        hf_dir = pathlib.Path(dirs[row])
        entry = greedy(row, hf_dir, torch.bfloat16)
        f32 = prefill_logits(hf_dir, entry["prompt_ids"], torch.float32)
        b16 = prefill_logits(hf_dir, entry["prompt_ids"], torch.bfloat16)
        entry["f32_last_logits_max_abs"] = float(np.abs(f32 - b16).max())
        print(row, "bf16 vs f32 prefill logits, max abs", entry["f32_last_logits_max_abs"])
        fixture[row] = entry
    (OUT / "transformers_bf16.json").write_text(json.dumps(fixture) + "\n")


def main() -> None:
    if sys.argv[1:2] == ["--bf16"]:
        bf16_main(sys.argv[2:])
        return
    if sys.argv[1:2] == ["--long"]:
        long_main(sys.argv[2:])
        return
    onnx_dir = pathlib.Path(sys.argv[1])
    dirs = dict(arg.split("=", 1) for arg in sys.argv[2:])
    assert sorted(dirs) == sorted(ROWS), f"need {ROWS}, got {sorted(dirs)}"
    OUT.mkdir(parents=True, exist_ok=True)

    reference = onnx_reference(onnx_dir)
    (OUT / "onnx_reference.json").write_text(json.dumps(reference) + "\n")
    print("onnx step id", reference["step_id"])

    fixture = {row: greedy(row, pathlib.Path(dirs[row])) for row in ROWS}
    fixture["qwen3-0.6b"]["lm_head_vs_embed_max_abs"] = head_distance(
        pathlib.Path(dirs["qwen3-0.6b"]))
    (OUT / "transformers.json").write_text(json.dumps(fixture) + "\n")


if __name__ == "__main__":
    main()
