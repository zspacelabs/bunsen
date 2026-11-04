#!/usr/bin/env python3
"""Generate the Gemma reference fixtures.

Run once; commit the outputs. Regenerating is not part of CI.

    python3 -m venv /tmp/pkvenv
    /tmp/pkvenv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
    /tmp/pkvenv/bin/pip install transformers safetensors tokenizers jinja2 numpy
    /tmp/pkvenv/bin/python validation/tools/gen_gemma_fixtures.py ROW=HF_DIR...

- `ROW=HF_DIR`: a bunsen row name and a directory holding that row's pinned files under their
  repo names (`config.json`, the safetensors, `tokenizer.json`, ...), for the rows of
  `ROWS`, `IT_ROWS`, `BF16_ROWS`, `CHAT_ROWS` and `LONG_ROWS` (revisions as bunsen's rows pin
  them, on the `unsloth/` mirrors). Each output is written when every row it needs is
  given.

Pinned versions used to produce the committed fixtures:
    transformers 5.19.0, torch 2.14.1+cpu, numpy 2.5.4, tokenizers 0.23.3, jinja2 3.1.6

Every model runs on the CPU with eager attention (`transformers`' sdpa path drops Gemma 2's
attention soft-cap). `gemma-3-4b-pt` is multimodal: `Gemma3ForConditionalGeneration` runs on
text ids, as a text prompt runs through it. No Gemma file is written here: only ids, texts
and logits.

Outputs, under `validation/models/gemma-model-validation/testdata/`:

`transformers.json`
    For each of `ROWS` and `IT_ROWS`: the row's tokenizer encoding `PROMPT` (`<bos>` added, as Gemma is
    trained) to `prompt_ids`, then `NEW_TOKENS` greedy tokens in float32 through
    `transformers`' own KV cache: the generated `ids`, their `text`, each step's `margins`
    (top-1 minus top-2 logit), and the prefill's last-position logits, `last_logits`
    (recorded as `shape`, `sum`, `abs_sum` and a strided sample).
`transformers_window.json`
    For each of `ROWS`: a prompt longer than the row's sliding window (a generated
    ledger, no third-party text; at least `window + WINDOW_PAST` ids), then
    `WINDOW_NEW_TOKENS` greedy tokens in float32: its `window`, `prompt_ids` and the
    decode's `ids`, `text`, `margins` and `last_logits`. A decode here runs past what a
    sliding layer's cache keeps.
`transformers_bf16.json`
    For each of `BF16_ROWS` (`gemma-2-9b`, which a 24 GB device holds only in bf16): the
    greedy decode of `PROMPT` in bfloat16, its `last_logits` as float32, and
    `f32_last_logits_max_abs`, how far the float32 model's prefill logits are from those: a
    bf16 run's own distance from float32, which bunsen's bf16 run is measured against.
`chat.json`
    For each of `CHAT_ROWS`: `encode`, `TEXTS` each with the ids `transformers`' fast
    tokenizer gives it (no special tokens added); and `templates`, the conversations the
    row's template accepts, under each `add_generation_prompt`, each with the text
    `apply_chat_template` renders and its ids.
`transformers_long.json`
    `conversation`: a long multi-turn conversation over a generated warehouse ledger (no
    third-party text), its last turn a question whose answer is in the ledger, far behind it,
    as `gen_qwen3_fixtures.py` writes Qwen3's; rendered by each of `LONG_ROWS`' chat
    templates (generation prompt on) to at least `LONG_MIN_POSITIONS` ids. For each row: its
    `prompt_ids`; the greedy decode of `LONG_NEW_TOKENS` tokens through `transformers`' KV
    cache on the CPU in float32 (`f32`) and in bfloat16 (`bf16`), each its `ids`, `text`,
    `margins` and prefill `last_logits` (read as float32); how far the two prefills' logits
    are apart, `bf16_vs_f32_last_logits_max_abs`; and how many leading tokens the two decodes
    agree on, `bf16_f32_agree`. bf16's distance from float32 is what bunsen's bf16 run is
    measured against.
`chat_replies.json`
    For each of `CHAT_ROWS`, its `SESSIONS`: conversations of user turns, each answered by
    `transformers` (CPU, float32) greedily through its own KV cache from the chat-templated
    prompt (generation prompt on), until a stop token or `budget` tokens. The stop tokens are
    bunsen's for the row: its `generation_config.json`'s, and the id its tokenizer gives
    `<end_of_turn>`. Each reply: its `ids` (the stop token included), its `text` (without it),
    and each step's `margins`. The reply joins the conversation as an assistant turn of its
    text.
"""

import json
import pathlib
import random
import sys

import numpy as np
import torch
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer, Gemma3ForConditionalGeneration
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "validation/models/gemma-model-validation/testdata"

N_RECORDED = 2048
PROMPT = "The lighthouse keeper climbed the stairs every evening, and"
NEW_TOKENS = 32
ROWS = ["gemma-3-270m", "gemma-3-1b-pt", "gemma-3-4b-pt", "gemma-2-2b"]
IT_ROWS = ["gemma-3-1b-it", "gemma-2-2b-it"]
BF16_ROWS = ["gemma-2-9b"]
CHAT_ROWS = ["gemma-3-1b-it", "gemma-2-2b-it"]
END_OF_TURN = "<end_of_turn>"
LONG_ROWS = ["gemma-3-1b-it", "gemma-2-2b-it", "gemma-2-9b-it"]
LONG_NEW_TOKENS = 48
LONG_MIN_POSITIONS = 2048
LONG_SEED = 20261103
WINDOW_PAST = 96
WINDOW_NEW_TOKENS = 16
WINDOW_SEED = 20261104

TEXTS = [
    "",
    "Hello, world!",
    PROMPT,
    "  leading spaces, trailing spaces   ",
    "tabs\tand\nnewlines\r\n\n\nand more",
    "numbers: 12345 3.14159 -42 1e10 0x7fff",
    "fn main() {\n    println!(\"{}\", 1 + 2);\n}\n",
    "Ünïcödé façade naïve résumé",
    "中文测试：你好，世界！",
    "emoji: 🦀🔥👩‍💻",
    "<start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model\n",
    "an unknown rune: \U00010330 and a control: \u0007",
]

SESSIONS = {
    "gemma-3-1b-it": [
        {"name": "two_turns", "budget": 48,
         "turns": ["Name three colours of the rainbow.", "Which of them is warmest?"]},
        {"name": "arithmetic", "budget": 24, "turns": ["What is 17 + 25?"]},
    ],
    "gemma-2-2b-it": [
        {"name": "two_turns", "budget": 32,
         "turns": ["Write one sentence about the sea.", "Now one about the sky."]},
    ],
}

CONVERSATIONS = {
    "one-turn": [{"role": "user", "content": "What is the capital of France?"}],
    "multi-turn": [
        {"role": "user", "content": "Name a colour."},
        {"role": "assistant", "content": "Blue."},
        {"role": "user", "content": "  And another, please.  "},
    ],
    "system": [
        {"role": "system", "content": "Answer in one word."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
}


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


def load(hf_dir: pathlib.Path, dtype):
    config = AutoConfig.from_pretrained(hf_dir)
    cls = (Gemma3ForConditionalGeneration if config.model_type == "gemma3"
           else AutoModelForCausalLM)
    return cls.from_pretrained(hf_dir, dtype=dtype, attn_implementation="eager").eval()


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
            out = model(input_ids=torch.tensor([[ids[-1]]]), past_key_values=past,
                        use_cache=True)
            last, past = out.logits[0, -1].float(), out.past_key_values
    return {"ids": ids, "text": tok.decode(ids), "margins": margins,
            "last_logits": record(last_logits)}, last_logits


def ledger_text(items: int) -> str:
    """A generated warehouse ledger of `items` lines, and a question about an early one."""
    rng = random.Random(WINDOW_SEED)
    materials = ["copper", "steel", "oak", "brass", "nylon", "cedar", "glass", "rubber"]
    goods = ["wire", "bolts", "planks", "hinges", "rope", "beams", "panes", "gaskets"]
    lines = [f"Item {i:03d}: {rng.choice(materials)} {rng.choice(goods)}, "
             f"{rng.randint(3, 480)} units, aisle {rng.choice('ABCDEFGH')}, "
             f"shelf {rng.randint(1, 6)}." for i in range(1, items + 1)]
    return ("Warehouse ledger:\n" + "\n".join(lines)
            + "\n\nQuestion: which aisle holds item 004?\nAnswer: Item 004 is in aisle")


def window_prompt(tok, window: int) -> list[int]:
    """The fewest ledger lines (in steps of 10) whose text encodes to at least
    `window + WINDOW_PAST` ids, `<bos>` first."""
    for items in range(10, 2000, 10):
        ids = tok(ledger_text(items))["input_ids"]
        if len(ids) >= window + WINDOW_PAST:
            return ids
    raise AssertionError("no ledger is long enough")


def main() -> None:
    dirs = {row: pathlib.Path(d) for row, d in (arg.split("=", 1) for arg in sys.argv[1:])}
    unknown = set(dirs) - set(ROWS + IT_ROWS + BF16_ROWS + CHAT_ROWS + LONG_ROWS)
    assert not unknown, f"not a fixture row: {sorted(unknown)}"
    OUT.mkdir(parents=True, exist_ok=True)
    if all(row in dirs for row in ROWS + IT_ROWS):
        write_f32(dirs)
    if all(row in dirs for row in BF16_ROWS):
        write_bf16(dirs)
    if all(row in dirs for row in CHAT_ROWS):
        write_chat(dirs)
        write_sessions(dirs)
    if all(row in dirs for row in LONG_ROWS):
        write_long(dirs)


def write_f32(dirs: dict) -> None:
    greedy, window = {}, {}
    for row in ROWS + IT_ROWS:
        tok = AutoTokenizer.from_pretrained(dirs[row])
        model = load(dirs[row], torch.float32)
        prompt_ids = tok(PROMPT)["input_ids"]
        assert prompt_ids[0] == tok.bos_token_id, f"{row}: <bos> first"
        entry, _ = decode(model, tok, prompt_ids, NEW_TOKENS)
        greedy[row] = {"prompt": PROMPT, "prompt_ids": prompt_ids, **entry}
        print(row, entry["text"][:80].replace("\n", " "), "| min margin", min(entry["margins"]))
        if row in IT_ROWS:
            del model
            continue

        config = AutoConfig.from_pretrained(dirs[row])
        text = getattr(config, "text_config", config)
        w = text.sliding_window
        prompt_ids = window_prompt(tok, w)
        entry, _ = decode(model, tok, prompt_ids, WINDOW_NEW_TOKENS)
        window[row] = {"window": w, "prompt_ids": prompt_ids, **entry}
        print(row, f"window {w}, {len(prompt_ids)} positions:",
              entry["text"][:80].replace("\n", " "), "| min margin", min(entry["margins"]))
        del model
    (OUT / "transformers.json").write_text(json.dumps(greedy) + "\n")
    (OUT / "transformers_window.json").write_text(json.dumps(window) + "\n")


def write_bf16(dirs: dict) -> None:
    bf16 = {}
    for row in BF16_ROWS:
        tok = AutoTokenizer.from_pretrained(dirs[row])
        prompt_ids = tok(PROMPT)["input_ids"]
        model = load(dirs[row], torch.bfloat16)
        entry, b16 = decode(model, tok, prompt_ids, NEW_TOKENS)
        del model
        model = load(dirs[row], torch.float32)
        with torch.no_grad():
            f32 = model(input_ids=torch.tensor([prompt_ids])).logits[0, -1].float().numpy()
        del model
        entry["f32_last_logits_max_abs"] = float(np.abs(f32 - b16[0]).max())
        bf16[row] = {"prompt": PROMPT, "prompt_ids": prompt_ids, **entry}
        print(row, "bf16:", entry["text"][:80].replace("\n", " "),
              "| bf16 vs f32 prefill logits, max abs", entry["f32_last_logits_max_abs"])
    (OUT / "transformers_bf16.json").write_text(json.dumps(bf16) + "\n")


def write_chat(dirs: dict) -> None:
    chat = {}
    for row in CHAT_ROWS:
        tok = AutoTokenizer.from_pretrained(dirs[row])
        encode = [{"text": t, "ids": tok(t, add_special_tokens=False)["input_ids"]}
                  for t in TEXTS]
        templates = []
        for name, messages in CONVERSATIONS.items():
            for add in [True, False]:
                try:
                    text = tok.apply_chat_template(messages, add_generation_prompt=add,
                                                   tokenize=False)
                except Exception as e:  # Gemma 2's template refuses a system turn.
                    print(row, name, "refused:", e)
                    continue
                ids = tok(text, add_special_tokens=False)["input_ids"]
                templates.append({"name": name, "messages": messages,
                                  "add_generation_prompt": add, "text": text, "ids": ids})
        chat[row] = {"encode": encode, "templates": templates}
        print(row, len(encode), "texts,", len(templates), "renders")
    (OUT / "chat.json").write_text(json.dumps(chat) + "\n")


def stop_ids(tok, hf_dir: pathlib.Path) -> list[int]:
    """bunsen's stop tokens for a row: its `generation_config.json`'s end of text, and the id
    its tokenizer gives a turn's end."""
    eos = json.loads((hf_dir / "generation_config.json").read_text())["eos_token_id"]
    stops = eos if isinstance(eos, list) else [eos]
    end = tok.convert_tokens_to_ids(END_OF_TURN)
    return stops + ([end] if end not in stops else [])


def greedy_reply(model, tok, prompt_ids: list[int], stops: list[int], budget: int) -> dict:
    ids, margins = [], []
    with torch.no_grad():
        out = model(input_ids=torch.tensor([prompt_ids]), use_cache=True)
        last, past = out.logits[0, -1].float(), out.past_key_values
        for _ in range(budget):
            top = torch.topk(last, 2)
            ids.append(int(top.indices[0]))
            margins.append(float(top.values[0] - top.values[1]))
            if ids[-1] in stops:
                break
            out = model(input_ids=torch.tensor([[ids[-1]]]), past_key_values=past,
                        use_cache=True)
            last, past = out.logits[0, -1].float(), out.past_key_values
    body = ids[:-1] if ids and ids[-1] in stops else ids
    return {"ids": ids, "text": tok.decode(body), "margins": margins}


def write_sessions(dirs: dict) -> None:
    replies = {}
    for row in CHAT_ROWS:
        hf_dir = pathlib.Path(dirs[row])
        tok = AutoTokenizer.from_pretrained(hf_dir)
        model = load(hf_dir, torch.float32)
        stops = stop_ids(tok, hf_dir)
        sessions = []
        for session in SESSIONS[row]:
            messages, turns = [], []
            for user in session["turns"]:
                messages.append({"role": "user", "content": user})
                text = tok.apply_chat_template(messages, add_generation_prompt=True,
                                               tokenize=False)
                prompt_ids = tok(text, add_special_tokens=False)["input_ids"]
                reply = greedy_reply(model, tok, prompt_ids, stops, session["budget"])
                turns.append({"user": user, "prompt_ids": prompt_ids, **reply})
                messages.append({"role": "assistant", "content": reply["text"]})
                print(row, session["name"], repr(reply["text"][:60]), "| min margin",
                      min(reply["margins"]))
            sessions.append({**session, "replies": turns})
        replies[row] = sessions
        del model
    (OUT / "chat_replies.json").write_text(json.dumps(replies, ensure_ascii=False) + "\n")


def ledger_conversation(items: int) -> list[dict]:
    """A conversation over a generated ledger of `items` lines: the ledger, two questions
    answered from it, then a last question for the model, about lines far behind it. The
    same conversation as `gen_qwen3_fixtures.py`'s, for the same `items`."""
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
    `LONG_MIN_POSITIONS` ids, `<bos>` first."""
    for items in range(60, 400, 10):
        conversation = ledger_conversation(items)
        text = tok.apply_chat_template(conversation, add_generation_prompt=True,
                                       tokenize=False)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) >= LONG_MIN_POSITIONS:
            assert ids[0] == tok.bos_token_id, "<bos> first"
            return conversation, ids
    raise AssertionError("no ledger is long enough")


def write_long(dirs: dict) -> None:
    fixture = {"new_tokens": LONG_NEW_TOKENS, "rows": {}}
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
            model = load(hf_dir, dtype)
            entry[name], logits[name] = decode(model, tok, prompt_ids, LONG_NEW_TOKENS)
            del model
            print(row, name, len(prompt_ids), "positions:",
                  entry[name]["text"][:80].replace("\n", " "),
                  "| min margin", min(entry[name]["margins"]))
        entry["bf16_vs_f32_last_logits_max_abs"] = float(
            np.abs(logits["f32"] - logits["bf16"]).max())
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


if __name__ == "__main__":
    main()
