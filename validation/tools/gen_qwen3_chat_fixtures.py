#!/usr/bin/env python3
"""Generate the Qwen3 tokenizer and chat-template fixtures.

Run once; commit the output. Regenerating is not part of CI.

    python3 -m venv /tmp/pkvenv
    /tmp/pkvenv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
    /tmp/pkvenv/bin/pip install transformers tokenizers jinja2
    /tmp/pkvenv/bin/python validation/tools/gen_qwen3_chat_fixtures.py ROW=HF_DIR...

- `ROW=HF_DIR`: a bunsen row name and a directory holding that row's pinned files under their
  repo names, for `qwen3-0.6b` and `qwen3-4b-instruct-2507` (revisions as bunsen's rows pin
  them). `qwen3-1.7b` and `qwen3-4b` carry `qwen3-0.6b`'s tokenizer files.

Pinned versions used to produce the committed fixture:
    transformers 5.19.0, tokenizers 0.23.3, jinja2 3.1.6, torch 2.14.1+cpu

Outputs, under `validation/models/qwen3-model-validation/testdata/`, by row:

`chat.json`
    - `encode`: `TEXTS`, each with the ids `transformers`' fast tokenizer gives it, no
      special tokens added (added tokens in the text are matched as `transformers` matches
      them).
    - `templates`: `CONVERSATIONS` under each setting of `add_generation_prompt` and
      `enable_thinking` (`None` is not passed), each with the text `apply_chat_template`
      renders and its ids.
`chat_replies.json`
    `SESSIONS`: conversations of user turns, each answered by `transformers`'
    `Qwen3ForCausalLM` (CPU, float32) greedily through its own KV cache from the
    chat-templated prompt (generation prompt on, `enable_thinking` as the session says), until
    a stop token of its `generation_config.json` or `budget` tokens. Each reply: its `ids` (the
    stop token included), its `text` (without it), and each step's `margins` (top-1 minus
    top-2 logit). The reply joins the conversation as an assistant turn of its text.
"""

import json
import pathlib
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "validation/models/qwen3-model-validation/testdata/chat.json"
REPLIES = ROOT / "validation/models/qwen3-model-validation/testdata/chat_replies.json"

ROWS = ["qwen3-0.6b", "qwen3-4b-instruct-2507"]

TEXTS = [
    "",
    "Hello, world!",
    "The lighthouse keeper climbed the stairs every evening, and",
    "  leading spaces, trailing spaces   ",
    "tabs\tand\nnewlines\r\n\n\nand more",
    "numbers: 12345 3.14159 -42 1e10 0x7fff",
    "fn main() {\n    println!(\"{}\", 1 + 2);\n}\n",
    "def f(x):\n\treturn x ** 2  # square\n",
    "Ünïcödé façade naïve résumé",
    "é versus é versus Ȩ́",
    "中文测试：你好，世界！",
    "日本語のテキストです。カタカナとひらがな。",
    "한국어 텍스트",
    "emoji 🎉🚀👩‍💻 and flags 🇺🇸🇯🇵",
    "Ελληνικά και Русский текст",
    "<|im_start|>user\nhi<|im_end|>\n",
    "<think>\nreasoning\n</think>\n\nanswer",
    "glued<|im_end|>text<|endoftext|>more",
    "a <|im_start broken special",
    "<tool_call>\n{\"name\": \"f\"}\n</tool_call>",
    "don't won't I'm you're they've we'll he'd",
    "URLs: https://example.com/path?q=1&r=2#frag",
    "     ",
    "\n",
    "mixed 中文 and English and 123 numbers",
    "zero​width‍joiner﻿bom",
    "a" * 300,
    "repeated words words words words words",
]

CONVERSATIONS = {
    "one_turn": [{"role": "user", "content": "What is 2 + 2?"}],
    "system": [
        {"role": "system", "content": "You are a terse assistant."},
        {"role": "user", "content": "Name a prime number."},
    ],
    "multi_turn_thinking": [
        {"role": "user", "content": "Pick a colour."},
        {"role": "assistant", "content": "<think>\nBlue is common.\n</think>\n\nBlue."},
        {"role": "user", "content": "Another?"},
    ],
    "reasoning_content": [
        {"role": "user", "content": "Pick a number."},
        {"role": "assistant", "content": "Seven.", "reasoning_content": "Seven is lucky."},
        {"role": "user", "content": "Why?"},
    ],
    "ends_with_assistant": [
        {"role": "user", "content": "Say hi."},
        {"role": "assistant", "content": "<think>\n\n</think>\n\nHi!"},
    ],
}

SETTINGS = [
    {"add_generation_prompt": True, "enable_thinking": None},
    {"add_generation_prompt": True, "enable_thinking": True},
    {"add_generation_prompt": True, "enable_thinking": False},
    {"add_generation_prompt": False, "enable_thinking": None},
]


SESSIONS = {
    "qwen3-0.6b": [
        {"name": "no_thinking", "enable_thinking": False, "budget": 48,
         "turns": ["Name three colours of the rainbow.", "Which of them is warmest?"]},
        {"name": "thinking", "enable_thinking": True, "budget": 40,
         "turns": ["What is 17 + 25?"]},
    ],
    "qwen3-4b-instruct-2507": [
        {"name": "instruct", "enable_thinking": None, "budget": 32,
         "turns": ["Write one sentence about the sea.", "Now one about the sky."]},
    ],
}


def greedy_reply(model, tok, prompt_ids, stops, budget) -> dict:
    ids, margins = [], []
    with torch.no_grad():
        out = model(input_ids=torch.tensor([prompt_ids]), use_cache=True)
        last, past = out.logits[0, -1], out.past_key_values
        for _ in range(budget):
            top = torch.topk(last, 2)
            ids.append(int(top.indices[0]))
            margins.append(float(top.values[0] - top.values[1]))
            if ids[-1] in stops:
                break
            out = model(input_ids=torch.tensor([[ids[-1]]]), past_key_values=past, use_cache=True)
            last, past = out.logits[0, -1], out.past_key_values
    body = ids[:-1] if ids and ids[-1] in stops else ids
    return {"ids": ids, "text": tok.decode(body), "margins": margins}


def replies_fixture(row: str, hf_dir: pathlib.Path) -> list:
    tok = AutoTokenizer.from_pretrained(hf_dir)
    model = AutoModelForCausalLM.from_pretrained(hf_dir, dtype=torch.float32).eval()
    stops = json.loads((hf_dir / "generation_config.json").read_text())["eos_token_id"]
    stops = stops if isinstance(stops, list) else [stops]
    sessions = []
    for session in SESSIONS[row]:
        messages, replies = [], []
        for user in session["turns"]:
            messages.append({"role": "user", "content": user})
            kwargs = {"add_generation_prompt": True}
            if session["enable_thinking"] is not None:
                kwargs["enable_thinking"] = session["enable_thinking"]
            text = tok.apply_chat_template(messages, tokenize=False, **kwargs)
            prompt_ids = tok(text, add_special_tokens=False)["input_ids"]
            reply = greedy_reply(model, tok, prompt_ids, stops, session["budget"])
            replies.append({"user": user, "prompt_ids": prompt_ids, **reply})
            messages.append({"role": "assistant", "content": reply["text"]})
            print(row, session["name"], repr(reply["text"][:60]), "min margin", min(reply["margins"]))
        sessions.append({**session, "replies": replies})
    return sessions


def row_fixture(hf_dir: pathlib.Path) -> dict:
    tok = AutoTokenizer.from_pretrained(hf_dir)
    encode = [{"text": t, "ids": tok(t, add_special_tokens=False)["input_ids"]} for t in TEXTS]
    templates = []
    for name, messages in CONVERSATIONS.items():
        for setting in SETTINGS:
            kwargs = {"add_generation_prompt": setting["add_generation_prompt"]}
            if setting["enable_thinking"] is not None:
                kwargs["enable_thinking"] = setting["enable_thinking"]
            text = tok.apply_chat_template(messages, tokenize=False, **kwargs)
            templates.append({
                "name": name,
                "messages": messages,
                **setting,
                "text": text,
                "ids": tok(text, add_special_tokens=False)["input_ids"],
            })
    return {"encode": encode, "templates": templates}


def main() -> None:
    dirs = dict(arg.split("=", 1) for arg in sys.argv[1:])
    assert sorted(dirs) == sorted(ROWS), f"need {ROWS}, got {sorted(dirs)}"
    fixture = {row: row_fixture(pathlib.Path(dirs[row])) for row in ROWS}
    OUT.write_text(json.dumps(fixture, ensure_ascii=False, indent=1) + "\n")
    replies = {row: replies_fixture(row, pathlib.Path(dirs[row])) for row in ROWS}
    REPLIES.write_text(json.dumps(replies, ensure_ascii=False) + "\n")
    print("wrote", OUT, {r: (len(f["encode"]), len(f["templates"])) for r, f in fixture.items()})


if __name__ == "__main__":
    main()
