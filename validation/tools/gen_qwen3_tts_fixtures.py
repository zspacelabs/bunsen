#!/usr/bin/env python3
"""Generate `qwen3-tts-model-validation`'s fixtures: `qwen_tts` over Qwen3-TTS-12Hz-0.6B-CustomVoice.

Run once; commit the outputs. Regenerating is not part of CI.

    # Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice @ 85e237c12c027371202489a0ec509ded67b5e4b5
    hf download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \\
        --revision 85e237c12c027371202489a0ec509ded67b5e4b5 --local-dir model
    python3.12 -m venv venv
    venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu \\
        torch==2.9.1 torchaudio==2.9.1
    venv/bin/pip install transformers==4.57.3 accelerate==1.12.0 safetensors numpy \\
        einops librosa soundfile sox onnxruntime
    venv/bin/pip install --no-deps qwen-tts==0.1.1
    venv/bin/python -I validation/tools/gen_qwen3_tts_fixtures.py --model model

Pinned versions used to produce the committed fixtures:
    qwen-tts 0.1.1 (byte-identical to `QwenLM/Qwen3-TTS` @ 022e286b), transformers 4.57.3,
    torch 2.9.1+cpu, Python 3.12; the checkpoint's files checked against their sha256s.

Everything runs on the CPU in float32 (the bf16 weights upcast), eager attention:
`Qwen3TTSModel.from_pretrained(model, device_map="cpu", dtype=float32,
attn_implementation="eager")`, its processor's tokenizer (transformers' fast tokenizer, built
from the repository's slow files) and its speech tokenizer (the 12 Hz codec).

The outputs are safetensors files: integers as int32, values as float32, sums as float64, and
the strings (JSON) in the header's `__metadata__`, which carries the pins.

`reference.safetensors`: for each of `REQUESTS`, under `<name>.` (metadata `<name>`: the
request as JSON):

- `input_ids` `[n]`: the processor's ids for `<|im_start|>assistant\\n{text}<|im_end|>\\n
  <|im_start|>assistant\\n`;
- `prompt.norms` `[L]`: each prefill row's L2 norm; `prompt.sample`: every `SAMPLE_STRIDE`-th
  value of the `[L, 1024]` prefill input; `trailing.norms` `[t]`, the trailing text's;
- `codes` `[F, 16]`: a greedy take (`do_sample` and `subtalker_dosample` false, the release's
  repetition penalty, at most `FRAMES + 1` codebook-0 tokens), the EOS frame not among them;
- `talker.argmax`, `talker.margin` `[F + 1]`, `talker.top8.ids`, `talker.top8.values`
  `[F + 1, 8]`, `talker.sums` `[F + 1, 2]`: the talker's raw last-position logits at the
  prefill and at each step (row `k` chose frame `k`'s codebook 0);
- `predictor.argmax`, `predictor.margin` `[F, 15]`, `predictor.top8.ids`,
  `predictor.top8.values` `[F, 15, 8]`: the code predictor's raw logits for codebooks 1-15 of
  each frame.

`reference_bf16.safetensors`: the same model in bfloat16 (`dtype=bfloat16`, the codec
untouched), each request's take dictated `reference.safetensors`' codes (forward hooks keep each
logits row, then leave only the dictated code to choose): the `talker.*` and `predictor.*`
summaries alone, under the same names.

`codec.safetensors`: the audio the codec decodes (`speech_tokenizer.decode`, upstream's
`chunked_decode` of 300 frames with 25 of context) under `<case>.`: `codes` `[F, 16]`;
`audio.stats` `[4]` (samples, sum, absolute sum, RMS); `audio.sample` (every
`SAMPLE_STRIDE`-th sample); `audio.windows` `[3, AUDIO_WINDOW]` from `audio.window_starts`.
The cases (metadata `cases`): `take:<name>` for each request's take, and `long`: `LONG_FRAMES`
frames of seeded random codes, past one chunk, its windows across the first chunk's end.

`tokenizer.safetensors`: `<name>.ids`, the processor's tokenizer's ids (no special tokens
added) for each of `TEXTS`.
"""

import argparse
import hashlib
import importlib.metadata
import json
import pathlib
import struct
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "validation/models/qwen3-tts-model-validation/testdata"
AUDIO = ROOT / "crates/dev/bunsen-test-audio/testdata"

PINS = {"qwen-tts": "0.1.1", "transformers": "4.57.3", "torch": "2.9.1+cpu"}
REVISION = "85e237c12c027371202489a0ec509ded67b5e4b5"
FILES = {
    "model.safetensors": "bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb",
    "speech_tokenizer/model.safetensors": (
        "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258"
    ),
    "config.json": "81aca2b6fac304944d8acf345272d8a9a727d5fc2e2e66b222ab4729340c7455",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
}

CHINESE = "其实我真的有发现，我是一个特别善于观察别人情绪的人。"
"""The model README's own Chinese example."""

ENGLISH = "We choose to go to the Moon in this decade and do the other things."
"""From `jfk_moon.txt`."""

FRENCH = "La Déclaration universelle des droits de l'homme des Nations unies."
"""From `udhr_fr.txt`."""

REQUESTS = {
    # name: (text, speaker, language, non_streaming)
    "ryan_english": (ENGLISH, "ryan", "english", False),
    "vivian_chinese_upfront": (CHINESE, "vivian", "chinese", True),
    "eric_auto_dialect": (CHINESE, "eric", "auto", False),
    "aiden_french_upfront": (FRENCH, "aiden", "french", True),
}
"""A stock speaker with its language, text interleaved; another, upfront; a dialect speaker
under `auto` (upstream's override to its dialect, `sichuan_dialect`); a speaker in another
voice's language, upfront."""

FRAMES = 12
SAMPLE_STRIDE = 97
AUDIO_WINDOW = 1024
LONG_FRAMES = 330


def texts() -> dict:
    """`TEXTS`: the UDHR texts, the JFK transcript, the Chinese example."""
    out = {f"udhr_{lang}": (AUDIO / f"udhr_{lang}.txt").read_text() for lang in ("de", "es", "fr")}
    out["jfk_moon"] = (AUDIO / "jfk_moon.txt").read_text()
    out["chinese"] = CHINESE
    return out


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_pins(model: pathlib.Path) -> None:
    for name, want in PINS.items():
        got = torch.__version__ if name == "torch" else importlib.metadata.version(name)
        if got != want:
            sys.exit(f"{name} is {got}, not the pinned {want}")
    for name, digest in FILES.items():
        got = sha256(model / name)
        if got != digest:
            sys.exit(f"{model / name}: sha256 {got}, not the pinned {digest}")


def i32(x) -> torch.Tensor:
    return torch.as_tensor(x).to(torch.int32)


def f32(x) -> torch.Tensor:
    return torch.as_tensor(x).to(torch.float32)


def summary(logits: torch.Tensor, prefix: str) -> dict:
    """Rows of logits `[..., V]`: the argmax and its margin over the runner-up, the 8 best,
    and each row's sum and absolute sum."""
    top = logits.topk(8, dim=-1)
    return {
        f"{prefix}.argmax": i32(top.indices[..., 0]),
        f"{prefix}.margin": f32(top.values[..., 0] - top.values[..., 1]),
        f"{prefix}.top8.ids": i32(top.indices),
        f"{prefix}.top8.values": f32(top.values),
        f"{prefix}.sums": torch.stack(
            [logits.double().sum(-1), logits.double().abs().sum(-1)], dim=-1
        ),
    }


class Forced:
    """Keeps a module's last-position logits at each call, then leaves only the code the
    script dictates for that call to choose."""

    def __init__(self, module, script):
        self.logits: list = []
        self.script = script
        module.register_forward_hook(self._post)

    def _post(self, _m, _a, out):
        row = out.logits[0, -1].detach().clone()
        self.logits.append(row)
        code = self.script(len(self.logits) - 1)
        forced = torch.full_like(out.logits[:, -1:], float("-inf"))
        forced[..., code] = 0.0
        out.logits = torch.cat([out.logits[:, :-1], forced], dim=1)
        return out


def forced_fixture(tts, takes):
    """`reference_bf16.safetensors`' summaries: each request's take dictated `takes`."""
    model = tts.model
    tensors = {}
    for name, (text, speaker, language, non_streaming) in REQUESTS.items():
        codes = takes[name]
        frames = codes.shape[0]
        eos = model.config.talker_config.codec_eos_token_id
        talker = Forced(model.talker, lambda i: int(codes[i, 0]) if i < frames else eos)
        predictor = Forced(model.talker.code_predictor, lambda i: int(codes[i // 15, 1 + i % 15]))
        ids = tts._tokenize_texts([tts._build_assistant_text(text)])[0]
        with torch.inference_mode():
            got, _ = model.generate(
                input_ids=[ids],
                instruct_ids=[None],
                languages=[language],
                speakers=[speaker],
                non_streaming_mode=non_streaming,
                max_new_tokens=FRAMES + 1,
                do_sample=False,
                subtalker_dosample=False,
                repetition_penalty=1.05,
            )
        assert torch.equal(got[0], codes), f"{name}: the dictated take"
        model.talker._forward_hooks.clear()
        model.talker.code_predictor._forward_hooks.clear()
        rows = torch.stack(talker.logits).float()
        assert rows.shape[0] == frames + 1, (name, rows.shape)
        tensors.update(summary(rows, f"{name}.talker"))
        tensors.update(
            summary(torch.stack(predictor.logits).float().view(frames, 15, -1), f"{name}.predictor")
        )
        print(f"{name}: bf16, dictated {frames} frames")
    return tensors


class Calls:
    """Each call of a module: its keyword arguments and its output's last-position logits."""

    def __init__(self, module):
        self.kwargs: list = []
        self.logits: list = []
        module.register_forward_pre_hook(self._pre, with_kwargs=True)
        module.register_forward_hook(self._post)

    def _pre(self, _m, _a, kwargs):
        self.kwargs.append(kwargs)

    def _post(self, _m, _a, out):
        self.logits.append(out.logits[0, -1].detach().clone())

    def clear(self):
        self.kwargs.clear()
        self.logits.clear()


def reference_fixture(tts):
    model = tts.model
    talker_calls = Calls(model.talker)
    predictor_calls = Calls(model.talker.code_predictor)
    prompts: list = []
    model.talker.model.register_forward_pre_hook(
        lambda _m, _a, kw: prompts.append(kw["inputs_embeds"].detach().clone()),
        with_kwargs=True,
    )
    tensors, metadata, takes = {}, {}, {}
    for name, (text, speaker, language, non_streaming) in REQUESTS.items():
        talker_calls.clear()
        predictor_calls.clear()
        prompts.clear()
        ids = tts._tokenize_texts([tts._build_assistant_text(text)])[0]
        with torch.inference_mode():
            codes, _ = model.generate(
                input_ids=[ids],
                instruct_ids=[None],
                languages=[language],
                speakers=[speaker],
                non_streaming_mode=non_streaming,
                max_new_tokens=FRAMES + 1,
                do_sample=False,
                subtalker_dosample=False,
                repetition_penalty=1.05,
            )
        codes = codes[0]
        frames = codes.shape[0]
        prompt = prompts[0][0]
        trailing = talker_calls.kwargs[0]["trailing_text_hidden"][0]
        talker = torch.stack(talker_calls.logits)
        predictor = torch.stack(predictor_calls.logits).view(-1, 15, 2048)
        assert predictor.shape[0] >= frames
        tensors.update(
            {
                f"{name}.input_ids": i32(ids[0]),
                f"{name}.prompt.norms": f32(prompt.norm(dim=-1)),
                f"{name}.prompt.sample": f32(prompt.flatten()[::SAMPLE_STRIDE]),
                f"{name}.trailing.norms": f32(trailing.norm(dim=-1)),
                f"{name}.codes": i32(codes),
                **summary(talker[: frames + 1], f"{name}.talker"),
                **summary(predictor[:frames], f"{name}.predictor"),
            }
        )
        metadata[name] = json.dumps(
            {
                "text": text,
                "speaker": speaker,
                "language": language,
                "non_streaming": non_streaming,
                "frames": frames,
            },
            ensure_ascii=False,
        )
        takes[name] = codes
        print(f"{name}: {len(ids[0])} ids, prompt {list(prompt.shape)}, {frames} frames")
    return tensors, metadata, takes


def audio_summary(audio: torch.Tensor, prefix: str, starts: list) -> dict:
    a = audio.double()
    return {
        f"{prefix}.audio.stats": torch.tensor(
            [a.numel(), a.sum(), a.abs().sum(), a.pow(2).mean().sqrt()], dtype=torch.float64
        ),
        f"{prefix}.audio.sample": f32(audio[::SAMPLE_STRIDE]),
        f"{prefix}.audio.windows": f32(torch.stack([audio[s : s + AUDIO_WINDOW] for s in starts])),
        f"{prefix}.audio.window_starts": i32(starts),
    }


def codec_fixture(tts, takes):
    codec = tts.model.speech_tokenizer
    rate = 1920
    cases = {f"take:{name}": codes for name, codes in takes.items()}
    g = torch.Generator().manual_seed(17)
    cases["long"] = torch.randint(0, 2048, (LONG_FRAMES, 16), generator=g)
    tensors = {}
    for case, codes in cases.items():
        with torch.inference_mode():
            wavs, _ = codec.decode([{"audio_codes": codes}])
        audio = torch.as_tensor(wavs[0]).float()
        assert audio.numel() == codes.shape[0] * rate
        n = audio.numel()
        if case == "long":
            boundary = 300 * rate
            starts = [0, boundary - AUDIO_WINDOW // 2, n - AUDIO_WINDOW]
        else:
            starts = [0, (n - AUDIO_WINDOW) // 2, n - AUDIO_WINDOW]
        tensors[f"{case}.codes"] = i32(codes)
        tensors.update(audio_summary(audio, case, starts))
        print(f"{case}: {codes.shape[0]} frames, {n} samples")
    return tensors, {"cases": json.dumps(list(cases))}


def tokenizer_fixture(tts):
    tok = tts.processor.tokenizer
    return {
        f"{name}.ids": i32(tok(text, add_special_tokens=False)["input_ids"])
        for name, text in texts().items()
    }


DTYPES = {torch.int32: "I32", torch.float32: "F32", torch.float64: "F64"}


def write(name: str, tensors: dict, metadata: dict) -> None:
    """A safetensors file, its header's keys sorted, so a regeneration is byte-identical."""
    header, blobs, offset = {"__metadata__": dict(sorted(metadata.items()))}, [], 0
    for key in sorted(tensors):
        t = tensors[key].contiguous()
        blob = t.numpy().tobytes()
        header[key] = {
            "dtype": DTYPES[t.dtype],
            "shape": list(t.shape),
            "data_offsets": [offset, offset + len(blob)],
        }
        blobs.append(blob)
        offset += len(blob)
    raw = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
    raw += b" " * (-len(raw) % 8)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=pathlib.Path, required=True)
    args = parser.parse_args()
    check_pins(args.model)
    torch.set_num_threads(32)
    from qwen_tts import Qwen3TTSModel

    tts = Qwen3TTSModel.from_pretrained(
        str(args.model), device_map="cpu", dtype=torch.float32, attn_implementation="eager"
    )
    meta = {
        "revision": REVISION,
        "torch": torch.__version__,
        "qwen_tts": PINS["qwen-tts"],
        "frames": str(FRAMES),
        "sample_stride": str(SAMPLE_STRIDE),
    }
    reference, metadata, takes = reference_fixture(tts)
    write("reference.safetensors", reference, {**meta, **metadata})
    codec, cases = codec_fixture(tts, takes)
    write("codec.safetensors", codec, {**meta, **cases})
    write("tokenizer.safetensors", tokenizer_fixture(tts), meta)
    del tts
    tts = Qwen3TTSModel.from_pretrained(
        str(args.model), device_map="cpu", dtype=torch.bfloat16, attn_implementation="eager"
    )
    write("reference_bf16.safetensors", forced_fixture(tts, takes), {**meta, "dtype": "bf16"})


if __name__ == "__main__":
    main()
