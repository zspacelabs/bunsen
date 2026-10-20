#!/usr/bin/env python3
"""Generate the Qwen3-ASR reference fixtures.

Run once; commit the outputs. Regenerating is not part of CI.

    python3 -m venv /tmp/pkvenv
    /tmp/pkvenv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
    /tmp/pkvenv/bin/pip install transformers onnxruntime safetensors soundfile numpy
    /tmp/pkvenv/bin/python tools/gen_qwen3_asr_fixtures.py \\
        ONNX_DIR HF_DIR FERMION_ENGINE_CPU_DIR PHONON1_DIR...

- `ONNX_DIR`: `OpenVoiceOS/qwen3-asr-0.6b-onnx` at 05ac7b041a8e2acbfe8ffc895664ced67a2a4c58
  (`encoder.onnx`, `embed_tokens.onnx`, `decoder.onnx` + `decoder.onnx_data`).
- `HF_DIR`: `Qwen/Qwen3-ASR-0.6B-hf` at 7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c.
- `FERMION_ENGINE_CPU_DIR`: `fermion/_speech/_engine_cpu/` from `fermion-research` 0.2.11
  (Apache-2.0); its `artifact._unpack_symbols` and `hybrid.dequantize_affine` read the
  Phonon-1 decoder codes and affine records.
- `PHONON1_DIR...`: unpacked `FermionResearch/Phonon-1*` archives
  (`package_release_bps.py unpack`), named after their profile (`audio6`, `micro`, `parity`).

Pinned versions used to produce the committed fixtures:
    transformers 5.19.0, torch 2.14.1+cpu, onnxruntime 1.31.0, numpy 2.5.4, soundfile 0.14.0

Outputs, under `crates/validation/qwen3-asr-model-validation/testdata/`:

`onnx_reference.json`
    The three graphs, each run once by onnxruntime on deterministic synthetic input the Rust
    side rebuilds from the same formulas: the encoder over 9 chunks (two attention windows),
    the embedding over a few ids, and the decoder's prefill then one step. Each output is
    recorded as its `shape`, `sum`, `abs_sum` and a strided sample.
`jfk_moon.json`
    Greedy transcriptions of `bunsen-test-audio`'s `jfk_moon.mp3`, as two 30 s
    utterances (`WINDOWS`; Phonon-1 is built for utterances of at most 30 s), by
    `transformers`' `Qwen3ASRForConditionalGeneration`, on the CPU in float32: `base` from
    the -hf weights, and one entry per Phonon-1 profile from that archive expanded to dense
    float32 (quint5 symbols to their exact levels; affine records to `scales * q + biases`).
    Each window holds the `prompt_ids`, the generated `ids` and the decoded `text` (with its
    `language ...<asr_text>` prefix).
"""

import importlib.util
import json
import pathlib
import re
import sys

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from safetensors import safe_open

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "crates/validation/qwen3-asr-model-validation/testdata"
CLIP = ROOT / "crates/dev/bunsen-test-audio/testdata/jfk_moon.mp3"

N_MELS = 128
CHUNK = 100
FRAMES_PER_CHUNK = 13
WINDOW_CHUNKS = 8
D_TEXT = 1024
N_LAYERS = 28
N_KV = 8
HEAD_DIM = 128
N_RECORDED = 2048

# The utterances `jfk_moon.json` transcribes, in seconds.
WINDOWS = [(0, 30), (30, 60)]


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


def synthetic_features(frames: int) -> np.ndarray:
    m = np.arange(N_MELS)[:, None]
    f = np.arange(frames)[None, :]
    return ((((m * 7 + f * 13) % 211) / 105.0 - 1.0) * 0.8).astype(np.float32)[None]


def synthetic_embeds(seq: int, salt: int) -> np.ndarray:
    k = np.arange(seq * D_TEXT).reshape(1, seq, D_TEXT)
    return (((((k * 5 + salt) % 197) / 98.0) - 1.0) * 0.05).astype(np.float32)


def window_bias(n_frames: int) -> np.ndarray:
    """Block-diagonal attention over windows of WINDOW_CHUNKS chunks."""
    window = FRAMES_PER_CHUNK * WINDOW_CHUNKS
    ids = np.arange(n_frames) // window
    allowed = ids[:, None] == ids[None, :]
    return np.where(allowed, 0.0, -np.inf).astype(np.float32)[None, None]


def onnx_fixture(onnx_dir: pathlib.Path) -> dict:
    enc = ort.InferenceSession(str(onnx_dir / "encoder.onnx"))
    emb = ort.InferenceSession(str(onnx_dir / "embed_tokens.onnx"))
    dec = ort.InferenceSession(str(onnx_dir / "decoder.onnx"))
    out = {"source": "OpenVoiceOS/qwen3-asr-0.6b-onnx@05ac7b041a8e2acbfe8ffc895664ced67a2a4c58",
           "runtime": f"onnxruntime {ort.__version__}"}

    chunks = 9
    feats = synthetic_features(chunks * CHUNK)
    n = chunks * FRAMES_PER_CHUNK
    (audio,) = enc.run(None, {
        "input_features": feats,
        "valid_indices": np.arange(n, dtype=np.int64),
        "attn_bias": window_bias(n),
    })
    out["encoder"] = {"chunks": chunks, "audio_embeds": record(audio)}

    ids = np.array([[151644, 8948, 198, 22574, 151645, 151669, 0, 77091]], dtype=np.int64)
    (embeds,) = emb.run(None, {"input_ids": ids})
    out["embed"] = {"ids": ids[0].tolist(), "inputs_embeds": record(embeds)}

    seq = 6
    names = [i.name for i in dec.get_inputs()]
    empty = np.zeros((1, N_KV, 0, HEAD_DIM), np.float32)
    causal = np.where(np.tril(np.ones((seq, seq), bool)), 0.0, -np.inf).astype(np.float32)
    feed = {"inputs_embeds": synthetic_embeds(seq, 0), "attn_bias": causal[None, None],
            "position_ids": np.arange(seq, dtype=np.int64)[None]}
    feed.update({name: empty for name in names if name.startswith("past_key_values")})
    prefill = dec.run(None, feed)
    present = dict(zip([o.name for o in dec.get_outputs()], prefill))

    feed = {"inputs_embeds": synthetic_embeds(1, 1), "attn_bias": np.zeros((1, 1, 1, seq + 1), np.float32),
            "position_ids": np.array([[seq]], dtype=np.int64)}
    for name in names:
        if name.startswith("past_key_values"):
            layer, kind = re.match(r"past_key_values\.(\d+)\.(key|value)", name).groups()
            feed[name] = present[f"present.{layer}.{kind}"]
    step = dict(zip([o.name for o in dec.get_outputs()], dec.run(None, feed)))
    out["decoder"] = {
        "seq": seq,
        "prefill_logits": record(present["logits"]),
        "prefill_key_0": record(present["present.0.key"]),
        "prefill_value_27": record(present["present.27.value"]),
        "step_logits": record(step["logits"]),
        "step_key_27": record(step["present.27.key"]),
    }
    return out


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def phonon1_state(model_dir: pathlib.Path, engine: pathlib.Path) -> dict:
    """A Phonon-1 archive's tensors, dense float32, under the -hf names."""
    artifact = load_module("fermion_artifact", engine / "artifact.py")
    hybrid = load_module("fermion_hybrid", engine / "hybrid.py")
    manifest = json.loads((model_dir / "packed_manifest.json").read_text())
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    affine = hybrid.quantized_modules(manifest) if manifest.get("hybrid_quantization") else {}
    packed = {row["name"]: row for row in manifest["modules"]}

    tensors = {}
    for shard in sorted(set(index["weight_map"].values())):
        with safe_open(model_dir / shard, framework="pt") as handle:
            for name in handle.keys():
                tensors[name] = handle.get_tensor(name)

    dense = {}
    for name, row in packed.items():
        symbols = artifact._unpack_symbols(tensors[name + ".quint5_q"], row["in_features"])
        alpha = tensors[name + ".base_alpha"].to(torch.float32)[:, None]
        residual = tensors[name + ".residual_scale"].to(torch.float32).reshape(-1)
        residual = residual[:, None] if residual.numel() > 1 else residual
        levels = torch.cat([-alpha - residual, -alpha + residual, torch.zeros_like(alpha),
                            alpha - residual, alpha + residual], dim=1)
        dense[name + ".weight"] = torch.gather(levels, 1, symbols.to(torch.int64))
    for name, row in affine.items():
        dense[name + ".weight"] = hybrid.dequantize_affine(
            tensors[name + ".weight"], tensors[name + ".scales"], tensors[name + ".biases"],
            bits=row["bits"], group_size=row["group_size"], in_features=row["in_features"])
    for name, tensor in tensors.items():
        base, suffix = name.rsplit(".", 1)
        if base in packed and suffix in ("quint5_q", "base_alpha", "residual_scale"):
            continue
        if base in affine and suffix in ("weight", "scales", "biases"):
            continue
        if "conv2d" in name and name.endswith(".weight") and tensor.ndim == 4:
            tensor = tensor.permute(0, 3, 1, 2)
        dense[name] = tensor.to(torch.float32)

    renamed = {}
    for name, tensor in dense.items():
        if name.startswith("audio_tower.proj1."):
            name = "model.multi_modal_projector.linear_1." + name.rsplit(".", 1)[1]
        elif name.startswith("audio_tower.proj2."):
            name = "model.multi_modal_projector.linear_2." + name.rsplit(".", 1)[1]
        elif name.startswith("audio_tower."):
            name = "model." + name
        elif name.startswith("model."):
            name = "model.language_model." + name[len("model."):]
        renamed[name] = tensor.contiguous()
    return renamed


def transcribe_windows(model, processor, wave) -> list:
    return [dict(start=a, end=b, **transcribe(model, processor, wave[a * 16000:b * 16000]))
            for a, b in WINDOWS]


def transcribe(model, processor, wave) -> dict:
    conversation = [{"role": "user", "content": [{"type": "audio", "audio": wave}]}]
    inputs = processor.apply_chat_template(conversation, tokenize=True, return_dict=True,
                                           add_generation_prompt=True, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    prompt = inputs["input_ids"].shape[1]
    ids = out[0, prompt:].tolist()
    text = processor.decode(out[0, prompt:], skip_special_tokens=True)
    return {"prompt_ids": inputs["input_ids"][0].tolist(), "ids": ids, "text": text}


def speech_fixture(hf_dir: pathlib.Path, engine: pathlib.Path, phonon_dirs: list) -> dict:
    from transformers import AutoProcessor, Qwen3ASRConfig, Qwen3ASRForConditionalGeneration

    wave, sr = sf.read(str(CLIP), dtype="float32")
    assert sr == 16000 and wave.ndim == 1
    processor = AutoProcessor.from_pretrained(hf_dir)
    out = {"clip": CLIP.name}

    model = Qwen3ASRForConditionalGeneration.from_pretrained(hf_dir, dtype=torch.float32).eval()
    out["base"] = transcribe_windows(model, processor, wave)
    print("base:", [w["text"] for w in out["base"]])

    config = Qwen3ASRConfig.from_pretrained(hf_dir)
    for model_dir in phonon_dirs:
        profile = model_dir.name.rsplit("-", 1)[-1]
        state = phonon1_state(model_dir, engine)
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [k for k in missing if k != "lm_head.weight"]
        assert not missing and not unexpected, (profile, missing[:5], unexpected[:5])
        model.tie_weights()
        assert torch.equal(model.lm_head.weight, state["model.language_model.embed_tokens.weight"])
        out[profile] = transcribe_windows(model, processor, wave)
        print(f"{profile}:", [w["text"] for w in out[profile]])
    return out


def main(onnx_dir, hf_dir, engine, *phonon_dirs) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "onnx_reference.json").write_text(json.dumps(onnx_fixture(onnx_dir), indent=1) + "\n")
    (OUT / "jfk_moon.json").write_text(
        json.dumps(speech_fixture(hf_dir, engine, list(phonon_dirs)), indent=1) + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    main(*(pathlib.Path(a) for a in sys.argv[1:]))
