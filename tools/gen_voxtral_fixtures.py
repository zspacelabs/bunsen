#!/usr/bin/env python3
"""Generate the Voxtral Realtime reference fixtures.

Run once; commit the outputs. Regenerating is not part of CI.

    python3 -m venv /tmp/vxvenv
    /tmp/vxvenv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
    /tmp/vxvenv/bin/pip install transformers 'mistral_common[audio]' onnxruntime soundfile numpy
    /tmp/vxvenv/bin/python tools/gen_voxtral_fixtures.py ONNX_DIR HF_DIR

Either directory may be `-`, to leave its fixture as it is.

- `ONNX_DIR`: `onnx-community/Voxtral-Mini-4B-Realtime-2602-ONNX` at
  db34633130c460bd650644fc18dc90710d9f3e38: `audio_encoder.onnx`, `decoder_model_merged.onnx`,
  `embed_tokens.onnx` and their `.onnx_data*` files from its `onnx/`, the published graphs, not
  the ones `bind-voxtral-onnx` rewrites. onnxruntime refuses external data reached through a
  symlink that leaves the directory: hard links work.
- `HF_DIR`: `mistralai/Voxtral-Mini-4B-Realtime-2602` at 2769294da9567371363522aac9bbcfdd19447add:
  `config.json`, `generation_config.json`, `processor_config.json`, `tekken.json` and
  `model.safetensors`.

Pinned versions used to produce the committed fixtures:
    onnxruntime 1.31.1, numpy 2.3.5, transformers 5.22.1, torch 2.15.0+cpu,
    mistral_common 1.13.0, soundfile 0.14.0

Outputs, under `crates/validation/voxtral-model-validation/testdata/`:

`onnx_reference.json`
    The three published graphs, each run by onnxruntime (CPU, graph optimizations off) on
    deterministic synthetic input the Rust side rebuilds from the same formulas
    (`synthetic_features`, `synthetic_embeds`):

    - the audio encoder over `ENCODER_FRAMES` mel frames from empty caches (past its 750-frame
      attention window), then over `ENCODER_STEP_FRAMES` more through the caches it returned;
    - the token embedding over `EMBED_IDS`;
    - the decoder's prefill of `DECODER_PREFILL` positions from an empty cache, then one cached
      step.

    Each output is recorded as its `shape`, `sum`, `abs_sum` and a strided sample.
`jfk_moon.json`
    `transformers`' `VoxtralRealtimeForConditionalGeneration` on the CPU in float32, greedy, on
    `bunsen-test-audio`'s `jfk_moon.mp3` cut into `WINDOWS` (seconds), each window processed by
    its `VoxtralRealtimeProcessor` (offline: padded, a 39-id prompt, the default 480 ms delay):
    the `prompt_ids`, the whole sequence `ids` (the prompt first), each generated id's `margins`
    (top-1 minus top-2 logit), and the `text`. And `front_end`: the processor's `input_features`
    and the model's `audio_embeds` for `FRONT_END_SAMPLES` of a synthetic waveform the Rust side
    rebuilds from the same formula (`synthetic_wave`), so the front end is compared on the same
    samples and not through two MP3 decoders.
"""

import json
import pathlib
import sys

import numpy as np
import onnxruntime as ort

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "crates/validation/voxtral-model-validation/testdata"
SOURCE = "onnx-community/Voxtral-Mini-4B-Realtime-2602-ONNX@db34633130c460bd650644fc18dc90710d9f3e38"

N_RECORDED = 2048
N_MELS = 128
D_TEXT = 3072
# The packed conv carry: conv1's 128 input channels and conv2's 1280, two frames each.
PADDING_CACHE = (1, 1408, 2)
ENCODER_LAYERS, ENCODER_HEADS, ENCODER_HEAD_DIM = 32, 32, 64
DECODER_LAYERS, DECODER_KV_HEADS, DECODER_HEAD_DIM = 26, 8, 128

# Multiples of 8 mel frames: one decoder position each.
ENCODER_FRAMES = 1600
ENCODER_STEP_FRAMES = 96
EMBED_IDS = [1, 32, 33, 1656, 1278, 24361, 131071]
DECODER_PREFILL = 39


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


def synthetic_features(frames: int, offset: int) -> np.ndarray:
    """`[1, N_MELS, frames]` stand-in log-mels, in the extractor's range, from frame `offset`."""
    m = np.arange(N_MELS)[:, None]
    f = np.arange(offset, offset + frames)[None, :]
    return ((((m * 7 + f * 13) % 211) / 105.0 - 1.0) * 0.8 + 0.3).astype(np.float32)[None]


def synthetic_embeds(seq: int, salt: int) -> np.ndarray:
    """`[1, seq, D_TEXT]` stand-in decoder inputs."""
    k = np.arange(seq * D_TEXT).reshape(1, seq, D_TEXT)
    return (((((k * 5 + salt) % 197) / 98.0) - 1.0) * 0.05).astype(np.float32)


def session(path: pathlib.Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def encoder_reference(onnx_dir: pathlib.Path) -> dict:
    sess = session(onnx_dir / "audio_encoder.onnx")
    names = [o.name for o in sess.get_outputs()]

    def run(features, past, padding):
        seq = features.shape[2] // 2
        past_len = past[0][0].shape[2]
        feed = {
            "input_features": features,
            "attention_mask": np.ones((1, past_len + seq), np.int64),
            "position_ids": np.arange(past_len, past_len + seq, dtype=np.int64)[None],
            "past_padding_cache": padding,
        }
        for i in range(ENCODER_LAYERS):
            feed[f"past_key_values.{i}.key"], feed[f"past_key_values.{i}.value"] = past[i]
        d = dict(zip(names, sess.run(None, feed)))
        cache = [(d[f"present.{i}.key"], d[f"present.{i}.value"]) for i in range(ENCODER_LAYERS)]
        return d["audio_embeds"], cache, d["present_padding_cache"]

    empty = [(np.zeros((1, ENCODER_HEADS, 0, ENCODER_HEAD_DIM), np.float32),) * 2
             for _ in range(ENCODER_LAYERS)]
    embeds, cache, padding = run(synthetic_features(ENCODER_FRAMES, 0), empty,
                                 np.zeros(PADDING_CACHE, np.float32))
    step_embeds, step_cache, step_padding = run(
        synthetic_features(ENCODER_STEP_FRAMES, ENCODER_FRAMES), cache, padding)
    return {
        "frames": ENCODER_FRAMES,
        "step_frames": ENCODER_STEP_FRAMES,
        "embeds": record(embeds),
        "padding_cache": record(padding),
        "key_0": record(cache[0][0]),
        "value_31": record(cache[31][1]),
        "step_embeds": record(step_embeds),
        "step_padding_cache": record(step_padding),
        "step_key_31": record(step_cache[31][0]),
    }


def embed_reference(onnx_dir: pathlib.Path) -> dict:
    sess = session(onnx_dir / "embed_tokens.onnx")
    (embeds,) = sess.run(None, {"input_ids": np.array([EMBED_IDS], np.int64)})
    return {"ids": EMBED_IDS, "embeds": record(embeds)}


def decoder_reference(onnx_dir: pathlib.Path) -> dict:
    sess = session(onnx_dir / "decoder_model_merged.onnx")
    names = [o.name for o in sess.get_outputs()]

    def run(embeds, past):
        seq = embeds.shape[1]
        past_len = past[0][0].shape[2]
        feed = {"inputs_embeds": embeds, "attention_mask": np.ones((1, past_len + seq), np.int64)}
        for i in range(DECODER_LAYERS):
            feed[f"past_key_values.{i}.key"], feed[f"past_key_values.{i}.value"] = past[i]
        d = dict(zip(names, sess.run(None, feed)))
        cache = [(d[f"present.{i}.key"], d[f"present.{i}.value"]) for i in range(DECODER_LAYERS)]
        return d["logits"], cache

    empty = [(np.zeros((1, DECODER_KV_HEADS, 0, DECODER_HEAD_DIM), np.float32),) * 2
             for _ in range(DECODER_LAYERS)]
    logits, cache = run(synthetic_embeds(DECODER_PREFILL, 0), empty)
    step_logits, step_cache = run(synthetic_embeds(1, DECODER_PREFILL), cache)
    return {
        "prefill": DECODER_PREFILL,
        "logits": record(logits),
        "argmax": [int(i) for i in logits[0].argmax(-1)],
        "key_0": record(cache[0][0]),
        "value_25": record(cache[25][1]),
        "step_logits": record(step_logits),
        "step_argmax": int(step_logits[0, -1].argmax()),
        "step_key_25": record(step_cache[25][0]),
    }


CLIP = ROOT / "crates/dev/bunsen-test-audio/testdata/jfk_moon.mp3"
# The utterances `jfk_moon.json` transcribes, in seconds.
WINDOWS = [(0, 30), (30, 60)]


# The synthetic waveform's length: 2.5 s.
FRONT_END_SAMPLES = 40_000


def synthetic_wave(n: int) -> np.ndarray:
    """A deterministic, broadband `[n]` waveform: an integer sawtooth, exact in float32."""
    k = np.arange(n)
    return ((((k * 7919 + 13) % 2003) / 1001.5 - 1.0) * 0.25).astype(np.float32)


def front_end_fixture(model, processor) -> dict:
    import torch

    inputs = processor(synthetic_wave(FRONT_END_SAMPLES), return_tensors="pt")
    with torch.no_grad():
        audio = model.get_audio_features(input_features=inputs["input_features"]).pooler_output
    return {
        "samples": FRONT_END_SAMPLES,
        "input_features": record(inputs["input_features"].numpy()),
        "audio_embeds": record(audio.numpy()),
    }


def transcribe(model, processor, wave) -> dict:
    import torch

    inputs = processor(wave, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, do_sample=False, output_scores=True,
                             return_dict_in_generate=True)
    ids = out.sequences[0].tolist()
    margins = []
    for scores in out.scores:
        top = torch.topk(scores[0].float(), 2).values
        margins.append(float(top[0] - top[1]))
    return {
        "prompt_ids": inputs["input_ids"][0].tolist(),
        "ids": ids,
        "margins": margins,
        "text": processor.batch_decode(out.sequences, skip_special_tokens=True)[0],
    }


def speech_fixture(hf_dir: pathlib.Path) -> dict:
    import soundfile as sf
    import torch
    from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration

    wave, sr = sf.read(str(CLIP), dtype="float32")
    assert sr == 16000 and wave.ndim == 1
    processor = AutoProcessor.from_pretrained(hf_dir)
    model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
        hf_dir, dtype=torch.float32).eval()
    windows = []
    for a, b in WINDOWS:
        entry = transcribe(model, processor, wave[a * 16000:b * 16000])
        print(f"{a}-{b}s:", entry["text"], "min margin", min(entry["margins"]))
        windows.append(dict(start=a, end=b, **entry))
    return {"clip": CLIP.name, "front_end": front_end_fixture(model, processor),
            "windows": windows}


def main(onnx_dir: str, hf_dir: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if onnx_dir != "-":
        onnx_dir = pathlib.Path(onnx_dir)
        reference = {
            "source": SOURCE,
            "encoder": encoder_reference(onnx_dir),
            "embed": embed_reference(onnx_dir),
            "decoder": decoder_reference(onnx_dir),
        }
        (OUT / "onnx_reference.json").write_text(json.dumps(reference) + "\n")
        print("decoder step argmax", reference["decoder"]["step_argmax"])
    if hf_dir != "-":
        fixture = speech_fixture(pathlib.Path(hf_dir))
        (OUT / "jfk_moon.json").write_text(json.dumps(fixture) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
