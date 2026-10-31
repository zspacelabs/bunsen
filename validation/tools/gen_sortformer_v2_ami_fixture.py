#!/usr/bin/env python3
"""Generate the Sortformer v2 real-speech fixture: NeMo's own diarization of AMI.

Run once; commit the output. Regenerating is not part of CI.

    uv venv --python 3.12 /tmp/nemovenv
    uv pip install --python /tmp/nemovenv/bin/python \
        --index-url https://download.pytorch.org/whl/cpu torch torchaudio
    uv pip install --python /tmp/nemovenv/bin/python \
        "nemo_toolkit[asr] @ git+https://github.com/NVIDIA-NeMo/Speech@6f558408dbe1c2a7e6b730184e8d46e1474d6dca"
    /tmp/nemovenv/bin/python -I validation/tools/gen_sortformer_v2_ami_fixture.py V2_NEMO V2_1_NEMO AMI_WAV AMI_RTTM

`V2_NEMO` and `V2_1_NEMO` are `nvidia/diar_streaming_sortformer_4spk-v2`'s
`.nemo` at revision 84edd514b8ef68004c10086918cd62f2148cbd59 (sha256
b371afce...) and `-v2.1`'s at cd03eee90fbec18297ac31b8c21546e596b7f71c
(sha256 8abd3283...): the files bunsen's well-known rows pin. `AMI_WAV` and
`AMI_RTTM` are `bunsen-test-audio`'s `ami` fixture (NeMo-Speech.cpp's
`test_files/diar/ami_en2002d_2132.{wav,rttm}` at 2bdfb03b).

Pinned versions used to produce the committed fixture:
    nemo_toolkit 3.1.0+6f558408d, torch 2.14.1+cpu, numpy 2.5.4

NeMo itself, on the CPU in float32, streams the clip through its sync
`forward_streaming` (`streaming_update`) at each latency preset bunsen names,
batch one: one frame of left context, three silence slots per speaker, a
cache of 188. Recorded per model and preset:

    probs   the 80 ms speaker probabilities, `[1, frames, 4]`, as a
            `TensorRecord` (shape, sums, every `stride`-th value)
    der     NeMo's md-eval DER (and its parts) of the turns NeMo's
            post-processing gives without filtering (onset = offset = 0.5),
            against the RTTM, collar 0.25 s per side, overlap scored

Output: `validation/models/sortformer-model-validation/testdata/ami_reference_v2.json`.
"""

import json
import pathlib
import sys
import wave

import numpy as np
import torch
from nemo.collections.asr.metrics.der import score_labels_from_rttm_labels
from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.collections.asr.parts.utils.vad_utils import binarization_vectorized

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "validation/models/sortformer-model-validation/testdata/ami_reference_v2.json"

MODELS = {
    "v2": "nvidia/diar_streaming_sortformer_4spk-v2@84edd514b8ef68004c10086918cd62f2148cbd59",
    "v2_1": "nvidia/diar_streaming_sortformer_4spk-v2.1@cd03eee90fbec18297ac31b8c21546e596b7f71c",
}

# (chunk_len, chunk_right_context, fifo_len, spkcache_update_period): bunsen's
# `SortformerStreamingConfig::sortformer_v2_*` presets, each with
# `chunk_left_context` 1, `spkcache_sil_frames_per_spk` 3 and `spkcache_len` 188.
PRESETS = {
    "low_latency": (6, 7, 188, 144),
    "very_high_latency": (340, 40, 40, 300),
}
N_RECORDED = 4096
FRAME = 0.08


def record(x: np.ndarray) -> dict:
    flat = np.asarray(x, dtype=np.float64).reshape(-1)
    stride = max(1, flat.size // N_RECORDED)
    return {
        "shape": list(x.shape),
        "sum": float(flat.sum()),
        "abs_sum": float(np.abs(flat).sum()),
        "stride": stride,
        "samples": [float(v) for v in flat[::stride]],
    }


def read_wav(path: pathlib.Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return (pcm.astype(np.float32) / 32768.0)


def turns(probs: np.ndarray) -> list:
    params = {"onset": 0.5, "offset": 0.5, "pad_onset": 0.0, "pad_offset": 0.0, "frame_length_in_sec": FRAME}
    out = []
    for speaker in range(probs.shape[1]):
        segments = binarization_vectorized(torch.from_numpy(probs[:, speaker].copy()), params)
        out += [f"{s:.3f} {e:.3f} speaker_{speaker}" for s, e in segments.tolist()]
    return out


def main(v2: pathlib.Path, v2_1: pathlib.Path, wav: pathlib.Path, rttm: pathlib.Path) -> None:
    audio = torch.from_numpy(read_wav(wav))[None]
    lengths = torch.tensor([audio.shape[1]])
    reference = [
        f"{float(f[3]):.3f} {float(f[3]) + float(f[4]):.3f} {f[7]}"
        for f in (line.split() for line in rttm.read_text().splitlines())
        if f and f[0] == "SPEAKER"
    ]

    out = {
        "runtime": "nemo_toolkit 3.1.0+6f558408d, sync forward_streaming, float32 CPU",
        "clip": "ami_en2002d_2132.wav",
        "models": {},
    }
    for key, path in (("v2", v2), ("v2_1", v2_1)):
        model = SortformerEncLabelModel.restore_from(str(path), map_location="cpu").float().eval()
        m = model.sortformer_modules
        presets = {}
        for name, (chunk, rc, fifo, period) in PRESETS.items():
            m.chunk_len, m.chunk_right_context, m.chunk_left_context = chunk, rc, 1
            m.fifo_len, m.spkcache_update_period, m.spkcache_len = fifo, period, 188
            m.spkcache_sil_frames_per_spk = 3
            model.streaming_mode, model.async_streaming = True, False
            with torch.no_grad():
                probs = model.forward(audio_signal=audio, audio_signal_length=lengths)
            probs = probs.float().numpy()
            _, _, (der, confusion, fa, miss) = score_labels_from_rttm_labels(
                [("ami", reference)], [("ami", turns(probs[0]))], collar=0.25, ignore_overlap=False, verbose=False
            )
            presets[name] = {
                "probs": record(probs),
                "der": {"der": der, "confusion": confusion, "false_alarm": fa, "missed": miss},
            }
            print(f"{key} {name}: probs {probs.shape}, DER {der:.4f} (conf {confusion:.4f} fa {fa:.4f} miss {miss:.4f})")
        out["models"][key] = {"source": f"{MODELS[key]} (.nemo)", "presets": presets}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main(*(pathlib.Path(a) for a in sys.argv[1:5]))
