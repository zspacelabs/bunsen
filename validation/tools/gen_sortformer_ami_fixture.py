#!/usr/bin/env python3
"""Generate the Sortformer real-speech fixture: NeMo's own diarization of AMI.

Run once; commit the output. Regenerating is not part of CI.

    uv venv --python 3.12 /tmp/nemovenv
    uv pip install --python /tmp/nemovenv/bin/python \
        --index-url https://download.pytorch.org/whl/cpu torch torchaudio
    uv pip install --python /tmp/nemovenv/bin/python \
        "nemo_toolkit[asr] @ git+https://github.com/NVIDIA-NeMo/Speech@6f558408dbe1c2a7e6b730184e8d46e1474d6dca"
    /tmp/nemovenv/bin/python validation/tools/gen_sortformer_ami_fixture.py NEMOTRON_NEMO AMI_WAV AMI_RTTM

`NEMOTRON_NEMO` is `nvidia/Nemotron-3-Diarization`'s
`Nemotron-3-Diarization.nemo` at revision
f667ed73aee57d40cc39428eb768b4fd87a0a29e (sha256 867c53f5...). `AMI_WAV` and
`AMI_RTTM` are `bunsen-test-audio`'s `ami` fixture (NeMo-Speech.cpp's
`test_files/diar/ami_en2002d_2132.{wav,rttm}` at 2bdfb03b).

Pinned versions used to produce the committed fixture:
    nemo_toolkit 3.1.0+6f558408d, torch 2.14.1+cpu, numpy 2.5.4

NeMo itself, on the CPU in float32, streams the clip through its sync
`forward_streaming` (`streaming_update`) at each latency preset bunsen names,
batch one, its mel front end computing with the bf16 tables the checkpoint
stores. Recorded per preset:

    probs   the 10 ms speaker probabilities, `[1, frames, 8]`, as a
            `TensorRecord` (shape, sums, every `stride`-th value)
    der     NeMo's md-eval DER (and its parts) of the turns NeMo's
            post-processing gives without filtering (onset = offset = 0.5),
            against the RTTM, collar 0.25 s per side, overlap scored

Output: `validation/models/sortformer-model-validation/testdata/ami_reference.json`.
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
OUT = ROOT / "validation/models/sortformer-model-validation/testdata/ami_reference.json"

# (chunk_len, chunk_right_context, fifo_len, spkcache_update_period, spkcache_len): bunsen's
# `SortformerStreamingConfig::nemotron_*` presets.
PRESETS = {
    "low_latency": (9, 4, 264, 222, 264),
    "offline": (340, 40, 40, 300, 264),
}
N_RECORDED = 4096
FRAME = 0.01


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


def main(nemo: pathlib.Path, wav: pathlib.Path, rttm: pathlib.Path) -> None:
    model = SortformerEncLabelModel.restore_from(str(nemo), map_location="cpu").float().eval()
    audio = torch.from_numpy(read_wav(wav))[None]
    lengths = torch.tensor([audio.shape[1]])
    reference = [
        f"{float(f[3]):.3f} {float(f[3]) + float(f[4]):.3f} {f[7]}"
        for f in (line.split() for line in rttm.read_text().splitlines())
        if f and f[0] == "SPEAKER"
    ]

    out = {
        "source": "nvidia/Nemotron-3-Diarization@f667ed73aee57d40cc39428eb768b4fd87a0a29e (.nemo)",
        "runtime": "nemo_toolkit 3.1.0+6f558408d, sync forward_streaming, float32 CPU",
        "clip": "ami_en2002d_2132.wav",
        "presets": {},
    }
    m = model.sortformer_modules
    for name, (chunk, rc, fifo, period, cache) in PRESETS.items():
        m.chunk_len, m.chunk_right_context, m.chunk_left_context = chunk, rc, 0
        m.fifo_len, m.spkcache_update_period, m.spkcache_len = fifo, period, cache
        model.streaming_mode, model.async_streaming = True, False
        with torch.no_grad():
            probs = model.forward(audio_signal=audio, audio_signal_length=lengths)
        probs = probs.float().numpy()
        _, _, (der, confusion, fa, miss) = score_labels_from_rttm_labels(
            [("ami", reference)], [("ami", turns(probs[0]))], collar=0.25, ignore_overlap=False, verbose=False
        )
        out["presets"][name] = {
            "probs": record(probs),
            "der": {"der": der, "confusion": confusion, "false_alarm": fa, "missed": miss},
        }
        print(f"{name}: probs {probs.shape}, DER {der:.4f} (conf {confusion:.4f} fa {fa:.4f} miss {miss:.4f})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main(*(pathlib.Path(a) for a in sys.argv[1:4]))
