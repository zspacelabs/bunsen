"""Dump per-hop ten-vad speech probabilities from the reference Python API.

Drives the shipped `libten_vad.so` through `include/ten_vad.py` -- the same
path every binding takes -- over a 16 kHz mono 16-bit WAV, and writes one
probability per 256-sample hop as a JSON array.
"""

import json
import sys
import wave

import numpy as np

TENVAD = "/home/crutcher/git/ten-vad"
sys.path.insert(0, f"{TENVAD}/include")

from ten_vad import TenVad  # noqa: E402

HOP = 256


def read_wav_i16(path):
    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 1, "need mono"
        assert w.getframerate() == 16000, "need 16 kHz"
        assert w.getsampwidth() == 2, "need 16-bit"
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def main(wav_path, out_path):
    pcm = read_wav_i16(wav_path)
    vad = TenVad(HOP)

    probs, flags = [], []
    for start in range(0, len(pcm) - HOP + 1, HOP):
        p, f = vad.process(pcm[start : start + HOP])
        probs.append(float(p))
        flags.append(int(f))

    with open(out_path, "w") as fh:
        json.dump(probs, fh)

    voiced = sum(flags)
    print(f"hops={len(probs)} voiced={voiced} ({100.0*voiced/len(probs):.1f}%)")
    print(f"prob range = [{min(probs):.6f}, {max(probs):.6f}]")
    print(f"first 6 = {[round(p, 6) for p in probs[:6]]}")
