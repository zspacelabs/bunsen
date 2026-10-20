#!/usr/bin/env python3
"""Generate the speaker-turn and diarization-error fixtures.

Run once; commit the output. Regenerating is not part of CI.

    uv venv --python 3.12 /tmp/nemovenv
    uv pip install --python /tmp/nemovenv/bin/python \
        --index-url https://download.pytorch.org/whl/cpu torch torchaudio
    uv pip install --python /tmp/nemovenv/bin/python \
        "nemo_toolkit[asr] @ git+https://github.com/NVIDIA-NeMo/Speech@6f558408dbe1c2a7e6b730184e8d46e1474d6dca"
    /tmp/nemovenv/bin/python tools/gen_turn_fixtures.py

Pinned versions used to produce the committed fixture:
    nemo_toolkit 3.1.0+6f558408d, torch 2.14.1+cpu, numpy 2.5.4

The reference is NeMo's own diarization post-processing and scoring:

`turns`
    Seeded per-speaker probability tracks, each run through
    `vad_utils.binarization_vectorized` then `vad_utils.filtering` (the pair
    `predlist_to_timestamps` applies, without its rounding to two decimals),
    under each `TurnFilterConfig` preset bunsen names, NeMo's default filter
    order and the other.

`der`
    Seeded reference and hypothesis turns, scored by
    `der.score_labels_from_rttm_labels` (NeMo's port of NIST md-eval-22) with
    and without a collar (half-width, md-eval's convention) and with and
    without overlap scored.

Output: `crates/public/bunsen/testdata/diarization/nemo.json`.
"""

import json
import pathlib

import numpy as np
import torch
from nemo.collections.asr.metrics.der import score_labels_from_rttm_labels
from nemo.collections.asr.parts.utils.vad_utils import binarization_vectorized, filtering

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "crates/public/bunsen/testdata/diarization/nemo.json"

FRAME = 0.08
N_FRAMES = 200
N_SPEAKERS = 3

# The presets of `TurnFilterConfig`, and two more that exercise the fill-first
# order and keep-between hysteresis with padding.
CONFIGS = {
    "none": dict(onset=0.5, offset=0.5, pad_onset=0.0, pad_offset=0.0,
                 min_duration_on=0.0, min_duration_off=0.0, filter_speech_first=True),
    "dihard3_v2": dict(onset=0.56, offset=1.0, pad_onset=0.063, pad_offset=0.002,
                       min_duration_on=0.007, min_duration_off=0.151, filter_speech_first=True),
    "callhome_v2": dict(onset=0.641, offset=0.561, pad_onset=0.229, pad_offset=0.079,
                        min_duration_on=0.511, min_duration_off=0.296, filter_speech_first=True),
    "callhome_v2_fill_first": dict(onset=0.641, offset=0.561, pad_onset=0.229, pad_offset=0.079,
                                   min_duration_on=0.511, min_duration_off=0.296,
                                   filter_speech_first=False),
    "wide": dict(onset=0.7, offset=0.3, pad_onset=0.15, pad_offset=0.25,
                 min_duration_on=0.3, min_duration_off=0.4, filter_speech_first=True),
}


def track(rng, frames, n_speakers):
    """Each speaker drifts between speech and silence, with jitter around the
    thresholds. float32, as a model emits them."""
    levels = np.zeros(n_speakers, dtype=np.float32)
    out = np.zeros((frames, n_speakers), dtype=np.float32)
    for i in range(frames):
        flip = rng.random(n_speakers) < 0.08
        levels = np.where(flip, np.where(levels > 0.5, 0.1, 0.9), levels).astype(np.float32)
        out[i] = np.clip(levels + (rng.random(n_speakers).astype(np.float32) - 0.5) * 0.7, 0.0, 1.0)
    return out


def json_f32(x):
    return float(np.format_float_positional(np.float32(x), unique=True, trim="-"))


def nemo_turns(probs, c):
    params = {
        "onset": c["onset"],
        "offset": c["offset"],
        "pad_onset": c["pad_onset"],
        "pad_offset": c["pad_offset"],
        "frame_length_in_sec": FRAME,
        "min_duration_on": c["min_duration_on"],
        "min_duration_off": c["min_duration_off"],
        "filter_speech_first": 1.0 if c["filter_speech_first"] else 0.0,
    }
    turns = []
    for speaker in range(probs.shape[1]):
        segments = binarization_vectorized(torch.from_numpy(probs[:, speaker].copy()), params)
        if segments.numel() > 0:
            segments = filtering(segments, params)
        for start, end in segments.tolist():
            turns.append({"speaker": speaker, "start": start, "end": end})
    turns.sort(key=lambda t: (t["start"], t["speaker"]))
    return turns


def random_turns(rng, names, length):
    """Turns on a 10 ms grid: each speaker talks in a few stretches."""
    turns = []
    for name in names:
        t = round(float(rng.uniform(0, 2)), 2)
        while t < length:
            d = round(float(rng.uniform(0.3, 3.0)), 2)
            turns.append((t, min(t + d, length), name))
            t = round(t + d + float(rng.uniform(0.2, 4.0)), 2)
    return turns


def jitter(rng, turns, rename):
    """A hypothesis: the reference's turns, moved, renamed, some dropped, some
    given to the wrong speaker, and a stray turn."""
    out = []
    names = sorted(set(rename.values()))
    for start, end, name in turns:
        if rng.random() < 0.1:
            continue
        start = max(0.0, round(start + float(rng.normal(0, 0.15)), 2))
        end = round(end + float(rng.normal(0, 0.15)), 2)
        if end <= start:
            continue
        who = rename[name] if rng.random() > 0.15 else names[int(rng.integers(len(names)))]
        out.append((start, end, who))
    out.append((5.0, 5.7, "stray"))
    return out


def main():
    rng = np.random.default_rng(20261020)

    turn_cases = []
    for name, config in CONFIGS.items():
        for trial in range(2):
            probs = track(rng, N_FRAMES, N_SPEAKERS)
            turn_cases.append({
                "config": name,
                "frame_seconds": FRAME,
                "n_speakers": N_SPEAKERS,
                # Shortest round-trip float32 spellings: each parses back to
                # the float32 the reference saw.
                "probs": [json_f32(x) for x in probs.reshape(-1)],
                "turns": nemo_turns(probs, config),
            })

    der_cases = []
    for trial in range(6):
        names = ["A", "B", "C", "D"][: 2 + trial % 3]
        reference = random_turns(rng, names, 30.0)
        rename = {n: f"speaker_{(i + 1) % len(names)}" for i, n in enumerate(names)}
        hypothesis = jitter(rng, reference, rename)
        for collar in (0.0, 0.25):
            for ignore_overlap in (False, True):
                labels = lambda ts: [f"{s:.2f} {e:.2f} {n}" for s, e, n in ts]
                _, _, (der, confusion, fa, miss) = score_labels_from_rttm_labels(
                    [("rec", labels(reference))],
                    [("rec", labels(hypothesis))],
                    collar=collar,
                    ignore_overlap=ignore_overlap,
                    verbose=False,
                )
                der_cases.append({
                    "reference": [list(t) for t in reference],
                    "hypothesis": [list(t) for t in hypothesis],
                    "collar": collar,
                    "skip_overlap": ignore_overlap,
                    "der": der,
                    "confusion": confusion,
                    "false_alarm": fa,
                    "missed": miss,
                })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "source": "NVIDIA-NeMo/Speech@6f558408dbe1c2a7e6b730184e8d46e1474d6dca",
        "turns": turn_cases,
        "der": der_cases,
    }) + "\n")
    print(f"wrote {OUT}: {len(turn_cases)} turn cases, {len(der_cases)} der cases")


if __name__ == "__main__":
    main()
