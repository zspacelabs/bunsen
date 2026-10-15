#!/usr/bin/env python3
"""Generate the Phonon-2 transcription reference.

Run once; commit the output. Regenerating is not part of CI.

    python3 -m venv /tmp/pkvenv
    /tmp/pkvenv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
    /tmp/pkvenv/bin/pip install transformers librosa soundfile numpy
    /tmp/pkvenv/bin/python tools/gen_phonon2_fixture.py \\
        REFERENCE_TRANSFORMERS_PY MODEL_FERMION BASE_DIR

- `REFERENCE_TRANSFORMERS_PY`: `reference_transformers.py` from
  `FermionResearch/Phonon-2` (Apache-2.0), beside its `fermion_container.py`.
- `MODEL_FERMION`: `model.fermion` from `phonon-2.bps.tar.zst`.
- `BASE_DIR`: `nvidia/parakeet-tdt-0.6b-v3`'s `config.json`,
  `generation_config.json`, `processor_config.json`, `tokenizer.json` and
  `tokenizer_config.json`.

Pinned versions used to produce the committed fixture:
    transformers 5.19.0, torch 2.14.1+cpu, soundfile 0.14.0, numpy 2.5.4

This is Phonon-2's own reference path: the container expanded to float32 and
loaded into `transformers`' `ParakeetForTDT`, then the greedy TDT `generate`
behind `ParakeetProcessor`, on the CPU in float32.

One output, `crates/validation/parakeet-model-validation/testdata/phonon2_jfk_moon.json`:
the `text`, the emitted token `ids` (blanks dropped), and the encoder `frames`
each was emitted at, for `whisper-model-validation`'s `jfk_moon.mp3`.
"""

import importlib.util
import json
import pathlib
import sys

import numpy as np
import soundfile as sf
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
CLIP = ROOT / "crates/validation/whisper-model-validation/testdata/jfk_moon.mp3"
OUT = ROOT / "crates/validation/parakeet-model-validation/testdata/phonon2_jfk_moon.json"


def main(reference: pathlib.Path, container: pathlib.Path, base_dir: pathlib.Path) -> None:
    sys.path.insert(0, str(reference.parent))
    spec = importlib.util.spec_from_file_location("reference_transformers", reference)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)

    model, processor, _ = ref.load_model(str(container), str(base_dir))
    wave, sr = sf.read(str(CLIP), dtype="float32")
    assert sr == 16000 and wave.ndim == 1, (sr, wave.shape)
    inputs = processor([wave], sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model.generate(
            input_features=inputs["input_features"],
            attention_mask=inputs.get("attention_mask"),
        )
    seq = out.sequences[0].tolist()
    durations = out.durations[0].tolist()
    blank = model.config.blank_token_id
    # `sequences[k]` was chosen at the frame the durations before it reach.
    frames = np.cumsum([0] + durations[:-1]).tolist()
    ids, at = [], []
    for k in range(1, len(seq)):
        if seq[k] != blank:
            ids.append(int(seq[k]))
            at.append(int(frames[k]))
    text = processor.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {"clip": CLIP.name, "model": "FermionResearch/Phonon-2", "text": text,
             "ids": ids, "frames": at},
            indent=1,
        )
        + "\n"
    )
    print(f"wrote {OUT}: {len(ids)} ids")
    print(text)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*(pathlib.Path(a) for a in sys.argv[1:]))
