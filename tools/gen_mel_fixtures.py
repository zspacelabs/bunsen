#!/usr/bin/env python3
"""Generate mel-converter reference fixtures.

Run once; commit the outputs. Regenerating is not part of CI.

    python3 -m venv /tmp/melvenv
    /tmp/melvenv/bin/pip install librosa
    /tmp/melvenv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
    /tmp/melvenv/bin/pip install openai-whisper
    /tmp/melvenv/bin/python tools/gen_mel_fixtures.py

Pinned versions used to produce the committed fixtures:
    librosa 1.0.0, numpy 2.5.2, torch 2.13.0+cpu, openai-whisper 20250625

CPU torch is deliberate: the fixture only has to be correct, and a CPU run is
deterministic and a far smaller download than the CUDA wheels.

All outputs are flat little-endian f32, row-major.
"""

import pathlib
import numpy as np
import librosa

OUT = pathlib.Path(__file__).resolve().parent.parent / "crates/public/bunsen/testdata/perceptive_audio"

SR, N_FFT, HOP, N_MELS = 16000, 400, 160, 80
SECONDS = 2


def write(name, arr):
    path = OUT / name
    path.write_bytes(np.asarray(arr, dtype="<f4").tobytes(order="C"))
    print(f"{name:34s} {arr.shape} {path.stat().st_size:>8d} B")


def deterministic_signal(n):
    """Seeded noise + two chirps + a 200 ms silence.

    The silence matters: it is what exercises the log floor and the
    dynamic-range clamp, which a pure-noise signal never reaches.
    """
    rng = np.random.default_rng(0xB0BA)
    t = np.arange(n) / SR
    x = 0.05 * rng.standard_normal(n)
    x += 0.4 * np.sin(2 * np.pi * (200 + 600 * t) * t)
    x += 0.2 * np.sin(2 * np.pi * (3000 - 900 * t) * t)
    x[int(0.9 * SR): int(1.1 * SR)] = 0.0
    return np.clip(x, -1.0, 1.0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Periodic Hann, the analysis window.
    write("hann_400_periodic.f32", librosa.filters.get_window("hann", N_FFT, fftbins=True))

    # Filterbanks: the Whisper/librosa default, and the HTK+no-norm variant.
    write("mel_fb_slaney_16k_400_80.f32", librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS))
    write(
        "mel_fb_htk_16k_400_80_nonorm.f32",
        librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, htk=True, norm=None),
    )

    y = deterministic_signal(SECONDS * SR)
    write("signal_2s_16k.f32", y)

    # `center=True` with reflect padding is the geometry the streaming
    # converter reproduces: reflect start padding, reflect end padding.
    # librosa defaults to `pad_mode="constant"`, so it is passed explicitly.
    for name, center in [("logmel_center_true.f32", True), ("logmel_center_false.f32", False)]:
        S = librosa.feature.melspectrogram(
            y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
            center=center, pad_mode="reflect", power=2.0,
            window="hann", htk=False, norm="slaney",
        )
        # `[n_mels, frames]` -> `[frames, n_mels]`, matching the converter.
        write(name, np.log10(np.maximum(S, 1e-10)).T)

    write_whisper(y)


def write_whisper(y):
    """`whisper.audio.log_mel_spectrogram`, the packaged form.

    This is librosa's `center=True` spectrogram plus Whisper's own tail: drop
    the final frame, floor 8 log-units below the maximum, then `(x + 4) / 4`.
    Its filterbank is the same `librosa.filters.mel` call, shipped as an npz.
    """
    try:
        import torch
        import whisper.audio as wa
    except ImportError as e:
        print(f"skipping whisper_logmel.f32: {e}")
        return

    out = wa.log_mel_spectrogram(torch.from_numpy(y.astype(np.float32)), n_mels=N_MELS)

    # `[n_mels, frames]` -> `[frames, n_mels]`.
    write("whisper_logmel.f32", out.numpy().T)


if __name__ == "__main__":
    main()
