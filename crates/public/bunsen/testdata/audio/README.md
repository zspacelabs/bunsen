# Audio format fixtures

A short synthetic tone, used by `support::audio`'s own tests to check that each
decoder path works and that the two agree. Nothing here is speech; the model
fixtures live with the validation crates that own them.

Synthetic on purpose: these carry no licence and no provenance question, and a
format test does not need real audio.

| file | |
|---|---|
| `tone.wav` | 0.5 s, 16 kHz mono, 16-bit PCM — 8000 samples exactly |
| `tone.mp3` | the same tone, LAME 48 kbps |

Regenerate with:

```sh
ffmpeg -f lavfi -i "sine=frequency=440:sample_rate=16000:duration=0.5" \
  -af "volume=0.6" -ac 1 -ar 16000 -c:a pcm_s16le tone.wav
ffmpeg -i tone.wav -c:a libmp3lame -b:a 48k -ac 1 -ar 16000 tone.mp3
```

The mp3 decodes to slightly more than 8000 samples: gapless playback trims the
encoder delay, but mp3 frames in blocks of 576 samples, so the tail rounds up.
