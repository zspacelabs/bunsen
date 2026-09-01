# Audio fixtures

A short synthetic tone, used by `support::audio`'s own tests to check that each
decoder path works and that the two agree; and one four-second slice of
speech, used by the Whisper kit's speech gate to check the hysteresis machine
against Silero's real probabilities. The model fixtures live with the
validation crates that own them.

The tone is synthetic on purpose: it carries no licence and no provenance
question, and a format test does not need real audio.

| file | |
|---|---|
| `tone.wav` | 0.5 s, 16 kHz mono, 16-bit PCM — 8000 samples exactly |
| `tone.mp3` | the same tone, LAME 48 kbps |
| `jfk_moon_4s.mp3` | 4.0 s, 16 kHz mono, LAME 48 kbps — speech with a one-second pause |

## `jfk_moon_4s.mp3`

Seconds 7.0 to 11.0 of `whisper-model-validation`'s `jfk_moon.mp3` (see its
`testdata/README.md` for the source: John F. Kennedy at Rice University,
1962-09-12, public domain as a work of the U.S. federal government). The cut
holds the end of "why climb the highest mountain?", about a second of pause,
and "Why, thirty-five years ago, fly the Atlantic?", so a gate with a short
silence threshold finds two regions and a patient one finds one.

```sh
ffmpeg -i jfk_moon.mp3 -ss 7.0 -t 4.0 -ac 1 -ar 16000 -c:a libmp3lame -b:a 48k jfk_moon_4s.mp3
```

Regenerate with:

```sh
ffmpeg -f lavfi -i "sine=frequency=440:sample_rate=16000:duration=0.5" \
  -af "volume=0.6" -ac 1 -ar 16000 -c:a pcm_s16le tone.wav
ffmpeg -i tone.wav -c:a libmp3lame -b:a 48k -ac 1 -ar 16000 tone.mp3
```

The mp3 decodes to slightly more than 8000 samples: gapless playback trims the
encoder delay, but mp3 frames in blocks of 576 samples, so the tail rounds up.
