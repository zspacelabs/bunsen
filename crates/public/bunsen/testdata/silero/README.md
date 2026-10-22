# Silero VAD fixtures

What upstream's Silero VAD says about a real clip, so that the kits' tests of
what a driver or a gate does with speech probabilities need no model.

| file | |
|---|---|
| `jfk_moon_4s.json` | upstream's 16 kHz model (`silero/vad-16k`, `silero_vad_16k.safetensors`) over `bunsen_test_audio::JFK_MOON_4S`, zero-padded to whole chunks of 512 samples, in one call through the streaming context, on the CPU: one speech probability per chunk. Read by `RecordedVoiceActivity::silero_jfk_moon_4s`. |

`silero-model-validation` checks it against the model
(`test_the_kits_recording_is_upstreams`), and rewrites it when run by name:

```sh
cargo test -p silero-model-validation --features weights \
  -- --ignored write_the_kits_recording
```

The probabilities are the output of Silero's MIT-licensed model; the clip is
`bunsen-test-audio`'s.
