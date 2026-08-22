# ten-vad reference fixtures

## `pitch.json`

One pitch estimate in Hz per 256-sample hop — `0.0` meaning unvoiced — over
`../silero/test.wav` (16 kHz mono, 60 s, 3750 hops). Produced by the ten-vad
**C reference**, not by a port.

Consumed by
`kits::speech::ten_vad::cross_test::tests::test_pitch_estimator_reference_golden`,
which pins `TenVadPitchEstimator` against it.

### Regenerating

`dump_pitch.cc` reproduces the reference's pitch branch: it drives
`AUP_PE_proc` (`src/pitch_est.cc`) from the reference `AUP_Analyzer` STFT,
feeding the estimator the raw hop and the un-normalized bin powers — the same
wiring `AUP_Aed_runOneFrm` uses. It needs a checkout of the ten-vad reference
for its sources; only the front end is linked, so no ONNX runtime is involved.

`coeff.h` cannot be included directly (it pulls in `aed_st.h`, which needs
`onnxruntime_c_api.h`), so the Hann-768 analysis window is extracted from it
and given external linkage:

```sh
TENVAD=/path/to/ten-vad          # https://github.com/TEN-framework/ten-vad
awk '/^const float AUP_AED_STFTWindow_Hann768/,/};/' "$TENVAD/src/coeff.h" \
  | sed '1s/^const float/extern const float/' > window.cc

g++ -O2 -w -I"$TENVAD/src" -o dump_pitch dump_pitch.cc window.cc \
    "$TENVAD/src/stft.cc" "$TENVAD/src/pitch_est.cc" \
    "$TENVAD/src/biquad.cc" "$TENVAD/src/fftw.c"

./dump_pitch ../silero/test.wav \
  | awk '{printf "%s%s", (NR>1 ? ", " : "["), $2} END {print "]"}' > pitch.json
```

The dump prints `frameIndex pitchHz voiced` per line; only the pitch column is
checked in, since the voicing flag is recoverable as `pitch > 0`.
