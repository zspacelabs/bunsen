# ten-vad reference fixtures

Two goldens over the same audio, `../silero/test.wav` (16 kHz mono, 60 s,
3750 hops of 256 samples):

| file | what it pins | produced by |
|---|---|---|
| `probs.json` | the **whole driver** — front end and model | the reference Python binding |
| `pitch.json` | feature `40` alone | a harness around the reference C pitch estimator |

`probs.json` is the stronger of the two: nothing is shared between it and
bunsen except the audio. `pitch.json` isolates one feature, which is what makes
a pitch regression legible instead of showing up as a drifting probability.

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


## `probs.json`

One speech probability per hop, from the reference implementation end to end:
its own front end, its own inference engine. Consumed by
`kits::speech::ten_vad::cross_test::tests::test_reference_probability_golden`.

### Regenerating

`gen_probs.py` drives the shipped `lib/Linux/x64/libten_vad.so` through
`include/ten_vad.py` — the same `ten_vad_process` entry point every binding
uses — from the reference repo's own venv.

```sh
TENVAD=/path/to/ten-vad          # https://github.com/TEN-framework/ten-vad
cd "$TENVAD" && .venv/bin/python /path/to/gen_probs.py   # see the snippet below
```

`gen_probs.py` exposes `main(wav_path, out_path)`; call it with this repo's
fixture and `probs.json`.

**The prebuilt `.so` needs LLVM's libc++, which Ubuntu does not install by
default.** It is not in the base image and the failure is an opaque
`OSError: libc++.so.1: cannot open shared object file`. You do not need root —
fetch the packages and unpack them locally:

```sh
mkdir -p /tmp/libcxx && cd /tmp/libcxx
apt-get download libc++1-18 libc++abi1-18 libunwind-18
for d in *.deb; do dpkg -x "$d" root; done
export LD_LIBRARY_PATH=/tmp/libcxx/root/usr/lib/x86_64-linux-gnu
```

Note `libunwind-18` specifically, not `libunwind8`: `libc++abi` wants
`libunwind.so.1`, and Ubuntu's `libunwind8` provides `libunwind.so.8`.
Verify with `ldd "$TENVAD/lib/Linux/x64/libten_vad.so" | grep "not found"`
returning nothing before running the generator.

Building from source instead is *not* currently an option here: the ONNX
Runtime C headers `examples_onnx` needs are not on this machine, and the venv
ships the runtime shared library without them.

At the time of writing the fixture yields 3750 hops, 76.2% flagged voiced, with
probabilities spanning `[0.158018, 0.992463]`.
