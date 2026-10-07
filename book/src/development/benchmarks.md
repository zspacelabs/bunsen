# Benchmarks

This page lists where bunsen measures performance, and the two things that
make a measurement honest: a real backend, and knowing whether a number is
cold or warm. It quotes no numbers; they depend on the hardware, and the
benchmarks print their own.

## What there is

The criterion benches live in
[`crates/public/bunsen/benches`](https://github.com/zspacelabs/bunsen/tree/main/crates/public/bunsen/benches):

| Bench | Measures | Runs on |
|---|---|---|
| `contracts` | shape-contract unpacks, asserts and periodic asserts ([cost](bunsen::contracts#cost)) | the host |
| `drop_block` | 2D DropBlock over a small image batch | the CPU backend, always |
| `math` | [`maybe_iroot`](bunsen::support::math::maybe_iroot) against a floating-point reference | the host |
| `lbm` | the D2Q9 lattice-Boltzmann collision, streaming and update steps, in `f32` and `f64` | `performance_device()` |
| `whisper_decode` | the Whisper encoder and decode loop at `base.en`'s shapes, with random weights, cold and warm | `performance_device()` |

Outside the benches:

- [`silero-bench`](https://github.com/zspacelabs/bunsen/tree/main/crates/dev/silero-bench)
  runs a mono WAV file through Silero VAD's streaming context on
  `performance_device()`, prints the per-chunk speech probabilities, and
  optionally checks them against an expected JSON file.
- [`conway`](https://github.com/zspacelabs/bunsen/tree/main/examples/conway)'s
  `benchmark` subcommand reports the sustained steps per second of the 2D or
  3D Game of Life on the device `--device` selects. It measures the tensor
  kernels behind the simulation kits.
- [whisper-cli](https://github.com/zspacelabs/bunsen/blob/main/examples/whisper-cli/README.md#benchmarks)'s
  `transcribe` reports, over a set of files, the mean audio length, the mean
  decode time and the ratio of the two. Its README runs it over a speech
  dataset.

## Cold and warm

The CubeCL backends autotune their kernels, and the tuning is keyed on
shapes. The first call at a new shape runs the tuner; later calls at that
shape use what it found. A cold call and a warm one can differ by more than
the optimization you are about to try, so measure both before deciding what
to optimize, and know which one a number is.

- Criterion warms up before it measures, so its numbers are warm.
  `whisper_decode` times each case's first call once and prints it as
  `cold`.
- `conway benchmark` leaves a warmup fraction of its steps out of the
  timing.
- whisper-cli sizes every push to the stream driver's grain, so the front
  end and the autotuned kernels behind it see one shape, not a drift of
  remainders.
- A benchmark whose shapes change every iteration measures the tuner, not
  the kernel.

## Always pass a backend

A benchmark on
[`performance_device`](bunsen::support::testing::performance_device)
measures whatever bunsen's backend feature selected, and without one that is
the CPU. It doesn't fail; it reports CPU numbers
([Testing and backends](./testing.md#the-silent-cpu-fallback)). Pass the
backend explicitly: `--features wgpu`, or another backend, for bunsen's own
benches, and `--features bunsen/wgpu` for a binary that reaches
`performance_device()` through bunsen, such as silero-bench. The examples,
whisper-cli and `conway` among them, pick their device at run time with
`--device`; a backend is selectable only when the example was built with its
feature, and without one `auto` is the CPU. The examples' READMEs and the
`whisper_decode` bench's docs give exact commands.

Benches build optimized already. Run the binaries with `--release` too: the
work is inside burn's kernels, and an unoptimized build measures the wrong
thing.
