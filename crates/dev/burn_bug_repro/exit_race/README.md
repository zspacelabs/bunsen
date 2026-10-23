# burn-exit-race-repro

Exiting with work still queued on a burn wgpu device (`vulkan` or `wgpu`) can segfault inside NVIDIA's Vulkan driver, after the program's output is written.

The device's work runs on the wgpu runtime's own device thread. When a program frees its tensors and returns from `main`, that thread can still be freeing memory-pool pages, through `wgpu_hal::vulkan::Device::destroy_buffer`, while libc's `exit()` runs the driver's exit handler. The handler tears the driver down underneath the thread.

Seen on Linux with NVIDIA driver 595.99.02 (RTX 3090) and burn 0.22. CUDA and the CPU (`flex`) don't crash. bunsen hit it with Whisper `medium` and `large`, and works around it with `bunsen::burner::device::DeviceSession`, which waits for the device before exit.

**Status: this doesn't reproduce the crash yet.** bunsen's own binary crashes on most launches; this program hasn't crashed on Vulkan in 40 runs. Four workloads have been tried, each matching more of the state in bunsen's core dump:
- v1: 3.4 GB of weights, freed, and a few tail allocations;
- v2: weights uploaded from host data, a decode-like loop, and a 200 MiB tensor;
- v3: 4,000 tail allocations handed to the device thread with `flush`;
- v4: the 200 MiB tensor last, so the weights sit in pools retired by the growth.

The next step is to compare cubecl's memory logs from bunsen's crashing run and from this program.

The package is standalone: it depends only on `burn`, from crates.io, and has its own `[workspace]`.

```sh
cargo run --release --features vulkan -- --device vulkan --runs 10          # counts crashes
cargo run --release --features vulkan -- --device vulkan --runs 10 --sync   # the workaround
```

`--runs N` runs the binary N times as a child, with its output piped, and reports each child's exit status. The crash is a race: piped output shows it, while output to files rarely does, and gdb never does. Without `--runs`, it's a single run.

Other flags:
- `--device vulkan|wgpu|cuda|metal|flex`;
- `--tensors N` (default 400) tensors of 1 to `--max-mib M` (default 16) MiB each;
- `--sync`, to wait for the device before returning.
