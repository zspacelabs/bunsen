# Conway's Game of Life

This demonstrate real-time volumetric simulations, using "Conway's Game of Life".

Select `BACKEND` from:

* `cuda` - nvidia backend.
* `metal` - apple backend.
* `vulkan` - vulkan backend.
* `wgpu` - web-gpu backend.
* `flex` - cpu backend.

## Sub-Commands

### `conway visual`

This example demonstrates a real-time visualization of Conway's Game of Life, a
cellular automaton devised by mathematician John Conway. The simulation runs on
a background thread as a GPU-tensor automaton, while the main thread renders the
latest published frame in an OpenGL window (via `piston` / `opengl_graphics`).
The grid is periodically re-seeded with a small amount of noise to keep the
field lively, and most parameters (grid shape, density, noise, FPS, ticks/sec,
zoom, opacity) are configurable from the CLI.

```bash
$ cargo run --release -p conway --features BACKEND -- visual [FLAGS/ARGS]
```

## Running the Example w/Autotune

```bash
$ cargo run --release -p conway --features BACKEND,burn/autotune
```

### `conway benchmark`

A headless throughput benchmark for Conway's Game of Life implemented as a
GPU-tensor cellular automaton. It seeds a random grid, runs a fixed number of
steps (with a warmup fraction discounted from the timing), and reports the
sustained `steps/sec` for the selected backend. Both the 2D and 3D life rules
can be benchmarked, across grids of arbitrary size.

This is useful for comparing backend/device performance and for measuring the
cost of the tensor kernels behind the simulation kits.

```bash
$ cargo run --release -p conway --features BACKEND -- benchmark [FLAGS/ARGS]
```
