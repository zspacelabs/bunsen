# Conway's Game of Life

This example demonstrates a real-time visualization of Conway's Game of Life, a
cellular automaton devised by mathematician John Conway. The simulation runs on
a background thread as a GPU-tensor automaton, while the main thread renders the
latest published frame in an OpenGL window (via `piston` / `opengl_graphics`).
The grid is periodically re-seeded with a small amount of noise to keep the
field lively, and most parameters (grid shape, density, noise, FPS, ticks/sec,
zoom, opacity) are configurable from the CLI.

## Bunsen features exercised

- `bunsen::kits::sims::conway::life2d` — the `ConwayLife2DConfig` /
  `ConwayLife2DState` 2D life simulator (`init`, `fuzz`, `step`).
- `bunsen::support::validators::parse_grid_shape` — a `clap` value-parser that
  accepts either `HEIGHT,WIDTH` or a single `SIZE`.
- `bunsen::zspace::ravel_dims` — flattens 2D grid coordinates into a linear
  index when reading back the published frame data.

It demonstrates running a backend-generic bunsen simulator on a worker thread
and streaming its `TensorData` out for display.

## Running the Example

Select `BACKEND` from:

* `wgpu` - web-gpu backend.
* `cuda` - nvidia backend.
* `metal` - apple backend.
* `flex` - cpu backend.

```bash
$ cargo run --release -p conway_vis --features BACKEND
```
