# Lattice Boltzmann 2D Demo

This example runs a 2D fluid simulation using the Lattice Boltzmann method on a
D2Q9 lattice and visualizes the resulting flow field in real time. The solver
steps on a background thread (streaming + BGK collision with a configurable
relaxation `tau`, solid-wall masks, and a periodic source/outflow forcing),
while the main thread renders the macroscopic momentum field as a colored grid
in an OpenGL window (`piston` / `opengl_graphics`). Mass is conserved by
tracking and re-injecting drift.

## Bunsen features exercised

- `bunsen::kits::sims::lbm::d2q9` — the D2Q9 solver: `LBMD2Q9Config` /
  `LBMD2Q9State` (`init`, `advance_step`, `solid_mask`, mass-conservation
  helpers), `RelaxationParam`, the `LbmTables` lattice constants, the
  `SPEED_OF_SOUND` constant, and `macroscopic_momentum` for deriving velocity
  from the distribution tensor.
- `bunsen::burner::tensor::TensorDataIndexView` — ergonomic multi-dimensional
  indexing into `TensorData` (`view`, `[&[y, x, c]]`) used by the renderer.
- `bunsen::support::validators::parse_grid_shape` — `clap` value-parser for
  `HEIGHT,WIDTH` or a single `SIZE`.

It demonstrates a more involved physics kit driven through Burn tensor slicing
(`slice_fill`, `slice_assign`) and dtype casting.

## Running the Example

Select `BACKEND` from:

* `wgpu` - web-gpu backend.
* `cuda` - nvidia backend.
* `metal` - apple backend.
* `flex` - cpu backend.

```bash
$ cargo run --release -p lbm2d_vis --features BACKEND
```
