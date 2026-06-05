# conway_benchmark example

A headless throughput benchmark for Conway's Game of Life implemented as a
GPU-tensor cellular automaton. It seeds a random grid, runs a fixed number of
steps (with a warmup fraction discounted from the timing), and reports the
sustained `steps/sec` for the selected backend. Both the 2D and 3D life rules
can be benchmarked, across grids of arbitrary size.

This is useful for comparing backend/device performance and for measuring the
cost of the tensor kernels behind the simulation kits.

## Bunsen features exercised

- `bunsen::kits::sims::conway::life2d` — the `ConwayLife2DConfig` / `ConwayLife2DState`
  2D life simulator (`init`, `fuzz`, `step`), built entirely on Burn tensors.
- `bunsen::kits::sims::conway::life3d` — the `ConwayLife3DConfig` / `ConwayLife3DState`
  3D simulator with configurable `LifeRules`.

It demonstrates that the `kits::sims` simulators are backend-generic and run on
any Burn backend (CPU or GPU).

## Running the Example

Select `BACKEND` from:

* `wgpu` - web-gpu backend.
* `cuda` - nvidia backend.
* `metal` - apple backend.
* `flex` - cpu backend.

```bash
$ cargo run --release -p conway_benchmark --features BACKEND -- \
  --steps 1000 --dims 2 --grid-size 100 --progress
```
