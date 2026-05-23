# `bunsen::kits::sims`

Iterative tensor simulations. These are runnable physics / cellular
kernels expressed as tensor operations: each step is a pure function
from one state tensor to the next, which makes them straightforward to
batch, GPU-accelerate, and feed back into ML pipelines.

API: <https://docs.rs/bunsen/latest/bunsen/kits/sims/>

## Current simulations

### `conway` &mdash; Conway's Game of Life

Cellular-automaton implementations expressed as windowed sums over a
boolean state tensor:

- `life2d` &mdash; classic 2D Conway's Game of Life over an
  $\text{H} \times \text{W}$ board.
- `life3d` &mdash; a 3D generalization over an
  $\text{H} \times \text{W} \times \text{Z}$ board, with a configurable
  spawn/survive ruleset (`LifeRules`).

Both expose an `next_interior_*` step kernel that takes the current
state and produces the next interior state, plus padding-aware wrappers
for the boundary.

### `lbm::d2q9` &mdash; Lattice-Boltzmann Fluid

A 2D, 9-velocity (D2Q9) lattice-Boltzmann fluid solver
([Wikipedia](https://en.wikipedia.org/wiki/Lattice_Boltzmann_methods)).
The simulation is split into the orthogonal operations that compose a
single LBM step:

- **streaming** &mdash; particle distributions propagate to neighbour
  cells along their velocity directions,
- **collision** &mdash; on-cell relaxation toward equilibrium,
- **reflection** &mdash; boundary handling,
- **thermal** and **relaxation** &mdash; configurable physics
  parameters,
- **simulation** &mdash; the driver that runs a sequence of steps.

Each piece is a tensor kernel; the whole thing runs on any `burn`
backend.
