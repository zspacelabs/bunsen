# Simulations

This is a demo of various constant-volume-grid simulations using `burn` and `bunsen`.

# Demos

## lbm2d_vis - Fluid Flow Demo

```terminaloutput
Fluid Flow demo for Burn

Usage: lbm2d_vis [OPTIONS]

Options:
      --grid-shape <GRID_SHAPE>            The grid shape as `HEIGHT,WIDTH`, or `SIZE` [default: 800]
      --fps <FPS>                          The max frames per second [default: 60]
      --tps <TPS>                          The tics per second [default: 0]
      --zoom <ZOOM>                        The initial window zoom [default: 1.5]
      --opacity <OPACITY>                  The display opacity of updates [default: 0.8]
      --init-skip-steps <INIT_SKIP_STEPS>  The number of steps to skip on init [default: 1]
      --tau <TAU>                          The collision relaxation tau [default: 0.9]
  -h, --help                               Print help
```

You must build with one of the following features:

* `wgpu`
* `cuda`
* `metal`
* `flex`

Example:

```terminaloutput
$ cargo run --release -p lbm2d_vis --features wgpu
```

## conway_vis - Conway's Game of Life Demo

```terminaloutput
Conway's Game of Life demo for Burn

Usage: conway_vis [OPTIONS]

Options:
      --grid-shape <GRID_SHAPE>
          The grid shape as `HEIGHT,WIDTH`, or `SIZE` [default: 800]
      --init-skip-steps <INIT_SKIP_STEPS>
          The number of steps to skip on init [default: 10]
      --initial-density <INITIAL_DENSITY>
          The initial density of the grid [default: 0.1]
      --update-noise <UPDATE_NOISE>
          The noise to apply to the grid on each step [default: 0.0001]
      --fps <FPS>
          The frames per second [default: 60]
      --tps <TPS>
          The tics per second [default: 500]
      --zoom <ZOOM>
          The initial window zoom [default: 1.5]
      --opacity <OPACITY>
          The opacity between frames [default: 0.8]
  -h, --help
          Print help
```

You must build with one of the following features:

* `wgpu`
* `cuda`
* `metal`
* `flex`

Example:

```terminaloutput
$ cargo run --release -p conway_vis --features wgpu
```

## conway_benchmark - Conway's Game of Life Benchmark Demo

```terminaloutput
Conway's Game of Life demo for Burn

Usage: conway_benchmark [OPTIONS]

Options:
      --steps <STEPS>                      The number of steps to run [default: 1000]
      --dims <DIMS>                        The number of dimensions [default: 2]
      --grid-size <GRID_SIZE>              The width and height of the grid [default: 100]
      --warmup-fraction <WARMUP_FRACTION>  The fraction of steps to use for warmup [default: 10]
  -p, --progress                           Show progress bar
  -h, --help 
```

You must build with one of the following features:

* `wgpu`
* `cuda`
* `metal`
* `flex`

Example:

```terminaloutput
$ cargo run --release -p conway_benchmark --features cuda
Args {
    steps: 1000,
    dims: 2,
    grid_size: 100,
    warmup_fraction: 10,
    progress: false,
}
2170.06 steps/sec
```

