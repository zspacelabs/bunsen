# Conway's Game of Life

This example demonstrates a visualization of Conway's Game of Life, a cellular automaton devised by mathematician John
Conway. The simulation is rendered using the `conway` crate and visualized with `wgpu`.

Select `BACKEND` from:

* `wpgu` - web-gpu backend.
* `cuda` - nvidia backend.
* `metal` - apple backend.
* `flex` - cpu backend.

```bash
$ cargo run --release -p conwawy_vis --features BACKEND
```