# Lattice Boltzmann 2D Demo

Select `BACKEND` from:

* `wpgu` - web-gpu backend.
* `cuda` - nvidia backend.
* `metal` - apple backend.
* `flex` - cpu backend.

```bash
$ cargo run --release -p lbm2d_vis --features BACKEND
```