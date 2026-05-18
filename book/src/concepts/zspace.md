# Z-Space Indexing

The *z-space* model is `bunsen`'s coordinate system for talking about index
positions inside batched, sharded, or otherwise structured tensors without
falling back to raw `usize` arithmetic.

> **TODO:** explain the z-space coordinate model and link it to the
> `bunsen::zspace` types.

See [`bunsen::zspace`](../components/zspace.md) for the API reference.
