# Bunsen Style Guide

Conventions for code and documentation in the `bunsen` crate. This file is the
source of truth; assertions added under each chapter are applied across the code
base.

> See also [`book/src/contributing/style.md`](book/src/contributing/style.md)
> for prose/Book conventions. This file governs in-source `rustdoc`.

## rustdoc

### Base Style

The base style for rustdoc is [rfc1574].

[rfc1574]: https://github.com/rust-lang/rfcs/blob/master/text/1574-more-api-documentation-conventions.md

### Coverage

* Every public interface requires rustdoc.
* Objects in a lifecycle ('Config'>'Module') require a short situational relationship description with cross-links.

Every public item needs rustdoc. This chapter defines the structure we expect,
the cross-links between paired types, and how tensor shapes are written.

### Tensor shape notation

A tensor shape is written as a **single** backtick code span wrapping the whole
shape in **square brackets**. Dimension expressions and inline annotations live
*inside* the brackets:

```text
`[batch, time, embed]`
`[batch, in_channels, height, width]`
`[batch, height/2 * width/2, 2 * channels]`
`[batch, out=planes*expansion, h, w]`
`[VY=3, VX=3, (VY, VX)=2]`
```

Do **not** use any of these forms for a tensor shape:

```text
``[batch, time, embed]``              (double backticks)
[`batch`, `time`, `embed`]            (per-element backticks)
(`b_nw`, `num_heads`, ws*ws)          (parentheses + per-element backticks)
(batch, time, embed)                  (parentheses)
```

Prefer **shape-first** phrasing — put the shape *before* the object it
describes (`SHAPE object`) rather than trailing it (`object of shape SHAPE`).
Articles (`a`/`an`/`the`) stay first; other modifiers follow the shape:

```text
prefer:  a `[batch, time, embed]` input tensor
         a `[2*h-1, 2*w-1, 2]` 3D tensor containing the offsets
not:     an input tensor of shape `[batch, time, embed]`
         a 3D tensor of shape `[2*h-1, 2*w-1, 2]` containing the offsets
```

These are **not** tensor shapes — leave them as written:

- Coordinate / value tuples and pairs: `(y, x)`, `(k, v)`,
  `(density, velocity)`, `(height, width)` tuples.
- Rust type code spans: `&[usize; D]`, `Param<Tensor<B, R, K>>`.
- Half-open ranges and indexing: `[start, end)`, `env[$VAR]`.
- Intra-doc link syntax: `[text](url)`.

<!-- Assertions go here. -->
