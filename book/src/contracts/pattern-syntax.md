# Pattern Syntax

A shape contract is written in a small DSL: a comma-separated list
of *dimension matchers*, each describing one dimension of the
expected shape.

## Dimension matchers

Each matcher is one of:

- **`_`** &mdash; any dimension size; requires the position to exist
  but does not constrain its size.
- **`...`** &mdash; an ellipsis matching any number of dimensions. At
  most one ellipsis may appear in a pattern.
- a **dimension expression** &mdash; an integer-valued expression
  over named parameters that must equal the dimension's size.

Dimension expressions are written in ordinary infix notation with
string-literal identifiers, and may be given an optional label:

```bnf
ShapeContract => <LabeledExpr> { ',' <LabeledExpr> }* ','?
LabeledExpr   => { Param '=' }? <Expr>
Expr          => <Term> { <AddOp> <Term> }
Term          => <Power> { <MulOp> <Power> }
Power         => <Factor> [ '^' <usize> ]
Factor        => <Param> | '(' <Expression> ')' | NegOp <Factor>
Param         => '"' <identifier> '"'
identifier    => { <alpha> | '_' } { <alphanumeric> | '_' }*
NegOp         => '+' | '-'
AddOp         => '+' | '-'
MulOp         => '*'
```

For example, a `(B, H, W, C)` image with windowed height and width
is:

```rust
# use bunsen::contracts::{ShapeContract, shape_contract};
static CONTRACT: ShapeContract = shape_contract![
    "batch",
    "height" = "h_wins" * "window_size",
    "width"  = "w_wins" * "window_size",
    "channels",
];
```

`"height"` here is a *label* on the expression
`"h_wins" * "window_size"`. The label is what appears in error
messages and what you can ask `unpack_shape` to return.

A few patterns worth knowing:

- `["batch", _, _, "c"]` &mdash; a rank-4 shape where you care about
  batch and channels but not spatial dims.
- `[..., "c"]` &mdash; any rank, with channels last.
- `["b", _, "h" * "w" * "c"]` &mdash; a flattened channel-spatial
  dimension that must equal `h * w * c`.
- `["a", "a"]` &mdash; a square shape: the same parameter `"a"`
  must satisfy both dimensions.

## What counts as a shape: `ShapeView`

The contract methods accept anything that implements
[`ShapeView`](https://docs.rs/bunsen/latest/bunsen/contracts/trait.ShapeView.html).
Out-of-the-box that includes:

- `&[usize]`, `&[usize; D]`
- `&[u32]`,   `&[u32; D]`
- `&[i32]`,   `&[i32; D]`
- `&Vec<usize>`, `&Vec<u32>`, `&Vec<i32>`

With `features = ["burner"]` enabled it also includes:

- `burner::prelude::Shape`, `&burner::prelude::Shape`
- `&burner::prelude::Tensor<_, _, _>`

The last one is the common case: you hand a `&Tensor` directly to
the macro and the contract reads its shape.
