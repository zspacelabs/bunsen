# Module Introspection

When you want to do something with a *subset* of a model's parameters
— group them for different optimizers, apply weight decay to some but
not others, audit which tensors of which shapes a model contains —
`burn` itself gives you very little to work with. A `Module` is a Rust
type; it tells the compiler about its sub-modules but doesn't expose a
queryable structure at runtime.

`bunsen::burner::module::reflection` fills that gap. It walks a
`Module` and produces an XML document mirroring its structure, then
hands you an XPath-based query API to select pieces of it. The result
is that "select every rank-2 weight tensor in this submodule" becomes
a one-line query instead of a hand-rolled visitor.

This chapter covers the user-facing surface. The full reference is at
<https://docs.rs/bunsen/latest/bunsen/burner/module/reflection/> and
the module rustdoc has a long, executable walkthrough that exercises
every method.

Enable with `features = ["reflection"]`.

## Building a tree

`XmlModuleTree::build(&module)` walks a `&impl Module<B>` and produces
the XML model:

```rust,ignore
use bunsen::burner::module::reflection::XmlModuleTree;

let module: Linear<B> = LinearConfig::new(2, 3).init(&device);
let mut mtree = XmlModuleTree::build(&module);
```

`mtree.to_xml(true)` dumps the underlying document pretty-printed,
which is the right first step when you're figuring out what to query
against. For a single `Linear` it looks like:

```xml
<XmlModuleTree version="...">
  <Structure>
    <Linear id="n:1" class="struct">
      <Param id="n:2" name="weight" param_id="..."
             class="tensor" kind="Float" dtype="..." shape="2 3" rank="2"/>
      <Param id="n:3" name="bias"   param_id="..."
             class="tensor" kind="Float" dtype="..." shape="3"   rank="1"/>
    </Linear>
  </Structure>
</XmlModuleTree>
```

A few things to notice:

- The query "context" is always `/XmlModuleTree/Structure`; that's the
  implicit prefix every query starts under.
- Each structural element has the **type name** as its tag
  (`Linear`, `Vec`, `Array`, `Tuple`, …) and a `class` attribute
  (`struct`, `builtin`, …) distinguishing user-defined modules from
  the container types `burn`'s derive uses.
- Each parameter is a `<Param>` leaf with `name`, `param_id`, `kind`,
  `dtype`, `shape`, and `rank` attributes.
- Container children (`Vec`, `Array`, `Tuple`) have no `@name`; they
  have positional indices instead (XPath indexes from `1`).

## Querying

`mtree.query()` returns an `XPathModuleQuery` you can chain methods
against:

- `.select(expr)` — append `"/expr"` to the current XPath.
- `.filter(expr)` — append `"[expr]"`.
- `.params()` — descend to all `Param` elements (shorthand for
  `descendant-or-self::Param`).

Terminators:

- `.to_param_ids()` — `Result<Vec<ParamId>, _>`.
- `.to_param_descs()` — `Result<Vec<TensorParamDesc>, _>`.
- `.to_fragments(pretty)` — `Result<Vec<String>, _>`, the matched
  nodes serialized as XML. Useful for debugging the query itself.
- `.expr()` — the XPath string accumulated so far.

`XmlModuleTree` also offers convenience shortcuts that wrap the
common cases:

- `mtree.param_ids()` — every `ParamId` in the tree.
- `mtree.param_descs()` — every `TensorParamDesc` in the tree.
- `mtree.select(expr)` — equivalent to `mtree.query().select(expr)`.
- `mtree.select_params(expr)` — `select(expr)` then `.params()`, which
  is the right starting point for almost every parameter-selection
  query.

## A small XPath crib

You don't need to learn all of XPath to use this. The pieces that
come up:

| Pattern | Meaning |
| --- | --- |
| `Linear` | Children of the current context named `Linear`. |
| `*` | All children, regardless of name. |
| `*//Linear` | All `Linear` descendants (anywhere below). |
| `*[@name='weight']` | Children whose `@name` attribute is `weight`. |
| `*[@rank=2]` | Children whose `@rank` attribute is `2`. |
| `*[2]` | The second child (1-indexed). |
| `descendant-or-self::Param` | Every `<Param>` at or below the current context (what `.params()` does for you). |

Predicates can be combined: `*[@name='weight'][@rank=2]` selects
2-D weights.

## Worked example

A `Linear` module wrapped in some container shapes:

```rust,ignore
let module = (
    LinearConfig::new(2, 3).init::<B>(&device),
    [LinearConfig::new(4, 5).init::<B>(&device)],
    vec![
        LinearConfig::new(6, 7).init::<B>(&device),
        LinearConfig::new(8, 9).init::<B>(&device),
    ],
);

let mut mtree = XmlModuleTree::build(&module);

// All Linear modules anywhere in the tree:
let linear_ids = mtree
    .select("*//Linear")
    .params()
    .to_param_ids()?;

// Just the 2-D weight tensors (skips bias, which is rank-1):
let weight_ids = mtree
    .query()
    .params()
    .filter("@rank=2")
    .to_param_ids()?;

// Everything under the third top-level child (the Vec) — by position:
let vec_param_ids = mtree
    .select("*/*[3]")
    .params()
    .to_param_ids()?;
```

## When to use it

Reflection is heavier than just calling fields on a module. Reach for
it when:

- you need a set of `ParamId`s defined by *structure* rather than by
  the variable names in your code (e.g., "every rank-2 weight under
  the transformer blocks"), or
- you're writing tooling that doesn't know the model's type up front
  and needs to walk it generically.

For the typical "give these layers a different learning rate" use
case, this machinery feeds directly into the
[`GroupOptimizerAdaptor{N}`](./composite-optimizers.md) family.
