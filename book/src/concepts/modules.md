# Modules and Lifecycle

`burn::module::Module` is the unit of composition in the `burn` framework.
`bunsen` extends the lifecycle around modules with `bunsen::burner` so that
construction, parameter loading, and finalization can be expressed declaratively.

> **TODO:** describe the `burner` lifecycle stages and show a `Module` that
> opts into them.

See [`bunsen::burner`](../components/burner.md) for the API reference.
