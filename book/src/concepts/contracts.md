# Tensor Contracts

Shape errors in tensor code are notoriously hard to diagnose. `bunsen::contracts`
provides a runtime contract system that lets you describe the shape of a tensor
the way you would describe it in a paper:

$$
x \in \RR^{B \times C \times H \times W}
$$

and then check, at module boundaries, that the runtime tensor actually matches.

> **TODO:** worked example showing `Contract::new(...)` against a real tensor,
> with the error message produced on mismatch.

See [`bunsen::contracts`](../components/contracts.md) for the API reference.
