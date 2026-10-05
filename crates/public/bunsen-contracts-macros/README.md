# bunsen-contracts-macros

[![Crates.io Version](https://img.shields.io/crates/v/bunsen-contracts-macros)](https://crates.io/crates/bunsen-contracts-macros)
[![docs.rs](https://img.shields.io/docsrs/bunsen-contracts-macros)](https://docs.rs/bunsen-contracts-macros/latest/bunsen_contracts_macros/)

The `shape_contract!` proc-macro behind [`bunsen`](https://crates.io/crates/bunsen)'s runtime tensor-shape contracts.

Don't depend on this crate directly: use `bunsen::contracts::shape_contract!`, which wraps this macro and brings the
names its expansion uses into scope. `bunsen`'s [`contracts`](https://docs.rs/bunsen/latest/bunsen/contracts/index.html) module
documents the contracts, their pattern language, and the macros built on this one. This crate has no other public
API.

## License

Distributed under the terms of both the MIT license and the Apache License (Version 2.0).
