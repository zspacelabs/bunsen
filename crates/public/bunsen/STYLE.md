# `bunsen` test fixtures

Where the `bunsen` crate's test support lives. The crate's naming, placement
and kit-layout rules are invariants of the code, so they are rustdoc: the
**Design Rules** sections of the crate root, `burner::module`, `blocks`,
`ops` and `kits`. The repository's [style guide](../../../STYLE.md) covers
the rest.

## Test fixtures

* A fixture may stay local to the tests that use it.
* A fixture that establishes a general mechanism, independent of any one test, goes to a
  `testing` module:
  * `support::testing` or `errors::testing`, for the crate-wide ones;
  * a module's own `testing` submodule, for one module's fixtures
    (`blocks::transformers::embedding::testing`).
* A `testing` module is `pub mod testing`, gated `#[cfg(any(test, feature = "testing"))]`, and
  is not glob-re-exported: callers name its path.
