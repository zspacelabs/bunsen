# bunsen-catalog

bunsen's built-in pretrained catalog: the layers of the build catalog bunsen
imports (`bunsen::data::catalog::builtin_stack`), as
`catalog_builder` builds them from their probe-contract sources in the
repository's `validation/` workspace.

`catalog/MANIFEST.txt` lists the layers in import order, each with the bunsen
feature it needs; `files()` hands them out in that order. The directory is
generated, by `cargo catalog update`, and checked against its sources by
`cargo catalog check`; it isn't edited by hand.
