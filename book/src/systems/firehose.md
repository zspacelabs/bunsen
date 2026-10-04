# Firehose

`bunsen-firehose` is a column-oriented data pipeline for feeding burn
models, and `bunsen-firehose-image` is its set of image operators. This
chapter is orientation only; the crate docs are the reference:
[`bunsen_firehose`] and
[`bunsen_firehose_image`].

A firehose pipeline is declared, not hand-wired. A
[`FirehoseTableSchema`](bunsen_firehose::core::schema::FirehoseTableSchema)
lists typed columns, and a
[`BuildPlan`](bunsen_firehose::core::schema::BuildPlan) says how a derived
column is computed from others by a named operator, built from a
serializable config. Planning a column checks the operator's parameter
types against the schema, so a pipeline such as "load the image at this
path, resize it, augment it, turn it into tensor data" is checked when it
is put together, not when the first batch fails. An executor then runs each
batch of rows through the plans in dependency order.

Operators register themselves globally when their crate is linked, and
[`init_default_operator_environment`](bunsen_firehose::ops::init_default_operator_environment)
collects every registration. A crate publishes operators just by being a
dependency: that is how `bunsen-firehose-image` adds image
[loading](bunsen_firehose_image::loader),
seeded [augmentation](bunsen_firehose_image::augmentation) and
[conversion to tensor data](bunsen_firehose_image::burn_support). The bridge
to burn is
[`FirehoseExecutorBatcher`](bunsen_firehose::burn_support::batcher::FirehoseExecutorBatcher),
a burn `Batcher` that runs the executor over each batch and hands the
tensors to burn's data loader.

Firehose is independent of `bunsen`, so it works with any burn model. The
[`resnet_tiny`](https://github.com/zspacelabs/bunsen/tree/main/examples/resnet_tiny)
and [`swin_tiny`](https://github.com/zspacelabs/bunsen/tree/main/examples/swin_tiny)
examples train image classifiers through the same firehose pipeline, and
are the place to see it end to end.
