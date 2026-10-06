# Errors

Every fallible bunsen function returns a
[`BunsenResult`](bunsen::errors::BunsenResult), and every failure is a
[`BunsenError`](bunsen::errors::BunsenError). This section explains how the
error is shaped and why, from three points of view, one per chapter:

- [Consuming errors](./errors/consumers.md): code that calls bunsen and
  receives errors. It needs to know what to do with one, and how to show one
  to a person.
- [Producing errors](./errors/producers.md): code that builds errors. It
  needs to know which kind to pick, what to put in the message, and what
  belongs in a cause.
- [Complex handlers](./errors/handlers.md): subsystems with a recovery
  strategy of their own, such as the cache that tries another mirror. They
  need typed facts that generic code never reads.

The reference for all of it, with compiled examples, is the
[`bunsen::errors`](bunsen::errors) module docs.

## The shape of an error

An error has five parts:

- a **kind**, a [`BunsenErrorKind`](bunsen::errors::BunsenErrorKind), which
  says what a generic consumer should do;
- a one-line **message**, which says what went wrong;
- optional multi-line **details**, the evidence: a diff, a dump, a list;
- the **frames** of context it gathered on its way up, each saying what a
  layer was doing, with that layer's source location;
- an optional **cause**: a typed fact, such as which key a lookup missed, or
  the foreign error underneath.

The kind is the only part generic code branches on. Kinds are reaction
classes, not descriptions of a failure. There are few of them on purpose: a
generic consumer must be able to handle each one without knowing which
subsystem raised it. What went wrong in detail is the job of the message and
the cause.

## Why kinds, and not a variant per failure

A variant per failure grows with every subsystem. It forces each consumer to
know every subsystem's vocabulary, and turns every new failure into a
breaking change. It also invites a catch-all variant that ends up holding
unrelated failures, which callers then tell apart by reading message text.

Bunsen's errors separate the two questions. *What should I do about it?* is
the kind, which is small and stable. *What exactly happened?* is the cause,
which a subsystem defines for itself and only the code that wants it reads.
The [complex handlers](./errors/handlers.md) chapter shows the second
question at work.

## Why context is separate from the message

An error passes through several layers on its way to a person, and each
layer knows something the layer below it does not: the model being loaded,
the shard set, the block index. Adding that knowledge by formatting the
error into a new message loses the original kind and cause, and the result
cannot be matched or retried correctly. Frames add the knowledge without
touching the error itself. Together they make the report a *logical
traceback*: the layers that chose to say something, with their source
locations, in release builds, and across threads.
