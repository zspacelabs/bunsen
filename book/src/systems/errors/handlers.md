# Complex handlers

Most code handles an error by its kind alone
([Consuming errors](./consumers.md)). Some code has a recovery strategy of
its own, and needs facts that the kind does not carry. The disk cache, for
example, decides between retrying a URL and moving to the next mirror. This
chapter is about that code.

## Kind and cause together

A specialist reads two things:

- the **kind**, which it shares with every other consumer;
- a **cause type** it knows, which it reaches with
  [`find`](bunsen::errors::BunsenError::find). `find` walks the cause chain
  and returns the first error of the requested type.

The kind keeps the ontology small. Generic consumers never see a
subsystem's cause types, so a subsystem can define them freely without
growing the set of kinds everyone has to handle. The cause gives the
specialist a precise vocabulary that only it reads.

## The cache's mirror loop

The cache fetches a file from one or more URLs. Each failure is a
`BunsenError` whose cause is the cache's own
[`FetchFailure`](bunsen::data::cache::FetchFailure), and the loop reads kind
and cause together:

- An `Unavailable` failure, such as a timeout, a server error or a
  truncated transfer, retries the same URL, within the cache's retry budget,
  and waits as long as a `Retry-After` header asked.
- A `Lookup` failure means the URL is missing or forbidden. The loop moves to
  the next mirror.
- A download whose digest does not match the pin is `InvalidResource` to
  everyone else. To the cache, its cause says "this mirror served the wrong
  bytes", and the loop moves to the next mirror instead of retrying the same
  one.
- Anything else, such as `Policy` or `Sys`, stops the loop. No other mirror
  would help.

If every mirror fails, the loop returns one error that holds each mirror's
error, labelled by URL. Outside the cache, a consumer sees only the kind of
that aggregate: it is `Unavailable` only if every mirror was unavailable, so
a generic retry happens only when a retry could help.

## Defining a cause type

A subsystem that needs this kind of handling defines a cause type next to
the code that uses it:

- **It owns its kind mapping.** It provides `From<Cause> for BunsenError`,
  which picks the kind from the variant in one place. A site cannot then
  build an error whose kind and cause disagree, and `?` works.
- **It prints once.** The `From` impl builds the error with
  [`from_cause`](bunsen::errors::BunsenError::from_cause), so the cause's
  text becomes the message and is not repeated in the report. A cause with
  more to say than one line implements
  [`Detailed`](bunsen::errors::Detailed).
- **It is public, `#[non_exhaustive]`, and documented** in the `# Errors`
  sections of the functions that return it, so callers can find it.

The shared cause types in [`bunsen::errors`], such as
[`LookupError`](bunsen::errors::LookupError) and
[`DigestMismatch`](bunsen::errors::DigestMismatch), follow the same rules.

## Aggregates

A batch reports its failures as one error with a
[`Multiple`](bunsen::errors::Multiple) cause, whose members are the
`BunsenError`s themselves, each labelled. The aggregate's kind is derived
from the members. A specialist reads each member's kind and cause, for
example to retry only the parts of a batch that can succeed.

## Keep the chain intact

`find` sees only what is still a typed error. Text loses it:

- formatting an error into a new message, instead of adding a frame;
- joining a list of errors into one string, instead of using `Multiple`;
- sending an error across a thread as a `String`. `BunsenError` is `Clone`,
  shared behind an `Arc`, so send the error itself.

One boundary cannot be helped. A serde (de)serializer accepts only a message
for its own error type, so an error that crosses serde arrives as text.

`find` matches type identity: a cause type from a second,
semver-incompatible copy of bunsen in the same build is a different type,
and is not found.

## Run containers

A long-running service, or one that loads and runs graphs dynamically, calls
the `try_x` half of bunsen's operations, and treats `Illegal` and `Internal`
as the failure of one job, not the process: it logs the report and moves on.
To contain the panics of code that does panic, it wraps the job in a
`catch_unwind` boundary and turns the panic into an `Illegal` error, with the
panic text as its message. Re-marking applies at that boundary like any
other. The container knows whether a job's input came from outside, and
re-marks accordingly.
