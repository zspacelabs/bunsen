# Consuming errors

This chapter is for code that calls bunsen and gets an error back: an
application, a training script, a kit that uses another subsystem, a test.

## React to the kind

A consumer that is not a specialist reads one thing, the
[kind](bunsen::errors::BunsenErrorKind), and has a small, fixed set of
reactions to choose from:

- **A bug, `Illegal` or `Internal`.**
  [`is_bug`](bunsen::errors::BunsenErrorKind::is_bug) is true. Escalate it.
  A script panics. A long-running service logs the report and fails the one
  request or graph, not the process. `Illegal` means the calling code broke a
  documented rule. `Internal` means bunsen broke its own, and is worth an
  issue.
- **Retry, `Unavailable`.**
  [`is_retryable`](bunsen::errors::BunsenErrorKind::is_retryable) is true for
  this kind only. Back off and try again, within a budget. No other kind
  improves with time, so retrying any other kind only delays the same
  failure.
- **Fall back, `Lookup`.** A key did not resolve. Try the next source, or
  tell the person which keys exist. The
  [`LookupError`](bunsen::errors::LookupError) cause carries the candidates.
- **Fix the settings, `Policy`.** The request is misconfigured: a config
  breaks a rule, downloads are turned off, values do not match a baseline.
  Tell the person which setting to change.
- **Report the source, `InvalidResource`.** Something from outside the
  program is not what it should be: a corrupt download, a file in the wrong
  format, a checkpoint that is not the model it claims to be. Name the
  source. Retrying the same source will not help.
- **Take another path, `Unsupported`.** The request is valid, but this code
  or this build does not do it.
- **Tell the operator, `Sys`.** The host failed: a full disk, a thread that
  would not spawn.
- **`Other`.** Unclassified; treat it as a bug in whatever produced it.

Do not branch on message text. Messages are written for people, and are
free to change. Anything a program should branch on is either the kind or a
typed cause. Reading causes is the subject of the
[complex handlers](./handlers.md) chapter.

## Show it to a person

An error prints two ways:

- **`Display` (`{}`)** is one line: the frames of context, outermost first,
  then the message. Use it in log lines and status text.
- **The report** ([`report`](bunsen::errors::BunsenError::report), `{:#}`,
  or `Debug`) is the whole story: the kind, the message and where it was
  built, the details, each frame with its source location, the cause chain,
  and a backtrace for a bug. Print it wherever a person will debug the
  failure.

`main() -> BunsenResult<()>` prints the report on failure, because Rust
prints a returned error's `Debug`. So does
[`ok_or_panic`](bunsen::errors::WithOkOrPanic::ok_or_panic).

## `try_x` or `x`

Bunsen's fallible operations come in pairs
([the convention](bunsen::errors#convention-try_x-and-x)): `try_x` returns a
`BunsenResult`, and `x` panics with the report. Call `x` when the input is
yours and known to be good, or when nothing above you could recover anyway.
Call `try_x` when the input comes from outside the program, and when you are
a service that must outlive one bad request.

## Pass it up with context

A consumer is usually also a layer that the error passes through on its way
to someone else. Add what you know with
[`ResultContext`](bunsen::errors::ResultContext): `.context("loading the
corpus")`, or `.with_context(|| ..)` to build the text only on failure. This
costs nothing on success, and does not change the kind or the cause.

Two situations call for more than context, because you know something about
the *blame* that the code below you could not:

- You passed values you read from outside the program, such as a config
  file or a command line, and got an `Illegal` back. The rule that broke is
  the input's fault, so re-mark it with
  [`as_policy`](bunsen::errors::ResultContext::as_policy).
- You passed a name your own code chose, and got a `Lookup` back. That is a
  bug on your side, so re-mark it with
  [`as_illegal`](bunsen::errors::ResultContext::as_illegal).

The [re-marking rules](bunsen::errors#re-marking) keep a bunsen bug from
ever becoming the caller's fault.

## Testing what you got

A test that expects a failure checks the parts it is about, with
[`ErrorMatcher`](bunsen::errors::testing::ErrorMatcher) (feature
`testing`): the kind, and perhaps a phrase of the message or a typed cause.
A test that pins the whole message breaks every time a word changes.
`BunsenError` is not `PartialEq` for that reason.
