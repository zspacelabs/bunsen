# Producing errors

This chapter is for code that builds errors: any fallible function in bunsen,
or in a crate built on it that returns `BunsenResult`. What each constructor
and cause type does is in the
[`bunsen::errors`](bunsen::errors#building-an-error) docs. This chapter is
about the decisions those calls encode.

## Choose the kind by who has to change

The kind answers one question: who or what has to change for the operation
to succeed? The [`errors` docs](bunsen::errors#which-kind) ask it as an
ordered list:

1. If the **code** has to change, the kind is `Illegal`, or `Internal` when
   the code at fault is bunsen's own.
2. If the **request or its settings** have to change, the kind is `Policy`.
3. Otherwise the **world** disagrees: `Lookup` (a key did not resolve),
   `Unavailable` (not now), or `InvalidResource` (present, but wrong).

`Unsupported`, `Sys` and the `Other` bail-out cover what is left.

### Locality, and why blame is not final

The hard case is the boundary between `Illegal` and `Policy`, because the
same check can be either. A config whose fields disagree is a bug when a
program built it, and a bad setting when it was read from a file. The
function that checks the config cannot know which.

The rule is **locality**:

- a rule documented by the function that checks it (in its `# Errors` or
  `# Panics`) is `Illegal` when broken;
- a rule set elsewhere, which this function only enforces, is `Policy`.

So the checking function reports `Illegal`, because the rule is its own.
The code that read the config from outside the program is an *input
boundary*, and it re-marks what comes back with `as_policy`. A handful of
boundaries do this: loaders, command-line parsers, deserializers. The
alternative would be to make every check guess where its input came from.

Rules also move. A policy that becomes part of an API's documented contract
turns `Illegal`, and a refusal that turns out to mean "not yet" turns
`Unavailable`. Pick the kind that is true today. Re-marking exists because
the answer depends on the caller.

## One line, then the evidence

A message is one line, written for a person: what went wrong, naming the
thing it went wrong with. Evidence goes in the
[details](bunsen::errors::BunsenError::with_details): the values that
differ, the config that was rejected, the shards that are missing. Details
print only in the report.

This split keeps log lines readable and test expectations stable. A
`{:#?}` dump in a message breaks both.

## Put facts in causes

A fact that a program may want to read belongs in a typed cause, not only in
the text:

- A check that a value breaks is a
  [`ConstraintError`](bunsen::errors::ConstraintError): the owner, the field,
  and the [`Rule`](bunsen::errors::Rule).
- A name that is not in a table is a
  [`LookupError`](bunsen::errors::LookupError), with the names there are.
  The four "no such X; there are: ..." messages in the crate all come from
  it.
- Text that did not parse is a [`ParseError`](bunsen::errors::ParseError),
  and the site chooses the kind by where the text came from: outside data
  gives `InvalidResource`, a request gives `Policy`, and the program's own
  text gives `Illegal`.
- A foreign error stays the cause. Do not format it into the message. An
  `io::Error` goes through [`sys_at`](bunsen::errors::sys_at), which also
  names the path. There is deliberately no `From<io::Error>`, because a bare
  `?` could not say which file failed.

A subsystem with failures of its own that a caller may act on defines its
own cause type. The [complex handlers](./handlers.md) chapter covers that.

## Add context; do not re-wrap

A function that calls another and passes its error up adds a frame with
[`ResultContext`](bunsen::errors::ResultContext). It never rebuilds the
error with a longer message. Rebuilding loses the kind, unless every
variant is copied by hand, and loses the cause in any case. A specialist
further up then cannot find what it is looking for.

## Document it, and test it

A `try_x` function's `# Errors` section names each kind it returns and when,
and the cause type where a caller may want it.
[STYLE.md](https://github.com/zspacelabs/bunsen/blob/main/STYLE.md#errors-try_x-and-x)
sets the rule. Its `x` twin panics with the report.

Each kind a function documents gets a test that names it, with
[`ErrorMatcher`](bunsen::errors::testing::ErrorMatcher): the kind, plus the
one phrase or cause field the test is about.
