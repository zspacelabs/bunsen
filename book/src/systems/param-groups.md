# Parameter groups

Training recipes often step different parts of a model differently. bunsen
handles this in two pieces that compose. Reflection selects parameters by
where they sit in the model, and the group optimizers step each selection
with its own optimizer and learning rate. This chapter explains how the two
fit together and where they bite. The reference is
[`bunsen::burner::module::reflection`]
and [`bunsen::burner::optim`].

## Why

A burn optimizer steps every parameter of a module with one rule. Real
recipes want more than that:

- 2-D weight matrices train well with Muon, while embeddings, norms and
  biases are better served by AdamW;
- some groups want their own learning rates: the NanoChat recipe scales the
  `lm_head` and embedding rates separately;
- some groups want their own weight decay or betas.

Grouping needs a way to name "every rank-2 weight under the transformer
blocks". A burn module is a Rust type. It tells the compiler about its
sub-modules, but it has no structure you can query at run time, so without
help each grouping is a hand-written `ModuleVisitor`. Reflection builds that
structure once, as a document, and the groups are queries against it.

## The model as an XML document

[`XmlModuleTree`](bunsen::burner::module::reflection::XmlModuleTree) mirrors
a built module as an XML document, and you select from it with XPath.
**The element name is the module's type name, and the field name is the
`@name` attribute.** A field `gpt: NanoChatGpt<B>` is the element
`<NanoChatGpt name="gpt">`, and each parameter is a `<Param>` leaf with its
`rank`, `shape` and `dtype` as attributes. So "every rank-2 weight of a
`Linear` under the field `h`" is a short path:
`*[@name='h']//Linear/*[@name='weight'][@rank=2]`. A selection returns
burn `ParamId`s, which are what an optimizer group is made of.

[The document](bunsen::burner::module::reflection#the-document) covers the
rest of the mapping (enums, containers, empty modules), and the
[XPath crib](bunsen::burner::module::reflection#xpath-crib) covers the few
XPath forms you need.

## The pattern

1. Build an `XmlModuleTree` over the live module.
2. Select each group's `ParamId`s with a query, by structure.
3. Gather every parameter that no group claimed into a remnant group, and
   check that no group is empty.
4. Wrap each set in an [`OptimizerGroup`](bunsen::burner::optim::OptimizerGroup)
   with its optimizer (any burn optimizer, such as
   `AdamWConfig::new().build()`) and, if it needs them, a learning-rate rule
   ([`LrSelector`](bunsen::burner::optim::LrSelector)) and a gradient
   clipping.
5. Plan the module and the groups with
   [`GroupOptimizerPlan::try_new`](bunsen::burner::optim::GroupOptimizerPlan::try_new).
6. Build burn's `ModuleOptimizer` with `plan.optimizer()` and its
   `ModuleLrScheduler` with `plan.lr_scheduler(schedule)`, and hand both to
   a `Learner`, or step them in your own loop.

Groups of any number of optimizer types go in one `Vec`: burn's
`ModuleOptimizer` erases the type. The plan checks that the groups
partition the module's float parameters. It rejects a parameter that two
groups claim
([`DuplicateParamId`](bunsen::burner::optim::GroupOptimizerError::DuplicateParamId)),
so nothing is stepped twice, and a float parameter that no group claims
([`UnassignedParamIds`](bunsen::burner::optim::GroupOptimizerError::UnassignedParamIds),
listing each one's module path), so nothing is silently left out. To keep
parameters fixed on purpose, put them in a frozen group
([`OptimizerGroup::frozen`](bunsen::burner::optim::OptimizerGroup::frozen)),
whose optimizer never moves them. A group's learning rate is a function of
the scheduled rate, so one schedule drives every group and each group shapes
its own rate.

The module docs have a compiled example
([optim](bunsen::burner::optim#example)) and a full
[lifecycle](bunsen::burner::optim#lifecycle).
[`examples/train-chat`](https://github.com/zspacelabs/bunsen/blob/main/examples/train-chat/src/main.rs)
is the full-size case: `ParamGroups::select` carves a `NanoChatGpt` into
matrix, embedding, `lm_head` and remnant groups for Muon and three AdamW
groups, and a test checks that the groups partition the model.

## Pitfalls

**Element names are type names, not field names.** A path that names a field
as an element matches nothing. In train-chat the field `gpt` holds a
`NanoChatGpt`, so the element is `NanoChatGpt` and its queries start
`GptHost/NanoChatGpt/...`. Queries that started `GptHost/gpt/...` would all
come back empty, every parameter would fall into the remnant group, and Muon
would never step a single weight. Nothing would fail: training would run,
with AdamW stepping everything at the remnant group's settings. Select a field
with `*[@name='gpt']`, or its type with `NanoChatGpt`, never with `gpt`.

**An empty selection is not an error.** A query that matches nothing
returns an empty set, and the remnant group quietly absorbs the parameters
it was meant to claim, with the remnant's settings. Renaming a field or a
type elsewhere in the model can do this to a query that used to work.
Assert that every group is non-empty. When a query comes back empty, print
the document with
[`to_xml`](bunsen::burner::module::reflection::XmlModuleTree::to_xml) and
read the names off it.

**`[a][b]`, not `[a,b]`.** In XPath a comma builds a sequence, not a
conjunction. `*[@name='weight', @rank=2]` parses, so building the query
succeeds. It fails with a type error only when it is evaluated, and over an
empty selection it is never evaluated at all, so a wrong path in front of it
hides the mistake. Stack the predicates, `[@name='weight'][@rank=2]`, or use
`and` ([XPath crib](bunsen::burner::module::reflection#xpath-crib)).

**The plan holds paths, not ids.** Groups are selected by `ParamId`, but
the plan keys each group by the module paths of its parameters, such as
`gpt.h.0.attn.c_q.weight`. A model with the same structure is stepped the
same way, whatever its ids. A parameter at a path the plan doesn't hold, such
as a head added under a new field after the plan was built, goes to the
fallback optimizer. By default that fallback panics at `step`, naming the
parameter's rank and shape, rather than train part of the model in silence.
Plan the groups after surgery. An
[`UnknownParamPolicy`](bunsen::burner::optim::UnknownParamPolicy) of
`Freeze` accepts unclaimed parameters at `try_new` and leaves them unchanged
at `step`. To keep parameters fixed on purpose, use a frozen group instead.

**Resuming builds the same plan over a fresh model.** Ids persist across a
checkpoint: `load_record` gives each parameter the id saved in the record,
so a restored model has the saved run's ids, not the ones it was built with.
The optimizer record saves each parameter's state with its module path, and
loading it hands each state to the group that holds that path. So a resumed
run builds a fresh model and plans it the same way the first run did, then
loads the model record and the optimizer record; a `Learner` resuming from a
checkpoint does the loading for you. The scheduler's record is matched by
position, so build the resumed scheduler from a plan with the same groups,
in the same order, over the same schedule
([resuming](bunsen::burner::optim#resuming-from-a-checkpoint)).
