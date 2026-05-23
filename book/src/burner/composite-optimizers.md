# Composite Optimizers

`burn`'s `Optimizer<M, B>` trait owns a single optimizer that touches
every parameter in a `Module`. For real training runs that isn't
always what you want:

- 2-D matrix parameters benefit from
  [`Muon`](https://kellerjordan.github.io/posts/muon/), while
  embeddings and scalars are better served by `AdamW`.
- Different parameter groups want different learning-rate schedules
  (the NanoChat recipe scales `lm_head` and embedding learning rates
  differently, and applies a `d_model` factor on top).
- You may want different weight-decay or `beta` settings per group.

`bunsen::burner::optim` provides the `GroupOptimizerAdaptor{N}`
family: a single `Optimizer<M, B>` that mounts `N` *kinds* of
optimizer, each with one or more parameter groups, dispatching each
parameter's gradient to the optimizer it belongs to. Pair it with
the [Module Introspection](./module-introspection.md) machinery and
you can carve a model into parameter groups with XPath queries.

Enable with `features = ["train"]`.

API: <https://docs.rs/bunsen/latest/bunsen/burner/optim/>

## The building blocks

### `OptimizerGroup<B, O>`

One *group* — a `HashSet<ParamId>` plus an optimizer of type `O`
plus an optional per-group `LrSelector` for learning-rate mapping:

```rust,ignore
use bunsen::burner::optim::OptimizerGroup;

let group = OptimizerGroup::from_adaptor(
    param_ids,                       // anything IntoIterator<Item = ParamId>
    &AdamWConfig::new()
        .with_weight_decay(0.01)
        .init::<B, MyModel<B>>(),
)
.with_fixed_lr(3e-4);                // or .with_lr_selector(closure)
```

`.with_lr_selector(closure)` takes any `FnMut(LearningRate, &HashMap<String, LearningRate>) -> LearningRate`,
so per-group warmup, decay, or scaling factors live inside the group
itself.

### `GroupOptimizerAdaptor{N}`

`GroupOptimizerAdaptor2`, `…3`, `…4`, … (defined up through 6) each
take `N` `Vec<OptimizerGroup<B, O_i>>` arguments — one vector per
*kind* of optimizer:

```rust,ignore
use bunsen::burner::optim::GroupOptimizerAdaptor2;

let optimizer = GroupOptimizerAdaptor2::new(
    /* groups of optimizer kind 1: */ vec![adamw_group_a, adamw_group_b],
    /* groups of optimizer kind 2: */ vec![muon_group],
)?;
```

The adaptor implements `Optimizer<M, B>`, so it slots into
`burn::train::Learner` exactly where a single optimizer would.

Constructor validation: each `ParamId` may appear in *at most one*
group across all kinds. A duplicate returns
`GroupOptimizerError::DuplicateParamId` with the conflicting
positions.

## The pattern

```text
1. Build an XmlModuleTree over the live module.
2. Use XPath to extract disjoint HashSet<ParamId>s for each group.
3. Wrap each set in an OptimizerGroup with its optimizer + LR selector.
4. Compose with GroupOptimizerAdaptorN::new(...).
5. Hand the result to Learner.
```

The disjointness check at step 4 is your guard that nothing is
double-counted or accidentally dropped.

## Worked example: the NanoChat recipe

The [`demos/chat/examples/train`](https://github.com/zspacelabs/bunsen/tree/main/demos/chat/examples/train)
example trains a `NanoChatGpt` with two optimizer kinds and four
groups — three driven by `AdamW`, one by `Muon`. Stripped to the
essentials:

```rust,ignore
use std::collections::HashSet;
use bunsen::{
    burner::{
        module::reflection::XmlModuleTree,
        optim::{GroupOptimizerAdaptor2, OptimizerGroup},
    },
    public::burn::{module::ParamId, optim::LearningRate},
};

let mut mtree = XmlModuleTree::build(&host);

// 1. Carve the model into disjoint parameter sets using XPath.

// 2-D weight matrices inside the transformer block sequence.
let matrix_params: HashSet<ParamId> = mtree
    .select_params("GptHost/GPT/*[@name='h']/Linear/*[@name='weight',@rank=2]")
    .to_param_ids()?
    .into_iter()
    .collect();

let embedding_params: HashSet<ParamId> = mtree
    .select_params("GptHost/GPT/*[@name='wte']")
    .to_param_ids()?
    .into_iter()
    .collect();

let lm_head_params: HashSet<ParamId> = mtree
    .select_params("GptHost/GPT/*[@name='lm_head']")
    .to_param_ids()?
    .into_iter()
    .collect();

// Everything left over (norms, biases, scalars, ...).
let remnant_params: HashSet<ParamId> = mtree
    .param_ids()?
    .into_iter()
    .collect::<HashSet<_>>()
    .difference(&matrix_params).cloned().collect::<HashSet<_>>()
    .difference(&embedding_params).cloned().collect::<HashSet<_>>()
    .difference(&lm_head_params).cloned().collect();

// 2. Build groups: AdamW with three flavours, Muon for matrix params.

let optimizer = GroupOptimizerAdaptor2::new(
    // Kind 1: AdamW, three groups with different LR scales + betas.
    vec![
        OptimizerGroup::from_adaptor(
            lm_head_params,
            &AdamWConfig::new()
                .with_beta_1(0.8).with_beta_2(0.96)
                .with_weight_decay(0.01)
                .init::<B, GptHost<B>>(),
        )
        .with_lr_selector(move |lr: f64, _| lr * lm_head_lr),

        OptimizerGroup::from_adaptor(
            embedding_params,
            &AdamWConfig::new()
                .with_beta_1(0.8).with_beta_2(0.995)
                .with_weight_decay(0.001)
                .init::<B, GptHost<B>>(),
        )
        .with_lr_selector(move |lr, _| lr * embedding_lr),

        OptimizerGroup::from_adaptor(
            remnant_params,
            &AdamWConfig::new()
                .with_beta_1(0.8).with_beta_2(0.96)
                .with_weight_decay(0.01)
                .init::<B, GptHost<B>>(),
        )
        .with_lr_selector(move |lr, _| lr * scalar_lr),
    ],
    // Kind 2: Muon for the 2-D matrices.
    vec![
        OptimizerGroup::from_adaptor(
            matrix_params,
            &MuonConfig::new()
                .with_weight_decay(Some(WeightDecayConfig { penalty }))
                .init::<B, GptHost<B>>(),
        )
        .with_lr_selector(move |lr, _| lr * matrix_lr),
    ],
)?;

// 3. Use exactly as a single Optimizer:
let result = training.launch(Learner::new(host, optimizer, warmup_scheduler));
```

What this buys you over a hand-rolled solution:

1. **Selection is declarative.** Each group's membership is an XPath
   expression. Renaming a field elsewhere can't silently drop
   parameters from a group — the XPath either still matches or it
   doesn't, and the `param_ids()` cross-check makes the gap obvious.
2. **Disjointness is verified.** `GroupOptimizerAdaptor::new` returns
   `Err(DuplicateParamId)` if two groups claim the same parameter, so
   you can't accidentally optimize a tensor twice.
3. **Per-group LR is a closure**, not a separate scheduler tree. The
   global learning rate flowing in from `burn::train::Learner` is
   handed to every group's `LrSelector`, and the group decides how to
   shape it.

## Selecting the right `N`

`GroupOptimizerAdaptor2` if you need two kinds of optimizer
(AdamW + Muon, AdamW + SGD-with-momentum, …). The family runs up
through `GroupOptimizerAdaptor6` — pick the smallest `N` that fits
your kinds. Adding more groups *of the same kind* doesn't increase
`N`; that just extends the `Vec` for that kind.
