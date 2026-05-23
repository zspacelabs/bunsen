# `bunsen::kits::bimm`

*Bunsen/Burn Image Models* &mdash; an incremental port of the
[`timm`](https://github.com/huggingface/pytorch-image-models) (Torch
Image Models) ecosystem to `burn`.

`bimm` is the home for full image-recognition model families inside
`bunsen`. Each model lives next to its prefab configurations and
pretrained-weight loaders, so picking one up is closer to "pick a
prefab, fetch weights, init" than "wire together a stack of blocks."

API: <https://docs.rs/bunsen/latest/bunsen/kits/bimm/>

## Current models

### `ResNet`

The classic `ResNet` family
([arXiv:1512.03385](https://arxiv.org/abs/1512.03385)), with prefab
configurations and a pretrained-weight loader that pulls from the
`torchvision` checkpoints.

A representative usage:

```rust,no_run
use bunsen::{
    cache::DiskCacheConfig,
    kits::bimm::resnet::{PREFAB_RESNET_MAP, ResNet},
};
use burn::backend::Flex;

let device = Default::default();

let prefab = PREFAB_RESNET_MAP.expect_lookup_prefab("resnet18");

let weights = prefab
    .expect_lookup_pretrained_weights("tv_in1k")
    .fetch_weights(&DiskCacheConfig::default())
    .expect("Failed to fetch weights");

let model: ResNet<Flex> = prefab
    .to_config()
    .to_structure()
    .init(&device)
    .load_pytorch_weights(weights)
    .expect("Failed to load weights")
    // Re-head to 10 classes:
    .with_classes(10)
    // Stochastic block drops for training:
    .with_stochastic_drop_block(0.2)
    // Stochastic depth for training:
    .with_stochastic_path_depth(0.1);
```

### Swin Transformer V2

The Swin Transformer V2 family ([reference
implementation](https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer_v2.py)),
implemented in terms of windowed self-attention blocks, patch merging,
and relative-position biases.

See the [`bunsen::kits::bimm::swin::v2`](https://docs.rs/bunsen/latest/bunsen/kits/bimm/swin/v2/index.html)
module for the full configuration API.
