# resnet-finetune example

This example fine-tunes a pretrained ImageNet ResNet for multi-label image
classification. It downloads pretrained weights and a sample fine-tune dataset,
rebuilds the classifier head, and trains with the Burn `Learner` (gradient
accumulation, label smoothing, a multi-label Hamming-score metric, and
early-stopping). It also showcases bunsen's model-surgery options: swapping the
network's activation (e.g. LeakyReLU / GELU), enabling DropBlock / stochastic
depth regularization, optionally freezing layers, and cautious weight decay.
Running with `--help` (or with no dataset present) will also list the available
pretrained ResNet variants.

The data loading in this example is based on the tracel-ai models repository:
https://github.com/tracel-ai/models/blob/main/resnet-burn/examples/finetune/examples/finetune.rs

## Bunsen features exercised

- `bunsen::kits::images::classification::resnet` — the `ResNet` model and the
  `ResNetContractConfig` policy config; its `pretrained` module has the
  prefab geometries (`ResNetContractConfig::resnet18()` and its siblings, listed by `Prefabs`) and `default_resnet_factory` for the
  pretrained checkpoints by name.
- `bunsen::burner::module` — `ModuleInit` for building/initializing the module
  and `DTypeMapper` for dtype remapping during load.
- `bunsen::data::resources::ResourceCache` — the digest-pinned cache the
  pretrained weights are fetched into.

It demonstrates pretrained-weight loading plus post-hoc module rewriting
(activation replacement, regularization injection, layer freezing) on a ResNet
backbone.

## Running the Example

This will download both model weights and a fine-tune dataset:

```bash
cargo run --release -p resnet_finetune --features cuda
```

## List Available Models

This lists the prefabs, each with the rows that state its geometry, by their
`provider:ref` as `--pretrained` takes it (`torchvision/resnet50` and bare
`resnet50` also resolve). A timm row (`timm/resnet50.a1_in1k`) takes its
geometry from its `config.json`, so it is not listed here:

```terminaloutput
Available pretrained models:
* "resnet18"
ResNetContractConfig { layers: [2, 2, 2, 2], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Single, stem_width: 64, conv_stem_width: 64, avg_down: false, attention: None, anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: None, normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
  - "catalog:torchvision/resnet18": TorchVision ResNet-18.
* "resnet26"
ResNetContractConfig { layers: [2, 2, 2, 2], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Single, stem_width: 64, conv_stem_width: 64, avg_down: false, attention: None, anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "resnet34"
ResNetContractConfig { layers: [3, 4, 6, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Single, stem_width: 64, conv_stem_width: 64, avg_down: false, attention: None, anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: None, normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
  - "catalog:torchvision/resnet34": TorchVision ResNet-34.
* "resnet50"
ResNetContractConfig { layers: [3, 4, 6, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Single, stem_width: 64, conv_stem_width: 64, avg_down: false, attention: None, anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
  - "catalog:torchvision/resnet50": TorchVision ResNet-50.
* "resnet50_gn"
ResNetContractConfig { layers: [3, 4, 6, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Single, stem_width: 64, conv_stem_width: 64, avg_down: false, attention: None, anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Group(GroupNormConfig { num_groups: 32, num_channels: 0, epsilon: 1e-5, affine: true }), activation: Relu }
* "resnet101"
ResNetContractConfig { layers: [3, 4, 23, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Single, stem_width: 64, conv_stem_width: 64, avg_down: false, attention: None, anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
  - "catalog:torchvision/resnet101": TorchVision ResNet-101.
* "resnet152"
ResNetContractConfig { layers: [3, 8, 36, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Single, stem_width: 64, conv_stem_width: 64, avg_down: false, attention: None, anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
  - "catalog:torchvision/resnet152": TorchVision ResNet-152.
* "resnet10t"
ResNetContractConfig { layers: [1, 1, 1, 1], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Tiered, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: None, anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: None, normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "resnet14t"
ResNetContractConfig { layers: [1, 1, 1, 1], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Tiered, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: None, anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "resnet18d"
ResNetContractConfig { layers: [2, 2, 2, 2], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Deep, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: None, anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: None, normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "resnet26d"
ResNetContractConfig { layers: [2, 2, 2, 2], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Deep, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: None, anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "resnet26t"
ResNetContractConfig { layers: [2, 2, 2, 2], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Tiered, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: None, anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "resnet34d"
ResNetContractConfig { layers: [3, 4, 6, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Deep, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: None, anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: None, normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "resnet50c"
ResNetContractConfig { layers: [3, 4, 6, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Deep, stem_width: 32, conv_stem_width: 64, avg_down: false, attention: None, anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "resnet50d"
ResNetContractConfig { layers: [3, 4, 6, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Deep, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: None, anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "resnet50s"
ResNetContractConfig { layers: [3, 4, 6, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Deep, stem_width: 64, conv_stem_width: 64, avg_down: false, attention: None, anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "resnet50t"
ResNetContractConfig { layers: [3, 4, 6, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Tiered, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: None, anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "resnet101c"
ResNetContractConfig { layers: [3, 4, 23, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Deep, stem_width: 32, conv_stem_width: 64, avg_down: false, attention: None, anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "resnet101d"
ResNetContractConfig { layers: [3, 4, 23, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Deep, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: None, anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "resnet101s"
ResNetContractConfig { layers: [3, 4, 23, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Deep, stem_width: 64, conv_stem_width: 64, avg_down: false, attention: None, anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "resnet152c"
ResNetContractConfig { layers: [3, 8, 36, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Deep, stem_width: 32, conv_stem_width: 64, avg_down: false, attention: None, anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "resnet152d"
ResNetContractConfig { layers: [3, 8, 36, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Deep, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: None, anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "resnet152s"
ResNetContractConfig { layers: [3, 8, 36, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Deep, stem_width: 64, conv_stem_width: 64, avg_down: false, attention: None, anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "resnet200d"
ResNetContractConfig { layers: [3, 24, 36, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Deep, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: None, anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "resnext50d_32x4d"
ResNetContractConfig { layers: [3, 4, 6, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Deep, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: None, anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 32, base_width: 4 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "resnext50_32x4d"
ResNetContractConfig { layers: [3, 4, 6, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Single, stem_width: 64, conv_stem_width: 64, avg_down: false, attention: None, anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 32, base_width: 4 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "resnext101_32x4d"
ResNetContractConfig { layers: [3, 4, 23, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Single, stem_width: 64, conv_stem_width: 64, avg_down: false, attention: None, anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 32, base_width: 4 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "resnext101_32x8d"
ResNetContractConfig { layers: [3, 4, 23, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Single, stem_width: 64, conv_stem_width: 64, avg_down: false, attention: None, anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 32, base_width: 8 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "resnext101_32x16d"
ResNetContractConfig { layers: [3, 4, 23, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Single, stem_width: 64, conv_stem_width: 64, avg_down: false, attention: None, anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 32, base_width: 16 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "resnext101_32x32d"
ResNetContractConfig { layers: [3, 4, 23, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Single, stem_width: 64, conv_stem_width: 64, avg_down: false, attention: None, anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 32, base_width: 32 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "resnext101_64x4d"
ResNetContractConfig { layers: [3, 4, 23, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Single, stem_width: 64, conv_stem_width: 64, avg_down: false, attention: None, anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 64, base_width: 4 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "wide_resnet50_2"
ResNetContractConfig { layers: [3, 4, 6, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Single, stem_width: 64, conv_stem_width: 64, avg_down: false, attention: None, anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 128 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "wide_resnet101_2"
ResNetContractConfig { layers: [3, 4, 23, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Single, stem_width: 64, conv_stem_width: 64, avg_down: false, attention: None, anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 128 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "ecaresnet26t"
ResNetContractConfig { layers: [2, 2, 2, 2], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Tiered, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: Some(Eca), anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "ecaresnet50d"
ResNetContractConfig { layers: [3, 4, 6, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Deep, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: Some(Eca), anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "ecaresnet50t"
ResNetContractConfig { layers: [3, 4, 6, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Tiered, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: Some(Eca), anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "ecaresnet200d"
ResNetContractConfig { layers: [3, 24, 36, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Deep, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: Some(Eca), anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "ecaresnetlight"
ResNetContractConfig { layers: [1, 1, 11, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Single, stem_width: 64, conv_stem_width: 64, avg_down: true, attention: Some(Eca), anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "ecaresnet101d"
ResNetContractConfig { layers: [3, 4, 23, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Deep, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: Some(Eca), anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "ecaresnet269d"
ResNetContractConfig { layers: [3, 30, 48, 8], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Deep, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: Some(Eca), anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "ecaresnext26t_32x4d"
ResNetContractConfig { layers: [2, 2, 2, 2], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Tiered, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: Some(Eca), anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 32, base_width: 4 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "ecaresnext50t_32x4d"
ResNetContractConfig { layers: [3, 4, 6, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Tiered, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: Some(Eca), anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 32, base_width: 4 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "seresnet18"
ResNetContractConfig { layers: [2, 2, 2, 2], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Single, stem_width: 64, conv_stem_width: 64, avg_down: false, attention: Some(Se { rd_ratio: 0.0625 }), anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: None, normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "seresnet34"
ResNetContractConfig { layers: [3, 4, 6, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Single, stem_width: 64, conv_stem_width: 64, avg_down: false, attention: Some(Se { rd_ratio: 0.0625 }), anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: None, normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "seresnet50"
ResNetContractConfig { layers: [3, 4, 6, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Single, stem_width: 64, conv_stem_width: 64, avg_down: false, attention: Some(Se { rd_ratio: 0.0625 }), anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "seresnet50t"
ResNetContractConfig { layers: [3, 4, 6, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Tiered, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: Some(Se { rd_ratio: 0.0625 }), anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "seresnet101"
ResNetContractConfig { layers: [3, 4, 23, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Single, stem_width: 64, conv_stem_width: 64, avg_down: false, attention: Some(Se { rd_ratio: 0.0625 }), anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "seresnet152"
ResNetContractConfig { layers: [3, 8, 36, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Single, stem_width: 64, conv_stem_width: 64, avg_down: false, attention: Some(Se { rd_ratio: 0.0625 }), anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "seresnet152d"
ResNetContractConfig { layers: [3, 8, 36, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Deep, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: Some(Se { rd_ratio: 0.0625 }), anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "seresnet200d"
ResNetContractConfig { layers: [3, 24, 36, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Deep, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: Some(Se { rd_ratio: 0.0625 }), anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "seresnet269d"
ResNetContractConfig { layers: [3, 30, 48, 8], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Deep, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: Some(Se { rd_ratio: 0.0625 }), anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "seresnext26d_32x4d"
ResNetContractConfig { layers: [2, 2, 2, 2], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Deep, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: Some(Se { rd_ratio: 0.0625 }), anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 32, base_width: 4 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "seresnext26t_32x4d"
ResNetContractConfig { layers: [2, 2, 2, 2], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Tiered, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: Some(Se { rd_ratio: 0.0625 }), anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 32, base_width: 4 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "seresnext50_32x4d"
ResNetContractConfig { layers: [3, 4, 6, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Single, stem_width: 64, conv_stem_width: 64, avg_down: false, attention: Some(Se { rd_ratio: 0.0625 }), anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 32, base_width: 4 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "seresnext101_32x4d"
ResNetContractConfig { layers: [3, 4, 23, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Single, stem_width: 64, conv_stem_width: 64, avg_down: false, attention: Some(Se { rd_ratio: 0.0625 }), anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 32, base_width: 4 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "seresnext101_32x8d"
ResNetContractConfig { layers: [3, 4, 23, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Single, stem_width: 64, conv_stem_width: 64, avg_down: false, attention: Some(Se { rd_ratio: 0.0625 }), anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 32, base_width: 8 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "seresnext101d_32x8d"
ResNetContractConfig { layers: [3, 4, 23, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Deep, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: Some(Se { rd_ratio: 0.0625 }), anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 32, base_width: 8 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "seresnext101_64x4d"
ResNetContractConfig { layers: [3, 4, 23, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Single, stem_width: 64, conv_stem_width: 64, avg_down: false, attention: Some(Se { rd_ratio: 0.0625 }), anti_alias: None, replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 64, base_width: 4 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "resnetblur18"
ResNetContractConfig { layers: [2, 2, 2, 2], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Single, stem_width: 64, conv_stem_width: 64, avg_down: false, attention: None, anti_alias: Some(Blur), replace_stem_pool: false, output_stride: 32, bottleneck_policy: None, normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "resnetblur50"
ResNetContractConfig { layers: [3, 4, 6, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Single, stem_width: 64, conv_stem_width: 64, avg_down: false, attention: None, anti_alias: Some(Blur), replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "resnetblur50d"
ResNetContractConfig { layers: [3, 4, 6, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Deep, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: None, anti_alias: Some(Blur), replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "resnetblur101d"
ResNetContractConfig { layers: [3, 4, 23, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Deep, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: None, anti_alias: Some(Blur), replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "resnetaa34d"
ResNetContractConfig { layers: [3, 4, 6, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Deep, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: None, anti_alias: Some(Avg), replace_stem_pool: false, output_stride: 32, bottleneck_policy: None, normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "resnetaa50"
ResNetContractConfig { layers: [3, 4, 6, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Single, stem_width: 64, conv_stem_width: 64, avg_down: false, attention: None, anti_alias: Some(Avg), replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "resnetaa50d"
ResNetContractConfig { layers: [3, 4, 6, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Deep, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: None, anti_alias: Some(Avg), replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "resnetaa101d"
ResNetContractConfig { layers: [3, 4, 23, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Deep, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: None, anti_alias: Some(Avg), replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "seresnetaa50d"
ResNetContractConfig { layers: [3, 4, 6, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Deep, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: Some(Se { rd_ratio: 0.0625 }), anti_alias: Some(Avg), replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "seresnextaa101d_32x8d"
ResNetContractConfig { layers: [3, 4, 23, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Deep, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: Some(Se { rd_ratio: 0.0625 }), anti_alias: Some(Avg), replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 32, base_width: 8 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "seresnextaa201d_32x8d"
ResNetContractConfig { layers: [3, 24, 36, 4], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Deep, stem_width: 64, conv_stem_width: 64, avg_down: true, attention: Some(Se { rd_ratio: 0.0625 }), anti_alias: Some(Avg), replace_stem_pool: false, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 32, base_width: 8 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "resnetrs50"
ResNetContractConfig { layers: [3, 4, 6, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Deep, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: Some(Se { rd_ratio: 0.25 }), anti_alias: None, replace_stem_pool: true, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "resnetrs101"
ResNetContractConfig { layers: [3, 4, 23, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Deep, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: Some(Se { rd_ratio: 0.25 }), anti_alias: None, replace_stem_pool: true, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "resnetrs152"
ResNetContractConfig { layers: [3, 8, 36, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Deep, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: Some(Se { rd_ratio: 0.25 }), anti_alias: None, replace_stem_pool: true, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "resnetrs200"
ResNetContractConfig { layers: [3, 24, 36, 3], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Deep, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: Some(Se { rd_ratio: 0.25 }), anti_alias: None, replace_stem_pool: true, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "resnetrs270"
ResNetContractConfig { layers: [4, 29, 53, 4], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Deep, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: Some(Se { rd_ratio: 0.25 }), anti_alias: None, replace_stem_pool: true, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "resnetrs350"
ResNetContractConfig { layers: [4, 36, 72, 4], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Deep, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: Some(Se { rd_ratio: 0.25 }), anti_alias: None, replace_stem_pool: true, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
* "resnetrs420"
ResNetContractConfig { layers: [4, 44, 87, 4], num_classes: 1000, in_channels: 3, channels: [64, 128, 256, 512], stem_type: Deep, stem_width: 32, conv_stem_width: 64, avg_down: true, attention: Some(Se { rd_ratio: 0.25 }), anti_alias: None, replace_stem_pool: true, output_stride: 32, bottleneck_policy: Some(BottleneckPolicyConfig { pinch_factor: 4, cardinality: 1, base_width: 64 }), normalization: Batch(BatchNormConfig { num_features: 0, epsilon: 1e-5, momentum: 0.1 }), activation: Relu }
```

## Various Options to Rewrite Models

```terminaloutput

$ cargo run --release -p resnet_finetune --features cuda -- --cautious-weight-decay
| Split | Metric                         | Min.     | Epoch    | Max.     | Epoch    |
|-------|--------------------------------|----------|----------|----------|----------|
| Train | CPU Memory                     | 16.340   | 1        | 19.275   | 59       |
| Train | CPU Usage                      | 1.752    | 26       | 3.208    | 59       |
| Train | Hamming Score @ Threshold(0.5) | 82.597   | 1        | 97.160   | 48       |
| Train | Learning Rate                  | 1.284e-8 | 58       | 4.999e-5 | 1        |
| Train | Loss                           | 0.147    | 50       | 0.642    | 1        |
| Valid | CPU Memory                     | 16.268   | 1        | 19.412   | 59       |
| Valid | CPU Usage                      | 1.682    | 7        | 3.643    | 58       |
| Valid | Hamming Score @ Threshold(0.5) | 84.843   | 1        | 95.235   | 38       |
| Valid | Loss                           | 0.146    | 52       | 0.624    | 1        |
Args {
    seed: 0,
    train_percentage: 70,
    artifact_dir: "/tmp/resnet_finetune",
    batch_size: 24,
    grads_accumulation: 8,
    smoothing: Some(
        0.1,
    ),
    num_workers: 4,
    num_epochs: 60,
    patience: 20,
    pretrained: "torchvision/resnet50",
    replace_activation: None,
    freeze_layers: false,
    drop_block_prob: 0.2,
    stochastic_depth_prob: 0.05,
    learning_rate: 5e-5,
    cautious_weight_decay: true,
    weight_decay: 0.02,
}

$ cargo run --release -p resnet_finetune --features cuda --
| Split | Metric                         | Min.     | Epoch    | Max.     | Epoch    |
|-------|--------------------------------|----------|----------|----------|----------|
| Train | CPU Memory                     | 16.192   | 1        | 18.592   | 59       |
| Train | CPU Usage                      | 1.704    | 57       | 2.672    | 47       |
| Train | Hamming Score @ Threshold(0.5) | 82.580   | 1        | 97.126   | 49       |
| Train | Learning Rate                  | 1.284e-8 | 58       | 4.999e-5 | 1        |
| Train | Loss                           | 0.147    | 50       | 0.643    | 1        |
| Valid | CPU Memory                     | 16.235   | 1        | 18.607   | 59       |
| Valid | CPU Usage                      | 1.704    | 1        | 3.640    | 41       |
| Valid | Hamming Score @ Threshold(0.5) | 84.902   | 1        | 95.196   | 53       |
| Valid | Loss                           | 0.148    | 55       | 0.621    | 1        |
Args {
    seed: 0,
    train_percentage: 70,
    artifact_dir: "/tmp/resnet_finetune",
    batch_size: 24,
    grads_accumulation: 8,
    smoothing: Some(
        0.1,
    ),
    num_workers: 4,
    num_epochs: 60,
    patience: 20,
    pretrained: "torchvision/resnet50",
    replace_activation: None,
    freeze_layers: false,
    drop_block_prob: 0.2,
    stochastic_depth_prob: 0.05,
    learning_rate: 5e-5,
    cautious_weight_decay: false,
    weight_decay: 0.02,
}

$ cargo run --release -p resnet_finetune --features cuda -- --replace-activation leaky-relu --cautious-weight-decay
| Split | Metric                         | Min.     | Epoch    | Max.     | Epoch    |
|-------|--------------------------------|----------|----------|----------|----------|
| Train | CPU Memory                     | 16.486   | 1        | 18.592   | 60       |
| Train | CPU Usage                      | 1.492    | 31       | 2.761    | 3        |
| Train | Hamming Score @ Threshold(0.5) | 82.647   | 1        | 97.168   | 55       |
| Train | Learning Rate                  | 1.284e-8 | 58       | 4.999e-5 | 1        |
| Train | Loss                           | 0.146    | 50       | 0.642    | 1        |
| Valid | CPU Memory                     | 16.585   | 1        | 18.600   | 60       |
| Valid | CPU Usage                      | 1.452    | 32       | 5.302    | 3        |
| Valid | Hamming Score @ Threshold(0.5) | 84.941   | 1        | 95.294   | 39       |
| Valid | Loss                           | 0.147    | 55       | 0.620    | 1        |

Training completed in 12m53s
Args {
    seed: 0,
    train_percentage: 70,
    artifact_dir: "/tmp/resnet_finetune",
    batch_size: 24,
    grads_accumulation: 8,
    smoothing: Some(
        0.1,
    ),
    num_workers: 4,
    num_epochs: 60,
    patience: 20,
    pretrained: "torchvision/resnet50",
    replace_activation: Some(
        LeakyRelu,
    ),
    freeze_layers: false,
    drop_block_prob: 0.2,
    stochastic_depth_prob: 0.05,
    learning_rate: 5e-5,
    cautious_weight_decay: true,
    weight_decay: 0.02,
}

$ cargo run --release -p resnet_finetune --features cuda -- --replace-activation gelu --cautious-weight-decay
| Split | Metric                         | Min.     | Epoch    | Max.     | Epoch    |
|-------|--------------------------------|----------|----------|----------|----------|
| Train | CPU Memory                     | 16.276   | 1        | 18.669   | 60       |
| Train | CPU Usage                      | 1.491    | 46       | 2.209    | 2        |
| Train | Hamming Score @ Threshold(0.5) | 83.244   | 1        | 95.790   | 58       |
| Train | Learning Rate                  | 1.284e-8 | 58       | 4.999e-5 | 1        |
| Train | Loss                           | 0.171    | 50       | 0.654    | 1        |
| Valid | CPU Memory                     | 16.409   | 1        | 18.673   | 60       |
| Valid | CPU Usage                      | 1.494    | 1        | 2.902    | 15       |
| Valid | Hamming Score @ Threshold(0.5) | 83.373   | 1        | 94.706   | 57       |
| Valid | Loss                           | 0.159    | 55       | 0.637    | 1        |

Training completed in 13m12s
Args {
    seed: 0,
    train_percentage: 70,
    artifact_dir: "/tmp/resnet_finetune",
    batch_size: 24,
    grads_accumulation: 8,
    smoothing: Some(
        0.1,
    ),
    num_workers: 4,
    num_epochs: 60,
    patience: 20,
    pretrained: "torchvision/resnet50",
    replace_activation: Some(
        Gelu,
    ),
    freeze_layers: false,
    drop_block_prob: 0.2,
    stochastic_depth_prob: 0.05,
    learning_rate: 5e-5,
    cautious_weight_decay: true,
    weight_decay: 0.02,
}
```