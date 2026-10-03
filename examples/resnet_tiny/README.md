# resnet_tiny example

This example trains a ResNet image classifier from scratch on the CINIC-10
dataset (a CIFAR-10-style 10-class image set). It scans an on-disk image folder,
streams batches through a bunsen-firehose data pipeline with on-the-fly image
decoding, resizing, tensor conversion, and randomized augmentation (horizontal
flip), and trains with the Burn `Learner`. It is the lighter-weight companion to
the `swin_tiny` example, sharing the same data-loading machinery.

## Bunsen features exercised

- `bunsen::kits::images::resnet` — the `ResNet` model; its `pretrained`
  module has the `RESNET_PREFABS` table of geometries, and
  `default_resnet_factory` for an optional pretrained checkpoint.
- `bunsen::burner::module` — `ToStructureConfig` / `ModuleInit` for module init
  and `DTypeMapper` for dtype mapping.
- `bunsen::data::pretrained::PretrainedCache` — the digest-pinned cache a
  pretrained checkpoint is fetched into.
- `bunsen-firehose` — the columnar batch data engine: `FirehoseTableSchema` /
  `ColumnSchema`, row readers/writers, path scanning, and the Burn
  `FirehoseExecutorBatcher` bridge.
- `bunsen-firehose-image` — image loading/augmentation operators
  (`ImageLoader`, `ResizeSpec`, `HorizontalFlipStage`, `WithProbStage`) and
  `ImageToTensorData` / `stack_tensor_data_column` Burn conversion helpers.

It demonstrates wiring a bunsen-firehose image pipeline into a Burn training
loop.

## Installing the Dataset

See:

* https://github.com/BayesWatch/cinic-10
* https://datashare.ed.ac.uk/handle/10283/3192

1. Download the dataset, and unpack it.
2. Set the environment variable `CINIC10_PATH` to the path of the unpacked dataset.

## Running the Example

Run the training:

```bash
cargo run --release -p resnet_tiny --features cuda -- \
  --training-root $CINIC10_PATH/train \
  --validation-root $CINIC10_PATH/valid 
```


