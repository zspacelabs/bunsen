# swin_tiny example

This example trains a Swin Transformer V2 Tiny model for image classification on
the CINIC-10 dataset. Like `resnet_tiny`, it scans an on-disk image folder and
streams batches through a bunsen-firehose pipeline (decode, resize, tensor
conversion, randomized horizontal-flip augmentation), then trains the
hierarchical windowed-attention transformer with the Burn `Learner`. It
additionally applies DropBlock regularization to the model.

## Bunsen features exercised

- `bunsen::kits::bimm::swin::v2` — the bimm `SwinTransformerV2` model with
  `SwinTransformerV2Config` / `LayerConfig`.
- `bunsen::blocks::images::drop::drop_block` — the `DropBlock2d` /
  `DropBlock2dConfig` structured-dropout regularizer.
- `bunsen::burner::module::ModuleInit` — module initialization.
- `bunsen::errors` — `BunsenResult` and the `WithOkOrPanic` error helpers.
- `bunsen-firehose` — the columnar batch data engine (`FirehoseTableSchema`,
  row readers/writers, path scanning, `FirehoseExecutorBatcher`).
- `bunsen-firehose-image` — image loading/augmentation operators
  (`ImageLoader`, `ResizeSpec`, `HorizontalFlipStage`, `WithProbStage`) and
  Burn tensor-conversion helpers.

It demonstrates assembling a transformer backbone plus regularization blocks fed
by a bunsen-firehose image pipeline.

## Installing the Dataset

See:

* https://github.com/BayesWatch/cinic-10
* https://datashare.ed.ac.uk/handle/10283/3192

1. Download the dataset, and unpack it.
2. Set the environment variable `CINIC10_PATH` to the path of the unpacked dataset.

## Running the Example

Run the training:

```bash
cargo run --release -p swin_tiny --features cuda -- \
  --training-root $CINIC10_PATH/train \
  --validation-root $CINIC10_PATH/val 
```


