# Examples

The `examples/` directory holds runnable programs. Each one exercises a kit
or a system end to end, and its README says which bunsen features it uses
and how to run it. They are not published, and they are where a change to a
public API is felt first, so CI builds all of them.

Speech to text is [`bunsen-tool asr`](https://github.com/zspacelabs/bunsen/tree/main/crates/public/bunsen-tool),
a published binary rather than an example: files or the microphone, through
the Parakeet, Qwen3-ASR and Whisper stream drivers, and who spoke when
through the Sortformer diarizer, with commands that fetch and inspect models
and manage the weights cache. It is built on the same
`bunsen-app` plumbing.

| Example | What it shows |
|---|---|
| [`train-chat`](https://github.com/zspacelabs/bunsen/tree/main/examples/train-chat) | Trains a NanoChat GPT from scratch on streamed dataset [shards](../systems/data/datasets.md), with nanochat's Muon + AdamW [parameter groups](../systems/param-groups.md). |
| [`resnet_finetune`](https://github.com/zspacelabs/bunsen/tree/main/examples/resnet_finetune) | Fine-tunes a pretrained ResNet for multi-label classification, with model surgery: a new head, a swapped activation, DropBlock and stochastic depth. |
| [`resnet_tiny`](https://github.com/zspacelabs/bunsen/tree/main/examples/resnet_tiny) | Trains a ResNet from scratch on CINIC-10 through a [firehose](../systems/firehose.md) pipeline, loading a checkpoint into a modified prefab. |
| [`swin_tiny`](https://github.com/zspacelabs/bunsen/tree/main/examples/swin_tiny) | Trains a Swin Transformer V2 on CINIC-10, sharing `resnet_tiny`'s data pipeline. |
| [`conway`](https://github.com/zspacelabs/bunsen/tree/main/examples/conway) | Conway's Game of Life in 2D and 3D, on any backend: an interactive view and a benchmark. |
| [`lbm2d_vis`](https://github.com/zspacelabs/bunsen/tree/main/examples/lbm2d_vis) | A D2Q9 lattice-Boltzmann fluid simulation, rendered live. |

## Choosing a device

Every example picks its device at run time with `--device` (`auto`, `cuda`,
`metal`, `vulkan`, `wgpu` or `flex`) and `--device-index`, through
`bunsen-app`'s `DeviceArgs`. A backend is offered only when the example was
built with its cargo feature (`cuda`, `metal`, `vulkan`, `wgpu`); `flex`, the
CPU, always is, and `auto`, the default, takes the first accelerator built
in. Each README has the exact command.

The device's default float dtype is `f32`. An example that simulates in
another dtype takes `--init-dtype` (`lbm2d_vis` simulates its flow in `f16`
by default, where the backend has it); the training examples train in `f32`.
The training examples also draw burn's terminal
dashboard (their default `tui` feature) when run in a terminal.
[Features, backends and the network](./features.md) explains bunsen's own
backend features and why some examples also need `fetch`.
