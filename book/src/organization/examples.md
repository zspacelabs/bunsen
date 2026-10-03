# Examples

The `examples/` directory holds runnable programs. Each one exercises a kit
or a system end to end, and its README says which bunsen features it uses
and how to run it. They are not published, and they are where a change to a
public API is felt first, so CI builds all of them.

| Example | What it shows |
|---|---|
| [`whisper-cli`](https://github.com/zspacelabs/bunsen/tree/main/examples/whisper-cli) | Speech to text through the Whisper stream driver: a file, a file pushed in chunks as a live loop would, or the microphone. Its `models` subcommand is the operational view of [pretrained models](../systems/pretrained.md): list, fetch and inspect named checkpoints. |
| [`train-chat`](https://github.com/zspacelabs/bunsen/tree/main/examples/train-chat) | Trains a NanoChat GPT from scratch on streamed dataset [shards](../systems/shards.md), with nanochat's Muon + AdamW [parameter groups](../systems/param-groups.md). |
| [`resnet_finetune`](https://github.com/zspacelabs/bunsen/tree/main/examples/resnet_finetune) | Fine-tunes a pretrained ResNet for multi-label classification, with model surgery: a new head, a swapped activation, DropBlock and stochastic depth. |
| [`resnet_tiny`](https://github.com/zspacelabs/bunsen/tree/main/examples/resnet_tiny) | Trains a ResNet from scratch on CINIC-10 through a [firehose](../systems/firehose.md) pipeline, loading a checkpoint into a modified prefab. |
| [`swin_tiny`](https://github.com/zspacelabs/bunsen/tree/main/examples/swin_tiny) | Trains a Swin Transformer V2 on CINIC-10, sharing `resnet_tiny`'s data pipeline. |
| [`conway`](https://github.com/zspacelabs/bunsen/tree/main/examples/conway) | Conway's Game of Life in 2D and 3D, on any backend: an interactive view and a benchmark. |
| [`lbm2d_vis`](https://github.com/zspacelabs/bunsen/tree/main/examples/lbm2d_vis) | A D2Q9 lattice-Boltzmann fluid simulation, rendered live. |

## Choosing a backend

Every example selects its backend with a cargo feature (`wgpu`, `vulkan`,
`cuda`, `metal`, or `flex` for the CPU); see each README for the exact
command. [Features, backends and the network](./features.md) explains the
backend features and why some examples also need `fetch`.
