# Bunsen examples

Runnable programs, one or more per kit or system. They are not published, and CI builds all of them. The book's
[Examples](https://zspacelabs.ai/bunsen/book/organization/examples.html) page says what each one shows and how to
choose a backend; each example's own README says how to run it.

- [`conway`](conway): Conway's Game of Life in 2D and 3D, with `visual` and `benchmark` subcommands.
  Uses `kits::sims::conway`, `support::geometry::GridShape2D` and `zspace::ravel_dims`.
- [`lbm2d_vis`](lbm2d_vis): a D2Q9 lattice-Boltzmann fluid simulation, rendered live.
  Uses `kits::sims::lbm::d2q9`, `burner::tensor::TensorDataView` and `support::geometry::GridShape2D`.
- [`resnet_finetune`](resnet_finetune): fine-tunes a pretrained ResNet for multi-label classification, with model
  surgery. Uses `kits::images::resnet::pretrained` (`default_resnet_factory`, `RESNET_PREFABS`),
  `data::pretrained::PretrainedCache` and `burner::module` (`ModuleInit`, `DTypeMapper`).
- [`resnet_tiny`](resnet_tiny): trains a ResNet on CINIC-10 through a firehose image pipeline, optionally from a
  pretrained checkpoint. Uses `kits::images::resnet`, `data::pretrained::PretrainedCache`, `burner::module`
  (`ToStructureConfig`, `ModuleInit`, `DTypeMapper`), `bunsen-firehose` and `bunsen-firehose-image`.
- [`swin_tiny`](swin_tiny): trains a Swin Transformer V2 on CINIC-10 with DropBlock, sharing `resnet_tiny`'s
  pipeline. Uses `kits::images::swin::v2`, `blocks::images::drop::drop_block`, `bunsen-firehose` and
  `bunsen-firehose-image`.
- [`train-chat`](train-chat): trains a NanoChat GPT on fineweb-edu shards, with Muon and AdamW parameter groups
  selected by module-tree reflection. Uses `kits::gpts::nanochat` and its `datasets`, `data::shards`,
  `data::cache::BunsenDiskCache`, `burner::module::reflection::XmlModuleTree`, `burner::optim` and
  `bunsen-arrow-dataloaders`.
- [`whisper-cli`](whisper-cli): transcribes a file or the microphone through the Whisper stream driver, with the
  model named as `openai-whisper` names it, and a `models` subcommand that lists, fetches and inspects models.
  Uses `kits::speech::whisper` (`pretrained`, `driver`), `kits::speech::silero_vad` and `data::pretrained`.
