# Bunsen Examples

This directory collects complex, runnable demos for `bunsen`. The goal is to
showcase the capabilities of the library — across simulators, vision models,
language-model training, data loading, and weight import — while also surfacing
a working edge of problems that further development should improve.

Each example has its own `README.md` with a full description and run
instructions. The overview below summarizes what each one does and which bunsen
features it exercises.

### Example: conway_benchmark

Headless throughput benchmark for Conway's Game of Life (2D and 3D), reporting
`steps/sec` for a chosen backend and grid size.

- **Bunsen coverage:** `kits::sims::conway::life2d` / `life3d` simulators on
  backend-generic Burn tensors.

### Example: conway_vis

Real-time OpenGL visualization of Conway's Game of Life, with the automaton
running on a worker thread and frames streamed to a `piston` window.

- **Bunsen coverage:** `kits::sims::conway::life2d`,
  `support::validators::parse_grid_shape`, `zspace::ravel_dims`.

### Example: lbm2d_vis

Real-time 2D Lattice Boltzmann (D2Q9) fluid simulation with an OpenGL flow-field
visualization and mass-conserving source/outflow forcing.

- **Bunsen coverage:** `kits::sims::lbm::d2q9` (solver, `LbmTables`,
  `macroscopic_momentum`), `burner::tensor::TensorDataIndexView`,
  `support::validators::parse_grid_shape`.

### Example: resnet_finetune

Fine-tunes a pretrained ImageNet ResNet for multi-label classification, with
model surgery (activation swap, DropBlock/stochastic depth, layer freezing,
cautious weight decay).

- **Bunsen coverage:** `kits::bimm::resnet` (model, `PREFAB_RESNET_MAP`,
  `ResNetContractConfig`), `burner::module` (`ModuleInit`, `DTypeMapper`),
  `data::cache::BunsenDiskCache`.

### Example: resnet_tiny

Trains a ResNet from scratch on CINIC-10 using a bunsen-firehose image pipeline.

- **Bunsen coverage:** `kits::bimm::resnet`, `burner::module`,
  `data::cache::BunsenDiskCache`, plus `bunsen-firehose` /
  `bunsen-firehose-image` data loading and augmentation.

### Example: swin_tiny

Trains a Swin Transformer V2 Tiny on CINIC-10 with DropBlock regularization,
sharing the firehose image pipeline with `resnet_tiny`.

- **Bunsen coverage:** `kits::bimm::swin::v2`,
  `blocks::images::drop::drop_block::DropBlock2d`, `burner::module::ModuleInit`,
  `errors`, plus `bunsen-firehose` / `bunsen-firehose-image`.

### Example: train-chat

Trains a NanoChat-style GPT on the fineweb-edu corpus, using per-group
optimizers (Muon for matrices, AdamW for embeddings/head/scalars) selected via
module-tree reflection.

- **Bunsen coverage:** `kits::gpts::nanochat`,
  `burner::module::reflection::XmlModuleTree`, `burner::optim`
  (`GroupOptimizerAdaptor2`, `OptimizerGroup`),
  `bunsen-preview-chat-dataloader`, `zsl-data-cache`.

### Example: whisper-dev

Development utility that imports an OpenAI Whisper model from a PyTorch
checkpoint and prints the inferred config.

- **Bunsen coverage:** `kits::speech::whisper::pretrained::PytorchWhisperScanner`.

### Example: zsl-data-cache

Reusable on-disk shard download/cache for the nanochat (fineweb) dataset,
consumed by `train-chat`; includes a `pull_shards` CLI.

- **Bunsen coverage:** support library for bunsen training examples
  (`DatasetCacheConfig`, `DatasetSource`), built on Burn `Config` and
  `parquet`/`arrow`.
