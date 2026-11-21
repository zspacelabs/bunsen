//! # Laya's interop: its checkpoints' ecosystem
//!
//! - [`laya`]: a Laya release's files, its head's `rl_agent_config.json` and
//!   its `model.safetensors`' names; its encoder's `config.json` is
//!   `ModernBERT`'s
//!   ([`modernbert::interop::transformers`](crate::kits::language::encoders::modernbert::interop::transformers)).

pub mod laya;
