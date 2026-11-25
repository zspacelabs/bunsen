//! # Swin interop
//!
//! [`timm`]: what timm's Swin checkpoints share, V1's and V2's: their state
//! dicts' names onto the staged vision encoder's, and the loading. Each
//! version's kit reads its own `config.json`
//! ([`v2::interop`](super::v2::interop)).

pub mod timm;
