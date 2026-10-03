//! burn [`DataLoader`](burn::data::dataloader::DataLoader)s over Parquet
//! shards.
//!
//! - [`chat`]: [`ChatDataLoader`](chat::ChatDataLoader), packed token blocks
//!   for language-model training.

pub mod chat;
