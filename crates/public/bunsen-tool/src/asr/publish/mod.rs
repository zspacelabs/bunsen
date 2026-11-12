//! `--publish`, and the publisher `transcribe` and `live` push into.
mod clock;
mod config;
mod event;
#[cfg(feature = "mqtt")]
mod mqtt;
mod publisher;
mod record;
#[cfg(feature = "webhook")]
mod webhook;
mod writer;

pub use clock::*;
use config::Opts;
pub use config::PublishArgs;
pub use event::*;
pub use publisher::Publisher;
use publisher::{
    Cx,
    Sink,
};

#[cfg(test)]
mod tests;
