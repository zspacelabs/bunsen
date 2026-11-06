//! Speech to text, and who spoke when, as `bunsen-tool asr` runs them.
//!
//! - [`AsrConstruct`] and [`AsrConstructConfig`]: presets, a model chosen by
//!   what it is for (the task, the speech, the latency) rather than by which
//!   model does it, resolved from a table (`asr_construct.yml`) to a family's
//!   checkpoint.
//! - [`Family`] and [`Kit`]: the model families and the kits, and loading a
//!   model with its info card. Each family's module ([`parakeet`],
//!   [`qwen3_asr`], [`voxtral`], [`whisper`]) builds its [`Driver`] and
//!   transcribes through it; [`stream()`] streams a recording through a driver,
//!   with voice activity and speakers when asked (and with `live`, a `Stream`
//!   is one live stream's context).
//! - [`Transcript`]: a transcript as the outputs carry it, its words, segments
//!   and speaker turns.
//! - [`publish`]: where a transcript goes (stdout, a file, a webhook, an MQTT
//!   broker), and in which form (text, JSON lines, RTTM).
//! - With `live`: the microphone, captured at a model's rate (`Capture`), and
//!   resampled when the device's rate differs (`Resampler`).
//! - [`cli`]: the `asr` command tree.

#[cfg(feature = "live")]
mod capture;
pub mod cli;
mod construct;
mod driver;
mod family;
pub mod parakeet;
pub mod publish;
pub mod qwen3_asr;
#[cfg(feature = "live")]
mod resample;
mod stream;
mod transcript;
pub mod voxtral;
pub mod whisper;

#[cfg(feature = "live")]
pub use capture::*;
pub use construct::*;
pub use driver::*;
pub use family::*;
#[cfg(feature = "live")]
pub use resample::*;
pub use stream::*;
pub use transcript::*;
