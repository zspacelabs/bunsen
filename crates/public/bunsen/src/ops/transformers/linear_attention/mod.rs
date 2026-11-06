//! # Linear-attention operations
//!
//! The gated `DeltaNet`'s token mixing, as functions and plain values: what a
//! hybrid model's linear-attention layers (Qwen3.5's, Qwen3-Next's) compute
//! between their projections, and the state a decode carries between steps.
//! Nothing here owns trainable weights.
//!
//! - [`causal_depthwise_conv1d`]: the short causal convolution over time, one
//!   kernel per channel, continuing from the inputs a previous call saw.
//! - [`GatedDeltaRule`]: the gated delta rule, `S ← exp(g) S + β k (v - exp(g)
//!   S^T k)^T`, read out as `S^T q`; [`recurrent`](GatedDeltaRule::recurrent)
//!   steps it one position at a time, from a carried state, and
//!   [`chunked`](GatedDeltaRule::chunked) a chunk at a time (a prefill's form),
//!   which agree.
//! - [`DeltaNetState`]: one linear-attention layer's carried state, the
//!   convolution's last inputs and the delta rule's matrix, made by
//!   [`DeltaNetStateConfig`].
//!
//! As a KV cache is, the state is *injected*: the caller makes it, passes it
//! in, and the model never owns it. Unlike a KV cache it is a summary of the
//! whole prefix, not a history: it cannot be rewound to an earlier position,
//! only [`reset`](DeltaNetState::reset) and run again.
//!
//! Prefill in parts and prefill in one pass agree: each op continues exactly
//! from the state the last call left ("chunking must not be observable").

mod causal_conv;
mod delta_rule;
mod state;

pub use causal_conv::*;
pub use delta_rule::*;
pub use state::*;
