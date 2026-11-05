//! # The activation seam
//!
//! [`BunsenActivationConfig`] and [`BunsenActivation`] are bunsen's activation
//! layers: the kinds bunsen's blocks and examples use, each burn's layer, and
//! the kinds bunsen adds, behind one seam that bunsen owns. A block that takes
//! an activation takes an [`BunsenActivationConfig`], so new kinds are bunsen's
//! to add without waiting on burn's releases.
//!
//! - **burn's kinds** keep burn's `BunsenActivationConfig` names
//!   ([`Identity`](BunsenActivationConfig::Identity),
//!   [`Relu`](BunsenActivationConfig::Relu),
//!   [`Gelu`](BunsenActivationConfig::Gelu),
//!   [`GeluApproximate`](BunsenActivationConfig::GeluApproximate),
//!   [`Sigmoid`](BunsenActivationConfig::Sigmoid),
//!   [`Tanh`](BunsenActivationConfig::Tanh),
//!   [`PRelu`](BunsenActivationConfig::PRelu),
//!   [`LeakyRelu`](BunsenActivationConfig::LeakyRelu),
//!   [`Selu`](BunsenActivationConfig::Selu),
//!   [`HardSigmoid`](BunsenActivationConfig::HardSigmoid),
//!   [`HardSwish`](BunsenActivationConfig::HardSwish),
//!   [`Softplus`](BunsenActivationConfig::Softplus),
//!   [`Softsign`](BunsenActivationConfig::Softsign),
//!   [`Elu`](BunsenActivationConfig::Elu),
//!   [`Celu`](BunsenActivationConfig::Celu),
//!   [`ThresholdedRelu`](BunsenActivationConfig::ThresholdedRelu),
//!   [`HardShrink`](BunsenActivationConfig::HardShrink),
//!   [`SoftShrink`](BunsenActivationConfig::SoftShrink),
//!   [`Shrink`](BunsenActivationConfig::Shrink)): every weightless, elementwise
//!   kind of burn's `ActivationConfig`, so a config's serde form and a native
//!   record's keys are burn's. Not burn's `SwiGlu`: it owns two `Linear`s and
//!   changes the width, a gated MLP rather than an activation (bunsen's is the
//!   gated MLP with [`Silu`](BunsenActivationConfig::Silu)).
//! - **bunsen's kinds**: [`Silu`](BunsenActivationConfig::Silu), which burn has
//!   as a layer but not as a kind; [`Pow`](BunsenActivationConfig::Pow), `x^p`;
//!   and [`Chain`](BunsenActivationConfig::Chain), kinds applied in order:
//!   nanochat's `relu(x)²` is `Chain([Relu, Pow(2)])`.

mod activation_impl;

pub use activation_impl::*;
