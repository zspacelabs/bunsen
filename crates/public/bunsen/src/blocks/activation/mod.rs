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
//!   [`LeakyRelu`](BunsenActivationConfig::LeakyRelu)), so a config's serde
//!   form and a native record's keys are burn's.
//! - **bunsen's kinds**: [`Silu`](BunsenActivationConfig::Silu), which burn has
//!   as a layer but not as a kind; [`Pow`](BunsenActivationConfig::Pow), `x^p`;
//!   and [`Chain`](BunsenActivationConfig::Chain), kinds applied in order:
//!   nanochat's `relu(x)²` is `Chain([Relu, Pow(2)])`.

mod activation_impl;

pub use activation_impl::*;
