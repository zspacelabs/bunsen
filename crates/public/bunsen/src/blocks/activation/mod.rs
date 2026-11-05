//! # The activation seam
//!
//! [`ActivationConfig`] and [`Activation`] are bunsen's activation layers:
//! the kinds bunsen's blocks and examples use, each burn's layer, and the
//! kinds bunsen adds, behind one seam that bunsen owns. A block that takes an
//! activation takes an [`ActivationConfig`], so new kinds are bunsen's to add
//! without waiting on burn's releases.
//!
//! - **burn's kinds** keep burn's `ActivationConfig` names
//!   ([`Identity`](ActivationConfig::Identity),
//!   [`Relu`](ActivationConfig::Relu), [`Gelu`](ActivationConfig::Gelu),
//!   [`GeluApproximate`](ActivationConfig::GeluApproximate),
//!   [`Sigmoid`](ActivationConfig::Sigmoid), [`Tanh`](ActivationConfig::Tanh),
//!   [`PRelu`](ActivationConfig::PRelu),
//!   [`LeakyRelu`](ActivationConfig::LeakyRelu)), so a config's serde form and
//!   a native record's keys are burn's.
//! - **bunsen's kinds**: [`Silu`](ActivationConfig::Silu), which burn has as a
//!   layer but not as a kind; [`Pow`](ActivationConfig::Pow), `x^p`; and
//!   [`Chain`](ActivationConfig::Chain), kinds applied in order: nanochat's
//!   `relu(x)²` is `Chain([Relu, Pow(2)])`.

mod activation_impl;

pub use activation_impl::*;
