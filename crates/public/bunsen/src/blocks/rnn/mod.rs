//! # Recurrent blocks
//!
//! burn ships a sequence-level [`Lstm`](burn::nn::Lstm) module. The blocks
//! here cover what a streaming model needs instead: one step at a time, with
//! the state carried by the caller between steps. See [`lstm`].

pub mod lstm;
