//! [`TokenMixer`], a decoder layer's token mixer.

use burn::{
    config::Config,
    module::Module,
    tensor::{
        Device,
        FloatDType,
    },
};

use crate::{
    blocks::transformers::{
        attention::csa::{
            CausalSelfAttention,
            CausalSelfAttentionConfig,
            CausalSelfAttentionMeta,
        },
        mixers::{
            GatedDeltaNet,
            GatedDeltaNetConfig,
            GatedDeltaNetMeta,
        },
    },
    burner::module::{
        InputDTypes,
        ResultDType,
    },
    errors::BunsenResult,
};

/// Config for [`TokenMixer`].
#[derive(Config, Debug)]
pub enum TokenMixerConfig {
    /// Causal self-attention.
    Attention(CausalSelfAttentionConfig),

    /// A gated `DeltaNet`, linear attention.
    GatedDeltaNet(GatedDeltaNetConfig),
}

impl From<CausalSelfAttentionConfig> for TokenMixerConfig {
    fn from(config: CausalSelfAttentionConfig) -> Self {
        Self::Attention(config)
    }
}

impl From<GatedDeltaNetConfig> for TokenMixerConfig {
    fn from(config: GatedDeltaNetConfig) -> Self {
        Self::GatedDeltaNet(config)
    }
}

impl TokenMixerConfig {
    /// The size of the input and output.
    pub fn n_embed(&self) -> usize {
        match self {
            Self::Attention(c) => c.n_embed(),
            Self::GatedDeltaNet(c) => c.n_embed(),
        }
    }

    /// The attention's config, if the mixer attends.
    pub fn as_attention(&self) -> Option<&CausalSelfAttentionConfig> {
        match self {
            Self::Attention(c) => Some(c),
            Self::GatedDeltaNet(_) => None,
        }
    }

    /// The attention's config, mutably, if the mixer attends.
    pub fn as_attention_mut(&mut self) -> Option<&mut CausalSelfAttentionConfig> {
        match self {
            Self::Attention(c) => Some(c),
            Self::GatedDeltaNet(_) => None,
        }
    }

    /// The attention's config.
    ///
    /// # Panics
    /// If the mixer does not attend.
    pub fn attention(&self) -> &CausalSelfAttentionConfig {
        self.as_attention().expect("a token mixer that attends")
    }

    /// The attention's config, mutably.
    ///
    /// # Panics
    /// If the mixer does not attend.
    pub fn attention_mut(&mut self) -> &mut CausalSelfAttentionConfig {
        self.as_attention_mut().expect("a token mixer that attends")
    }

    /// Initializes the mixer at `slot`: its slot in the decode's cache for
    /// its kind, a [`KVCache`](crate::ops::transformers::attention::KVCache)'s
    /// layer for attention, a
    /// [`DeltaNetCache`](crate::ops::transformers::linear_attention::DeltaNetCache)'s
    /// for a gated `DeltaNet`.
    pub fn try_init(
        &self,
        slot: usize,
        device: &Device,
    ) -> BunsenResult<TokenMixer> {
        Ok(match self {
            Self::Attention(c) => TokenMixer::Attention(c.try_init(slot, device)?),
            Self::GatedDeltaNet(c) => TokenMixer::GatedDeltaNet(c.try_init(slot, device)?),
        })
    }
}

/// A decoder layer's token mixer, what mixes information across positions:
/// [`CausalSelfAttention`], or a [`GatedDeltaNet`] (a hybrid model's linear
/// layers).
///
/// A loader skips enum variants, so a mixer's record paths name its module's
/// fields alone (`attn.branch.c_q`, `attn.branch.in_proj_qkv`).
///
/// Built by [`TokenMixerConfig`].
#[derive(Module, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum TokenMixer {
    /// Causal self-attention.
    Attention(CausalSelfAttention),

    /// A gated `DeltaNet`, linear attention.
    GatedDeltaNet(GatedDeltaNet),
}

impl From<CausalSelfAttention> for TokenMixer {
    fn from(attn: CausalSelfAttention) -> Self {
        Self::Attention(attn)
    }
}

impl From<GatedDeltaNet> for TokenMixer {
    fn from(net: GatedDeltaNet) -> Self {
        Self::GatedDeltaNet(net)
    }
}

impl TokenMixer {
    /// The size of the input and output.
    pub fn n_embed(&self) -> usize {
        match self {
            Self::Attention(m) => m.n_embed(),
            Self::GatedDeltaNet(m) => m.n_embed(),
        }
    }

    /// The attention, if the mixer attends.
    pub fn as_attention(&self) -> Option<&CausalSelfAttention> {
        match self {
            Self::Attention(m) => Some(m),
            Self::GatedDeltaNet(_) => None,
        }
    }

    /// The attention, mutably, if the mixer attends.
    pub fn as_attention_mut(&mut self) -> Option<&mut CausalSelfAttention> {
        match self {
            Self::Attention(m) => Some(m),
            Self::GatedDeltaNet(_) => None,
        }
    }

    /// The attention.
    ///
    /// # Panics
    /// If the mixer does not attend.
    pub fn attention(&self) -> &CausalSelfAttention {
        self.as_attention().expect("a token mixer that attends")
    }

    /// The attention, mutably.
    ///
    /// # Panics
    /// If the mixer does not attend.
    pub fn attention_mut(&mut self) -> &mut CausalSelfAttention {
        self.as_attention_mut().expect("a token mixer that attends")
    }

    /// The gated `DeltaNet`, if the mixer is one.
    pub fn as_gated_delta_net(&self) -> Option<&GatedDeltaNet> {
        match self {
            Self::Attention(_) => None,
            Self::GatedDeltaNet(m) => Some(m),
        }
    }
}

impl InputDTypes for TokenMixer {
    /// Its module's.
    fn preferred_dtype(&self) -> FloatDType {
        match self {
            Self::Attention(m) => m.preferred_dtype(),
            Self::GatedDeltaNet(m) => m.preferred_dtype(),
        }
    }
}

impl ResultDType for TokenMixer {
    /// Its module's.
    fn result_dtype(
        &self,
        input: FloatDType,
    ) -> FloatDType {
        match self {
            Self::Attention(m) => m.result_dtype(input),
            Self::GatedDeltaNet(m) => m.result_dtype(input),
        }
    }
}
