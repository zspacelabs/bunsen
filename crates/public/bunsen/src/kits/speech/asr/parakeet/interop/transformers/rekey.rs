//! The state-dict names.

use crate::{
    blocks::signals::{
        DwStridingFlattenOrder,
        DwStridingSubsamplingMeta,
    },
    burner::module::Rekey,
};

/// `transformers`' `ParakeetEncoderSubsampling` names onto a
/// [`DwStridingSubsampling`](crate::blocks::signals::DwStridingSubsampling)'s,
/// relative to it: the flat `layers.{k}` by
/// [`sequential_rekey`](DwStridingSubsamplingMeta::sequential_rekey),
/// everything else as it is, and the projection's weight permuted when
/// `striding` flattens otherwise than `transformers` does (channels outer).
pub fn subsampling_rekey(striding: &impl DwStridingSubsamplingMeta) -> Rekey {
    striding.reflatten_from(
        DwStridingFlattenOrder::ChannelsFreqs,
        striding.sequential_rekey("layers."),
    )
}

/// `transformers`' `ParakeetEncoderBlock` names onto a
/// [`ConformerBlock`](crate::blocks::transformers::encoders::conformer::ConformerBlock)'s,
/// relative to it: the attention's projections and positional biases, the
/// convolution module's depthwise convolution and norm, and the sublayers'
/// norms, renamed; everything else as it is.
pub fn encoder_block_rekey() -> Rekey {
    Rekey::keep_rest()
        .rename("self_attn.q_proj.", "attn.query.")
        .rename("self_attn.k_proj.", "attn.key.")
        .rename("self_attn.v_proj.", "attn.value.")
        .rename("self_attn.o_proj.", "attn.output.")
        .rename("self_attn.relative_k_proj.", "attn.pos.")
        .exact("self_attn.bias_u", "attn.pos_bias_u")
        .exact("self_attn.bias_v", "attn.pos_bias_v")
        .rename("conv.depthwise_conv.", "conv.depthwise.conv.")
        .rename("conv.norm.", "conv.depthwise.norm.")
        .rename("norm_feed_forward1.", "feed_forward1_norm.")
        .rename("norm_self_att.", "attn_norm.")
        .rename("norm_conv.", "conv_norm.")
        .rename("norm_feed_forward2.", "feed_forward2_norm.")
        .rename("norm_out.", "out_norm.")
        .rename("self_attn.", "attn.")
}

/// `transformers`' `ParakeetEncoder` names onto the
/// [`SubsampledEncoder`](crate::blocks::transformers::encoders::SubsampledEncoder)
/// a [`ConformerEncoderConfig`](crate::blocks::transformers::encoders::conformer::ConformerEncoderConfig) builds: [`subsampling_rekey`] and
/// [`encoder_block_rekey`] mounted where they sit, the batch-norm counters
/// dropped, and everything else as it is.
pub fn encoder_rekey(striding: &impl DwStridingSubsamplingMeta) -> Rekey {
    Rekey::keep_rest()
        .drop_suffix(".num_batches_tracked")
        .mount("subsampling.", "subsampling.", subsampling_rekey(striding))
        .mount_indexed("layers.", "transformer.layers.", encoder_block_rekey())
}

/// `transformers`' `ParakeetForTDT` state-dict names onto a
/// [`ParakeetTdt`](crate::kits::speech::asr::parakeet::ParakeetTdt)'s module
/// paths: the encoder by [`encoder_rekey`], the prediction network's
/// `nn.LSTM` by
/// [`lstm_stack_rekey`](crate::kits::support::torch::interop::lstm_stack_rekey),
/// the batch-norm counters dropped, and everything else as it is.
///
/// `striding` is the encoder's subsampling, a config or a built module: its
/// stage count and flatten order fix its map.
///
/// Normalization parameters keep their `weight` / `bias` spellings, which
/// [`Rekey::apply_to`] resolves to `gamma` / `beta`.
pub fn parakeet_tdt_rekey(striding: &impl DwStridingSubsamplingMeta) -> Rekey {
    Rekey::keep_rest()
        .drop_suffix(".num_batches_tracked")
        .mount("encoder.", "encoder.", encoder_rekey(striding))
        .mount(
            "decoder.lstm.",
            "decoder.lstm.",
            crate::kits::support::torch::interop::lstm_stack_rekey(),
        )
        .rename("decoder.decoder_projector.", "decoder.projector.")
}
