//! The state-dict names.

use crate::{
    blocks::signals::DwStridingSubsamplingMeta,
    burner::module::Rekey,
    kits::speech::support::nemo::interop::{
        conformer_encoder_rekey,
        transformer_encoder_rekey,
    },
};

/// `NeMo` `SortformerEncLabelModel` state-dict names, in Sortformer v2's
/// layout, onto a
/// [`Sortformer`](crate::kits::speech::diarization::sortformer::Sortformer)'s
/// module paths: the `FastConformer` by [`conformer_encoder_rekey`], the
/// `transformer_encoder` by [`transformer_encoder_rekey`], the projection and
/// head by prefix. Anything else is dropped: `num_batches_tracked`, the
/// featurizer's tables, and the unused `hidden_to_spks`.
///
/// `striding` is the encoder's subsampling, a config or a built module: its
/// stage count and flatten order fix its map.
pub fn sortformer_v2_rekey(striding: &impl DwStridingSubsamplingMeta) -> Rekey {
    Rekey::drop_rest()
        .mount("encoder.", "encoder.", conformer_encoder_rekey(striding))
        .mount(
            "transformer_encoder.",
            "transformer.",
            transformer_encoder_rekey(),
        )
        .rename("sortformer_modules.encoder_proj.", "projection.")
        .rename("sortformer_modules.first_hidden_to_hidden.", "head.dense.")
        .rename("sortformer_modules.single_hidden_to_spks.", "head.out.")
}
