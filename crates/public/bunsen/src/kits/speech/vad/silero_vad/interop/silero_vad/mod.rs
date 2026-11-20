//! # Silero VAD as upstream's `silero_vad` package saves it
//!
//! `snakers4/silero-vad`'s `silero_vad_16k.safetensors`, the 16 kHz model's
//! `PyTorch` state dict, as bunsen's
//! [`SileroVad`](crate::kits::speech::vad::silero_vad::SileroVad): its names
//! ([`silero_vad_rekey`]).
//!
//! - **Loading** (feature `store`): `load_silero_vad_tensors`, and
//!   `load_silero_vad_16khz_from_path`, a file with no cache.

use crate::{
    blocks::conv::ConvSeq1d,
    burner::module::Rekey,
};

#[cfg(feature = "store")]
mod weights;
#[cfg(feature = "store")]
pub use weights::*;

/// Upstream's `silero_vad_16k.safetensors` names (the 16 kHz model's
/// `PyTorch` state dict) onto a
/// [`SileroVad`](crate::kits::speech::vad::silero_vad::SileroVad)'s module
/// paths; anything else is dropped.
///
/// | upstream | bunsen |
/// |---|---|
/// | `stft_conv` | `stft` |
/// | `conv1` … `conv4` | `encoder.blocks.0` … `3.conv`, by [`ConvSeq1d::one_based_rekey`] |
/// | `lstm_cell` (an `nn.LSTMCell`) | `lstm`, by [`lstm_cell_rekey`](crate::kits::support::interop::torch::lstm_cell_rekey) |
/// | `final_conv` | `decoder` |
pub fn silero_vad_rekey() -> Rekey {
    Rekey::drop_rest()
        .rename("stft_conv.", "stft.")
        .rename("final_conv.", "decoder.")
        .mount(
            "lstm_cell.",
            "lstm.",
            crate::kits::support::interop::torch::lstm_cell_rekey(),
        )
        .mount("conv", "encoder.", ConvSeq1d::one_based_rekey())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silero_vad_rekey() {
        let rekey = silero_vad_rekey();
        for (upstream, bunsen) in [
            ("stft_conv.weight", Some("stft.weight")),
            ("conv1.weight", Some("encoder.blocks.0.conv.weight")),
            ("conv4.bias", Some("encoder.blocks.3.conv.bias")),
            ("lstm_cell.weight_ih", Some("lstm.features.weight")),
            ("lstm_cell.bias_ih", Some("lstm.features.bias")),
            ("lstm_cell.weight_hh", Some("lstm.hidden.weight")),
            ("lstm_cell.bias_hh", Some("lstm.hidden.bias")),
            ("final_conv.bias", Some("decoder.bias")),
            ("conv0.weight", None),
            ("lstm_cell.weight_xx", None),
            ("lstm_cell.weight_ih_l0", None),
            ("unknown", None),
        ] {
            assert_eq!(rekey.apply(upstream).as_deref(), bunsen, "{upstream}");
        }
    }
}
