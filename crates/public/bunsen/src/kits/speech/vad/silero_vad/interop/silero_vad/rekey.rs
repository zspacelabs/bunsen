//! The state-dict names.

use crate::{
    blocks::conv::ConvSeq1d,
    burner::module::Rekey,
};

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
