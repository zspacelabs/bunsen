use super::*;
use crate::burner::module::testing::assert_rekeys;

#[test]
fn test_silero_vad_rekey() {
    let rekey = silero_vad_rekey();
    assert_rekeys(
        &rekey,
        [
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
        ],
    );
}
