pub mod silero_vad_op18_ifless {
    include!(concat!(env!("OUT_DIR"), "/silero_vad_op18_ifless.rs"));

    /// Pretrained dual branch 16khz/8khs model.
    pub const BURNPACK_WEIGHTS: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/silero_vad_op18_ifless.bpk"));

    /// Convert the burnpack weights to a burn bytes.
    pub fn burnpack_as_burn_bytes() -> burn::tensor::Bytes {
        burn::tensor::Bytes::from_bytes_vec(BURNPACK_WEIGHTS.to_vec())
    }
}
