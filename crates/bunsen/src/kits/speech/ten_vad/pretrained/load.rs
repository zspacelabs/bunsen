use burn::prelude::Backend;

use crate::kits::speech::ten_vad::{
    TenVad,
    reference,
};

impl<B: Backend> TenVad<B> {
    /// Load the common pretrained TenVAD model.
    pub fn load_pretrained(device: &B::Device) -> Self {
        Self::from_bytes(reference::burnpack_as_burn_bytes(), device)
    }
}
