//! `embed run`.

use std::path::PathBuf;

use bunsen::{
    errors::{
        BunsenError,
        BunsenResult,
        sys_at,
    },
    kits::images::support::eval::try_load_eval_batch,
};
use bunsen_app::{
    cache::WeightsCacheArgs,
    device::BackendArgs,
    logging::{
        LogArgs,
        LogPolicy,
    },
    precision::MinPrecisionArgs,
};
use log::LevelFilter;

use crate::classify::{
    DEFAULT_IMAGE_MODEL,
    load_image_model,
};

/// Embeds image files.
#[derive(clap::Args, Debug)]
pub struct EmbedRunCommand {
    #[clap(flatten)]
    logging: LogArgs,

    #[clap(flatten)]
    cache: WeightsCacheArgs,

    /// The device.
    #[clap(flatten)]
    device: BackendArgs,

    /// The narrowest float the checkpoint's weights are placed in.
    #[clap(flatten)]
    precision: MinPrecisionArgs,

    /// The checkpoint: a name `classify list` shows, or `hf:timm/<repo>`.
    #[arg(long, default_value = DEFAULT_IMAGE_MODEL)]
    model: String,

    /// The safetensors file to write.
    ///
    /// Its `features` are `[images, d]` float32; its metadata names the image
    /// files, in order, as JSON.
    #[arg(long, short)]
    output: PathBuf,

    /// The image files (JPEG, PNG or GIF).
    #[arg(required = true)]
    images: Vec<PathBuf>,
}

impl EmbedRunCommand {
    /// Embeds the images and writes their features.
    pub fn run(&self) -> BunsenResult<()> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Warn, LevelFilter::Warn))?;
        let cache = self.cache.init()?;
        let device = self.device.open()?;
        let loaded = load_image_model(&self.model, &cache, &device, self.precision.load_policy())?;
        let batch = try_load_eval_batch(&loaded.card.transform, &self.images, &device)?;
        let features = loaded
            .model
            .forward_pooled(batch)
            .cast(burn::tensor::DType::F32);
        let [n, d] = features.dims();
        let values = features
            .into_data()
            .convert::<f32>()
            .try_into_vec::<f32>()
            .map_err(|e| BunsenError::internal(format!("the features: {e:?}")))?;
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let view =
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![n, d], &bytes)
                .map_err(|e| BunsenError::internal(format!("the features: {e}")))?;
        let images: Vec<String> = self
            .images
            .iter()
            .map(|p| p.display().to_string())
            .collect();
        let metadata = std::collections::HashMap::from([
            (
                "images".to_string(),
                serde_json::to_string(&images).expect("paths serialize"),
            ),
            ("model".to_string(), self.model.clone()),
        ]);
        let file = safetensors::serialize([("features", view)], Some(metadata))
            .map_err(|e| BunsenError::internal(format!("the features: {e}")))?;
        std::fs::write(&self.output, file).map_err(sys_at("write", &self.output))?;
        println!("{n} x {d} features: {}", self.output.display());
        Ok(())
    }
}
