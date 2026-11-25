//! `classify run`.

use std::path::PathBuf;

use bunsen::{
    errors::BunsenResult,
    kits::images::classification::driver::classify_files,
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
use serde::Serialize;

use crate::{
    classify::{
        DEFAULT_IMAGE_MODEL,
        load_image_model,
    },
    support::Format,
};

/// Classifies image files.
#[derive(clap::Args, Debug)]
pub struct ClassifyRunCommand {
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

    /// How many classes to print per image.
    #[arg(short, long, default_value_t = 5)]
    k: usize,

    /// The results' format: YAML, or JSON on one line.
    #[arg(long, value_enum, default_value_t = Format::Yml)]
    format: Format,

    /// The image files (JPEG, PNG or GIF).
    #[arg(required = true)]
    images: Vec<PathBuf>,
}

/// One image's top classes, as `classify run` prints them.
#[derive(Debug, Serialize)]
pub struct ImageClasses {
    /// The image file.
    pub image: String,

    /// Its top classes, most probable first.
    pub classes: Vec<ClassScore>,
}

/// One class of an image's top classes.
#[derive(Debug, Serialize)]
pub struct ClassScore {
    /// The class's index in the head.
    pub index: usize,

    /// Its softmax probability.
    pub probability: f32,

    /// Its readable name, when the checkpoint has labels.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Its `WordNet` synset, when the checkpoint has labels.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub synset: Option<String>,
}

impl ClassifyRunCommand {
    /// Classifies the images and prints their classes.
    pub fn run(&self) -> BunsenResult<()> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Warn, LevelFilter::Warn))?;
        let cache = self.cache.init()?;
        let device = self.device.open()?;
        let loaded = load_image_model(&self.model, &cache, &device, self.precision.load_policy())?;
        if loaded.model.n_classes() == 0 {
            return Err(bunsen::errors::BunsenError::unsupported(format!(
                "{}: a headless image tower has no classes; `bunsen-tool embed` writes its \
                 features",
                self.model
            )));
        }
        let top = classify_files(
            loaded.model.as_ref(),
            &loaded.card,
            &self.images,
            self.k,
            &device,
        )?;
        let results: Vec<ImageClasses> = self
            .images
            .iter()
            .zip(top)
            .map(|(image, classes)| ImageClasses {
                image: image.display().to_string(),
                classes: classes
                    .into_iter()
                    .map(|c| ClassScore {
                        index: c.index,
                        probability: c.probability,
                        name: c.name,
                        synset: c.synset,
                    })
                    .collect(),
            })
            .collect();
        print!("{}", self.format.render(&results, "the classes")?);
        Ok(())
    }
}
