//! # The workspace's test images
//!
//! The open images the workspace's tests read: photos of `ImageNet`-1k
//! classes, each released under CC0 1.0 by its author, found through
//! Openverse and downscaled to 512 pixels on the longer side.
//! `testdata/README.md` has each one's source, author and licence.
//!
//! They are for basic local tests (decoding, preprocessing, a classifier's
//! top class) and may be committed and shipped. `ImageNet`'s own images,
//! which a validation compares against, are pulled at run time and never
//! committed.
//!
//! ```
//! let image = bunsen_test_images::GOLDEN_RETRIEVER;
//! assert!(image.path().exists());
//! assert_eq!(image.imagenet_index, 207);
//! ```
//!
//! The crate is a leaf: it depends on nothing in the workspace, so `bunsen`
//! itself can take it as a dev-dependency. It hands out paths, not pixels:
//! decoding is `bunsen::support::images`'.

use std::path::{
    Path,
    PathBuf,
};

/// One committed image, under `testdata/`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageFixture {
    /// The image's file name.
    pub file: &'static str,
    /// Its width, in pixels.
    pub width: usize,
    /// Its height, in pixels.
    pub height: usize,
    /// The `ImageNet`-1k class it shows: its index in timm's class order.
    pub imagenet_index: usize,
    /// That class's `WordNet` synset.
    pub synset: &'static str,
    /// That class's lemmas, as timm names it.
    pub lemma: &'static str,
    /// The image's licence.
    pub licence: &'static str,
    /// The SHA-256 of the file.
    pub sha256: &'static str,
}

impl ImageFixture {
    /// The image's path.
    pub fn path(&self) -> PathBuf {
        testdata_dir().join(self.file)
    }
}

/// The directory the committed images are in.
pub fn testdata_dir() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/testdata"))
}

/// Golden retriever: `ImageNet`-1k class 207 (CC0 1.0).
pub const GOLDEN_RETRIEVER: ImageFixture = ImageFixture {
    file: "golden_retriever.jpg",
    width: 512,
    height: 340,
    imagenet_index: 207,
    synset: "n02099601",
    lemma: "golden retriever",
    licence: "CC0 1.0",
    sha256: "811dae8cf7672e1f8fa91df20032d3d20e08ba3d303861b1b98f88c1a315eade",
};

/// Tabby: `ImageNet`-1k class 281 (CC0 1.0).
pub const TABBY_CAT: ImageFixture = ImageFixture {
    file: "tabby_cat.jpg",
    width: 512,
    height: 342,
    imagenet_index: 281,
    synset: "n02123045",
    lemma: "tabby, tabby cat",
    licence: "CC0 1.0",
    sha256: "6b1335c0e89d0eebdc1e445f801c184991d153fc8fe73450ad45e806e1110e91",
};

/// Goldfish: `ImageNet`-1k class 1 (CC0 1.0).
pub const GOLDFISH: ImageFixture = ImageFixture {
    file: "goldfish.jpg",
    width: 341,
    height: 512,
    imagenet_index: 1,
    synset: "n01443537",
    lemma: "goldfish, Carassius auratus",
    licence: "CC0 1.0",
    sha256: "8163d059ef09c38fa526172f7d01b2c7d5c1e57e92cb90c9e87bcc7a89806dfb",
};

/// Pizza: `ImageNet`-1k class 963 (CC0 1.0).
pub const PIZZA: ImageFixture = ImageFixture {
    file: "pizza.jpg",
    width: 512,
    height: 342,
    imagenet_index: 963,
    synset: "n07873807",
    lemma: "pizza, pizza pie",
    licence: "CC0 1.0",
    sha256: "1645581b1856c25f21814418492bdd7459c47bdd6277fb86ffe94bb360cdbefd",
};

/// School bus: `ImageNet`-1k class 779 (CC0 1.0).
pub const SCHOOL_BUS: ImageFixture = ImageFixture {
    file: "school_bus.jpg",
    width: 512,
    height: 377,
    imagenet_index: 779,
    synset: "n04146614",
    lemma: "school bus",
    licence: "CC0 1.0",
    sha256: "b673a50cb2071838ac875ffab5d6c3d7836eb65cb37d53f0f160d4f25f209aef",
};

/// Banana: `ImageNet`-1k class 954 (CC0 1.0).
pub const BANANA: ImageFixture = ImageFixture {
    file: "banana.jpg",
    width: 512,
    height: 384,
    imagenet_index: 954,
    synset: "n07753592",
    lemma: "banana",
    licence: "CC0 1.0",
    sha256: "da6fdf8cac8508aef31ddc49836ce5b9ba23eb1a5b9a89c1f583c79e9cf8ef5b",
};

/// Every image, in the order above.
pub const ALL: [ImageFixture; 6] = [
    GOLDEN_RETRIEVER,
    TABBY_CAT,
    GOLDFISH,
    PIZZA,
    SCHOOL_BUS,
    BANANA,
];

#[cfg(test)]
mod tests {
    use sha2::{
        Digest,
        Sha256,
    };

    use super::*;

    /// Every image is on disk with its digest.
    #[test]
    fn test_every_image_is_pinned() {
        for image in ALL {
            let bytes = std::fs::read(image.path()).unwrap();
            let digest: String = Sha256::digest(&bytes)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            assert_eq!(digest, image.sha256, "{}", image.file);
        }
    }
}
