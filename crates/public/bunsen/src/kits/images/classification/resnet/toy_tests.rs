//! The toy models against timm's own `ResNet`.
//!
//! `testdata/resnet/` holds two tiny seeded timm `ResNet`s and their answers,
//! computed in float64 (`tools/gen_resnet_toy_fixtures.py`, timm's
//! `timm.models.resnet.ResNet`): four stages of one block at timm's
//! `channels=(8, 16, 32, 64)` on its 64-wide 7x7 stem, one of basic blocks and
//! one of bottlenecks. The weights, under timm's names, load through
//! [`load_resnet_tensors`]; each stage output, the pooled features and the
//! logits are checked against timm's.
//!
//! `ResNetContractConfig` derives its stage widths from the stem's, so the
//! geometry is the generic conv net's policy, stage by stage.

use std::path::PathBuf;

use burn::prelude::*;

use super::interop::torchvision::load_resnet_tensors;
use crate::{
    blocks::images::models::conv_net::{
        ConvNetContractConfig,
        ConvNetModel,
        ConvStageContractConfig,
        ConvStemContractConfig,
        ConvUnitKind,
    },
    burner::module::ModuleInit,
    kits::images::{
        classification::driver::ImageClassifier,
        support::encoder::ImageEncoder,
    },
    support::testing::{
        ToyFixture,
        assert_io_close,
        cpu_device,
    },
};

/// timm's stage widths, the mixer width of each stage's unit.
const CHANNELS: [usize; 4] = [8, 16, 32, 64];

/// The toys' geometry: a 64-wide `7x7` stem and four one-unit stages of
/// [`CHANNELS`] (a bottleneck's output 4x its mixer's), the first at stride
/// 1, 10 classes.
fn toy_policy(bottleneck: bool) -> ConvNetContractConfig {
    let (unit, expansion) = if bottleneck {
        (ConvUnitKind::Bottleneck, 4)
    } else {
        (ConvUnitKind::Basic, 1)
    };
    let stages =
        CHANNELS
            .iter()
            .enumerate()
            .map(|(idx, &mid)| {
                ConvStageContractConfig::new(1, unit, mid * expansion, mid)
                    .with_stride(if idx == 0 { 1 } else { 2 })
            })
            .collect();
    ConvNetContractConfig::new(ConvStemContractConfig::new(vec![64]), stages, 10)
}

/// The toy `name`, its weights loaded, and its fixture's answers.
fn load_toy(
    name: &str,
    bottleneck: bool,
    device: &Device,
) -> (ConvNetModel, ToyFixture) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/resnet")
        .join(format!("{name}.safetensors"));
    let mut fixture = ToyFixture::load(path, device);
    let mut model: ConvNetModel = toy_policy(bottleneck).init(device);
    let applied = load_resnet_tensors(&mut model, fixture.take_weights()).expect("load the toy");
    // Only the batch norms' counters are left: the rekey drops them.
    assert!(
        applied
            .unused
            .iter()
            .all(|n| n.ends_with(".num_batches_tracked")),
        "{name}: {:?}",
        applied.unused
    );
    (model, fixture)
}

/// Every stage, the pooled features and the logits are timm's.
fn check_toy(
    name: &str,
    bottleneck: bool,
) {
    let device = cpu_device();
    let (model, fixture) = load_toy(name, bottleneck, &device);
    let images: Tensor<4> = fixture.io("images");
    let stages = model.forward_intermediates(images.clone());
    assert_eq!(stages.len(), 5, "{name}: the stem and four stages");
    for (k, stage) in stages.into_iter().enumerate() {
        assert_io_close(
            &format!("{name} stage_{k}"),
            stage,
            fixture.io(&format!("stage_{k}")),
        );
    }
    assert_io_close(
        &format!("{name} pooled"),
        model.forward_pooled(images.clone()),
        fixture.io("pooled"),
    );
    assert_io_close(
        &format!("{name} logits"),
        model.forward_logits(images),
        fixture.io("logits"),
    );
}

#[test]
fn test_basic_toy_is_timms() {
    check_toy("toy_basic", false);
}

#[test]
fn test_bottleneck_toy_is_timms() {
    check_toy("toy_bottleneck", true);
}
