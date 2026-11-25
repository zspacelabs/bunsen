//! Each row's model against timm's, on `ImageNet`'s validation images.

use bunsen::{
    data::{
        datasets::try_read_webdataset,
        formats::LoadPolicy,
    },
    kits::images::{
        classification::{
            driver::ImageClassifier,
            resnet::pretrained::default_resnet_factory,
        },
        support::{
            encoder::ImageEncoder,
            eval::try_eval_input,
        },
    },
    support::{
        images::try_decode_image,
        testing::performance_device,
    },
};
use burn::{
    Tensor,
    tensor::{
        DType,
        FloatDType,
        TensorData,
    },
};

use super::*;

/// How far bunsen's float32 (on wgpu) may be from timm's float64, per
/// value: `|bunsen - timm| <= ABS + REL * |timm|`. Measured on the rows
/// (2026-11-24): at most 3.4e-5 on a logit (the group norm's
/// `resnet50_gn`), 2.6e-5 on a feature (the 101-layer
/// `seresnextaa101d_32x8d` at 288 pixels).
const ABS: f32 = 1e-4;
/// See [`ABS`].
const REL: f32 = 1e-4;

/// How far bunsen's own preprocessing may be from timm's input, on average
/// over every value, in normalised units: `image`'s resize against `PIL`'s.
/// Measured on the rows (2026-11-24): 0.0028 to 0.0034.
const PREPROCESS_MEAN_ABS: f32 = 0.01;

/// The largest `|a - b|`, and whether every value is within the tolerance.
fn compare(
    what: &str,
    actual: &[f32],
    expected: &[f32],
) {
    assert_eq!(actual.len(), expected.len(), "{what}: lengths");
    let mut worst = 0f32;
    let mut misses = 0usize;
    for (a, e) in actual.iter().zip(expected) {
        let d = (a - e).abs();
        worst = worst.max(d);
        if d > ABS + REL * e.abs() {
            misses += 1;
        }
    }
    println!("{what}: max |bunsen - timm| = {worst:.3e}");
    assert_eq!(
        misses, 0,
        "{what}: {misses} values out of tolerance (max {worst:.3e})"
    );
}

/// The top class of each row of `[n, classes]` values.
fn top1(
    values: &[f32],
    classes: usize,
) -> Vec<usize> {
    values
        .chunks_exact(classes)
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
        })
        .collect()
}

/// A tensor's values as `f32`s.
fn host<const D: usize>(x: Tensor<D>) -> Vec<f32> {
    x.into_data()
        .convert::<f32>()
        .try_into_vec::<f32>()
        .unwrap()
}

/// `row`'s model, its weights loaded in f32 at least, against its fixture.
fn check_row(row: &str) {
    let device = performance_device();
    let fixture = read_fixture(row);
    let cache = weights_cache();
    let construct = default_resnet_factory()
        .expect("the resnet factory")
        .load_construct(
            row,
            &cache,
            &device,
            LoadPolicy::min_precision(FloatDType::F32),
        )
        .unwrap_or_else(|e| panic!("load {row}: {e:#}"));
    let (model, card) = (&construct.model, &construct.card);
    let key = input_key(&card.transform);
    assert_eq!(
        key, fixture.input_key,
        "{row}: the card's transform is the fixture's"
    );

    let (inputs, shape, sha) = read_inputs(&inputs_path(&cache, &key));
    assert_eq!(
        sha, fixture.input_sha256,
        "{row}: the local inputs are the fixture's"
    );
    let images = Tensor::<4>::from_data(TensorData::new(inputs.clone(), shape), &device);

    let logits = host(model.forward_logits(images.clone()));
    compare(&format!("{row} logits"), &logits, &fixture.logits);
    compare(
        &format!("{row} pre-logits"),
        &host(model.forward_pooled(images)),
        &fixture.pre_logits,
    );
    let ours = top1(&logits, fixture.classes);
    assert_eq!(ours, top1(&fixture.logits, fixture.classes), "{row}: top-1");
    let correct = ours
        .iter()
        .zip(&fixture.labels)
        .filter(|(a, b)| a == b)
        .count();
    println!("{row}: {correct}/{IMAGES} top-1 correct against the labels");

    // bunsen's own preprocessing of the same images, against timm's input.
    let samples = try_read_webdataset(
        &validation_shard(&cache),
        Some(&["jpg", "cls"]),
        Some(IMAGES),
    )
    .expect("read the shard");
    assert_eq!(
        samples.iter().map(|s| s.key.as_str()).collect::<Vec<_>>(),
        fixture.keys,
        "{row}: the shard's first samples are the fixture's"
    );
    let ours: Vec<Tensor<3>> = samples
        .iter()
        .map(|s| {
            let image = try_decode_image(s.field("jpg").unwrap()).expect("decode");
            try_eval_input(&card.transform, &image, &device).expect("bunsen's eval input")
        })
        .collect();
    let ours = host(Tensor::stack::<4>(ours, 0).cast(DType::F32));
    let mean_abs = ours
        .iter()
        .zip(&inputs)
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / ours.len() as f32;
    println!("{row}: bunsen's preprocessing, mean |bunsen - timm| = {mean_abs:.4}");
    assert!(
        mean_abs < PREPROCESS_MEAN_ABS,
        "{row}: bunsen's preprocessing is {mean_abs} from timm's on average"
    );
}

/// The torchvision rows.
mod torchvision {
    use super::*;

    #[test]
    #[serial_test::serial]
    fn resnet18() {
        check_row("torchvision/resnet18");
    }

    #[test]
    #[serial_test::serial]
    fn resnet50() {
        check_row("torchvision/resnet50");
    }
}

/// The timm hub rows.
mod timm {
    use super::*;

    #[test]
    #[serial_test::serial]
    fn resnet18_a1_in1k() {
        check_row("timm/resnet18.a1_in1k");
    }

    #[test]
    #[serial_test::serial]
    fn resnet26_bt_in1k() {
        check_row("timm/resnet26.bt_in1k");
    }

    #[test]
    #[serial_test::serial]
    fn resnet50_a1_in1k() {
        check_row("timm/resnet50.a1_in1k");
    }

    #[test]
    #[serial_test::serial]
    fn resnet101_tv2_in1k() {
        check_row("timm/resnet101.tv2_in1k");
    }

    #[test]
    #[serial_test::serial]
    fn resnet50_gn_a1h_in1k() {
        check_row("timm/resnet50_gn.a1h_in1k");
    }

    #[test]
    #[serial_test::serial]
    fn resnext50_32x4d_a1h_in1k() {
        check_row("timm/resnext50_32x4d.a1h_in1k");
    }

    #[test]
    #[serial_test::serial]
    fn wide_resnet50_2_racm_in1k() {
        check_row("timm/wide_resnet50_2.racm_in1k");
    }

    #[test]
    #[serial_test::serial]
    fn resnet50d_a1_in1k() {
        check_row("timm/resnet50d.a1_in1k");
    }

    #[test]
    #[serial_test::serial]
    fn resnet26t_ra2_in1k() {
        check_row("timm/resnet26t.ra2_in1k");
    }

    #[test]
    #[serial_test::serial]
    fn resnet50s_gluon_in1k() {
        check_row("timm/resnet50s.gluon_in1k");
    }

    #[test]
    #[serial_test::serial]
    fn seresnet50_a1_in1k() {
        check_row("timm/seresnet50.a1_in1k");
    }

    #[test]
    #[serial_test::serial]
    fn ecaresnet50d_miil_in1k() {
        check_row("timm/ecaresnet50d.miil_in1k");
    }

    #[test]
    #[serial_test::serial]
    fn resnetblur50_bt_in1k() {
        check_row("timm/resnetblur50.bt_in1k");
    }

    #[test]
    #[serial_test::serial]
    fn resnetaa50_a1h_in1k() {
        check_row("timm/resnetaa50.a1h_in1k");
    }

    #[test]
    #[serial_test::serial]
    fn seresnextaa101d_32x8d_sw_in12k_ft_in1k_288() {
        check_row("timm/seresnextaa101d_32x8d.sw_in12k_ft_in1k_288");
    }

    #[test]
    #[serial_test::serial]
    fn resnetrs50_tf_in1k() {
        check_row("timm/resnetrs50.tf_in1k");
    }
}
