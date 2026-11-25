use alloc::{
    vec,
    vec::Vec,
};

use burn::prelude::Tensor;

use super::*;
use crate::{
    blocks::conv::ConvBlock2dMeta,
    burner::module::{
        ModuleInit,
        ToStructureConfig,
    },
    errors::{
        BunsenErrorKind,
        BunsenResult,
        ConstraintError,
        testing::ErrorMatcher,
    },
    ops::drop::DropBlockOptions,
    support::testing::cpu_device,
};

/// A `ResNet`'s policy at toy widths: a `stem`-wide `7x7` stem and its max
/// pool, then a stage per depth, the first at stride 1, each doubling the
/// width; a bottleneck's units 4x their mixer.
fn resnet_like(
    depths: &[usize],
    stem: usize,
    unit: ConvUnitKind,
) -> ConvNetContractConfig {
    let expansion = match unit {
        ConvUnitKind::Basic => 1,
        ConvUnitKind::Bottleneck => 4,
    };
    let stages = depths
        .iter()
        .enumerate()
        .map(|(idx, &depth)| {
            let mid = stem << idx;
            ConvStageContractConfig::new(depth, unit, mid * expansion, mid)
                .with_stride(if idx == 0 { 1 } else { 2 })
        })
        .collect();
    ConvNetContractConfig::new(ConvStemContractConfig::new(vec![stem]), stages, 7)
}

/// A conv block's `[in, out]`, kernel, stride, dilation, groups, and whether
/// it has an activation.
fn conv_shape(conv: &ResidualConvConfig) -> ([usize; 2], usize, usize, usize, usize, bool) {
    let ResidualConvConfig::Block(block) = conv;
    (
        [block.in_channels(), block.out_channels()],
        block.kernel_size()[0],
        block.stride()[0],
        block.dilation()[0],
        block.groups(),
        block.act.is_some(),
    )
}

/// The basic stage: two `3x3`s, the first the mixer and strided; the last
/// without an activation, which follows the add; a shortcut only where the
/// width or the size changes.
#[test]
fn test_basic_units_lower_to_resnets() {
    let structure = resnet_like(&[2, 2], 8, ConvUnitKind::Basic).to_structure();
    assert_eq!(structure.stem.convs.len(), 1);
    let stem = &structure.stem.convs[0];
    assert_eq!(
        (
            stem.in_channels(),
            stem.out_channels(),
            stem.kernel_size()[0]
        ),
        (3, 8, 7)
    );
    assert_eq!(structure.stem.stride(), 4);

    let first = &structure.stages[0].units[0];
    assert_eq!(first.mixer, 0);
    assert_eq!(
        first.convs.iter().map(conv_shape).collect::<Vec<_>>(),
        [([8, 8], 3, 1, 1, 1, true), ([8, 8], 3, 1, 1, 1, false)]
    );
    assert!(
        first.shortcut.is_none(),
        "8 -> 8 at stride 1 is the identity"
    );
    assert!(first.act.is_some());

    let down = &structure.stages[1].units[0];
    assert_eq!(
        down.convs.iter().map(conv_shape).collect::<Vec<_>>(),
        [([8, 16], 3, 2, 1, 1, true), ([16, 16], 3, 1, 1, 1, false)]
    );
    let Some(ConvShortcutConfig::Conv(shortcut)) = &down.shortcut else {
        panic!("a strided unit has a shortcut");
    };
    assert_eq!(
        (
            shortcut.in_channels(),
            shortcut.out_channels(),
            shortcut.kernel_size()[0],
            shortcut.stride()[0],
            shortcut.act.is_some(),
        ),
        (8, 16, 1, 2, false)
    );
    assert!(structure.stages[1].units[1].shortcut.is_none());
    assert_eq!((structure.d_features(), structure.num_classes()), (16, 7));
}

/// The bottleneck stage: `1x1`, a `3x3` mixer that strides and groups, and
/// a `1x1` expand; the first stage's first unit widens, so has a shortcut.
#[test]
fn test_bottleneck_units_lower_to_resnets() {
    let mut policy = resnet_like(&[1, 2], 8, ConvUnitKind::Bottleneck);
    policy.stages[1].groups = 2;
    let structure = policy.to_structure();

    let first = &structure.stages[0].units[0];
    assert_eq!(first.mixer, 1);
    assert_eq!(
        first.convs.iter().map(conv_shape).collect::<Vec<_>>(),
        [
            ([8, 8], 1, 1, 1, 1, true),
            ([8, 8], 3, 1, 1, 1, true),
            ([8, 32], 1, 1, 1, 1, false)
        ]
    );
    assert!(first.shortcut.is_some(), "8 -> 32 needs a projection");

    let down = &structure.stages[1].units[0];
    assert_eq!(
        down.convs.iter().map(conv_shape).collect::<Vec<_>>(),
        [
            ([32, 16], 1, 1, 1, 1, true),
            ([16, 16], 3, 2, 1, 2, true),
            ([16, 64], 1, 1, 1, 1, false)
        ]
    );
    assert_eq!(structure.stages[1].units[1].in_channels(), 64);
    assert_eq!(structure.d_features(), 64);
}

/// Past the output stride a stage's stride is dilation, as timm's: at 8, the
/// third stage's units dilate by 2 (its first unit's mixer by the 1 before
/// it), the fourth's by 4 (its first by 2), and neither strides; their first
/// units still project the widened channels.
#[test]
fn test_output_stride_dilates() {
    let structure = resnet_like(&[2, 2, 2, 2], 8, ConvUnitKind::Basic)
        .with_output_stride(8)
        .to_structure();
    let strides: Vec<usize> = structure.stages.iter().map(|s| s.stride()).collect();
    assert_eq!(strides, [1, 2, 1, 1]);
    let dilations = |stage: usize, unit: usize| -> [usize; 2] {
        let path = &structure.stages[stage].units[unit].convs;
        [conv_shape(&path[0]).3, conv_shape(&path[1]).3]
    };
    assert_eq!(dilations(2, 0), [1, 2]);
    assert_eq!(dilations(2, 1), [2, 2]);
    assert_eq!(dilations(3, 0), [2, 4]);
    assert_eq!(dilations(3, 1), [4, 4]);
    // A 3x3's padding keeps the size at its dilation.
    let ResidualConvConfig::Block(block) = &structure.stages[3].units[1].convs[0];
    assert_eq!(
        block.conv.padding,
        burn::nn::PaddingConfig2d::Explicit(4, 4, 4, 4)
    );
    assert!(structure.stages[2].units[0].shortcut.is_some());

    let device = cpu_device();
    let model: ConvNetModel = structure.init(&device);
    let map = model.feature_map(Tensor::<4>::zeros([1, 3, 64, 64], &device));
    assert_eq!(map.dims(), [1, 64, 8, 8]);
}

/// Asserts that `a` and `b` answer every [`ConvNetMeta`] method alike.
fn assert_meta_agrees(
    a: &impl ConvNetMeta,
    b: &impl ConvNetMeta,
) {
    assert_eq!(a.in_channels(), b.in_channels());
    assert_eq!(a.num_stages(), b.num_stages());
    assert_eq!(a.d_features(), b.d_features());
    assert_eq!(a.num_classes(), b.num_classes());
}

/// A policy builds the same model through its structure as through the
/// blanket `init`.
#[test]
fn test_policy_pathways_agree() {
    let device = cpu_device();
    let policy = resnet_like(&[1, 1], 8, ConvUnitKind::Bottleneck);
    let structure = policy.to_structure();
    assert_eq!(
        (
            structure.in_channels(),
            structure.num_stages(),
            structure.d_features()
        ),
        (3, 2, 64)
    );
    let lowered: ConvNetModel = structure.init(&device);
    let direct: ConvNetModel = policy.init(&device);
    assert_meta_agrees(&direct, &lowered);
    assert_meta_agrees(&direct, &structure);
}

/// The maps by stage are timm's: the stem's convs at stride 2, then each
/// stage; the pooled features are the head's width, the logits its classes.
#[test]
fn test_forward_shapes() {
    let device = cpu_device();
    let model: ConvNetModel = resnet_like(&[1, 1], 8, ConvUnitKind::Basic).init(&device);
    let images = Tensor::<4>::zeros([2, 3, 32, 32], &device);
    let maps: Vec<_> = model
        .stage_maps(images.clone())
        .iter()
        .map(|x| x.dims())
        .collect();
    assert_eq!(maps, [[2, 8, 16, 16], [2, 8, 8, 8], [2, 16, 4, 4]]);
    assert_eq!(model.feature_map(images.clone()).dims(), [2, 16, 4, 4]);
    assert_eq!(model.pooled_features(images.clone()).dims(), [2, 16]);
    assert_eq!(model.forward(images.clone()).dims(), [2, 7]);
    assert_eq!(model.with_classes(3).forward(images).dims(), [2, 3]);
}

/// A stride-2 stage takes an odd size, `ceil(n / 2)` as torch's: a
/// 40-pixel input's stages are 10, 5 and 3 wide.
#[test]
fn test_odd_stage_sizes() {
    let device = cpu_device();
    let model: ConvNetModel = resnet_like(&[1, 1, 1], 8, ConvUnitKind::Basic).init(&device);
    let sides: Vec<_> = model
        .stage_maps(Tensor::<4>::zeros([1, 3, 40, 40], &device))
        .iter()
        .map(|x| x.dims()[2])
        .collect();
    assert_eq!(sides, [20, 10, 5, 3]);
}

/// A stem with no pool strides by its convs alone.
#[test]
fn test_a_stem_without_a_pool() {
    let mut policy = resnet_like(&[1], 8, ConvUnitKind::Basic);
    policy.stem.pool = ConvStemPoolKind::None;
    let structure = policy.to_structure();
    assert!(structure.stem.pool.is_none());
    assert_eq!(structure.stem.stride(), 2);
    let device = cpu_device();
    let model: ConvNetModel = structure.init(&device);
    assert_eq!(model.stem.stride(), 2);
    let map = model.feature_map(Tensor::<4>::zeros([1, 3, 32, 32], &device));
    assert_eq!(map.dims(), [1, 8, 16, 16]);
}

/// A kind not built yet is refused when lowered, naming the stage.
#[test]
fn test_unbuilt_kinds_are_unsupported() {
    let mut policy = resnet_like(&[1, 1], 8, ConvUnitKind::Basic);
    policy.layers.shortcut = ConvShortcutKind::AvgConv;
    ErrorMatcher::kind(BunsenErrorKind::Unsupported)
        .message_contains("AvgConv")
        .frame_contains("conv net stage 1")
        .assert_err(&policy.try_to_structure());
}

/// A malformed tree is an error from `try_init`, not a panic: no stages, an
/// empty stage, a stage that does not chain, a mixer past the path, a
/// widening unit with no shortcut.
#[test]
fn test_try_init_rejects_bad_trees() {
    let device = cpu_device();
    let structure = resnet_like(&[1, 1], 8, ConvUnitKind::Basic).to_structure();
    let illegal = || ErrorMatcher::kind(BunsenErrorKind::Illegal).has_cause::<ConstraintError>();
    let init = |s: ConvNetStructureConfig| -> BunsenResult<ConvNetModel> { s.try_init(&device) };

    illegal()
        .message_contains("ConvNetStructureConfig.stages")
        .assert_err(&init(ConvNetStructureConfig {
            stages: vec![],
            ..structure.clone()
        }));

    // The policy reaches the same checks through the blanket `try_init`.
    let empty: BunsenResult<ConvNetModel> =
        resnet_like(&[1, 0], 8, ConvUnitKind::Basic).try_init(&device);
    illegal()
        .frame_contains("conv net stage 1")
        .message_contains("ConvStageConfig.units")
        .assert_err(&empty);

    let mut unchained = structure.clone();
    unchained.stages.swap(0, 1);
    illegal()
        .message_contains("ConvNetStructureConfig.stages")
        .assert_err(&init(unchained));

    let mut bad_mixer = structure.clone();
    bad_mixer.stages[0].units[0].mixer = 2;
    illegal()
        .message_contains("ResidualUnitConfig.mixer")
        .frame_contains("unit 0")
        .assert_err(&init(bad_mixer));

    let mut no_shortcut = structure.clone();
    no_shortcut.stages[1].units[0].shortcut = None;
    illegal()
        .message_contains("ResidualUnitConfig.shortcut")
        .assert_err(&init(no_shortcut));

    let mut wrong_head = structure;
    wrong_head.head.d_features = 9;
    illegal()
        .message_contains("ConvNetStructureConfig.head")
        .assert_err(&init(wrong_head));
}

/// timm's stochastic-depth schedule (`make_blocks` in `resnet.py`) for a
/// `[2, 2, 2, 2]` net at a rate of 0.1: unit `i` of the net's `n` gets
/// `0.1 * i / (n - 1)`, counting every unit. The values are python3's.
const TIMM_DROP_PATH_2222_AT_0_1: [f64; 8] = [
    0.0,
    0.014285714285714287,
    0.028571428571428574,
    0.042857142857142864,
    0.05714285714285715,
    0.07142857142857142,
    0.08571428571428573,
    0.1,
];

/// Each unit's drop-path probability in a structure, in net order.
fn structure_drop_path_probs(structure: &ConvNetStructureConfig) -> Vec<f64> {
    structure
        .stages
        .iter()
        .flat_map(|stage| stage.units.iter())
        .map(|unit| unit.drop_path_prob)
        .collect()
}

/// Each unit's drop-path probability in a model, in net order; 0 for a unit
/// without a `DropPath`.
fn module_drop_path_probs(model: &ConvNetModel) -> Vec<f64> {
    model
        .stages
        .iter()
        .flat_map(|stage| stage.units.iter())
        .map(|unit| unit.drop_path.as_ref().map_or(0.0, |d| d.drop_prob))
        .collect()
}

fn assert_rates_eq(
    actual: &[f64],
    expected: &[f64],
) {
    assert_eq!(actual.len(), expected.len(), "{actual:?} != {expected:?}");
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < 1e-12, "{actual:?} != {expected:?}");
    }
}

/// The structure's and the module's stochastic-depth schedules are timm's.
#[test]
fn test_drop_path_schedules_match_timm() {
    let structure = resnet_like(&[2, 2, 2, 2], 8, ConvUnitKind::Basic).to_structure();
    let model: ConvNetModel = structure.init(&cpu_device());
    assert_rates_eq(
        &module_drop_path_probs(&model.with_stochastic_path_depth(0.1)),
        &TIMM_DROP_PATH_2222_AT_0_1,
    );
    assert_rates_eq(
        &structure_drop_path_probs(&structure.with_stochastic_depth_drop_path_rate(0.1)),
        &TIMM_DROP_PATH_2222_AT_0_1,
    );

    // Two units: timm gives `[0.0, 0.1]`.
    let structure = resnet_like(&[1, 1], 8, ConvUnitKind::Basic).to_structure();
    let model: ConvNetModel = structure.init(&cpu_device());
    assert_rates_eq(
        &module_drop_path_probs(&model.with_stochastic_path_depth(0.1)),
        &[0.0, 0.1],
    );
    assert_rates_eq(
        &structure_drop_path_probs(&structure.with_stochastic_depth_drop_path_rate(0.1)),
        &[0.0, 0.1],
    );
}

/// timm's `DropBlock` schedule (`drop_blocks` in `resnet.py`) for a
/// `[2, 2, 2, 2]` net at 0.1: none in the first two stages; stage 3's units
/// a block size of 5 and a gamma scale of 0.25, stage 4's 3 and 1.0.
fn timm_drop_blocks_2222_at_0_1() -> Vec<Option<DropBlockOptions>> {
    let stage3 = DropBlockOptions::default()
        .with_drop_prob(0.1)
        .with_block_size(5)
        .with_gamma_scale(0.25);
    let stage4 = DropBlockOptions::default()
        .with_drop_prob(0.1)
        .with_block_size(3)
        .with_gamma_scale(1.0);
    vec![
        None,
        None,
        None,
        None,
        Some(stage3.clone()),
        Some(stage3),
        Some(stage4.clone()),
        Some(stage4),
    ]
}

fn structure_drop_blocks(structure: &ConvNetStructureConfig) -> Vec<Option<DropBlockOptions>> {
    structure
        .stages
        .iter()
        .flat_map(|stage| stage.units.iter())
        .map(|unit| unit.drop_block.clone())
        .collect()
}

fn module_drop_blocks(model: &ConvNetModel) -> Vec<Option<DropBlockOptions>> {
    model
        .stages
        .iter()
        .flat_map(|stage| stage.units.iter())
        .map(|unit| unit.drop_block.as_ref().map(|d| d.options.clone()))
        .collect()
}

/// The structure's `DropBlock` schedule is timm's, a model built from it
/// keeps it, and the module's schedule is the same.
#[test]
fn test_drop_block_schedules_match_timm() {
    let device = cpu_device();
    let structure = resnet_like(&[2, 2, 2, 2], 8, ConvUnitKind::Basic).to_structure();
    let model: ConvNetModel = structure.init(&device);
    assert_eq!(
        module_drop_blocks(&model.with_stochastic_drop_block(0.1)),
        timm_drop_blocks_2222_at_0_1()
    );

    let structure = structure.with_standard_drop_block_prob(0.1);
    assert_eq!(
        structure_drop_blocks(&structure),
        timm_drop_blocks_2222_at_0_1()
    );
    let model: ConvNetModel = structure.init(&device);
    assert_eq!(module_drop_blocks(&model), timm_drop_blocks_2222_at_0_1());

    // A model with drop blocks still runs (in eval they are identities).
    let logits = model.forward(Tensor::<4>::zeros([1, 3, 64, 64], &device));
    assert_eq!(logits.dims(), [1, 7]);
}

/// With fewer than 2 stages there is no second-to-last stage for the
/// `DropBlock` schedule: a nonzero rate is an error from both `try_` forms;
/// a rate of 0 places nothing.
#[test]
fn test_drop_block_schedules_reject_one_stage() {
    let structure = resnet_like(&[2], 8, ConvUnitKind::Basic).to_structure();
    let model: ConvNetModel = structure.init(&cpu_device());

    let one_stage =
        ErrorMatcher::kind(BunsenErrorKind::Illegal).message_contains("at least 2 stages");
    one_stage.assert_err(&structure.clone().try_with_standard_drop_block_prob(0.1));
    one_stage.assert_err(&model.clone().try_with_stochastic_drop_block(0.1));
    ErrorMatcher::kind(BunsenErrorKind::Illegal)
        .frame_contains("drop_prob")
        .assert_err(&structure.clone().try_with_standard_drop_block_prob(1.5));

    let structure = structure.try_with_standard_drop_block_prob(0.0).unwrap();
    assert_eq!(structure_drop_blocks(&structure), vec![None, None]);
    let model = model.try_with_stochastic_drop_block(0.0).unwrap();
    assert_eq!(module_drop_blocks(&model), vec![None, None]);
}

/// The panicking twin panics with a message that names the problem.
#[test]
#[should_panic(expected = "at least 2 stages")]
fn test_drop_block_schedule_panics_on_one_stage() {
    let _ = resnet_like(&[2], 8, ConvUnitKind::Basic)
        .to_structure()
        .with_standard_drop_block_prob(0.1);
}
