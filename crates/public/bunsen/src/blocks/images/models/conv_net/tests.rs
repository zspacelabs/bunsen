use alloc::{
    vec,
    vec::Vec,
};

use burn::{
    nn::{
        PaddingConfig2d,
        conv::Conv2dConfig,
        pool::MaxPool2dConfig,
    },
    prelude::Tensor,
    tensor::Distribution,
};

use super::*;
use crate::{
    blocks::{
        activation::BunsenActivationConfig,
        conv::{
            ConvBlock2dConfig,
            ConvBlock2dMeta,
        },
        images::{
            anti_alias::AntiAliasConfig,
            attention::{
                ChannelAttentionConfig,
                SqueezeExciteConfig,
            },
        },
    },
    burner::module::{
        ModuleInit,
        ToStructureConfig,
    },
    errors::{
        BunsenResult,
        ConstraintError,
        testing::ErrorMatcher,
    },
    ops::drop::DropBlockOptions,
    support::testing::{
        cpu_device,
        expect_variant,
    },
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
        ConvUnitKind::Basic | ConvUnitKind::PreBasic => 1,
        ConvUnitKind::Bottleneck | ConvUnitKind::PreBottleneck => 4,
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
    expect_variant!(let ResidualConvConfig::Block(block) = conv, "a conv block");
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
    let Some(ConvShortcutConfig {
        pool: None,
        block: shortcut,
    }) = &down.shortcut
    else {
        panic!("a strided unit has a conv shortcut");
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
    expect_variant!(
        let ResidualConvConfig::Block(block) = &structure.stages[3].units[1].convs[0],
        "a conv block",
    );
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

/// The average-pool shortcut is timm's `downsample_avg`: no pool where the
/// unit only widens, a `2x2` ceil-mode pool of its stride where it strides,
/// and a bottom-right padded `2x2` pool of stride 1 where a dilated stage
/// would have strided. Its conv is a `1x1` at stride 1; odd sizes halve up.
#[test]
fn test_avg_conv_shortcuts_are_timms() {
    let mut policy = resnet_like(&[1, 1, 1], 8, ConvUnitKind::Bottleneck).with_output_stride(8);
    policy.layers.shortcut = ConvShortcutKind::AvgConv;
    let structure = policy.to_structure();
    let pools: Vec<_> = structure
        .stages
        .iter()
        .map(|stage| {
            expect_variant!(let Some(shortcut) = &stage.units[0].shortcut, "an avg-conv shortcut");
            assert_eq!(shortcut.block.stride(), [1, 1]);
            shortcut.pool.as_ref().map(|p| {
                (
                    p.strides[0],
                    p.ceil_mode,
                    p.count_include_pad,
                    format!("{:?}", p.padding),
                )
            })
        })
        .collect();
    assert_eq!(
        pools,
        [
            None,
            Some((2, true, false, "Valid".into())),
            Some((1, false, true, "Explicit(0, 0, 1, 1)".into())),
        ]
    );

    let device = cpu_device();
    let model: ConvNetModel = structure.init(&device);
    let maps = model.stage_maps(Tensor::<4>::ones([1, 3, 36, 36], &device));
    let sizes: Vec<_> = maps.iter().map(|m| m.dims()[2]).collect();
    // The stem's 18, its pool's 9; stage 1 halves 9 up to 5; stage 2 dilates.
    assert_eq!(sizes, [18, 9, 5, 5]);
}

/// A malformed tree is an error from `try_init`, not a panic: a head that
/// doesn't read the pooled features (a net of no stages pools its stem's),
/// an empty stage, a stage that does not chain, a mixer past the path, a
/// widening unit with no shortcut.
#[test]
fn test_try_init_rejects_bad_trees() {
    let device = cpu_device();
    let structure = resnet_like(&[1, 1], 8, ConvUnitKind::Basic).to_structure();
    let illegal = || ErrorMatcher::illegal().has_cause::<ConstraintError>();
    let init = |s: ConvNetStructureConfig| -> BunsenResult<ConvNetModel> { s.try_init(&device) };

    illegal()
        .message_contains("ConvNetStructureConfig.head")
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

/// The model's stochastic-depth schedule is timm's.
#[test]
fn test_drop_path_schedules_match_timm() {
    let model: ConvNetModel =
        resnet_like(&[2, 2, 2, 2], 8, ConvUnitKind::Basic).init(&cpu_device());
    assert_rates_eq(
        &module_drop_path_probs(&model.with_stochastic_path_depth(0.1)),
        &TIMM_DROP_PATH_2222_AT_0_1,
    );

    // Two units: timm gives `[0.0, 0.1]`; one unit, 0 (timm divides by
    // zero).
    let model: ConvNetModel = resnet_like(&[1, 1], 8, ConvUnitKind::Basic).init(&cpu_device());
    assert_rates_eq(
        &module_drop_path_probs(&model.with_stochastic_path_depth(0.1)),
        &[0.0, 0.1],
    );
    let model: ConvNetModel = resnet_like(&[1], 8, ConvUnitKind::Basic).init(&cpu_device());
    assert_rates_eq(
        &module_drop_path_probs(&model.with_stochastic_path_depth(0.1)),
        &[0.0],
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

fn module_drop_blocks(model: &ConvNetModel) -> Vec<Option<DropBlockOptions>> {
    model
        .stages
        .iter()
        .flat_map(|stage| stage.units.iter())
        .map(|unit| unit.drop_block.as_ref().map(|d| d.options.clone()))
        .collect()
}

/// The model's `DropBlock` schedule is timm's.
#[test]
fn test_drop_block_schedules_match_timm() {
    let device = cpu_device();
    let model: ConvNetModel = resnet_like(&[2, 2, 2, 2], 8, ConvUnitKind::Basic).init(&device);
    let model = model.with_stochastic_drop_block(0.1);
    assert_eq!(module_drop_blocks(&model), timm_drop_blocks_2222_at_0_1());

    // A model with drop blocks still runs (in eval they are identities).
    let logits = model.forward(Tensor::<4>::zeros([1, 3, 64, 64], &device));
    assert_eq!(logits.dims(), [1, 7]);
}

/// With fewer than 2 stages there is no second-to-last stage for the
/// `DropBlock` schedule: a nonzero rate is an error; a rate of 0 places
/// nothing.
#[test]
fn test_drop_block_schedules_reject_one_stage() {
    let model: ConvNetModel = resnet_like(&[2], 8, ConvUnitKind::Basic).init(&cpu_device());

    ErrorMatcher::illegal()
        .constraint("standard_drop_block_options", "num_stages")
        .assert_err(&model.clone().try_with_stochastic_drop_block(0.1));
    ErrorMatcher::illegal()
        .frame_contains("drop_prob")
        .assert_err(&model.clone().try_with_stochastic_drop_block(1.5));

    let model = model.try_with_stochastic_drop_block(0.0).unwrap();
    assert_eq!(module_drop_blocks(&model), vec![None, None]);
}

/// The panicking twin panics with a message that names the problem.
#[test]
#[should_panic(expected = "standard_drop_block_options.num_stages")]
fn test_drop_block_schedule_panics_on_one_stage() {
    let model: ConvNetModel = resnet_like(&[2], 8, ConvUnitKind::Basic).init(&cpu_device());
    let _ = model.with_stochastic_drop_block(0.1);
}

/// `RegNet`'s attention: a squeeze-excite after the mixer, over the mixer's
/// channels, its reduction `round(in * se_ratio)` of the unit's input and its
/// activation the layers'.
#[test]
fn test_mixer_attention_lowers_as_regnets() {
    let mut policy = ConvNetContractConfig::new(
        ConvStemContractConfig::new(vec![8])
            .with_kernel_size(3)
            .with_pool(ConvStemPoolKind::None),
        vec![ConvStageContractConfig::new(
            2,
            ConvUnitKind::Bottleneck,
            24,
            24,
        )],
        5,
    );
    policy.layers = ConvLayersConfig::new()
        .with_act(BunsenActivationConfig::Silu)
        .with_attn(Some(ConvAttentionKind::SeFromInput { se_ratio: 0.25 }))
        .with_attn_place(ConvAttentionPlace::Mixer);
    let structure = policy.to_structure();
    for (unit, rd) in structure.stages[0].units.iter().zip([2, 6]) {
        assert_eq!(unit.attn_place, ConvAttentionPlace::Mixer);
        expect_variant!(let Some(ChannelAttentionConfig::Se(se)) = &unit.attn, "a squeeze-excite");
        assert_eq!((se.channels, se.rd_channels), (24, rd));
        assert!(matches!(se.act, BunsenActivationConfig::Silu));
    }

    let mut wrong = structure.stages[0].units[0].clone();
    wrong.attn = Some(ChannelAttentionConfig::Se(SqueezeExciteConfig::new(8, 2)));
    ErrorMatcher::illegal()
        .message_contains("mixer out_channels")
        .assert_err(&wrong.try_validate());
}

/// A unit's attention after the mixer is the path run with the attention
/// between the mixer and the last conv.
#[test]
fn test_mixer_attention_forward() {
    let device = cpu_device();
    let unit: ResidualUnit = ResidualUnitConfig::new(
        vec![
            ConvBlock2dConfig::new(Conv2dConfig::new([4, 6], [1, 1]))
                .with_norm(None)
                .with_act(Some(BunsenActivationConfig::Relu))
                .into(),
            ConvBlock2dConfig::new(
                Conv2dConfig::new([6, 6], [3, 3])
                    .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1, 1, 1)),
            )
            .with_norm(None)
            .with_act(Some(BunsenActivationConfig::Relu))
            .into(),
            ConvBlock2dConfig::new(Conv2dConfig::new([6, 4], [1, 1]))
                .with_norm(None)
                .into(),
        ],
        1,
    )
    .with_attn(Some(ChannelAttentionConfig::Se(SqueezeExciteConfig::new(
        6, 2,
    ))))
    .with_attn_place(ConvAttentionPlace::Mixer)
    .init(&device);
    let x = crate::support::testing::seeded_tensor::<4>(
        7,
        [2, 4, 5, 5],
        burn::tensor::Distribution::Default,
        &device,
    );
    let mut path = x.clone();
    for (idx, conv) in unit.convs.iter().enumerate() {
        path = conv.forward(path);
        if idx == 1 {
            path = unit.attn.as_ref().unwrap().forward(path);
        }
    }
    let want = burn::tensor::activation::relu(path + x.clone());
    crate::support::testing::assert_tensors_close(
        &unit.forward(x),
        &want,
        burn::tensor::Tolerance::default(),
    );
}

/// A pre-activation bottleneck: the unit's input normed and activated once,
/// each later conv after its own norm, a shortcut conv with no norm, no
/// activation after the add. A `None` shortcut drops the residual of a unit
/// that changes shape; a unit with no residual takes no shortcut.
#[test]
fn test_preact_and_unshortcut_units() {
    let stages = vec![ConvStageContractConfig::new(
        2,
        ConvUnitKind::PreBottleneck,
        16,
        8,
    )];
    let policy = ConvNetContractConfig::new(
        ConvStemContractConfig::new(vec![8])
            .with_kernel_size(3)
            .with_pool(ConvStemPoolKind::None)
            .with_last_norm_act(false),
        stages,
        5,
    )
    .with_final_conv(Some(ConvFinalKind::Act));
    let structure = policy.to_structure();
    let stem = &structure.stem.convs[0];
    assert!(stem.norm.is_none() && stem.act.is_none());
    let unit = &structure.stages[0].units[0];
    assert!(unit.pre.is_some());
    assert!(unit.act.is_none());
    let norms: Vec<_> = unit
        .convs
        .iter()
        .map(|c| match c {
            ResidualConvConfig::PreAct(conv) => conv.norm.is_some(),
            other => panic!("a pre-activation conv, not {other:?}"),
        })
        .collect();
    assert_eq!(norms, [false, true, true]);
    assert!(unit.shortcut.as_ref().unwrap().block.norm.is_none());
    assert!(matches!(
        structure.final_conv,
        Some(ConvFinalConfig::Act(_))
    ));
    let device = cpu_device();
    let model: ConvNetModel = structure.init(&device);
    let images = Tensor::<4>::zeros([1, 3, 16, 16], &device);
    assert_eq!(model.forward(images).dims(), [1, 5]);

    let mut policy = resnet_like(&[1, 1], 8, ConvUnitKind::Bottleneck);
    policy.layers.shortcut = ConvShortcutKind::None;
    let structure = policy.to_structure();
    let reshaping = &structure.stages[0].units[0];
    assert!(!reshaping.residual && reshaping.shortcut.is_none());
    let mut wrong = structure.stages[1].units[0].clone();
    wrong.shortcut = structure.stages[0].units[0].shortcut.clone().or_else(|| {
        resnet_like(&[1], 8, ConvUnitKind::Bottleneck)
            .to_structure()
            .stages[0]
            .units[0]
            .shortcut
            .clone()
    });
    ErrorMatcher::illegal()
        .message_contains("a unit with no residual has no shortcut")
        .assert_err(&wrong.try_validate());
}

/// `ResNetV2`'s pieces: a pre-activation basic unit (a bare first conv, the
/// second after its norm) whose identity shortcut reads the raw input under
/// `pre_identity: false`; the zero-padded stem pool; a final norm-act; weight
/// standardization on every conv.
#[test]
fn test_resnetv2_pieces() {
    let mut policy = ConvNetContractConfig::new(
        ConvStemContractConfig::new(vec![8])
            .with_pool(ConvStemPoolKind::ZeroPadMax)
            .with_last_norm_act(false),
        vec![ConvStageContractConfig::new(2, ConvUnitKind::PreBasic, 8, 8).with_stride(1)],
        5,
    )
    .with_final_conv(Some(ConvFinalKind::NormAct));
    policy.layers.pre_identity = false;
    policy.layers.weight_std = Some(1e-8);
    let structure = policy.to_structure();
    assert!(matches!(
        structure.stem.pool,
        Some(ConvStemPoolConfig::ZeroPadMax(_))
    ));
    assert_eq!(structure.stem.convs[0].weight_std, Some(1e-8));
    let unit = &structure.stages[0].units[0];
    assert!(unit.pre.is_some() && unit.shortcut.is_none() && !unit.pre_identity);
    assert_eq!(unit.mixer, 0);
    expect_variant!(
        let ResidualConvConfig::PreAct(first) = &unit.convs[0],
        "a pre-activation conv",
    );
    assert!(first.norm.is_none() && first.weight_std == Some(1e-8));
    assert!(matches!(
        structure.final_conv,
        Some(ConvFinalConfig::NormAct(_))
    ));

    // The identity reads the raw input: with every conv zeroed, the unit is
    // its input.
    let device = cpu_device();
    let unit: ResidualUnit = unit.init(&device);
    let mut zeroed = unit.clone();
    for conv in &mut zeroed.convs {
        if let ResidualConv::PreAct(conv) = conv {
            conv.conv.weight = conv.conv.weight.clone().map(|w| w.zeros_like());
            conv.weight_std = None;
        }
    }
    let x = Tensor::<4>::random([1, 8, 4, 4], Distribution::Default, &device);
    zeroed
        .forward(x.clone())
        .into_data()
        .assert_eq(&x.into_data(), true);

    let model: ConvNetModel = structure.init(&device);
    let images = Tensor::<4>::zeros([1, 3, 16, 16], &device);
    assert_eq!(model.forward(images).dims(), [1, 5]);
}

/// The hybrid `ViT`'s backbone pieces: every conv SAME-padded (unpadded
/// itself, its input padded from its own size), the SAME-padded max pool,
/// and the map's size on odd images, `ceil(side / stride)` at every stride.
#[test]
fn test_same_padded_net() {
    let mut policy = ConvNetContractConfig::new(
        ConvStemContractConfig::new(vec![8]).with_pool(ConvStemPoolKind::SameMax),
        vec![ConvStageContractConfig::new(
            1,
            ConvUnitKind::Bottleneck,
            16,
            8,
        )],
        0,
    );
    policy.layers.same_pad = true;
    policy.layers.weight_std = Some(1e-8);
    let structure = policy.to_structure();
    assert!(matches!(
        structure.stem.pool,
        Some(ConvStemPoolConfig::SameMax(_))
    ));
    let unit = &structure.stages[0].units[0];
    let blocks = structure.stem.convs.iter().chain(
        unit.convs
            .iter()
            .map(|c| match c {
                ResidualConvConfig::Block(block) => block,
                other => panic!("a conv block, not {other:?}"),
            })
            .chain(unit.shortcut.as_ref().map(|s| &s.block)),
    );
    for block in blocks {
        assert!(block.same_pad && matches!(block.conv.padding, PaddingConfig2d::Valid));
    }
    // 19: the stem's stride-2 conv, 10; its pool, 5; the stage, 3.
    assert_eq!(structure.try_feature_size([19, 21]).unwrap(), [3, 3]);
    let device = cpu_device();
    let model: ConvNetModel = structure.init(&device);
    let images = Tensor::<4>::random([1, 3, 19, 21], Distribution::Default, &device);
    assert_eq!(model.feature_map(images).dims(), [1, 16, 3, 3]);
}

/// The SAME-padded max pool pads with `-inf`: over a negative map, the pad
/// never wins, where a zero pad would.
#[test]
fn test_same_max_pool_pads_with_neg_infinity() {
    let device = cpu_device();
    let pool: ConvStemPool =
        ConvStemPoolConfig::SameMax(MaxPool2dConfig::new([3, 3]).with_strides([2, 2]))
            .init(&device);
    let x = Tensor::<4>::ones([1, 2, 6, 5], &device).neg();
    let y = pool.forward(x);
    assert_eq!(y.dims(), [1, 2, 3, 3]);
    y.into_data().assert_eq(
        &Tensor::<4>::ones([1, 2, 3, 3], &device).neg().into_data(),
        true,
    );
}

/// A net of no stages is its stem: the hybrid `ViT`'s smallest backbone.
#[test]
fn test_stem_only_net() {
    let policy = ConvNetContractConfig::new(
        ConvStemContractConfig::new(vec![8]).with_pool(ConvStemPoolKind::SameMax),
        vec![],
        0,
    )
    .with_layers(ConvLayersConfig::new().with_same_pad(true));
    let structure = policy.to_structure();
    assert_eq!(structure.d_features(), 8);
    assert_eq!(structure.try_feature_size([32, 32]).unwrap(), [8, 8]);
    let device = cpu_device();
    let model: ConvNetModel = structure.init(&device);
    assert_eq!(model.d_features(), 8);
    let images = Tensor::<4>::zeros([2, 3, 32, 32], &device);
    assert_eq!(model.feature_map(images.clone()).dims(), [2, 8, 8, 8]);
    assert_eq!(model.forward(images).dims(), [2, 8]);
}

/// `Res2Net`'s mixer: a bottleneck's `3x3` becomes the scale split, its
/// branches apart and its last chunk pooled in a unit that strides or
/// reshapes, hierarchical in the others; it is a bottleneck's alone.
#[test]
fn test_res2_mixer() {
    let mut policy = resnet_like(&[2, 1], 8, ConvUnitKind::Bottleneck);
    policy.stages[0].mid_width = 12;
    policy.stages[1].mid_width = 12;
    policy.layers.mixer = ConvMixerKind::Res2 { scale: 3 };
    let structure = policy.to_structure();
    let res2 = |unit: &ResidualUnitConfig| match &unit.convs[1] {
        ResidualConvConfig::Res2(res2) => res2.clone(),
        other => panic!("a scale split, not {other:?}"),
    };
    let first = res2(&structure.stages[0].units[0]);
    assert_eq!((first.scale, first.width(), first.convs.len()), (3, 4, 2));
    assert!(!first.hierarchical && first.pool.as_ref().is_some_and(|p| p.strides == [1, 1]));
    let second = res2(&structure.stages[0].units[1]);
    assert!(second.hierarchical && second.pool.is_none());
    let strided = res2(&structure.stages[1].units[0]);
    assert!(strided.pool.as_ref().is_some_and(|p| p.strides == [2, 2]));
    let device = cpu_device();
    let model: ConvNetModel = structure.init(&device);
    let images = Tensor::<4>::random([1, 3, 33, 33], Distribution::Default, &device);
    assert_eq!(model.forward(images).dims(), [1, 7]);

    let mut basic = resnet_like(&[1], 8, ConvUnitKind::Basic);
    basic.layers.mixer = ConvMixerKind::Res2 { scale: 2 };
    ErrorMatcher::unsupported()
        .message_contains("a Res2 { scale: 2 } mixer in a Basic unit")
        .assert_err(&basic.try_to_structure());
    let mut uneven = resnet_like(&[1], 8, ConvUnitKind::Bottleneck);
    uneven.layers.mixer = ConvMixerKind::Res2 { scale: 3 };
    ErrorMatcher::illegal()
        .message_contains("a scale split of 3 chunks from 8 to 8 channels")
        .assert_err(&uneven.try_to_structure());
}

/// `ResNeSt`'s mixer: a bottleneck's `3x3` becomes the split attention in
/// the stage's groups; its `avd` pool takes the stride where a unit strides
/// 2 (not the first stage's reshaping unit), after the mixer or before it.
#[test]
fn test_split_attn_mixer_and_avd() {
    for aa_first in [false, true] {
        let mut policy = resnet_like(&[1, 1], 8, ConvUnitKind::Bottleneck);
        policy.layers.mixer = ConvMixerKind::SplitAttn {
            radix: 2,
            rd_ratio: 0.25,
            rd_divisor: 8,
        };
        policy.layers.aa = Some(ConvAaKind::Avd);
        policy.layers.aa_first = aa_first;
        let structure = policy.to_structure();
        let first = &structure.stages[0].units[0];
        assert!(first.aa.is_none(), "the first stage doesn't stride");
        let strided = &structure.stages[1].units[0];
        assert!(strided.aa_first == aa_first);
        expect_variant!(let Some(AntiAliasConfig::Avg(pool)) = &strided.aa, "an avd pool");
        assert_eq!((pool.kernel_size, pool.strides), ([3, 3], [2, 2]));
        assert!(pool.count_include_pad);
        expect_variant!(
            let ResidualConvConfig::SplitAttn(attn) = &strided.convs[1],
            "a split attention",
        );
        assert_eq!((attn.radix, attn.stride(), attn.out_channels()), (2, 1, 16));
        let device = cpu_device();
        let model: ConvNetModel = structure.init(&device);
        let images = Tensor::<4>::random([1, 3, 33, 33], Distribution::Default, &device);
        assert_eq!(model.forward(images).dims(), [1, 7]);
    }
}

/// `SKNet`'s mixer: a basic unit's first conv, a bottleneck's `3x3`, as the
/// selective kernel; its input split across the paths or not.
#[test]
fn test_selective_kernel_mixer() {
    let device = cpu_device();
    for (unit, idx, split_input) in [
        (ConvUnitKind::Basic, 0, true),
        (ConvUnitKind::Bottleneck, 1, false),
    ] {
        let mut policy = resnet_like(&[1, 1], 16, unit);
        policy.layers.mixer = ConvMixerKind::SelectiveKernel {
            kernel_size: [3, 5],
            keep_3x3: true,
            rd_ratio: 0.125,
            rd_divisor: 16,
            split_input,
        };
        let structure = policy.to_structure();
        let strided = &structure.stages[1].units[0];
        expect_variant!(
            let ResidualConvConfig::SelectiveKernel(sk) = &strided.convs[idx],
            "{unit:?}: a selective kernel",
        );
        assert_eq!(
            (sk.stride(), sk.paths.len(), sk.split_input),
            (2, 2, split_input)
        );
        let model: ConvNetModel = structure.init(&device);
        let images = Tensor::<4>::random([1, 3, 33, 33], Distribution::Default, &device);
        assert_eq!(model.forward(images).dims(), [1, 7], "{unit:?}");
    }
}
