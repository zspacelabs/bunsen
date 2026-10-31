use bunsen::{
    data::pretrained::SafetensorsCheckpoint,
    kits::speech::whisper::pretrained::{
        PytorchWhisperScanner,
        SafetensorsWhisperScanner,
    },
    support::testing::cpu_device,
};
use burn::store::{
    ModuleSnapshot,
    bridge,
};

/// `openai/whisper-base`'s `model.safetensors` is `transformers`' conversion
/// of `base.pt`, so through the kit's `transformers` loader it is `base.pt`
/// through the `.pt` loader: the same geometry, and every parameter the same
/// value, exactly.
#[test]
fn test_the_transformers_checkpoint_is_base_pt() {
    let device = cpu_device();
    let (pt, pt_cfg) = PytorchWhisperScanner::new()
        .load(bind_whisper_onnx::base_pt(), &device)
        .unwrap();
    let (hf, hf_cfg) = SafetensorsWhisperScanner::new()
        .load(
            &SafetensorsCheckpoint::single(bind_whisper_onnx::base_safetensors()),
            &device,
        )
        .unwrap();
    assert_eq!(hf_cfg.geometry(), pt_cfg.geometry());

    let params = |m: &bunsen::kits::speech::whisper::Whisper| {
        let mut ps = m.collect(None, None, true);
        ps.sort_by(|a, b| a.name.cmp(&b.name));
        ps
    };
    let (pt, hf) = (params(&pt), params(&hf));
    assert_eq!(
        hf.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(),
        pt.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(),
    );
    // `base.pt` stores `f16`; `transformers`' conversion widened it to
    // `f32`, which is exact, so the values agree once both are `f32`.
    for (a, b) in pt.iter().zip(&hf) {
        let f32_of = |t| bridge::to_data(t).unwrap().convert::<f32>();
        f32_of(b).assert_eq(&f32_of(a), true);
    }
}
