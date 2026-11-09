//! bunsen's Laya against upstream's, for every row, in float32: the prompts,
//! the encoder's states, the logits, and the calibrated answers.

use bunsen::{
    kits::{
        decision::laya::{
            LayaQuestionKind,
            LayaSpecials,
            build_prompt,
        },
        support::tokens::{
            BunsenTokenizer,
            Tokenizer,
        },
    },
    support::testing::{
        DeviceMemoryGuard,
        performance_device,
    },
};

use super::*;

/// The encoder's and the head's float32 tolerance against upstream's: two
/// float32 implementations of a 28-layer encoder, relative and absolute.
const ENCODER_TOL: (f64, f64) = (1e-3, 5e-3);

/// The logits' tolerance, relative and absolute: the act head's logits run
/// to thousands (its first act's probability is 1 to f32), where f32 holds
/// a few parts in 10^7.
const LOGITS_TOL: (f64, f64) = (1e-4, 5e-3);

/// The calibrated probabilities' tolerance, absolute.
const PROBABILITY_TOL: (f64, f64) = (0.0, 1e-3);

fn int_tensor<const D: usize>(
    values: Vec<i64>,
    shape: [usize; D],
    device: &Device,
) -> Tensor<D, Int> {
    Tensor::from_data(TensorData::new(values, shape), device)
}

/// Asserts `got` is `want`, each value within `atol + rtol * |want|`.
///
/// # Returns
/// The largest absolute difference.
fn close(
    what: &str,
    got: &[f64],
    want: &[f64],
    (rtol, atol): (f64, f64),
) -> f64 {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (g, w) in got.iter().zip(want) {
        assert!(
            (g - w).abs() <= atol + rtol * w.abs(),
            "{what}: {got:?} where upstream has {want:?}"
        );
    }
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0, f64::max)
}

fn to_f64(x: Tensor<2>) -> Vec<f64> {
    x.into_data()
        .convert::<f32>()
        .try_into_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect()
}

/// Every question of every case: the prompt is upstream's, exactly; the
/// encoder's states at the first token and the markers, the option logits
/// and the act logits are upstream's within tolerance, the question run
/// alone; and the case's questions decided in one padded batch give
/// upstream's calibrated probabilities.
fn check_row(row: &str) {
    let device = performance_device();
    let _memory = DeviceMemoryGuard::new(&device);
    let fixture = upstream_reference();
    let record = &fixture[row];
    let construct = load_construct(row, &device);
    let tokenizer = BunsenTokenizer::from_file(construct.tokenizer().expect("a tokenizer.json"))
        .expect("read the tokenizer");
    let config = std::fs::read_to_string(construct.tokenizer_config().expect("a config"))
        .expect("read the tokenizer config");
    let specials = LayaSpecials::from_tokenizer_config(&config, &tokenizer).expect("the specials");
    let agent = &construct.agent;
    let model = &construct.model;

    let (mut worst_encoder, mut worst_logits, mut worst_p) = (0.0f64, 0.0f64, 0.0f64);
    for (c, case) in record.cases.iter().enumerate() {
        for (q, (question, expected)) in case.questions.iter().zip(&case.prompts).enumerate() {
            let what = format!("{row} case {c} question {q}");
            let prompt = build_prompt(
                &tokenizer as &dyn Tokenizer,
                &specials,
                &case.state,
                question,
                agent.max_len,
                agent.head_max_len,
                false,
            )
            .unwrap();
            assert_eq!(prompt.ids, expected.ids, "{what}: ids");
            assert_eq!(prompt.markers, expected.markers, "{what}: markers");

            let (t, k) = (prompt.ids.len(), prompt.markers.len());
            let ids = int_tensor(prompt.ids.clone(), [1, t], &device);
            let rows: Vec<i64> = std::iter::once(0)
                .chain(prompt.markers.iter().map(|&m| m as i64))
                .collect();
            let states = model
                .encoder
                .forward(ids.clone(), None)
                .select(1, int_tensor(rows, [1 + k], &device))
                .squeeze_dim::<2>(0);
            let error = expected.encoder.assert_matches(
                &format!("{what}: encoder"),
                states,
                ENCODER_TOL.0,
                ENCODER_TOL.1,
            );
            worst_encoder = worst_encoder.max(error.abs);

            let markers = int_tensor(
                prompt.markers.iter().map(|&m| m as i64).collect(),
                [1, k],
                &device,
            );
            let options =
                Tensor::<2, Bool>::from_data(TensorData::new(vec![true; k], [1, k]), &device);
            let kind = int_tensor(vec![question.kind.index() as i64], [1], &device);
            let out = model.forward(ids, &[t], kind, markers, options);
            worst_logits = worst_logits.max(close(
                &format!("{what}: logits"),
                &to_f64(out.logits),
                &expected.logits,
                LOGITS_TOL,
            ));
            close(
                &format!("{what}: act logits"),
                &to_f64(out.act_logits),
                &expected.act_logits,
                LOGITS_TOL,
            );
        }

        let answers = model
            .decide(
                &tokenizer,
                &specials,
                agent,
                &case.state,
                &case.questions,
                &device,
            )
            .unwrap();
        for (q, (answer, expected)) in answers.iter().zip(&case.prompts).enumerate() {
            let what = format!("{row} case {c} question {q}");
            let got: Vec<f64> = answer.probabilities.iter().map(|(_, p)| *p).collect();
            worst_p = worst_p.max(close(
                &format!("{what}: probabilities"),
                &got,
                &expected.probabilities,
                PROBABILITY_TOL,
            ));
            close(
                &format!("{what}: act probability"),
                &[answer.act_probability],
                &[expected.act_probability],
                PROBABILITY_TOL,
            );
            if answer.kind == LayaQuestionKind::Noul {
                assert_eq!(answer.probabilities[1].0, "true", "{what}");
            }
        }
    }
    eprintln!(
        "{row}: worst encoder sample error {worst_encoder:.2e}, logit {worst_logits:.2e}, probability {worst_p:.2e}"
    );
}

#[test]
#[serial_test::serial]
fn laya() {
    check_row("laya");
}

#[test]
#[serial_test::serial]
fn laya_multilingual() {
    check_row("laya-multilingual");
}

#[test]
#[serial_test::serial]
fn laya_typed_decisions() {
    check_row("laya-typed-decisions");
}
