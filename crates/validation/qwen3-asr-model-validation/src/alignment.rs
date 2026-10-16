//! Where Qwen3-ASR's decoder looks while it writes each word.
//!
//! Qwen3-ASR does not time its output, but each reply token is predicted
//! under self-attention that reaches the window's audio placeholders, one per
//! encoder frame. These tests score every decoder head by how well the
//! placeholder it attends to most times each word, against
//! `testdata/jfk_moon_word_times.json`: Parakeet's word times on the same
//! clip (Parakeet's transducer times every token by construction).

use std::collections::BTreeMap;

use bunsen::{
    kits::speech::qwen3_asr::{
        Qwen3AsrBundle,
        Qwen3AsrFrontEndMeta,
    },
    support::testing::{
        DeviceMemoryGuard,
        performance_device,
    },
};
use serde::Deserialize;

use super::*;

/// The clip.
const CLIP: &str = "jfk_moon.mp3";

/// The windows decoded, in seconds.
const WINDOWS: [(usize, usize); 2] = [(0, 30), (30, 60)];

/// A reference word.
#[derive(Debug, Deserialize)]
struct TimedWord {
    word: String,
    start: f64,
}

/// `testdata/jfk_moon_word_times.json`.
#[derive(Debug, Deserialize)]
struct WordTimes {
    words: Vec<TimedWord>,
}

fn word_times() -> WordTimes {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/testdata/jfk_moon_word_times.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).expect("read the word times"))
        .expect("parse the word times")
}

/// A word for matching: lower case, letters and digits only.
fn normalize(word: &str) -> String {
    word.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// The longest common subsequence of `a` and `b`, as index pairs.
fn lcs_pairs(
    a: &[String],
    b: &[String],
) -> Vec<(usize, usize)> {
    let (n, m) = (a.len(), b.len());
    let mut t = vec![vec![0usize; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            t[i][j] = if a[i] == b[j] {
                t[i + 1][j + 1] + 1
            } else {
                t[i + 1][j].max(t[i][j + 1])
            };
        }
    }
    let (mut i, mut j, mut out) = (0, 0, Vec::new());
    while i < n && j < m {
        if a[i] == b[j] {
            out.push((i, j));
            i += 1;
            j += 1;
        } else if t[i + 1][j] >= t[i][j + 1] {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}

/// Seconds into the window of encoder frame `f`: a 1 s chunk of 100 mel
/// frames yields 13, each eight mel frames on from the last.
fn frame_seconds(f: usize) -> f64 {
    (f / FRAMES_PER_CHUNK) as f64 + ((f % FRAMES_PER_CHUNK) * 8 + 4) as f64 / 100.0
}

/// Monotonic alignment of rows to columns through `score` (`rows x cols`,
/// higher is better): the first column each row is aligned to.
fn dtw_starts(
    score: &[f64],
    rows: usize,
    cols: usize,
) -> Vec<usize> {
    let inf = f64::INFINITY;
    let mut cost = vec![inf; (rows + 1) * (cols + 1)];
    let mut step = vec![0u8; (rows + 1) * (cols + 1)];
    let at = |i: usize, j: usize| i * (cols + 1) + j;
    cost[0] = 0.0;
    for i in 1..=rows {
        for j in 1..=cols {
            let (diag, up, left) = (
                cost[at(i - 1, j - 1)],
                cost[at(i - 1, j)],
                cost[at(i, j - 1)],
            );
            let (best, s) = if diag <= up && diag <= left {
                (diag, 0)
            } else if up <= left {
                (up, 1)
            } else {
                (left, 2)
            };
            cost[at(i, j)] = best - score[(i - 1) * cols + (j - 1)];
            step[at(i, j)] = s;
        }
    }
    let mut starts = vec![0; rows];
    let (mut i, mut j) = (rows, cols);
    while i > 0 && j > 0 {
        starts[i - 1] = j - 1;
        match step[at(i, j)] {
            0 => {
                i -= 1;
                j -= 1;
            }
            1 => i -= 1,
            _ => j -= 1,
        }
    }
    starts
}

/// One window, decoded, with every layer's attention onto its audio.
struct Window {
    /// The window's start in the clip, in seconds.
    offset: f64,
    /// Per layer, `[heads, tokens, n_audio]` flattened.
    weights: Vec<Vec<f32>>,
    heads: usize,
    n_audio: usize,
    /// The transcript's tokens: their rows in `weights`.
    rows: Vec<usize>,
    /// The words: each its first token's index into `rows`, and its text.
    words: Vec<(usize, String)>,
}

fn decode_window(
    bundle: &Qwen3AsrBundle,
    samples: &[f32],
    (start, end): (usize, usize),
    device: &Device,
) -> Window {
    let rate = bundle.front_end.sample_rate();
    let piece = &samples[start * rate..(end * rate).min(samples.len())];
    let (features, valid) = bundle.front_end.forward(piece, device).unwrap();
    let reply = bundle
        .transcribe_features(features.clone(), valid)
        .unwrap()
        .tokens;
    let layers: Vec<usize> = (0..bundle.model.language_model.layers.len()).collect();
    let attention = bundle
        .audio_attention(features, valid, &reply, &layers)
        .unwrap();
    let [heads, _, n_audio] = attention[0].dims();
    let weights = attention
        .into_iter()
        .map(|w| w.into_data().convert::<f32>().try_to_vec().unwrap())
        .collect();

    let (_, text) = bundle.prompt.split_reply(&reply, &bundle.decoder.eos);
    let first = reply
        .iter()
        .position(|&t| t == bundle.prompt.text_token)
        .map_or(0, |at| at + 1);
    let rows: Vec<usize> = (first..first + text.len()).collect();
    let mut words: Vec<(usize, String)> = Vec::new();
    for (k, &t) in text.iter().enumerate() {
        let piece = bundle.vocab.text(&[t]).unwrap();
        match words.last_mut() {
            Some((_, word)) if !piece.starts_with(' ') => word.push_str(&piece),
            _ => words.push((k, piece.trim().to_string())),
        }
    }
    Window {
        offset: start as f64,
        weights,
        heads,
        n_audio,
        rows,
        words,
    }
}

/// The signed errors of `starts` (each word's start in the clip) against
/// the reference words of `window`, matched by text.
fn errors(
    window: &Window,
    starts: &[f64],
    reference: &WordTimes,
) -> Vec<f64> {
    let span = window.offset..window.offset + 30.0;
    let theirs: Vec<&TimedWord> = reference
        .words
        .iter()
        .filter(|w| span.contains(&w.start))
        .collect();
    let a: Vec<String> = window.words.iter().map(|(_, w)| normalize(w)).collect();
    let b: Vec<String> = theirs.iter().map(|w| normalize(&w.word)).collect();
    lcs_pairs(&a, &b)
        .into_iter()
        .map(|(i, j)| starts[i] - theirs[j].start)
        .collect()
}

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.total_cmp(b));
    xs[xs.len() / 2]
}

/// The median error, and the 90th percentile of the distances from it.
fn bias_and_spread(errors: &[f64]) -> (f64, f64) {
    let bias = median(&mut errors.to_vec());
    let mut spread: Vec<f64> = errors.iter().map(|e| (e - bias).abs()).collect();
    spread.sort_by(|a, b| a.total_cmp(b));
    (bias, spread[spread.len() * 9 / 10])
}

/// Each word's start in the clip under `frames`, a frame per transcript
/// token.
fn word_starts(
    window: &Window,
    frames: &[usize],
) -> Vec<f64> {
    window
        .words
        .iter()
        .map(|&(k, _)| window.offset + frame_seconds(frames[k]))
        .collect()
}

/// The heads whose attention times words best on the base model, by
/// [`test_measure_attention_alignment`]'s ranking: `(layer, head)`.
const HEADS: [(usize, usize); 8] = [
    (25, 10),
    (18, 5),
    (22, 7),
    (23, 14),
    (25, 5),
    (24, 15),
    (16, 14),
    (19, 5),
];

/// The words' errors when `heads`' rows, each renormalized over the audio,
/// are averaged and aligned to the audio monotonically.
fn aligned_errors(
    windows: &[Window],
    heads: &[(usize, usize)],
    reference: &WordTimes,
) -> Vec<f64> {
    let mut all = Vec::new();
    for w in windows {
        let tokens = w.weights[0].len() / (w.heads * w.n_audio);
        let rows = w.rows.len();
        let mut score = vec![0.0f64; rows * w.n_audio];
        for &(layer, head) in heads {
            for (i, &r) in w.rows.iter().enumerate() {
                let row = &w.weights[layer]
                    [(head * tokens + r) * w.n_audio..(head * tokens + r + 1) * w.n_audio];
                let sum: f32 = row.iter().sum();
                for (j, &x) in row.iter().enumerate() {
                    score[i * w.n_audio + j] += (x / sum.max(1e-9)) as f64 / heads.len() as f64;
                }
            }
        }
        let frames = dtw_starts(&score, rows, w.n_audio);
        all.extend(errors(w, &word_starts(w, &frames), reference));
    }
    all
}

/// Reports `errors` under `label`, and holds them to the gate: 90% of the
/// words within 0.28 s of the reference, once a constant offset is taken
/// out, and at least 70% within 0.16 s.
fn report_and_gate(
    label: &str,
    errors: &[f64],
    matched_at_least: usize,
) {
    let (bias, p90) = bias_and_spread(errors);
    let within = errors.iter().filter(|e| (*e - bias).abs() <= 0.16).count();
    eprintln!(
        "{label}: {} words matched, bias {bias:+.3} s, p90 {p90:.3} s, {within} within 0.16 s",
        errors.len()
    );
    assert!(
        errors.len() >= matched_at_least,
        "{label}: {} matched",
        errors.len()
    );
    assert!(p90 <= 0.28 + 1e-9, "{label}: p90 {p90}");
    assert!(
        within * 10 >= errors.len() * 7,
        "{label}: {within} within 0.16 s"
    );
}

/// Scores every head of the base model, and holds [`HEADS`] to the gate.
///
/// Measured on wgpu: a dozen heads in layers 16 to 26 put 90% of the words
/// within 0.20 s; [`HEADS`] averaged, 0.24 s, 83 of 114 within 0.16 s.
#[test]
#[serial_test::serial]
fn test_measure_attention_alignment() {
    let device = performance_device();
    let _memory = DeviceMemoryGuard::new(&device);
    let bundle = load_bundle("qwen/qwen3-asr-0.6b", &device);
    let samples = speech_clip(CLIP);
    let reference = word_times();
    let windows: Vec<Window> = WINDOWS
        .iter()
        .map(|&w| decode_window(&bundle, &samples, w, &device))
        .collect();
    let n_layers = windows[0].weights.len();
    let heads = windows[0].heads;

    // Per head: each token at the placeholder it attends to most.
    let mut scores: BTreeMap<(usize, usize), (f64, f64, usize)> = BTreeMap::new();
    for layer in 0..n_layers {
        for head in 0..heads {
            let mut all = Vec::new();
            for w in &windows {
                let tokens = w.weights[layer].len() / (heads * w.n_audio);
                let frames: Vec<usize> = w
                    .rows
                    .iter()
                    .map(|&r| {
                        let row = &w.weights[layer]
                            [(head * tokens + r) * w.n_audio..(head * tokens + r + 1) * w.n_audio];
                        (0..w.n_audio)
                            .max_by(|&a, &b| row[a].total_cmp(&row[b]))
                            .unwrap()
                    })
                    .collect();
                all.extend(errors(w, &word_starts(w, &frames), &reference));
            }
            let (bias, spread) = bias_and_spread(&all);
            scores.insert((layer, head), (spread, bias, all.len()));
        }
    }
    let mut ranked: Vec<_> = scores.iter().collect();
    ranked.sort_by(|a, b| a.1.0.total_cmp(&b.1.0));
    eprintln!("head (layer, head): p90 distance from the bias s, bias s, words matched");
    for ((layer, head), (spread, bias, n)) in ranked.iter().take(16) {
        eprintln!("  ({layer:2}, {head:2}): {spread:.3} {bias:+.3} {n}");
    }

    report_and_gate(
        "base, HEADS",
        &aligned_errors(&windows, &HEADS, &reference),
        100,
    );
}

/// Phonon-1's builds re-pack the base model's weights, and its heads time
/// words as well.
#[test]
#[serial_test::serial]
fn test_phonon1_builds_align_on_the_base_heads() {
    let device = performance_device();
    let samples = speech_clip(CLIP);
    let reference = word_times();
    for name in [
        "fermion/phonon-1",
        "fermion/phonon-1-micro",
        "fermion/phonon-1-big",
    ] {
        let _memory = DeviceMemoryGuard::new(&device);
        let bundle = load_bundle(name, &device);
        let windows: Vec<Window> = WINDOWS
            .iter()
            .map(|&w| decode_window(&bundle, &samples, w, &device))
            .collect();
        report_and_gate(name, &aligned_errors(&windows, &HEADS, &reference), 100);
    }
}

#[test]
fn test_lcs_pairs() {
    let s = |xs: &[&str]| xs.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    assert_eq!(
        lcs_pairs(&s(&["a", "b", "c", "d"]), &s(&["a", "x", "c", "d"])),
        [(0, 0), (2, 2), (3, 3)]
    );
}

#[test]
fn test_dtw_starts() {
    // Three rows, each peaked on its own column, in order.
    let score = [
        1.0, 0.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0,
    ];
    assert_eq!(dtw_starts(&score, 3, 4), [0, 2, 3]);
}
