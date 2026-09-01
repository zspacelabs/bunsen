//! # Batching: many streams, one decode.
//!
//! Server-batch mode is a function, not a type. The contexts are the state;
//! [`advance_ready`] gathers one due window from each of them, decodes the
//! lot in a single pass, and hands each context its own result to commit.
//! Nothing about a stream changes because it was batched: it commits the
//! same windows at the same times with the same ids it would have alone.
//!
//! Windows are batched by prompt, because a decode batch shares one. With
//! prompt carry off &mdash; which is how upstream batches too &mdash; every
//! stream's prompt is the same and one batch covers them all; with it on,
//! only first windows batch, and the rest go one per pass.

use burn::{
    Tensor,
    prelude::Backend,
};

use crate::{
    errors::BunsenResult,
    kits::speech::whisper::{
        driver::{
            WhisperDriver,
            WhisperStreamContext,
            context::Due,
        },
        emission::Emission,
    },
};

/// One context's due unit, packaged and ready to join a batch.
struct Pending<B: Backend> {
    /// Index into the contexts being advanced.
    context: usize,
    unit: Due,
    /// The prompt it decodes under; what it is batched by.
    prompt: Vec<i64>,
    /// `[1, n_mels, width]`.
    window: Tensor<B, 3>,
}

/// Advances every context that has a decode due, batching the decodes.
///
/// Repeats until no context has anything due, so a context with several
/// windows waiting gets them all. Returns each context's emissions, in the
/// order of `contexts`.
///
/// # Arguments
/// * `driver` - the driver the contexts were opened from.
/// * `contexts` - the streams, fed through [`feed`](WhisperStreamContext::feed)
///   rather than [`push`](WhisperStreamContext::push), so that nothing has been
///   decoded yet.
///
/// # Errors
/// As [`WhisperStreamContext::advance`].
pub fn advance_ready<B: Backend>(
    driver: &WhisperDriver<B>,
    contexts: &mut [WhisperStreamContext<B>],
) -> BunsenResult<Vec<Vec<Emission>>> {
    let mut out: Vec<Vec<Emission>> = vec![Vec::new(); contexts.len()];

    loop {
        // One due unit per context, with the prompt it would decode under.
        let mut pending: Vec<Option<Pending<B>>> = Vec::new();
        for (i, ctx) in contexts.iter_mut().enumerate() {
            ctx.skip_silence();
            if let Some(unit) = ctx.next_due() {
                let frames = ctx.frames_at(&unit);
                ctx.ensure_language(&frames);
                pending.push(Some(Pending {
                    context: i,
                    unit,
                    prompt: ctx.prompt_now(),
                    window: ctx.package_padded(frames),
                }));
            }
        }
        if pending.is_empty() {
            return Ok(out);
        }

        // Group by prompt, in order of first appearance.
        let mut groups: Vec<(Vec<i64>, Vec<usize>)> = Vec::new();
        for (k, item) in pending.iter().enumerate() {
            let prompt = &item.as_ref().expect("not yet taken").prompt;
            match groups.iter_mut().find(|(p, _)| p == prompt) {
                Some((_, members)) => members.push(k),
                None => groups.push((prompt.clone(), vec![k])),
            }
        }

        for (prompt, members) in groups {
            let windows: Vec<Tensor<B, 3>> = members
                .iter()
                .map(|&k| pending[k].as_ref().expect("not yet taken").window.clone())
                .collect();
            let batch = Tensor::cat(windows, 0);

            let config = driver.decode_config(prompt);
            let tokens = driver
                .model()
                .decode_windows(batch, &config, driver.filters());

            for (row, k) in members.into_iter().enumerate() {
                let item = pending[k].take().expect("taken once");
                out[item.context]
                    .extend(contexts[item.context].commit_due(item.unit, tokens[row].clone())?);
            }
        }
    }
}
