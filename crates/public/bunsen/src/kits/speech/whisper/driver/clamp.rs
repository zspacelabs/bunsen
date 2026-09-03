//! # The clamp policy: where a window's dynamic-range floor comes from.
//!
//! Whisper floors its log-mels 8 dB below a reference maximum, and upstream
//! takes that maximum over the **whole clip** before cutting windows. A
//! stream cannot see the whole clip, so the question of what the reference
//! is becomes a policy, and the policy is an injected object behind this
//! trait rather than a variant the driver understands: the driver knows only
//! that every arriving frame is offered to it once, and that a window's
//! reference is asked for immediately before packaging.
//!
//! The `&mut observe` / `&self reference` split is forced, not chosen. A
//! provisional decode packages a window without mutating the context, so the
//! call it makes on the way cannot mutate either; anything a draft reaches
//! must take `&self`. That is what keeps packaging a window twice from moving
//! it.
//!
//! Two implementations cover the four behaviours that came up in design.
//! [`MaxSeen`] fed everything before the first packaging is the global
//! reference upstream uses; fed incrementally it is the running one; and
//! since a speech region is decoded as its own context, it is the per-region
//! one too. [`PerWindow`] is today's
//! [`package_mels`](crate::kits::speech::whisper::blocks::WhisperFrontEndConfig::package_mels).
//!
//! Packaging itself, and the clamp range, live in [`mel`](super::mel); a
//! policy supplies only the reference.

use std::fmt::Debug;

use burn::{
    Tensor,
    module::Module,
    prelude::Backend,
};
use dyn_clone::DynClone;

/// Decides the reference maximum a window is floored against.
///
/// One reference per batch row, in the post-log domain: `[batch]`.
///
/// Implementors are `Clone`, through [`DynClone`], because the stream
/// context that holds one boxed is `Clone`. Nothing else is asked of them.
pub trait ClampPolicy<B: Backend>: Send + Sync + Debug + DynClone {
    /// Offers arriving frames to the policy. The arrival path, and the only
    /// place a policy may mutate.
    ///
    /// # Arguments
    /// * `frames` - `[batch, frames, n_mels]` log-mels, as they arrive.
    fn observe(
        &mut self,
        frames: &Tensor<B, 3>,
    );

    /// The reference maximum for a window about to be packaged.
    ///
    /// Takes `&self`: a provisional decode packages a window without touching
    /// the context, so this cannot touch the policy either.
    ///
    /// # Arguments
    /// * `window` - `[batch, frames, n_mels]` log-mels about to be packaged.
    ///
    /// # Returns
    /// `[batch]`, one reference per row.
    fn reference(
        &self,
        window: &Tensor<B, 3>,
    ) -> Tensor<B, 1>;
}

// `Box<dyn ClampPolicy<B>>: Clone`, through the concrete policy's `Clone`
// behind the vtable. A local `CloneBox<dyn ClampPolicy<B>>` supertrait cannot
// say this: a trait naming its own object type among its supertraits is a
// cycle (E0391), and a supertrait that does not name it cannot re-fatten the
// pointer without `dyn_clone`'s erasure.
dyn_clone::clone_trait_object!(<B: Backend> ClampPolicy<B>);

/// A boxed policy is a policy, so a context may be generic over a concrete
/// policy or hold a dynamic one; either way it is one type parameter.
impl<B: Backend> ClampPolicy<B> for Box<dyn ClampPolicy<B>> {
    fn observe(
        &mut self,
        frames: &Tensor<B, 3>,
    ) {
        (**self).observe(frames);
    }

    fn reference(
        &self,
        window: &Tensor<B, 3>,
    ) -> Tensor<B, 1> {
        (**self).reference(window)
    }
}

/// The maximum over each row: `[batch, frames, n_mels]` to `[batch]`.
fn row_max<B: Backend>(x: &Tensor<B, 3>) -> Tensor<B, 1> {
    let batch = x.dims()[0];
    x.clone().max_dims(&[1, 2]).reshape([batch])
}

/// Each window is floored against its own maximum.
///
/// Ignores [`observe`](ClampPolicy::observe). This is what
/// [`package_mels`](crate::kits::speech::whisper::blocks::WhisperFrontEndConfig::package_mels)
/// does today, and it is the right policy when a window *is* the whole clip.
#[derive(Debug, Default, Clone, Copy)]
pub struct PerWindow;

impl<B: Backend> ClampPolicy<B> for PerWindow {
    fn observe(
        &mut self,
        _frames: &Tensor<B, 3>,
    ) {
    }

    fn reference(
        &self,
        window: &Tensor<B, 3>,
    ) -> Tensor<B, 1> {
        row_max(window)
    }
}

/// The running maximum over everything observed so far, per row.
///
/// Fed the whole clip before the first packaging, this is exactly upstream's
/// global reference. Fed as audio arrives, it is the running one: a window is
/// floored against the loudest thing heard *so far*, which is the closest a
/// live stream can come.
///
/// The reference for a window is never below that window's own maximum, so
/// with nothing observed this degrades to [`PerWindow`] rather than to
/// something wrong.
#[derive(Module, Debug)]
pub struct MaxSeen<B: Backend> {
    /// `[batch]` running maximum, or `None` before the first observation.
    seen: Option<Tensor<B, 1>>,
}

impl<B: Backend> Default for MaxSeen<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> MaxSeen<B> {
    /// A policy that has seen nothing yet.
    pub fn new() -> Self {
        Self { seen: None }
    }

    /// The running maximum, `[batch]`, if anything has been observed.
    pub fn seen(&self) -> Option<&Tensor<B, 1>> {
        self.seen.as_ref()
    }
}

impl<B: Backend> ClampPolicy<B> for MaxSeen<B> {
    fn observe(
        &mut self,
        frames: &Tensor<B, 3>,
    ) {
        let arriving = row_max(frames);
        self.seen = Some(match self.seen.take() {
            Some(seen) => seen.max_pair(arriving),
            None => arriving,
        });
    }

    fn reference(
        &self,
        window: &Tensor<B, 3>,
    ) -> Tensor<B, 1> {
        let own = row_max(window);
        match &self.seen {
            Some(seen) => seen.clone().max_pair(own),
            None => own,
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::prelude::TensorData;

    use super::*;
    use crate::support::testing::{
        CpuBackend,
        assert_close_to_vec,
    };

    type B = CpuBackend;

    /// `[2, 2, 2]`: two rows, two frames, two mels, from a flat list.
    fn frames(values: [f64; 8]) -> Tensor<B, 3> {
        Tensor::from_data(
            TensorData::new(values.to_vec(), [2, 2, 2]),
            &Default::default(),
        )
    }

    fn to_vec(t: Tensor<B, 1>) -> Vec<f64> {
        t.cast(burn::tensor::DType::F64).to_data().to_vec().unwrap()
    }

    #[test]
    fn test_per_window_is_each_rows_own_maximum() {
        let mut policy = PerWindow;
        let window = frames([0.0, -3.0, -1.0, -20.0, 5.0, 4.0, -9.0, 1.0]);

        // Observing changes nothing, and rows do not see each other's peaks.
        ClampPolicy::<B>::observe(&mut policy, &frames([99.0; 8]));
        assert_close_to_vec(&to_vec(policy.reference(&window)), &[0.0, 5.0], 1e-12);
    }

    #[test]
    fn test_max_seen_runs_over_observations() {
        let mut policy = MaxSeen::<B>::new();
        assert!(policy.seen().is_none());

        policy.observe(&frames([0.0, -3.0, -1.0, -20.0, 5.0, 4.0, -9.0, 1.0]));
        assert_close_to_vec(&to_vec(policy.seen().unwrap().clone()), &[0.0, 5.0], 1e-12);

        // A later, louder frame in row 0 raises row 0 only.
        policy.observe(&frames([7.0, -3.0, -1.0, -20.0, -5.0, -4.0, -9.0, -1.0]));
        assert_close_to_vec(&to_vec(policy.seen().unwrap().clone()), &[7.0, 5.0], 1e-12);

        // A quiet window is floored against what was heard, not against
        // itself; a window louder than anything heard raises its own bar.
        let quiet = frames([-10.0; 8]);
        assert_close_to_vec(&to_vec(policy.reference(&quiet)), &[7.0, 5.0], 1e-12);
        let loud = frames([-10.0, -10.0, -10.0, -10.0, 9.0, -10.0, -10.0, -10.0]);
        assert_close_to_vec(&to_vec(policy.reference(&loud)), &[7.0, 9.0], 1e-12);
    }

    /// With nothing observed, `MaxSeen` is `PerWindow`.
    #[test]
    fn test_max_seen_degrades_to_per_window() {
        let policy = MaxSeen::<B>::new();
        let window = frames([0.0, -3.0, -1.0, -20.0, 5.0, 4.0, -9.0, 1.0]);

        assert_close_to_vec(
            &to_vec(policy.reference(&window)),
            &to_vec(PerWindow.reference(&window)),
            1e-12,
        );
    }

    /// `reference` takes `&self`, so asking twice is asking once.
    #[test]
    fn test_reference_does_not_move() {
        let mut policy = MaxSeen::<B>::new();
        policy.observe(&frames([1.0; 8]));
        let window = frames([0.0, -3.0, -1.0, -20.0, 5.0, 4.0, -9.0, 1.0]);

        let first = to_vec(policy.reference(&window));
        let second = to_vec(policy.reference(&window));
        assert_eq!(first, second);
        assert_close_to_vec(&to_vec(policy.seen().unwrap().clone()), &[1.0, 1.0], 1e-12);
    }

    /// The shape the driver will hold: boxed, dynamic, debuggable, and
    /// swappable without the driver knowing which it has.
    #[test]
    fn test_trait_object() {
        let window = frames([0.0, -3.0, -1.0, -20.0, 5.0, 4.0, -9.0, 1.0]);

        let mut policies: Vec<Box<dyn ClampPolicy<B>>> =
            vec![Box::new(PerWindow), Box::new(MaxSeen::new())];
        for policy in policies.iter_mut() {
            policy.observe(&window);
            assert_close_to_vec(&to_vec(policy.reference(&window)), &[0.0, 5.0], 1e-12);
            assert!(!format!("{policy:?}").is_empty());

            // A clone carries the observations with it, and diverges after.
            let mut copy = policy.clone();
            copy.observe(&frames([50.0; 8]));
            assert_close_to_vec(&to_vec(policy.reference(&window)), &[0.0, 5.0], 1e-12);
            assert!(to_vec(copy.reference(&window))[0] >= 0.0);
        }
    }
}
