use std::{
    fmt,
    sync::Arc,
};

use burn::optim::{
    LearningRate,
    lr_scheduler::{
        LrScheduler,
        LrSchedulerRecord,
        module_lr_scheduler::ModuleLrScheduler,
    },
};

/// Maps the global learning rate to a group's learning rate.
///
/// A `Send + Sync` closure from the global rate to the group's rate is an
/// `LrSelector`: `|lr| lr * 0.5`. [`SelectedLr`] applies one to a schedule.
pub trait LrSelector: Send + Sync {
    /// Selects this group's learning rate from the global learning rate.
    fn select(
        &self,
        lr: LearningRate,
    ) -> LearningRate;
}

impl<F> LrSelector for F
where
    F: Fn(LearningRate) -> LearningRate + Send + Sync,
{
    fn select(
        &self,
        lr: LearningRate,
    ) -> LearningRate {
        (self)(lr)
    }
}

/// [`LrSelector`] that selects the global learning rate itself.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GlobalLrSelector;

impl LrSelector for GlobalLrSelector {
    fn select(
        &self,
        lr: LearningRate,
    ) -> LearningRate {
        lr
    }
}

/// [`LrSelector`] that always selects a fixed learning rate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FixedLrSelector {
    lr: LearningRate,
}

impl FixedLrSelector {
    /// Creates a new selector.
    pub fn new(lr: LearningRate) -> Self {
        Self { lr }
    }

    /// Returns the fixed learning rate.
    pub fn lr(&self) -> LearningRate {
        self.lr
    }
}

impl LrSelector for FixedLrSelector {
    fn select(
        &self,
        _lr: LearningRate,
    ) -> LearningRate {
        self.lr
    }
}

/// An [`LrScheduler`] that derives its rate from a global schedule: each
/// step is the global rate of the base scheduler's step, mapped through an
/// [`LrSelector`].
///
/// This is "the global schedule × k": a group whose rate follows the warmup
/// and decay of the global schedule, at its own scale.
/// [`GroupOptimizerPlan::lr_scheduler`] mounts one per group that has a
/// selector, each over its own clone of the base. The clones step in
/// lockstep, so every group sees the same point of the schedule.
///
/// The base is a [`ModuleLrScheduler`], which is what a `burn` scheduler
/// config's `init()` returns; any [`LrScheduler`], including a constant
/// `LearningRate`, converts into one. Its global rate is the rate of its
/// first group, which matches every parameter
/// ([`ModuleLearningRate::base`](burn::optim::lr_scheduler::module_lr_scheduler::ModuleLearningRate::base)).
///
/// The record is the base scheduler's record.
///
/// [`GroupOptimizerPlan::lr_scheduler`]: crate::burner::optim::GroupOptimizerPlan::lr_scheduler
#[derive(Clone)]
pub struct SelectedLr {
    base: ModuleLrScheduler,
    selector: Arc<dyn LrSelector>,
}

impl SelectedLr {
    /// Creates a scheduler that steps `base` and maps its global rate
    /// through `selector`.
    pub fn new(
        base: impl Into<ModuleLrScheduler>,
        selector: Arc<dyn LrSelector>,
    ) -> Self {
        Self {
            base: base.into(),
            selector,
        }
    }

    /// Returns the base scheduler.
    pub fn base(&self) -> &ModuleLrScheduler {
        &self.base
    }

    /// Returns the selector.
    pub fn selector(&self) -> &Arc<dyn LrSelector> {
        &self.selector
    }
}

impl fmt::Debug for SelectedLr {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.debug_struct("SelectedLr").finish_non_exhaustive()
    }
}

impl LrScheduler for SelectedLr {
    fn step(&mut self) -> LearningRate {
        self.selector.select(self.base.step().base())
    }

    fn to_record(&self) -> LrSchedulerRecord {
        self.base.to_record()
    }

    fn load_record(
        &mut self,
        record: LrSchedulerRecord,
    ) {
        self.base = self.base.clone().load_record(record);
    }
}

#[cfg(test)]
mod tests {
    use burn::optim::lr_scheduler::linear::LinearLrSchedulerConfig;

    use super::*;

    #[test]
    fn test_fn_selector_impl() {
        let selector: Arc<dyn LrSelector> = Arc::new(|lr: LearningRate| lr * 0.5);

        assert_eq!(selector.select(1.0), 0.5);
    }

    #[test]
    fn test_global_selector() {
        let selector = GlobalLrSelector;
        assert_eq!(selector.select(0.25), 0.25);
    }

    #[test]
    fn test_fixed_selector() {
        let selector = FixedLrSelector::new(0.01);
        assert_eq!(selector.select(0.0), 0.01);
        assert_eq!(selector.lr(), 0.01);
    }

    #[test]
    fn test_selected_lr_scales_the_base_schedule() {
        // 1.0 → 0.0 over 4 steps: 1.0, 0.75, 0.5, ...
        let base = LinearLrSchedulerConfig::new(1.0, 0.0, 4).init().unwrap();
        let mut plain = base.clone();
        let mut scaled = SelectedLr::new(base, Arc::new(|lr: LearningRate| lr * 0.5));

        for _ in 0..3 {
            assert_eq!(scaled.step(), plain.step().base() * 0.5);
        }
    }

    #[test]
    fn test_selected_lr_records_the_base_schedule() {
        let base = LinearLrSchedulerConfig::new(1.0, 0.0, 4).init().unwrap();
        let mut first = SelectedLr::new(base.clone(), Arc::new(|lr: LearningRate| lr * 2.0));
        first.step();
        first.step();

        let mut resumed = SelectedLr::new(base, Arc::new(|lr: LearningRate| lr * 2.0));
        resumed.load_record(first.to_record());

        assert_eq!(resumed.step(), first.step());
    }
}
