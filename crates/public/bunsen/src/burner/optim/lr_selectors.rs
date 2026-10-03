use burn::optim::LearningRate;

/// Selection function for learning rates.
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

/// Learning rate selector that always returns the given learning rate.
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

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
    }
}
