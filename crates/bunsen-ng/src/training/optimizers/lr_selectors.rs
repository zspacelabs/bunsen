use burn::optim::LearningRate;
use hashbrown::HashMap;

/// Selection function for learning rates.
pub trait LrSelector: Send + Sync {
    /// Select the learning rate for this group.
    fn select(
        &self,
        lr: LearningRate,
        named_lrs: &HashMap<String, LearningRate>,
    ) -> LearningRate;
}

impl<F> LrSelector for F
where
    F: Fn(LearningRate, &HashMap<String, LearningRate>) -> LearningRate + Send + Sync,
{
    fn select(
        &self,
        lr: LearningRate,
        named_lrs: &HashMap<String, LearningRate>,
    ) -> LearningRate {
        (self)(lr, named_lrs)
    }
}

/// Learning rate selector that always returns the given learning rate.
pub struct GlobalLrSelector;

impl LrSelector for GlobalLrSelector {
    fn select(
        &self,
        lr: LearningRate,
        _named_lrs: &HashMap<String, LearningRate>,
    ) -> LearningRate {
        lr
    }
}

pub struct FixedLrSelector {
    lr: LearningRate,
}

impl FixedLrSelector {
    pub fn new(lr: LearningRate) -> Self {
        Self { lr }
    }

    pub fn lr(&self) -> LearningRate {
        self.lr
    }
}

impl LrSelector for FixedLrSelector {
    fn select(
        &self,
        _lr: LearningRate,
        _named_lrs: &HashMap<String, LearningRate>,
    ) -> LearningRate {
        self.lr
    }
}

impl LrSelector for NamedLrSelector {
    fn select(
        &self,
        _lr: LearningRate,
        named_lrs: &HashMap<String, LearningRate>,
    ) -> LearningRate {
        *named_lrs
            .get(&self.name)
            .unwrap_or_else(|| panic!("No learning rate for {}", self.name))
    }
}

/// [`LrSelector`] that always selects a given key.
pub struct NamedLrSelector {
    name: String,
}

impl NamedLrSelector {
    pub fn new(name: String) -> Self {
        Self { name }
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn test_fn_selector_impl() {
        let selector: Arc<dyn LrSelector> = Arc::new(
            |lr: LearningRate, named_lrs: &HashMap<String, LearningRate>| lr + named_lrs["foo"],
        );
        let lrs: HashMap<String, LearningRate> = [("foo".to_string(), 0.5)].into_iter().collect();

        assert_eq!(selector.select(1.0, &lrs), 1.5);
    }

    #[test]
    fn test_global_selector() {
        let selector = GlobalLrSelector;
        assert_eq!(selector.select(0.0, &HashMap::new()), 0.0);
    }

    #[test]
    fn test_fixed_selector() {
        let selector = FixedLrSelector::new(0.01);
        assert_eq!(selector.select(0.0, &HashMap::new()), 0.01);
    }

    #[test]
    fn test_named_selector() {
        let selector = NamedLrSelector::new("foo".to_string());
        let lrs: HashMap<String, LearningRate> = [("foo".to_string(), 0.5)].into_iter().collect();

        assert_eq!(selector.select(0.0, &lrs), 0.5);
    }
}
