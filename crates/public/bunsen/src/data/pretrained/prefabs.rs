//! # Config Prefabs for Well-Known Model Configurations

use alloc::{
    collections::BTreeMap,
    format,
    string::{
        String,
        ToString,
    },
    sync::Arc,
    vec::Vec,
};
use core::fmt::Debug;

use burn::config::Config;

use super::{
    PretrainedWeightsDescriptor,
    PretrainedWeightsMap,
    StaticPretrainedWeightsMap,
    not_found,
};
use crate::errors::BunsenResult;

/// Static builder for a [`PreFabConfig`]
pub struct StaticPreFabConfig<C>
where
    C: 'static + Config + Debug + Clone,
{
    /// Name of the model config pre-fab.
    pub name: &'static str,

    /// Description of the model config pre-fab.
    pub description: &'static str,

    /// Builder function for the config.
    pub builder: fn() -> C,

    /// Pretrained weights map.
    pub weights: Option<&'static StaticPretrainedWeightsMap<'static>>,
}

impl<C> StaticPreFabConfig<C>
where
    C: 'static + Config + Debug + Clone,
{
    /// Converts to a [`PreFabConfig<C>`].
    pub fn to_prefab(&self) -> PreFabConfig<C> {
        let builder = self.builder;
        PreFabConfig {
            name: self.name.to_string(),
            description: self.description.to_string(),
            builder: Arc::new(builder),
            weights: self.weights.map(|w| w.to_directory()),
        }
    }

    /// Builds a new config.
    pub fn to_config(&self) -> C {
        (self.builder)()
    }
}

impl<C> From<&StaticPreFabConfig<C>> for PreFabConfig<C>
where
    C: 'static + Config + Debug + Clone,
{
    fn from(config: &StaticPreFabConfig<C>) -> Self {
        config.to_prefab()
    }
}

impl<C> Debug for StaticPreFabConfig<C>
where
    C: 'static + Config + Debug + Clone,
{
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        self.to_prefab().fmt(f)
    }
}

/// A [`Config`] Well-Known Pre-Fab.
#[derive(Clone)]
pub struct PreFabConfig<C>
where
    C: 'static + Config + Debug + Clone,
{
    /// Name of the model config pre-fab.
    pub name: String,

    /// Description of the model config pre-fab.
    pub description: String,

    /// Builder function for the config.
    pub builder: Arc<dyn Fn() -> C + Send + Sync>,

    /// Pretrained weights map.
    pub weights: Option<PretrainedWeightsMap>,
}

impl<C> Debug for PreFabConfig<C>
where
    C: 'static + Config + Debug + Clone,
{
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        let pretty = f.alternate();

        let type_name = core::any::type_name::<C>();
        let mut handle = f.debug_struct(&format!("PreFabConfig<{}>", type_name));

        handle
            .field("name", &self.name)
            .field("description", &self.description);

        if pretty {
            handle.field("config", &self.to_config());
        }

        handle.finish()
    }
}

impl<C> PreFabConfig<C>
where
    C: 'static + Config + Debug + Clone,
{
    /// Builds a new config.
    pub fn to_config(&self) -> C {
        (self.builder)()
    }

    /// Looks up a descriptor.
    pub fn lookup_pretrained_weights(
        &self,
        name: &str,
    ) -> Option<PretrainedWeightsDescriptor> {
        match &self.weights {
            None => None,
            Some(m) => m.lookup_by_name(name),
        }
    }

    /// Looks up a descriptor.
    ///
    /// # Errors
    /// [`ResourceNotFound`](crate::errors::BunsenError::ResourceNotFound),
    /// naming the weights there are.
    pub fn try_lookup_pretrained_weights(
        &self,
        name: &str,
    ) -> BunsenResult<PretrainedWeightsDescriptor> {
        self.lookup_pretrained_weights(name).ok_or_else(|| {
            let names = self.weights.as_ref().map(|w| w.names()).unwrap_or_default();
            not_found(Some(&self.name), "pretrained weights", name, &names)
        })
    }

    /// Looks up a descriptor.
    pub fn expect_lookup_pretrained_weights(
        &self,
        name: &str,
    ) -> PretrainedWeightsDescriptor {
        match self.try_lookup_pretrained_weights(name) {
            Ok(p) => p,
            Err(e) => panic!("{}", e),
        }
    }
}

/// Static builder for a [`PreFabMap`].
#[derive(Debug)]
pub struct StaticPreFabMap<C>
where
    C: 'static + Config + Debug + Clone,
{
    /// Name of the prefab map.
    pub name: &'static str,

    /// Description of the prefab map.
    pub description: &'static str,

    /// List of prefabs.
    pub items: &'static [&'static StaticPreFabConfig<C>],
}

impl<C> StaticPreFabMap<C>
where
    C: 'static + Config + Debug + Clone,
{
    /// Converts to a [`PreFabMap`].
    pub fn to_prefab_map(&self) -> PreFabMap<C> {
        PreFabMap {
            name: self.name.to_string(),
            description: self.description.to_string(),
            items: self
                .items
                .iter()
                .map(|c| (c.name.to_string(), c.to_prefab()))
                .collect(),
        }
    }

    /// The prefabs, in listing order.
    pub fn iter(&self) -> impl Iterator<Item = &'static StaticPreFabConfig<C>> + '_ {
        self.items.iter().copied()
    }

    /// The first prefab whose built config satisfies `pred`: the reverse
    /// lookup, for a config that arrived without a name.
    pub fn find(
        &self,
        mut pred: impl FnMut(&C) -> bool,
    ) -> Option<&'static StaticPreFabConfig<C>> {
        self.iter().find(|p| pred(&p.to_config()))
    }

    /// Every prefab name, in listing order.
    pub fn names(&self) -> Vec<&'static str> {
        self.iter().map(|p| p.name).collect()
    }

    /// Looks up a prefab.
    pub fn lookup_prefab(
        &self,
        name: &str,
    ) -> Option<PreFabConfig<C>> {
        self.iter().find(|c| c.name == name).map(|c| c.to_prefab())
    }

    /// Looks up a prefab.
    ///
    /// # Errors
    /// [`ResourceNotFound`](crate::errors::BunsenError::ResourceNotFound),
    /// naming the prefabs there are.
    pub fn try_lookup_prefab(
        &self,
        name: &str,
    ) -> BunsenResult<PreFabConfig<C>> {
        self.lookup_prefab(name)
            .ok_or_else(|| not_found(Some(self.name), "prefab", name, &self.names()))
    }

    /// Looks up a prefab.
    ///
    /// # Panics
    /// If there is no such prefab.
    pub fn expect_lookup_prefab(
        &self,
        name: &str,
    ) -> PreFabConfig<C> {
        self.try_lookup_prefab(name)
            .unwrap_or_else(|e| panic!("{e}"))
    }
}

/// A map of [`PreFabConfig`]s.
#[derive(Debug, Clone)]
pub struct PreFabMap<C>
where
    C: 'static + Config + Debug + Clone,
{
    /// Name of the prefab map.
    pub name: String,

    /// Description of the prefab map.
    pub description: String,

    /// Map of prefabs.
    pub items: BTreeMap<String, PreFabConfig<C>>,
}

impl<C> PreFabMap<C>
where
    C: 'static + Config + Debug + Clone,
{
    /// Every prefab name, in name order.
    pub fn names(&self) -> Vec<&str> {
        self.items.keys().map(String::as_str).collect()
    }

    /// Looks up a prefab.
    pub fn lookup_prefab(
        &self,
        name: &str,
    ) -> Option<PreFabConfig<C>> {
        self.items.get(name).cloned()
    }

    /// Looks up a prefab.
    ///
    /// # Errors
    /// [`ResourceNotFound`](crate::errors::BunsenError::ResourceNotFound),
    /// naming the prefabs there are.
    pub fn try_lookup_prefab(
        &self,
        name: &str,
    ) -> BunsenResult<PreFabConfig<C>> {
        self.lookup_prefab(name)
            .ok_or_else(|| not_found(Some(&self.name), "prefab", name, &self.names()))
    }

    /// Looks up a prefab.
    ///
    /// # Panics
    /// If there is no such prefab.
    pub fn expect_lookup_prefab(
        &self,
        name: &str,
    ) -> PreFabConfig<C> {
        self.try_lookup_prefab(name)
            .unwrap_or_else(|e| panic!("{e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::BunsenError;

    #[derive(Config, Debug)]
    struct Toy {
        width: usize,
    }

    static NARROW: StaticPreFabConfig<Toy> = StaticPreFabConfig {
        name: "narrow",
        description: "width 1",
        builder: || Toy::new(1),
        weights: None,
    };
    static WIDE: StaticPreFabConfig<Toy> = StaticPreFabConfig {
        name: "wide",
        description: "width 2",
        builder: || Toy::new(2),
        weights: None,
    };
    static TOYS: StaticPreFabMap<Toy> = StaticPreFabMap {
        name: "toys",
        description: "two toys",
        items: &[&NARROW, &WIDE],
    };

    #[test]
    fn test_iter_find_and_names() {
        assert_eq!(TOYS.names(), ["narrow", "wide"]);
        assert_eq!(TOYS.iter().count(), 2);
        assert_eq!(TOYS.find(|c| c.width == 2).map(|p| p.name), Some("wide"));
        assert!(TOYS.find(|c| c.width == 3).is_none());
        assert_eq!(TOYS.lookup_prefab("wide").unwrap().to_config().width, 2);

        let owned = TOYS.to_prefab_map();
        assert_eq!(owned.names(), ["narrow", "wide"]);
        assert_eq!(owned.lookup_prefab("narrow").unwrap().to_config().width, 1);
    }

    /// A miss names the table and what it holds, in the static and the
    /// owned map alike, and in a prefab's weights.
    #[test]
    fn test_a_miss_names_what_there_is() {
        let m = match TOYS.try_lookup_prefab("huge") {
            Err(BunsenError::ResourceNotFound(m)) => m,
            other => panic!("{other:?}"),
        };
        assert_eq!(m, "toys: no prefab \"huge\"; there are: narrow, wide");

        let m = match TOYS.to_prefab_map().try_lookup_prefab("huge") {
            Err(BunsenError::ResourceNotFound(m)) => m,
            other => panic!("{other:?}"),
        };
        assert_eq!(m, "toys: no prefab \"huge\"; there are: narrow, wide");

        let m = match TOYS
            .expect_lookup_prefab("wide")
            .try_lookup_pretrained_weights("in1k")
        {
            Err(BunsenError::ResourceNotFound(m)) => m,
            other => panic!("{other:?}"),
        };
        assert_eq!(m, "wide: no pretrained weights \"in1k\"; there are: (none)");
    }
}
