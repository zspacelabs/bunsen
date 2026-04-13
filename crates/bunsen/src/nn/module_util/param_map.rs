//! Parameter Map
use std::collections::BTreeMap;

use burn::{
    Tensor,
    module::{
        Module,
        ModuleVisitor,
        Param,
        ParamId,
    },
    prelude::{
        Backend,
        Bool,
        Int,
    },
};

/// Encodes the kind of a Module Parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum ParamKind {
    /// A Bool Parameter.
    Bool,

    /// A Float Parameter.
    Float,

    /// An Int Parameter.
    Int,
}

/// A reference to a parameter.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ParamTag {
    /// The id of the parameter.
    id: ParamId,

    /// The kind of the parameter.
    kind: ParamKind,
}

impl ParamTag {
    /// Creates a new `ParamRef`.
    pub fn new(
        id: ParamId,
        kind: ParamKind,
    ) -> Self {
        Self { id, kind }
    }

    /// Returns the id of the parameter.
    pub fn id(&self) -> ParamId {
        self.id
    }

    /// Returns the kind of the parameter.
    pub fn kind(&self) -> ParamKind {
        self.kind
    }
}

/// Represents a node in a module tree path.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ModulePathNode {
    /// The name of the node.
    name: String,

    /// The name of the container type of the node.
    container: String,
}

impl ModulePathNode {
    /// Creates a new `ModulePathNode`.
    pub fn new(
        name: &str,
        container: &str,
    ) -> Self {
        Self {
            name: name.to_string(),
            container: container.to_string(),
        }
    }
}

/// Represents a path in a module tree.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ModulePath(Vec<ModulePathNode>);

/// A map from module paths to parameter kinds.
#[derive(Debug, Clone, Default)]
pub struct ParamMap {
    params: BTreeMap<ModulePath, ParamTag>,
}

impl ParamMap {
    /// Collects the parameter map from a module.
    pub fn collect<M: Module<B>, B: Backend>(module: &M) -> Self {
        let mut visitor = ParamMapBuildingVisitor::<B>::default();
        module.visit(&mut visitor);
        visitor.param_map
    }

    /// Adds a parameter to the map.
    pub fn add_param_id(
        &mut self,
        path: ModulePath,
        id: ParamId,
        kind: ParamKind,
    ) {
        self.params.insert(path, ParamTag::new(id, kind));
    }

    /// Returns an iterator over the parameter map.
    pub fn iter(&self) -> impl Iterator<Item = (&ModulePath, &ParamTag)> {
        self.params.iter()
    }

    /// Returns the number of parameters in the map.
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// Returns true if the map is empty.
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }
}

#[derive(Debug, Clone, Default)]
struct ParamMapBuildingVisitor<B: Backend> {
    stack: Vec<ModulePathNode>,
    param_map: ParamMap,
    phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> ParamMapBuildingVisitor<B> {
    /// Adds a parameter to the map.
    pub fn add_param_id(
        &mut self,
        id: ParamId,
        kind: ParamKind,
    ) {
        let path = ModulePath(self.stack.clone());
        self.param_map.add_param_id(path, id, kind);
    }
}

impl<B: Backend> ModuleVisitor<B> for ParamMapBuildingVisitor<B> {
    fn enter_module(
        &mut self,
        name: &str,
        container_type: &str,
    ) {
        self.stack.push(ModulePathNode::new(name, container_type));
    }

    fn exit_module(
        &mut self,
        _name: &str,
        _container_type: &str,
    ) {
        self.stack.pop();
    }

    fn visit_bool<const D: usize>(
        &mut self,
        param: &Param<Tensor<B, D, Bool>>,
    ) {
        self.add_param_id(param.id, ParamKind::Bool);
    }

    fn visit_float<const D: usize>(
        &mut self,
        param: &Param<Tensor<B, D>>,
    ) {
        self.add_param_id(param.id, ParamKind::Float);
    }

    fn visit_int<const D: usize>(
        &mut self,
        param: &Param<Tensor<B, D, Int>>,
    ) {
        self.add_param_id(param.id, ParamKind::Int);
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::Wgpu,
        nn::{
            Linear,
            LinearConfig,
        },
    };

    use super::*;

    #[test]
    fn test_param_kind() {
        assert_eq!(ParamKind::Bool, ParamKind::Bool);
        assert_ne!(ParamKind::Bool, ParamKind::Float);
    }

    #[test]
    fn test_param_ref() {
        let ref1 = ParamTag::new(1.into(), ParamKind::Bool);
        let ref1_dup = ParamTag::new(1.into(), ParamKind::Bool);
        let ref1_cp = ref1.clone();

        assert_eq!(ref1, ref1_dup);
        assert_eq!(ref1, ref1_cp);

        assert_eq!(ref1.id(), 1.into());
        assert_eq!(ref1.kind(), ParamKind::Bool);

        let ref2 = ParamTag::new(2.into(), ParamKind::Float);
        let ref3 = ParamTag::new(3.into(), ParamKind::Int);

        assert_eq!(ref2.id(), 2.into());
        assert_eq!(ref2.kind(), ParamKind::Float);

        assert_eq!(ref3.id(), 3.into());
        assert_eq!(ref3.kind(), ParamKind::Int);

        assert_ne!(ref1, ref2);
        assert_ne!(ref1, ref3);
    }

    #[derive(Module, Debug)]
    struct TestModule<B: Backend> {
        seq: Vec<Linear<B>>,
    }

    impl<B: Backend> TestModule<B> {
        fn init(device: &B::Device) -> Self {
            Self {
                seq: vec![LinearConfig::new(10, 10).init(device)],
            }
        }
    }

    #[test]
    fn test_module_path() {
        type B = Wgpu;
        let device = Default::default();

        let module = TestModule::<B>::init(&device);

        let param_map = ParamMap::collect(&module);

        assert_eq!(
            &param_map.iter().collect::<Vec<_>>(),
            &vec![
                (
                    &ModulePath(vec![
                        ModulePathNode::new("seq", "Struct:TestModule"),
                        ModulePathNode::new("0", "Vec"),
                        ModulePathNode::new("bias", "Struct:Linear"),
                    ]),
                    &ParamTag::new(module.seq[0].bias.as_ref().unwrap().id, ParamKind::Float)
                ),
                (
                    &ModulePath(vec![
                        ModulePathNode::new("seq", "Struct:TestModule"),
                        ModulePathNode::new("0", "Vec"),
                        ModulePathNode::new("weight", "Struct:Linear"),
                    ]),
                    &ParamTag::new(module.seq[0].weight.id, ParamKind::Float)
                ),
            ]
        );
    }
}
