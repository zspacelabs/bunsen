//! Experimental library for graph operations

use burn::{
    prelude::Shape,
    tensor::DType,
};

#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub shape: Shape,
    pub dtype: DType,
}

pub struct TensorStub {
    pub id: uuid::Uuid,
    pub meta: TensorMeta,
}

pub struct GraphOp {
    pub id: uuid::Uuid,
    pub inputs: Vec<TensorStub>,
    pub outputs: Vec<TensorStub>,
}

#[cfg(test)]
mod tests {}
