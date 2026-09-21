use burn::{
    Tensor,
    prelude::{
        Backend,
        Bool,
        Float,
        Int,
        Shape,
        SliceArg,
        TensorData,
    },
    tensor::{
        BasicOps,
        DType,
        Slice,
    },
};

use crate::{
    burner::{
        descriptors::TensorKindDesc,
        module::HasDType,
        tensor::dynamic::RankHandler,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
    support::CloneBox,
    zspace::check_slices_bounds,
};

/// Provides a dynamic version of [`Tensor::slice`].
pub fn slice_dyn<B: Backend, const R: usize, K: BasicOps<B>>(
    tensor: Tensor<B, R, K>,
    slices: &[Slice],
) -> Tensor<B, R, K> {
    let mut tensor = tensor;
    for (dim, slice) in slices.iter().enumerate() {
        tensor = tensor.slice_dim(dim, *slice);
    }
    tensor
}

/// Values conversion trait for [`DynTensor::slice_assign`].
pub trait ValuesArg<B: Backend>: Sized {
    /// Convert to a [`DynTensor`] on a given device.
    fn into_values(
        self,
        device: &B::Device,
    ) -> BunsenResult<DynTensor<B>>;
}

impl<B: Backend, T: Into<DynTensor<B>>> ValuesArg<B> for T {
    fn into_values(
        self,
        device: &B::Device,
    ) -> BunsenResult<DynTensor<B>> {
        self.into().to_device(device)
    }
}

impl<B: Backend> ValuesArg<B> for TensorData {
    fn into_values(
        self,
        device: &B::Device,
    ) -> BunsenResult<DynTensor<B>> {
        DynTensor::from_data(self, device)
    }
}

/// A dynamic [`Tensor`] wrapper that can be sliced.
#[derive(Debug, Clone)]
pub struct DynTensor<B: Backend> {
    shape: Shape,
    dtype: DType,
    kind: TensorKindDesc,
    device: B::Device,
    tensor: Box<dyn CloneBox>,
    phantom: std::marker::PhantomData<B>,
}

impl<B: Backend, const R: usize, K> From<Tensor<B, R, K>> for DynTensor<B>
where
    K: 'static + BasicOps<B>,
{
    fn from(val: Tensor<B, R, K>) -> Self {
        DynTensor::new(val)
    }
}

impl<B: Backend> HasDType for DynTensor<B> {
    fn dtype(&self) -> DType {
        self.dtype
    }
}

impl<B: Backend> DynTensor<B> {
    /// Create a new `TensorStub` from a tensor.
    pub fn new<const R: usize, K>(tensor: Tensor<B, R, K>) -> Self
    where
        K: BasicOps<B> + 'static,
    {
        Self {
            shape: tensor.shape(),
            dtype: tensor.dtype(),
            kind: tensor.dtype().into(),
            device: tensor.device(),
            tensor: Box::new(tensor),
            phantom: std::marker::PhantomData,
        }
    }

    /// Get the tensor rank.
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Get the tensor shape.
    pub fn shape(&self) -> Shape {
        self.shape.clone()
    }

    /// Get the number of elements in the tensor.
    pub fn num_elements(&self) -> usize {
        self.shape.num_elements()
    }

    /// Returns the size estimate of the tensor in bytes.
    ///
    /// This is `self.dtype().size() * self.num_elements()`.
    pub fn size_estimate(&self) -> usize {
        self.dtype.size() * self.num_elements()
    }

    /// Get the tensor data type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get the tensor kind.
    pub fn kind(&self) -> TensorKindDesc {
        self.kind
    }

    /// Get the tensor device.
    pub fn device(&self) -> B::Device {
        self.device.clone()
    }

    /// Downcasts the tensor to a specific rank and kind.
    ///
    /// # Result
    /// - `Some(Tensor<B, R, K>)`: if the params are correct,
    /// - `None`: otherwise.
    pub fn downcast_clone<const R: usize, K>(&self) -> Option<Tensor<B, R, K>>
    where
        K: 'static + BasicOps<B>,
    {
        self.tensor.downcast_ref::<Tensor<B, R, K>>().cloned()
    }

    /// Downcasts to a static tensor.
    ///
    /// # Result
    /// - the static tensor: if the params are correct,
    ///
    /// # Panics
    /// If the types are incorrect.
    pub fn unwrap_clone<const R: usize, K>(&self) -> Tensor<B, R, K>
    where
        K: 'static + BasicOps<B>,
    {
        self.downcast_clone::<R, K>()
            .expect("downcast_clone failed")
    }

    /// Slice the stub tensor.
    ///
    /// # Arguments
    /// - `slices`: a `SliceArg<R>`.
    ///
    /// # Result
    /// - `Ok(DynTensor)`: the sliced tensor.
    /// - `Err(DynTensorError)`: an error.
    pub fn slice<S>(
        self,
        slices: S,
    ) -> BunsenResult<Self>
    where
        S: SliceArg,
    {
        let rank = self.rank();
        let slices: Vec<Slice> = slices.into_slices(&self.shape);

        check_slices_bounds(&self.shape(), &slices).map_err(BunsenError::SliceError)?;

        struct SliceHandler<B: Backend> {
            this: DynTensor<B>,
            slices: Vec<Slice>,
        }
        impl<B: Backend> RankHandler for SliceHandler<B> {
            type Output = DynTensor<B>;

            fn call<const R: usize>(self) -> BunsenResult<Self::Output> {
                Ok(match self.this.kind {
                    TensorKindDesc::Float => self
                        .this
                        .unwrap_clone::<R, Float>()
                        .slice(&self.slices)
                        .into(),
                    TensorKindDesc::Int => self
                        .this
                        .unwrap_clone::<R, Int>()
                        .slice(&self.slices)
                        .into(),
                    TensorKindDesc::Bool => self
                        .this
                        .unwrap_clone::<R, Bool>()
                        .slice(&self.slices)
                        .into(),
                })
            }
        }
        SliceHandler { this: self, slices }.dyn_call(rank)
    }

    /// A dynamic version of [`DynTensor::slice`].
    ///
    /// # Arguments
    /// - `slices`: a dynamic slice of `Slice`.
    ///
    /// # Result
    /// - `Ok(DynTensor)`: the sliced tensor.
    /// - `Err(DynTensorError)`: an error.
    pub fn slice_dyn(
        self,
        slices: &[Slice],
    ) -> BunsenResult<Self> {
        let rank = self.rank();

        check_slices_bounds(&self.shape(), slices).map_err(BunsenError::SliceError)?;

        struct SliceDynHandler<'a, B: Backend> {
            this: DynTensor<B>,
            slices: &'a [Slice],
        }
        impl<'a, B: Backend> RankHandler for SliceDynHandler<'a, B> {
            type Output = DynTensor<B>;

            fn call<const R: usize>(self) -> BunsenResult<Self::Output> {
                Ok(match self.this.kind {
                    TensorKindDesc::Float => {
                        slice_dyn(self.this.unwrap_clone::<R, Float>(), self.slices).into()
                    }
                    TensorKindDesc::Int => {
                        slice_dyn(self.this.unwrap_clone::<R, Int>(), self.slices).into()
                    }
                    TensorKindDesc::Bool => {
                        slice_dyn(self.this.unwrap_clone::<R, Bool>(), self.slices).into()
                    }
                })
            }
        }
        SliceDynHandler { this: self, slices }.dyn_call(rank)
    }

    /// Assign values to a slice.
    ///
    /// # Arguments
    /// - `slices`: a `SlicesArg<R2>`.
    /// - `values`: a coercible value; see [`ValuesArg`].
    ///
    /// # Result
    /// - `Ok(DynTensor)`: a converted tensor.
    /// - `Err(DynTensorError)`: an error.
    pub fn slice_assign<const R2: usize, S, V>(
        self,
        slices: S,
        values: V,
    ) -> BunsenResult<Self>
    where
        S: SliceArg,
        V: ValuesArg<B>,
    {
        let rank = self.rank();
        let slices: [Slice; R2] = slices.into_slices(&self.shape).try_into().map_err(|_| {
            BunsenError::InvalidArgument {
                msg: format!("slice_assign rank ({R2}) does not match tensor rank ({rank})"),
            }
        })?;
        let values: DynTensor<B> = values.into_values(&self.device())?;

        check_slices_bounds(&self.shape(), &slices).map_err(BunsenError::SliceError)?;

        if rank != values.rank() {
            return Err(BunsenError::InvalidArgument {
                msg: format!(
                    "slice of rank ({}) cannot be assigned to tensor of rank ({})",
                    values.rank(),
                    rank
                ),
            });
        }

        let values = values.cast(self.dtype())?;

        // TODO: check that slices shape == source.shape

        struct SliceAssignHandler<B: Backend, const R2: usize> {
            this: DynTensor<B>,
            slices: [Slice; R2],
            values: DynTensor<B>,
        }
        impl<B: Backend, const R2: usize> RankHandler for SliceAssignHandler<B, R2> {
            type Output = DynTensor<B>;

            fn call<const R: usize>(self) -> BunsenResult<Self::Output> {
                Ok(match (self.this.kind, self.values.kind) {
                    (TensorKindDesc::Float, TensorKindDesc::Float) => {
                        let target = self.this.unwrap_clone::<R, Float>();
                        let source = self.values.unwrap_clone::<R, Float>();
                        let source = source.cast(target.dtype());
                        target.slice_assign(self.slices, source).into()
                    }
                    (TensorKindDesc::Int, TensorKindDesc::Int) => {
                        let target = self.this.unwrap_clone::<R, Int>();
                        let source = self.values.unwrap_clone::<R, Int>();
                        let source = source.cast(target.dtype());
                        target.slice_assign(self.slices, source).into()
                    }
                    (TensorKindDesc::Bool, TensorKindDesc::Bool) => {
                        let target = self.this.unwrap_clone::<R, Bool>();
                        let source = self.values.unwrap_clone::<R, Bool>();
                        // let source = source.cast(target.dtype());
                        target.slice_assign(self.slices, source).into()
                    }
                    _ => Err(BunsenError::Invalid(format!(
                        "target kind ({:?}) != value kind ({:?})",
                        self.this.kind, self.values.kind
                    )))?,
                })
            }
        }
        SliceAssignHandler {
            this: self.clone(),
            slices,
            values,
        }
        .dyn_call(rank)
    }

    /// Dynamic slice rank version of [`DynTensor::slice_assign`].
    ///
    /// # Arguments
    /// - `slices`: a dynamic slice of `Slice`; missing trailing dimensions are
    ///   taken in full, as in [`DynTensor::slice_dyn`].
    /// - `values`: a coercible value; see [`ValuesArg`].
    ///
    /// # Result
    /// - `Ok(DynTensor)`: a converted tensor.
    /// - `Err(DynTensorError)`: an error.
    pub fn slice_assign_dyn<V>(
        self,
        slices: &[Slice],
        values: V,
    ) -> BunsenResult<Self>
    where
        V: ValuesArg<B>,
    {
        struct SliceAssignDynHandler<B: Backend> {
            this: DynTensor<B>,
            slices: Vec<Slice>,
            values: DynTensor<B>,
        }
        impl<B: Backend> RankHandler for SliceAssignDynHandler<B> {
            type Output = DynTensor<B>;

            fn call<const R: usize>(self) -> BunsenResult<Self::Output> {
                self.this
                    .slice_assign::<R, _, _>(self.slices.as_slice(), self.values)
            }
        }
        let rank = self.rank();

        check_slices_bounds(&self.shape(), slices).map_err(BunsenError::SliceError)?;
        let mut slices = slices.to_vec();
        slices.resize(rank, Slice::full());

        let values = values.into_values(&self.device())?;
        SliceAssignDynHandler {
            this: self,
            slices,
            values,
        }
        .dyn_call(rank)
    }

    /// Flatten the tensor.
    ///
    /// # Result
    /// - `Ok(DynTensor)`: a flattened (rank=1) tensor.
    /// - `Err(DynTensorError)`: an error.
    pub fn flatten(self) -> BunsenResult<Self> {
        struct FlattenHandler<B: Backend> {
            tensor: DynTensor<B>,
        }
        impl<B: Backend> RankHandler for FlattenHandler<B> {
            type Output = DynTensor<B>;

            fn call<const R: usize>(self) -> BunsenResult<Self::Output> {
                Ok(match self.tensor.kind {
                    TensorKindDesc::Float => self
                        .tensor
                        .unwrap_clone::<R, Float>()
                        .flatten::<1>(0, self.tensor.rank() - 1)
                        .into(),
                    TensorKindDesc::Int => self
                        .tensor
                        .unwrap_clone::<R, Int>()
                        .flatten::<1>(0, self.tensor.rank() - 1)
                        .into(),
                    TensorKindDesc::Bool => self
                        .tensor
                        .unwrap_clone::<R, Bool>()
                        .flatten::<1>(0, self.tensor.rank() - 1)
                        .into(),
                })
            }
        }
        let rank = self.rank();
        FlattenHandler { tensor: self }.dyn_call(rank)
    }

    /// Cast the tensor.
    ///
    /// Auto-converts kind if necessary.
    ///
    /// # Arguments
    /// - `dtype`: the target data type.
    ///
    /// # Result
    /// - `Ok(DynTensor)`: a converted tensor.
    /// - `Err(DynTensorError)`: an error.
    pub fn cast(
        self,
        dtype: DType,
    ) -> BunsenResult<Self> {
        struct CastHandler<B: Backend> {
            this: DynTensor<B>,
            dtype: DType,
        }
        impl<B: Backend> RankHandler for CastHandler<B> {
            type Output = DynTensor<B>;

            fn call<const R: usize>(self) -> BunsenResult<Self::Output> {
                let target_kind: TensorKindDesc = self.dtype.into();
                Ok(match self.this.kind {
                    TensorKindDesc::Float => {
                        let tensor: Tensor<B, R, Float> = self.this.unwrap_clone();
                        match target_kind {
                            TensorKindDesc::Float => tensor.cast(self.dtype).into(),
                            TensorKindDesc::Int => tensor.int().cast(self.dtype).into(),
                            TensorKindDesc::Bool => tensor.bool().into(),
                        }
                    }
                    TensorKindDesc::Int => {
                        let tensor: Tensor<B, R, Int> = self.this.unwrap_clone();
                        match target_kind {
                            TensorKindDesc::Float => tensor.float().cast(self.dtype).into(),
                            TensorKindDesc::Int => tensor.cast(self.dtype).into(),
                            TensorKindDesc::Bool => tensor.bool().into(),
                        }
                    }
                    TensorKindDesc::Bool => {
                        let tensor: Tensor<B, R, Bool> = self.this.unwrap_clone();
                        match target_kind {
                            TensorKindDesc::Float => tensor.float().cast(self.dtype).into(),
                            TensorKindDesc::Int => tensor.int().cast(self.dtype).into(),
                            TensorKindDesc::Bool => self.this,
                        }
                    }
                })
            }
        }
        let rank = self.rank();
        CastHandler { this: self, dtype }.dyn_call(rank)
    }

    /// Move the tensor to the given device.
    ///
    /// Moving to the same device is an inexpensive no-op.
    ///
    /// # Arguments
    /// - `device`: the target device.
    ///
    /// # Result
    /// - `Ok(DynTensor<B>)`: the moved tensor.
    /// - `Err(DynTensorError)`: an error.
    pub fn to_device(
        self,
        device: &B::Device,
    ) -> BunsenResult<Self> {
        if &self.device() == device {
            return Ok(self);
        }

        struct ToDeviceHandler<'a, B: Backend> {
            this: DynTensor<B>,
            device: &'a B::Device,
        }
        impl<'a, B: Backend> RankHandler for ToDeviceHandler<'a, B> {
            type Output = DynTensor<B>;

            fn call<const R: usize>(self) -> BunsenResult<Self::Output> {
                Ok(match self.this.kind {
                    TensorKindDesc::Float => self
                        .this
                        .unwrap_clone::<R, Float>()
                        .to_device(self.device)
                        .into(),
                    TensorKindDesc::Int => self
                        .this
                        .unwrap_clone::<R, Int>()
                        .to_device(self.device)
                        .into(),
                    TensorKindDesc::Bool => self
                        .this
                        .unwrap_clone::<R, Bool>()
                        .to_device(self.device)
                        .into(),
                })
            }
        }
        let rank = self.rank();
        ToDeviceHandler { this: self, device }.dyn_call(rank)
    }

    /// Convert a [`TensorData`] to a [`DynTensor`].
    ///
    /// # Arguments
    /// - `data`: source [`TensorData`].
    /// - `device`: the target device.
    ///
    /// # Result
    /// - `Ok(DynTensor<B>)`: the converted tensor.
    /// - `Err(DynTensorError)`: an error.
    pub fn from_data(
        data: TensorData,
        device: &B::Device,
    ) -> BunsenResult<Self> {
        struct FromDataHandler<'a, B: Backend> {
            data: TensorData,
            device: &'a B::Device,
        }
        impl<'a, B: Backend> RankHandler for FromDataHandler<'a, B> {
            type Output = DynTensor<B>;

            fn call<const R: usize>(self) -> BunsenResult<Self::Output> {
                let kind: TensorKindDesc = self.data.dtype.into();
                Ok(match kind {
                    TensorKindDesc::Float => {
                        Tensor::<B, R, Float>::from_data(self.data, self.device).into()
                    }
                    TensorKindDesc::Int => {
                        Tensor::<B, R, Int>::from_data(self.data, self.device).into()
                    }
                    TensorKindDesc::Bool => {
                        Tensor::<B, R, Bool>::from_data(self.data, self.device).into()
                    }
                })
            }
        }
        let rank = data.rank();
        FromDataHandler { data, device }.dyn_call(rank)
    }

    /// Convert the tensor to a [`TensorData`].
    ///
    /// # Result
    /// - `Ok(TensorData)`: the converted data.
    /// - `Err(DynTensorError)`: an error.
    pub fn into_data(self) -> BunsenResult<TensorData> {
        struct ToDataHandler<B: Backend> {
            this: DynTensor<B>,
        }
        impl<B: Backend> RankHandler for ToDataHandler<B> {
            type Output = TensorData;

            fn call<const R: usize>(self) -> BunsenResult<Self::Output> {
                Ok(match self.this.kind {
                    TensorKindDesc::Float => self.this.unwrap_clone::<R, Float>().into_data(),
                    TensorKindDesc::Int => self.this.unwrap_clone::<R, Int>().into_data(),
                    TensorKindDesc::Bool => self.this.unwrap_clone::<R, Bool>().into_data(),
                })
            }
        }
        let rank = self.rank();
        ToDataHandler { this: self }.dyn_call(rank)
    }

    /// Convert the tensor to a [`TensorData`].
    ///
    /// # Result
    /// - `Ok(TensorData)`: the converted data.
    /// - `Err(DynTensorError)`: an error.
    pub fn to_data(self) -> BunsenResult<TensorData> {
        self.clone().into_data()
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        prelude::{
            Backend,
            s,
        },
        tensor::{
            Bool,
            Distribution,
            Float,
            Int,
            Slice,
            Tensor,
            TensorData,
        },
    };

    use crate::{
        burner::{
            descriptors::TensorKindDesc,
            tensor::dynamic::*,
        },
        errors::{
            BunsenError,
            SlicingError,
        },
        support::testing::{
            DeviceMemoryGuard,
            PerformanceBackend,
            backend_device,
        },
    };

    #[test]
    fn test_is_send() {
        type B = PerformanceBackend;

        fn assert_send<T: Send>() {}

        assert_send::<DynTensor<B>>();
    }

    #[test]
    #[serial_test::serial]
    fn test_stub_float() {
        type B = PerformanceBackend;
        let device = backend_device::<B>();
        let _memory = DeviceMemoryGuard::<B>::new(&device);

        let source: Tensor<B, 2> = Tensor::random([2, 3], Distribution::Default, &device);

        let stub = DynTensor::new(source.clone());

        assert_eq!(stub.rank(), 2);
        assert_eq!(stub.shape(), source.shape());
        assert_eq!(stub.num_elements(), 6);

        assert_eq!(stub.dtype(), source.dtype());
        assert_eq!(
            stub.size_estimate(),
            stub.num_elements() * source.dtype().size()
        );

        assert_eq!(stub.kind(), TensorKindDesc::Float);

        assert_eq!(stub.device(), device);

        assert!(stub.downcast_clone::<2, Int>().is_none());
        assert!(stub.downcast_clone::<2, Bool>().is_none());

        assert!(stub.downcast_clone::<3, Float>().is_none());

        let clone = stub.downcast_clone::<2, Float>().unwrap();
        clone.to_data().assert_eq(&source.clone().to_data(), true);

        stub.clone()
            .into_data()
            .unwrap()
            .assert_eq(&source.clone().to_data(), true);

        let flatten = stub.clone().flatten().unwrap();
        assert_eq!(flatten.shape(), [6].into());
        flatten
            .into_data()
            .unwrap()
            .assert_eq(&source.clone().flatten::<1>(0, 1).to_data(), true);
    }

    #[test]
    #[serial_test::serial]
    fn test_stub_int() {
        type B = PerformanceBackend;
        let device = backend_device::<B>();
        let _memory = DeviceMemoryGuard::<B>::new(&device);

        let source: Tensor<B, 2> = Tensor::random([2, 3], Distribution::Default, &device);
        let source = source.int();

        let stub = DynTensor::new(source.clone());

        assert_eq!(stub.rank(), 2);
        assert_eq!(stub.shape(), source.shape());
        assert_eq!(stub.num_elements(), 6);

        assert_eq!(stub.dtype(), source.dtype());
        assert_eq!(
            stub.size_estimate(),
            stub.num_elements() * source.dtype().size()
        );

        assert_eq!(stub.kind(), TensorKindDesc::Int);

        assert_eq!(stub.device(), device);

        assert!(stub.downcast_clone::<2, Float>().is_none());
        assert!(stub.downcast_clone::<2, Bool>().is_none());

        assert!(stub.downcast_clone::<3, Int>().is_none());

        let clone = stub.downcast_clone::<2, Int>().unwrap();
        clone.to_data().assert_eq(&source.clone().to_data(), true);

        stub.clone()
            .into_data()
            .unwrap()
            .assert_eq(&source.clone().to_data(), true);

        let flatten = stub.clone().flatten().unwrap();
        assert_eq!(flatten.shape(), [6].into());
        flatten
            .into_data()
            .unwrap()
            .assert_eq(&source.clone().flatten::<1>(0, 1).to_data(), true);
    }

    #[test]
    #[serial_test::serial]
    fn test_stub_bool() {
        type B = PerformanceBackend;
        let device = backend_device::<B>();
        let _memory = DeviceMemoryGuard::<B>::new(&device);

        let source: Tensor<B, 2> = Tensor::random([2, 3], Distribution::Bernoulli(0.5), &device);
        let source = source.bool();

        let stub = DynTensor::new(source.clone());

        assert_eq!(stub.rank(), 2);
        assert_eq!(stub.shape(), source.shape());
        assert_eq!(stub.num_elements(), 6);

        assert_eq!(stub.dtype(), source.dtype());
        assert_eq!(
            stub.size_estimate(),
            stub.num_elements() * source.dtype().size()
        );

        assert_eq!(stub.kind(), TensorKindDesc::Bool);

        assert_eq!(stub.device(), device);

        assert!(stub.downcast_clone::<2, Int>().is_none());
        assert!(stub.downcast_clone::<2, Float>().is_none());

        assert!(stub.downcast_clone::<3, Bool>().is_none());

        let clone = stub.downcast_clone::<2, Bool>().unwrap();
        clone.to_data().assert_eq(&source.clone().to_data(), true);

        stub.clone()
            .into_data()
            .unwrap()
            .assert_eq(&source.clone().to_data(), true);

        let flatten = stub.clone().flatten().unwrap();
        assert_eq!(flatten.shape(), [6].into());
        flatten
            .into_data()
            .unwrap()
            .assert_eq(&source.clone().flatten::<1>(0, 1).to_data(), true);
    }

    #[test]
    #[serial_test::serial]
    fn test_clone() {
        type B = PerformanceBackend;
        let device = backend_device::<B>();
        let _memory = DeviceMemoryGuard::<B>::new(&device);

        let source: Tensor<B, 2> = Tensor::random([2, 3], Distribution::Default, &device);

        let stub = DynTensor::new(source.clone());

        let stub_clone = stub.clone();

        assert!(stub_clone.downcast_clone::<3, Float>().is_none());
        assert!(stub_clone.downcast_clone::<2, Int>().is_none());
        let clone = stub_clone.downcast_clone::<2, Float>().unwrap();
        clone.to_data().assert_eq(&source.clone().to_data(), true);
    }

    #[test]
    #[serial_test::serial]
    fn test_slice() {
        type B = PerformanceBackend;
        let device = backend_device::<B>();
        let _memory = DeviceMemoryGuard::<B>::new(&device);

        let source: Tensor<B, 2> = Tensor::random([2, 3], Distribution::Default, &device);

        let stub = DynTensor::new(source.clone());

        let slice = stub.slice(s![.., 1..]).unwrap();
        assert_eq!(slice.shape(), [2, 2].into());
        slice
            .downcast_clone::<2, Float>()
            .unwrap()
            .to_data()
            .assert_eq(&source.clone().slice(s![.., 1..]).to_data(), true);
    }

    #[test]
    #[serial_test::serial]
    fn test_slice_dyn() {
        type B = PerformanceBackend;
        let device = backend_device::<B>();
        let _memory = DeviceMemoryGuard::<B>::new(&device);

        let source: Tensor<B, 2> = Tensor::random([2, 3], Distribution::Default, &device);

        let stub = DynTensor::new(source.clone());

        let slice = stub
            .slice_dyn(&vec![Slice::new(0, None, 1), Slice::new(1, None, 1)])
            .unwrap();
        assert_eq!(slice.shape(), [2, 2].into());
        slice
            .downcast_clone::<2, Float>()
            .unwrap()
            .to_data()
            .assert_eq(&source.clone().slice(s![.., 1..]).to_data(), true);
    }

    /// `[[0, 1, 2], [3, 4, 5]]` as a float [`DynTensor`].
    fn arange_2x3<B: Backend>(device: &B::Device) -> DynTensor<B> {
        Tensor::arange(0..6, device).reshape([2, 3]).float().into()
    }

    #[test]
    #[serial_test::serial]
    fn test_slice_assign() {
        type B = PerformanceBackend;
        let device = backend_device::<B>();
        let _memory = DeviceMemoryGuard::<B>::new(&device);

        let values: Tensor<B, 2> = Tensor::zeros([2, 1], &device);
        let result = arange_2x3(&device)
            .slice_assign::<2, _, _>(s![.., 1..2], values)
            .unwrap();
        assert_eq!(result.shape(), [2, 3].into());
        result.into_data().unwrap().assert_eq(
            &TensorData::from([[0.0f32, 0.0, 2.0], [3.0, 0.0, 5.0]]),
            true,
        );
    }

    #[test]
    #[serial_test::serial]
    fn test_slice_assign_casts_values() {
        type B = PerformanceBackend;
        let device = backend_device::<B>();
        let _memory = DeviceMemoryGuard::<B>::new(&device);

        let values: Tensor<B, 2, Int> = Tensor::from_data([[9], [9]], &device);
        let result = arange_2x3(&device)
            .slice_assign::<2, _, _>(s![.., 0..1], values)
            .unwrap();
        assert_eq!(result.kind(), TensorKindDesc::Float);
        result.into_data().unwrap().assert_eq(
            &TensorData::from([[9.0f32, 1.0, 2.0], [9.0, 4.0, 5.0]]),
            true,
        );
    }

    #[test]
    #[serial_test::serial]
    fn test_slice_assign_tensor_data() {
        type B = PerformanceBackend;
        let device = backend_device::<B>();
        let _memory = DeviceMemoryGuard::<B>::new(&device);

        let result: DynTensor<B> = arange_2x3(&device)
            .slice_assign::<2, _, _>(s![0..1, ..], TensorData::from([[7.0f32, 7.0, 7.0]]))
            .unwrap();
        result.into_data().unwrap().assert_eq(
            &TensorData::from([[7.0f32, 7.0, 7.0], [3.0, 4.0, 5.0]]),
            true,
        );
    }

    #[test]
    #[serial_test::serial]
    fn test_slice_assign_int_and_bool() {
        type B = PerformanceBackend;
        let device = backend_device::<B>();
        let _memory = DeviceMemoryGuard::<B>::new(&device);

        let source: Tensor<B, 1, Int> = Tensor::arange(0..4, &device);
        let values: Tensor<B, 1, Int> = Tensor::from_data([8, 9], &device);
        DynTensor::new(source)
            .slice_assign::<1, _, _>(s![1..3], values)
            .unwrap()
            .into_data()
            .unwrap()
            .convert::<i64>()
            .assert_eq(&TensorData::from([0i64, 8, 9, 3]), false);

        let source: Tensor<B, 1, Bool> = Tensor::from_data([false, false, false], &device);
        let values: Tensor<B, 1, Bool> = Tensor::from_data([true], &device);

        DynTensor::new(source)
            .slice_assign::<1, _, _>(s![2..3], values)
            .unwrap()
            .into_data()
            .unwrap()
            .assert_eq(&TensorData::from([false, false, true]), false);
    }

    #[test]
    #[serial_test::serial]
    fn test_slice_assign_rank_errors() {
        type B = PerformanceBackend;
        let device = backend_device::<B>();
        let _memory = DeviceMemoryGuard::<B>::new(&device);

        let values: Tensor<B, 1> = Tensor::zeros([3], &device);
        assert!(matches!(
            arange_2x3(&device).slice_assign::<2, _, _>(s![0..1, ..], values),
            Err(BunsenError::InvalidArgument { .. })
        ));

        let values: Tensor<B, 2> = Tensor::zeros([1, 3], &device);
        assert!(matches!(
            arange_2x3(&device).slice_assign::<1, _, _>(s![0..1], values),
            Err(BunsenError::InvalidArgument { .. })
        ));
    }

    #[test]
    #[serial_test::serial]
    fn test_slice_assign_out_of_bounds() {
        type B = PerformanceBackend;
        let device = backend_device::<B>();
        let _memory = DeviceMemoryGuard::<B>::new(&device);

        let values: Tensor<B, 2> = Tensor::zeros([1, 3], &device);
        assert!(matches!(
            arange_2x3(&device).slice_assign::<2, _, _>(s![5..6, ..], values),
            Err(BunsenError::SliceError(SlicingError::OutOfBounds { .. }))
        ));
    }

    #[test]
    #[serial_test::serial]
    fn test_slice_assign_dyn() {
        type B = PerformanceBackend;
        let device = backend_device::<B>();
        let _memory = DeviceMemoryGuard::<B>::new(&device);

        let values: Tensor<B, 2> = Tensor::ones([1, 3], &device);
        arange_2x3(&device)
            .slice_assign_dyn(&[Slice::new(1, None, 1), Slice::full()], values)
            .unwrap()
            .into_data()
            .unwrap()
            .assert_eq(
                &TensorData::from([[0.0f32, 1.0, 2.0], [1.0, 1.0, 1.0]]),
                true,
            );
    }

    #[test]
    #[serial_test::serial]
    fn test_slice_assign_dyn_partial_slices() {
        type B = PerformanceBackend;
        let device = backend_device::<B>();
        let _memory = DeviceMemoryGuard::<B>::new(&device);

        let values: Tensor<B, 2> = Tensor::ones([1, 3], &device);
        arange_2x3(&device)
            .slice_assign_dyn(&[Slice::new(0, Some(1), 1)], values)
            .unwrap()
            .into_data()
            .unwrap()
            .assert_eq(
                &TensorData::from([[1.0f32, 1.0, 1.0], [3.0, 4.0, 5.0]]),
                true,
            );
    }

    #[test]
    #[serial_test::serial]
    fn test_slice_assign_dyn_too_many_slices() {
        type B = PerformanceBackend;
        let device = backend_device::<B>();
        let _memory = DeviceMemoryGuard::<B>::new(&device);

        let values: Tensor<B, 2> = Tensor::ones([2, 3], &device);
        assert!(matches!(
            arange_2x3(&device)
                .slice_assign_dyn(&[Slice::full(), Slice::full(), Slice::full()], values),
            Err(BunsenError::SliceError(SlicingError::InvalidRank { .. }))
        ));
    }
}
