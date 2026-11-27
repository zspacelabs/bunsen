//! Named tolerances, resolved per backend.

use burn::tensor::{
    Device,
    Tolerance,
};

/// The backend family a test device runs on, as far as tolerances care.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TestBackend {
    /// A CPU backend: `Flex`, `NdArray`.
    Cpu,
    /// cubecl's wgpu runtime: WebGPU, Vulkan, Metal.
    Wgpu,
    /// cubecl's CUDA runtime, whose kernels run with fast math and whose
    /// matmuls may round through TF32.
    Cuda,
    /// Anything else.
    Other,
}

impl TestBackend {
    /// Every family.
    pub const ALL: [Self; 4] = [Self::Cpu, Self::Wgpu, Self::Cuda, Self::Other];

    /// The family of `device`'s backend, from burn's name for it.
    pub fn of(device: &Device) -> Self {
        use burn::backend::{
            Backend,
            Dispatch,
        };
        Self::parse(&Dispatch::name(device.as_dispatch()))
    }

    fn parse(name: &str) -> Self {
        if name.contains("cubecl<cuda") {
            Self::Cuda
        } else if name.contains("cubecl<wgpu") {
            Self::Wgpu
        } else if name.contains("flex") || name.contains("ndarray") {
            Self::Cpu
        } else {
            Self::Other
        }
    }

    const fn index(self) -> usize {
        match self {
            Self::Cpu => 0,
            Self::Wgpu => 1,
            Self::Cuda => 2,
            Self::Other => 3,
        }
    }
}

/// A tolerance that may resolve per device: a [`Tolerance`] is the same on
/// every device, a [`NamedTolerance`] has per-backend values.
///
/// The tensor assertions ([`assert_tensors_close`](super::assert_tensors_close)
/// and its siblings) take either, and resolve it on the tensor's device.
pub trait DeviceTolerance {
    /// The tolerance for a comparison on `device`.
    fn on(
        &self,
        device: &Device,
    ) -> Tolerance<f32>;
}

impl DeviceTolerance for Tolerance<f32> {
    fn on(
        &self,
        _device: &Device,
    ) -> Tolerance<f32> {
        *self
    }
}

impl<T: DeviceTolerance + ?Sized> DeviceTolerance for &T {
    fn on(
        &self,
        device: &Device,
    ) -> Tolerance<f32> {
        (**self).on(device)
    }
}

/// A tolerance with a name, a reason in its doc, and a value per backend:
/// one value everywhere unless an override names a backend.
///
/// The overrides are how a test says "this tight on the backends that hold
/// it, looser where the backend's arithmetic differs" (CUDA's TF32
/// matmuls), instead of keeping a list of expected failures.
#[derive(Clone, Copy, Debug)]
pub struct NamedTolerance {
    /// What the tolerance is for, in a failure message or a doc.
    pub name: &'static str,
    /// `(relative, absolute)`, on every backend without an override.
    pub rel_abs: (f64, f64),
    overrides: [Option<(f64, f64)>; 4],
}

impl NamedTolerance {
    /// `relative` and `absolute` on every backend (burn's
    /// [`Tolerance::rel_abs`]).
    pub const fn new(
        name: &'static str,
        relative: f64,
        absolute: f64,
    ) -> Self {
        Self {
            name,
            rel_abs: (relative, absolute),
            overrides: [None; 4],
        }
    }

    /// The same, but `relative` and `absolute` on `backend`.
    pub const fn on_backend(
        mut self,
        backend: TestBackend,
        relative: f64,
        absolute: f64,
    ) -> Self {
        self.overrides[backend.index()] = Some((relative, absolute));
        self
    }

    /// The `(relative, absolute)` pair on `backend`.
    pub const fn rel_abs_on(
        &self,
        backend: TestBackend,
    ) -> (f64, f64) {
        match self.overrides[backend.index()] {
            Some(pair) => pair,
            None => self.rel_abs,
        }
    }

    /// The tolerance on `backend`.
    pub fn for_backend(
        &self,
        backend: TestBackend,
    ) -> Tolerance<f32> {
        let (relative, absolute) = self.rel_abs_on(backend);
        Tolerance::rel_abs(relative, absolute)
    }
}

impl DeviceTolerance for NamedTolerance {
    fn on(
        &self,
        device: &Device,
    ) -> Tolerance<f32> {
        self.for_backend(TestBackend::of(device))
    }
}

/// burn's `Tolerance::default()` (`balanced`): relative `0.5%`, absolute
/// `1e-5`, the model tests' tolerance and the CUDA fallback of the tight ones.
const BALANCED: (f64, f64) = (0.005, 1e-5);

/// Bit-identical: values only uploaded, copied, sliced or rearranged, never
/// computed. The same on every backend.
pub const IDENTICAL: NamedTolerance = NamedTolerance::new("identical", 0.0, 0.0);

/// The same `f32` arithmetic up to rounding: relative and absolute `1e-6`.
/// CUDA, whose fast math and TF32 matmuls round differently, gets burn's
/// default.
pub const ROUNDING: NamedTolerance = NamedTolerance::new("rounding", 1e-6, 1e-6).on_backend(
    TestBackend::Cuda,
    BALANCED.0,
    BALANCED.1,
);

/// Two orders of one computation on one device (chunked against whole,
/// cached against uncached): relative `1e-4`, absolute `1e-5`. CUDA gets
/// burn's default.
pub const REORDER: NamedTolerance = NamedTolerance::new("reorder", 1e-4, 1e-5).on_backend(
    TestBackend::Cuda,
    BALANCED.0,
    BALANCED.1,
);

/// A toy's run against its float64 reference, stored as float32: relative
/// `1e-3`, absolute `1e-4`, the tolerance of
/// [`assert_io_close`](super::assert_io_close). CUDA gets burn's default.
pub const TOY: NamedTolerance =
    NamedTolerance::new("toy", 1e-3, 1e-4).on_backend(TestBackend::Cuda, BALANCED.0, BALANCED.1);

#[cfg(test)]
mod tests {
    use alloc::format;

    use super::*;
    use crate::support::testing::cpu_device;

    #[test]
    fn test_backend_of_a_name() {
        assert_eq!(TestBackend::of(&cpu_device()), TestBackend::Cpu);
        assert_eq!(
            TestBackend::parse("dispatch<fusion<cubecl<cuda<f32>>>>"),
            TestBackend::Cuda
        );
        assert_eq!(
            TestBackend::parse("dispatch<cubecl<wgpu<vulkan>>>"),
            TestBackend::Wgpu
        );
        assert_eq!(TestBackend::parse("dispatch<tch>"), TestBackend::Other);
    }

    /// The performance device is the backend its feature selects.
    #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
    #[test]
    fn test_performance_device_backend() {
        if std::env::var_os("BURN_DEVICE").is_none() {
            assert_eq!(
                TestBackend::of(&crate::support::testing::performance_device()),
                TestBackend::Wgpu
            );
        }
    }

    /// An override applies to its backend alone.
    #[test]
    fn test_overrides() {
        for backend in TestBackend::ALL {
            assert_eq!(IDENTICAL.rel_abs_on(backend), (0.0, 0.0));
        }
        assert_eq!(ROUNDING.rel_abs_on(TestBackend::Wgpu), (1e-6, 1e-6));
        assert_eq!(ROUNDING.rel_abs_on(TestBackend::Cpu), (1e-6, 1e-6));
        assert_eq!(ROUNDING.rel_abs_on(TestBackend::Cuda), BALANCED);
        let show = |t: Tolerance<f32>| format!("{t:?}");
        assert_eq!(
            show(TOY.on(&cpu_device())),
            show(Tolerance::rel_abs(1e-3, 1e-4))
        );
        let plain = Tolerance::<f32>::absolute(1e-3);
        assert_eq!(show(plain.on(&cpu_device())), show(plain));
    }
}
