//! Testing utilities.
pub mod asr;

use std::{
    any::{
        Any,
        TypeId,
    },
    fmt::Debug,
    sync::LazyLock,
};

use burn::{
    module::Param,
    prelude::{
        Backend,
        Tensor,
        TensorData,
    },
    tensor::{
        Tolerance,
        backend::DeviceOps,
    },
};
use dashmap::DashMap;
use num_traits::float::Float;

use crate::burner::tensor::TensorElemOpExt;

cfg_select! {
    feature = "cuda" => {
        /// Selected burn backend for compute-heavy tests.
        pub type PerformanceBackend = ::burn::backend::Cuda;
    }
    feature = "metal" => {
        /// Selected burn backend for compute-heavy tests.
        pub type PerformanceBackend = ::burn::backend::Metal;
    }
    feature = "vulkan" => {
        /// Selected burn backend for compute-heavy tests.
        pub type PerformanceBackend = ::burn::backend::Vulkan;
    }
    feature = "wgpu" => {
        /// Selected burn backend for compute-heavy tests.
        pub type PerformanceBackend = ::burn::backend::Wgpu;
    }
    _ => {
        /// Selected burn backend for compute-heavy tests.
        pub type PerformanceBackend = ::burn::backend::Flex;
    }
}

/// Type-erased cache values, keyed by the [`TypeId`] of the value's own type.
///
/// Spelled as an alias rather than inline: `clippy::type_complexity` fires on
/// the nested form, and the workspace denies warnings.
type ValueCache = DashMap<TypeId, Box<dyn Any + Send + Sync>>;

/// Process-global cache holding one value per value type.
static VALUE_CACHE: LazyLock<ValueCache> = LazyLock::new(DashMap::new);

/// Read the cached `V`, inserting `init()` if the slot is empty.
///
/// The slot is keyed by `V`'s own [`TypeId`], so a key can only ever name the
/// type it stores and the downcast cannot fail.
fn cached_value<V>(init: impl FnOnce() -> V) -> V
where
    V: Any + Clone + Send + Sync,
{
    let key = TypeId::of::<V>();

    // Scoped so the shard read lock is released before `entry` asks the same
    // shard for a write lock: dashmap's locks neither upgrade nor re-enter, so
    // a live `get` guard here would park the thread forever. Returning from
    // inside the `if let` drops the guard on the way out.
    if let Some(entry) = VALUE_CACHE.get(&key) {
        return downcast_value(&**entry);
    }

    // Built before the map is touched, so no shard lock is ever held across
    // `init`. Two threads racing a cold slot both build one; only one insert
    // wins, and reading back through the guard rather than returning `value`
    // has the loser return the winner's value instead of its own.
    let value: Box<dyn Any + Send + Sync> = Box::new(init());
    let entry = VALUE_CACHE.entry(key).or_insert(value);

    downcast_value(&**entry)
}

/// Clone a `V` out of a cache slot.
///
/// # Panics
///
/// Panics if the slot does not hold a `V`. Both writers file a `V` under
/// `TypeId::of::<V>()` and nothing else can reach the static, so a key holds
/// exactly one concrete type for the life of the process; this is unreachable.
fn downcast_value<V>(entry: &(dyn Any + Send + Sync)) -> V
where
    V: Any + Clone,
{
    entry
        .downcast_ref::<V>()
        .expect("cache slot holds a type other than the one keying it")
        .clone()
}

/// Replace the cached `V`.
fn set_cached_value<V>(value: V)
where
    V: Any + Send + Sync,
{
    VALUE_CACHE.insert(TypeId::of::<V>(), Box::new(value));
}

/// Empty the slot holding the cached `V`.
fn clear_cached_value<V>()
where
    V: Any,
{
    VALUE_CACHE.remove(&TypeId::of::<V>());
}

/// Get the shared default device of type `D`.
///
/// The first call for a given device type builds `D::default()` and caches it
/// process-wide; every later call returns a clone of that one device, so the
/// whole process shares a single handle per device type.
///
/// `D` is inferred from the surrounding use, which makes this a drop-in for
/// `Default::default()` at a call site that goes on to build a tensor:
///
/// ```
/// # use burn::prelude::{Tensor, TensorData};
/// # use bunsen::support::testing::{CpuBackend, default_device};
/// let device = default_device();
/// let tensor = Tensor::<CpuBackend, 1>::from_data(
///     TensorData::from([1.0, 2.0]),
///     &device,
/// );
/// ```
///
/// Use [`backend_device`] where a backend, rather than a device, is the type
/// in hand. [`set_default_device`] replaces the cached device, and
/// [`reset_default_device`] drops it.
///
/// The cache is keyed on the **device** type, so backends that share one — a
/// backend and its `Autodiff` wrapper, say — share a single cached device.
pub fn default_device<D: DeviceOps>() -> D {
    cached_value(D::default)
}

/// Get the shared default device for backend `B`.
///
/// A spelling of [`default_device`] for call sites holding a backend type
/// rather than a device type; both reach the same cache slot, since the slot
/// is keyed by `B::Device`.
///
/// ```
/// # use bunsen::support::testing::{CpuBackend, backend_device};
/// let device = backend_device::<CpuBackend>();
/// ```
pub fn backend_device<B: Backend>() -> B::Device {
    default_device::<B::Device>()
}

/// Pin the device that [`default_device`] returns for device type `D`.
///
/// Replaces any cached device; callers already holding a clone of the old one
/// keep it. Intended for a test binary or harness that selects a device once,
/// before the tests that use it run.
///
/// `D` is inferred from `device`, so this needs no turbofish.
pub fn set_default_device<D: DeviceOps>(device: D) {
    set_cached_value(device);
}

/// Drop the cached device of type `D`.
///
/// The next [`default_device`] call rebuilds it from `D::default()`, undoing a
/// [`set_default_device`].
pub fn reset_default_device<D: DeviceOps>() {
    clear_cached_value::<D>();
}

/// Selected burn backend for fast-setup tests.
pub type CpuBackend = ::burn::backend::Flex;

/// Asserts that two vectors of floating-point numbers are close to each other
/// within a given tolerance.
pub fn assert_close_to_vec<T>(
    actual: &[T],
    expected: &[T],
    tolerance: T,
) where
    T: Float + std::ops::Sub<Output = T> + std::ops::Add<Output = T> + Copy + Debug,
{
    let mut pass = actual.len() == expected.len();
    for (&a, &e) in actual.iter().zip(expected.iter()) {
        if !pass {
            break;
        }
        if (a - e).abs() > tolerance {
            pass = false;
            break;
        }
    }
    if !pass {
        panic!("Expected (+/- {tolerance:?}):\n{expected:?}\nActual:\n{actual:?}");
    }
}

/// Asserts that a tensor matches a row-major host buffer.
///
/// The comparison runs through [`TensorData::assert_approx_eq`], so a mismatch
/// reports the differing shape, or the relative and absolute error, rather than
/// a bare element index.
///
/// # Panics
///
/// Panics if `expected` does not hold exactly `actual.dims()` elements, or if
/// any pair differs by more than `tolerance`.
pub fn assert_tensor_close_to_vec<B, const D: usize>(
    actual: &Tensor<B, D>,
    expected: &[f64],
    tolerance: Tolerance<B::FloatElem>,
) where
    B: Backend,
    B::FloatElem: Float,
{
    let expected = TensorData::new(expected.to_vec(), actual.dims()).convert::<B::FloatElem>();
    actual
        .to_data_as::<B::FloatElem>()
        .assert_approx_eq::<B::FloatElem>(&expected, tolerance);
}

/// Asserts that two tensors of the same shape are approximately equal.
///
/// Both sides are read back at the backend's float element type, so tensors
/// that differ only in dtype still compare.
///
/// # Panics
///
/// Panics if the shapes differ, or if any pair of values differs by more than
/// `tolerance`.
pub fn assert_tensors_close<B, const D: usize>(
    actual: &Tensor<B, D>,
    expected: &Tensor<B, D>,
    tolerance: Tolerance<B::FloatElem>,
) where
    B: Backend,
    B::FloatElem: Float,
{
    actual
        .to_data_as::<B::FloatElem>()
        .assert_approx_eq::<B::FloatElem>(&expected.to_data_as::<B::FloatElem>(), tolerance);
}

/// Applies a parameter's **load**-path mapping to a tensor.
///
/// A [`Param`] can carry transformations that run only as it crosses a store
/// boundary — see [`repair_pytorch_strided_weight`] — which makes them
/// invisible from the outside. This exposes the load side, so a test can
/// assert *which* mappings a module attached, and to which parameters.
///
/// [`repair_pytorch_strided_weight`]:
///     crate::burner::store::repair_pytorch_strided_weight
pub fn param_load_mapping<B, const D: usize>(
    param: &Param<Tensor<B, D>>,
    tensor: Tensor<B, D>,
) -> Tensor<B, D>
where
    B: Backend,
{
    param.clone().consume().2.on_load(tensor)
}

#[cfg(test)]
mod tests {
    use std::any::TypeId;

    use burn::{
        prelude::{
            Device,
            Tensor,
            TensorData,
        },
        tensor::Tolerance,
    };
    use serial_test::serial;

    use crate::support::testing::{
        CpuBackend,
        PerformanceBackend,
        VALUE_CACHE,
        assert_close_to_vec,
        assert_tensor_close_to_vec,
        assert_tensors_close,
        backend_device,
        cached_value,
        clear_cached_value,
        default_device,
        reset_default_device,
        set_cached_value,
        set_default_device,
    };

    type B = PerformanceBackend;

    /// A `[2, 2]` tensor holding `values` in row-major order.
    fn square(
        values: [f64; 4],
        device: &Device<B>,
    ) -> Tensor<B, 2> {
        Tensor::from_data(TensorData::new(values.to_vec(), [2, 2]), device)
    }

    #[test]
    fn test_assert_close_to_vec() {
        let actual = vec![1.0, 2.0, 3.0];
        let expected = vec![1.0, 2.0, 3.0];
        assert_close_to_vec(&actual, &expected, 0.01);

        let actual = vec![1.0, 2.0, 3.1];
        let expected = vec![1.0, 2.0, 3.0];
        assert_close_to_vec(&actual, &expected, 0.2);
    }

    #[test]
    #[should_panic]
    fn test_assert_close_to_vec_bad_values() {
        let actual = vec![1.0, 2.0, 3.0];
        let expected = vec![1.0, 2.0, 3.5];
        assert_close_to_vec(&actual, &expected, 0.01);
    }

    #[test]
    #[should_panic]
    fn test_assert_close_to_vec_different_lengths() {
        let actual = vec![1.0, 2.0];
        let expected = vec![1.0, 2.0, 3.0];
        assert_close_to_vec(&actual, &expected, 0.01);
    }

    #[test]
    fn test_assert_tensor_close_to_vec() {
        let device = default_device();
        let t = square([1.0, 2.0, 3.0, 4.0], &device);
        assert_tensor_close_to_vec(&t, &[1.0, 2.0, 3.0, 4.0], Tolerance::default());
    }

    #[test]
    #[should_panic]
    fn test_assert_tensor_close_to_vec_bad_values() {
        let device = default_device();
        let t = square([1.0, 2.0, 3.0, 4.0], &device);
        assert_tensor_close_to_vec(&t, &[1.0, 2.0, 3.0, 9.0], Tolerance::default());
    }

    #[test]
    fn test_assert_tensors_close() {
        let device = default_device();
        let a = square([1.0, 2.0, 3.0, 4.0], &device);
        let b = square([1.0, 2.0, 3.0, 4.0], &device);
        assert_tensors_close(&a, &b, Tolerance::default());
    }

    #[test]
    #[should_panic]
    fn test_assert_tensors_close_bad_values() {
        let device = default_device();
        let a = square([1.0, 2.0, 3.0, 4.0], &device);
        let b = square([1.0, 2.0, 3.0, 9.0], &device);
        assert_tensors_close(&a, &b, Tolerance::default());
    }

    // The cache is process-global, and the `default_device` call sites across
    // the crate read the real device slots without serializing against
    // anything. So the tests that *mutate* the cache below use scratch value
    // types that nothing else touches, and the device-level tests only ever
    // restore a slot to the value it already holds.

    #[test]
    #[serial(value_cache)]
    fn test_cached_value_memoizes() {
        clear_cached_value::<u32>();

        assert_eq!(cached_value(|| 7_u32), 7);

        // The slot is populated now, so `init` no longer runs.
        assert_eq!(cached_value(|| 9_u32), 7);

        clear_cached_value::<u32>();
    }

    #[test]
    #[serial(value_cache)]
    fn test_cached_value_set_and_clear() {
        clear_cached_value::<u32>();
        assert!(!VALUE_CACHE.contains_key(&TypeId::of::<u32>()));

        // The setter populates the slot on its own, with no read in between.
        set_cached_value(42_u32);
        assert!(VALUE_CACHE.contains_key(&TypeId::of::<u32>()));
        assert_eq!(cached_value(|| 7_u32), 42);

        // Clearing sends the next read back to `init`.
        clear_cached_value::<u32>();
        assert!(!VALUE_CACHE.contains_key(&TypeId::of::<u32>()));
        assert_eq!(cached_value(|| 7_u32), 7);

        clear_cached_value::<u32>();
    }

    #[test]
    #[serial(value_cache)]
    fn test_cached_value_types_do_not_collide() {
        clear_cached_value::<u32>();
        clear_cached_value::<String>();

        assert_eq!(cached_value(|| 7_u32), 7);

        // A second type holds an independent slot. Were the two to share one,
        // this would fail the downcast rather than the assertion.
        assert_eq!(cached_value(|| "b".to_string()), "b");
        assert_eq!(cached_value(|| 0_u32), 7);

        clear_cached_value::<u32>();
        clear_cached_value::<String>();
    }

    #[test]
    fn test_device_entry_points_agree() {
        let device = backend_device::<CpuBackend>();

        // Reading populated the slot. Membership is the only observable signal
        // when the device type has a single inhabitant, as `FlexDevice` does:
        // under the CPU configurations every assertion on a device *value* is
        // vacuously true. Under `--features wgpu` the equalities below carry
        // their own weight.
        assert!(VALUE_CACHE.contains_key(&TypeId::of::<Device<CpuBackend>>()));

        assert_eq!(device, backend_device::<CpuBackend>());
        assert_eq!(device, default_device::<Device<CpuBackend>>());
    }

    #[test]
    fn test_default_device_set_and_reset_round_trip() {
        // Every write below restores the slot to the value it already holds,
        // so racing a concurrent reader is harmless: a reset it observes is
        // re-memoized to the same default.
        let device = backend_device::<CpuBackend>();

        set_default_device(device.clone());
        assert_eq!(backend_device::<CpuBackend>(), device);

        reset_default_device::<Device<CpuBackend>>();
        assert_eq!(backend_device::<CpuBackend>(), device);
    }
}
