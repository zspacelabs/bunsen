//! # Space Primitives
use burn::{
    Tensor,
    module::Module,
    prelude::{
        Backend,
        ElementConversion,
        Int,
    },
    tensor::{
        DType,
        DType::F32,
    },
};

use crate::contracts::unpack_shape_contract;

/// D2Q9 Direction Indices
///
/// # Returns
///
/// The ``[VY=3, VX=3, (VY, VX)=2]`` int direction indices.
pub fn direction_indices<B: Backend>(device: &B::Device) -> Tensor<B, 3, Int> {
    Tensor::<B, 3, Int>::from_data(
        [
            [[-1, -1], [-1, 0], [-1, 1]],
            [[0, -1], [0, 0], [0, 1]],
            [[1, -1], [1, 0], [1, 1]],
        ],
        device,
    )
}

/// D2Q9 Direction Vectors
///
/// # Returns
///
/// The ``[VY=3, VX=3, (VY, VX)=2]`` float direction vectors.
pub fn direction_vectors<B: Backend>(device: &B::Device) -> Tensor<B, 3> {
    direction_indices(device).float()
}

/// D2Q9 Equilibrium Weight Matrix
///
/// # Returns
///
/// The ``[VY=3, VX=3]`` equilibrium weight matrix.
pub fn weight_matrix<B: Backend>(device: &B::Device) -> Tensor<B, 2> {
    Tensor::<B, 2>::from_data(
        [
            [1.0 / 36.0, 1.0 / 9.0, 1.0 / 36.0],
            [1.0 / 9.0, 4.0 / 9.0, 1.0 / 9.0],
            [1.0 / 36.0, 1.0 / 9.0, 1.0 / 36.0],
        ],
        device,
    )
}

/// LBM Space Constants
#[derive(Module, Debug)]
pub struct LbmTables<B: Backend> {
    /// ``[H, W, (Y, X)=2]`` integer direction indices.
    e_idx: Tensor<B, 3, Int>,

    /// ``[H, W, (Y, X)=2]`` float direction vectors.
    e_vec: Tensor<B, 3>,

    /// ``[Y=3, X=3]`` equilibrium weights
    w: Tensor<B, 2>,
}

impl<B: Backend> LbmTables<B> {
    /// Create new space `lbm_tables`.
    pub fn init(device: &B::Device) -> Self {
        let e_idx = direction_indices(device);
        let e_vec = e_idx.clone().float();
        Self {
            e_idx,
            e_vec,
            w: weight_matrix(device),
        }
    }

    /// Get appropriate `lbm_tables` for the distribution.
    pub fn for_dist(dist: &Tensor<B, 4>) -> Self {
        Self::init(&dist.device()).to_dtype(dist.dtype())
    }

    /// Cast the `lbm_tables` to the given dtype.
    pub fn to_dtype(
        self,
        dtype: DType,
    ) -> Self {
        Self {
            e_vec: self.e_vec.cast(dtype),
            w: self.w.cast(dtype),
            ..self
        }
    }

    /// Get the direction indices.
    pub fn e_idx(&self) -> Tensor<B, 3, Int> {
        self.e_idx.clone()
    }

    /// Get the direction vectors.
    pub fn e_vec(&self) -> Tensor<B, 3> {
        self.e_vec.clone()
    }

    /// Get the equilibrium weight matrix.
    pub fn w(&self) -> Tensor<B, 2> {
        self.w.clone()
    }
}

/// Print a distribution.
///
/// Doesn't work well on big distributions.
///
/// # Arguments
///
/// - `label`: the label to use.
/// - `dist`: the dist to print.
pub fn dbg_dist<B: Backend>(
    label: &str,
    dist: Tensor<B, 4>,
) {
    let [height, width] = unpack_shape_contract!(
        ["h", "w", "vy", "vx"],
        dist.shape().as_slice(),
        &["h", "w"],
        &[("vy", 3), ("vx", 3)]
    );

    let total_energy: f32 = dist.clone().sum().into_scalar().elem();
    println!("{label}: {total_energy:<8.2e}");

    let data = dist.cast(F32).to_data().to_vec::<f32>().unwrap();

    fn cross_bar_line(width: usize) {
        print!("+");
        for _ in 0..width {
            for _ in 0..3 {
                print!(" --------");
            }
            print!(" +");
        }
        println!();
    }

    cross_bar_line(width);

    for h in 0..height {
        for vy in 0..3 {
            print!("|");
            for w in 0..width {
                for vx in 0..3 {
                    let v = data[(h * width * 3 * 3) + (w * 3 * 3) + (vy * 3) + vx];
                    print!(" {:>8.2e}", v);
                }
                print!(" |");
            }
            println!();
        }

        cross_bar_line(width);
    }
}

/// Population Density
///
/// # Arguments
///
/// - `dist`: a ``[H, W, VY=3, VX=3]`` population distribution.
///
/// # Returns
///
/// A ``[H, W]`` population density.
pub fn density<B: Backend>(dist: Tensor<B, 4>) -> Tensor<B, 2> {
    dist.sum_dims(&[2, 3]).squeeze_dims::<2>(&[2, 3])
}

/// Compute the directional macroscopic momentum.
///
/// This is the unnormalized macroscopic momentum.
///
/// # Arguments
///
/// - `dist`: a ``[H, W, VY, VX]`` population distribution.
/// - `e`: the D2Q9 direction vectors.
///
/// # Returns
///
/// The ``[H, W, (Y, X)=2]`` momentum.
pub fn macroscopic_momentum<B: Backend>(
    dist: Tensor<B, 4>,
    e: Tensor<B, 3>,
) -> Tensor<B, 3> {
    dist.unsqueeze_dims::<5>(&[-1])
        .mul(e.unsqueeze::<5>())
        .sum_dims(&[2, 3])
        .squeeze_dims::<3>(&[2, 3])
}

/// Computes directional velocity from macroscopic momentum.
///
/// # Arguments
///
/// - `m`: ``[H, W, (Y, X)=2]`` macroscopic momentum.
/// - `rho`: ``[H, W]`` population density.
///
/// # Returns
///
/// The ``[H, W, (Y, X)=2]`` velocity.
pub fn normalize_velocity<B: Backend>(
    m: Tensor<B, 3>,
    rho: Tensor<B, 2>,
) -> Tensor<B, 3> {
    // TODO: div-by-zero check?
    // .clamp_min(1e-15)?
    m.div(rho.unsqueeze_dim(2))
}

/// Compute the directional macroscopic velocity.
///
/// # Arguments
///
/// - `dist`: a ``[H, W, VY, VX]`` population distribution.
/// - `e`: the D2Q9 direction vectors.
///
/// # Returns
/// - ``[H, W, (Y, X)=2]`` velocity.
pub fn macroscopic_velocity<B: Backend>(
    dist: Tensor<B, 4>,
    rho: Tensor<B, 2>,
    e: Tensor<B, 3>,
) -> Tensor<B, 3> {
    normalize_velocity(macroscopic_momentum(dist, e), rho)
}

/// Compute the squared velocity field.
///
/// # Arguments
/// - `u`: ``[H, W, (Y, X)=2]`` macroscopic velocity
///
/// # Returns
/// - ``[H, W]`` velocity magnitude squared
pub fn velocity_squared<B: Backend>(u: Tensor<B, 3>) -> Tensor<B, 2> {
    u.square().sum_dim(2).squeeze_dims::<2>(&[2])
}

/// Compute the first (density) and second (macro velocity) moments.
///
/// # Arguments
///
/// - `dist`: a ``[H, W, VY, VX]`` population distribution.
/// - `e`: the D2Q9 direction vectors.
///
/// # Returns
/// ``(density, velocity)`` where:
/// - `density`: ``[H, W]``
/// - `velocity`: ``[H, W, (Y, X)=2]``
pub fn moments<B: Backend>(
    dist: Tensor<B, 4>,
    lbm_tables: &LbmTables<B>,
) -> (Tensor<B, 2>, Tensor<B, 3>) {
    let rho = density(dist.clone());
    let u = macroscopic_velocity(dist, rho.clone(), lbm_tables.e_vec());
    (rho, u)
}

/// Fold a distribution into windows.
///
/// Note the geometry: ``[H-2, W-2, VY=3, VX=3, WIN_Y=3, WIN_X=3]``
/// - ``(H, W)``: the spatial position of the "current" cell.
/// - ``(WIN_Y, WIN_X)``: the current window; with the current cell in the
///   center.
/// - ``(VY, VX)``: the 3x3 distribution in each cell.
///
/// # Arguments
/// - `dist`: a ``[H, W, VY=3, VX=3]`` distribution.
///
/// # Returns
/// ``[H, W, VY=3, VX=3, WIN_Y=3, WIN_X=3]`` folded windows.
pub fn dist_windows<B: Backend>(dist: Tensor<B, 4>) -> Tensor<B, 6> {
    dist.unfold::<5, _>(0, 3, 1).unfold::<6, _>(1, 3, 1)
}

#[cfg(test)]
mod tests {
    use burn::tensor::Tolerance;
    use serial_test::serial;

    use super::{
        super::velocity_squared,
        *,
    };
    use crate::support::testing::PerformanceBackend;

    #[test]
    #[serial]
    fn test_population_density() {
        type B = PerformanceBackend;
        let device = Default::default();

        let dist: Tensor<B, 4> = Tensor::from_data(
            [
                [
                    [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                    [[10., 20., 30.], [40., 50., 60.], [70., 80., 90.]],
                ],
                [
                    [[9., 10., 3.], [4., 5., 6.], [7., 8., 9.]],
                    [[0., -2., 0.], [0., 8., 0.], [0., 0., 0.]],
                ],
            ],
            &device,
        );

        let rho = density(dist.clone());

        rho.to_data().assert_approx_eq::<f32>(
            &Tensor::<B, 2>::from_data([[45., 450.], [61., 6.]], &device).to_data(),
            Tolerance::default(),
        )
    }

    #[test]
    #[serial]
    fn test_direction_vectors() {
        type B = PerformanceBackend;
        let device = Default::default();

        let e: Tensor<B, 3> = direction_vectors(&device);

        e.to_data().assert_eq(
            &Tensor::<B, 3>::from_data(
                [
                    [[-1., -1.], [-1., 0.], [-1., 1.]],
                    [[0., -1.], [0., 0.], [0., 1.]],
                    [[1., -1.], [1., 0.], [1., 1.]],
                ],
                &device,
            )
            .to_data(),
            false,
        );
    }

    #[test]
    #[serial]
    fn test_weight_matrix() {
        type B = PerformanceBackend;
        let device = Default::default();

        let w: Tensor<B, 2> = weight_matrix(&device);

        w.to_data().assert_eq(
            &Tensor::<B, 2>::from_data(
                [
                    [1.0 / 36.0, 1.0 / 9.0, 1.0 / 36.0],
                    [1.0 / 9.0, 4.0 / 9.0, 1.0 / 9.0],
                    [1.0 / 36.0, 1.0 / 9.0, 1.0 / 36.0],
                ],
                &device,
            )
            .to_data(),
            false,
        );
    }

    #[test]
    #[serial]
    fn test_momentum_and_velocity() {
        type B = PerformanceBackend;
        let device = Default::default();

        let dist: Tensor<B, 4> = Tensor::from_data(
            [[
                [[1., 0., 0.], [0., 10., 0.], [0., 0., 0.]],
                [[1., 2., 3.], [4., 10., 5.], [6., 7., 8.]],
            ]],
            &device,
        );

        let lbm_tables = LbmTables::for_dist(&dist);

        let momentum = macroscopic_momentum(dist.clone(), lbm_tables.e_vec());

        momentum.clone().to_data().assert_approx_eq::<f32>(
            &Tensor::<B, 3>::from_data([[[-1., -1.], [15., 5.]]], &device).to_data(),
            Tolerance::default(),
        );

        let (rho, u) = moments(dist.clone(), &lbm_tables);

        let rho_data = rho.to_data().to_vec::<f32>().unwrap();

        u.clone().to_data().assert_approx_eq::<f32>(
            &Tensor::<B, 3>::from_data(
                [[
                    [-1. / rho_data[0], -1. / rho_data[0]],
                    [15. / rho_data[1], 5. / rho_data[1]],
                ]],
                &device,
            )
            .to_data(),
            Tolerance::default(),
        );

        let v_sq = velocity_squared(u.clone());

        v_sq.clone().to_data().assert_approx_eq::<f32>(
            &Tensor::<B, 2>::from_data(
                [[
                    (1. + 1.) / rho_data[0].powi(2),
                    (15. * 15. + 5. * 5.) / rho_data[1].powi(2),
                ]],
                &device,
            )
            .to_data(),
            Tolerance::default(),
        );
    }
}
