#![allow(unused)]
use std::{
    sync::{
        Arc,
        Mutex,
        atomic::{
            AtomicBool,
            Ordering,
        },
    },
    thread,
    thread::JoinHandle,
    time::Duration,
};

use bunsen::{
    burner::tensor::TensorDataView,
    kits::sims::lbm::d2q9::{
        LBMD2Q9Config,
        LBMD2Q9State,
        LBMMeta,
        LbmTables,
        RelaxationParam,
        SPEED_OF_SOUND,
        macroscopic_momentum,
    },
    prelude::{
        TensorDataViewExt,
        TensorElemOpExt,
        TensorOpExt,
    },
    support::validators::parse_grid_shape,
};
use burn::{
    Tensor,
    prelude::{
        Backend,
        Bool,
        ElementConversion,
        TensorData,
        s,
    },
    tensor::DType,
};
use clap::Parser;
use glutin_window::GlutinWindow as Window;
use indicatif::ProgressBar;
use opengl_graphics::{
    GlGraphics,
    OpenGL,
};
use piston::{
    EventLoop,
    OpenGLWindow,
    RenderArgs,
    event_loop::{
        EventSettings,
        Events,
    },
    input::RenderEvent,
    window::WindowSettings,
};
use rand::RngExt;

/// Fluid Flow demo for Burn.
#[derive(Parser, Debug)]
#[command(long_about = None)]
pub struct Args {
    /// The grid shape as `HEIGHT,WIDTH`, or `SIZE`.
    #[arg(long, value_parser=parse_grid_shape, default_value="400")]
    pub grid_shape: [usize; 2],

    /// The max frames per second.
    #[arg(long, default_value_t = 60)]
    pub fps: u64,

    /// The tics per second.
    #[arg(long, default_value_t = 0.0)]
    pub tps: f32,

    /// The initial window zoom.
    #[arg(long, default_value_t = 1.5)]
    pub zoom: f64,

    /// The display opacity of updates.
    #[arg(long, default_value_t = 0.8)]
    pub opacity: f32,

    /// The number of steps to skip on init.
    #[arg(long, default_value_t = 1)]
    pub init_skip_steps: usize,

    /// The collision relaxation tau.
    #[arg(long, default_value_t = 0.9)]
    pub tau: f64,
}

fn main() {
    let args = Args::parse();
    println!("{:#?}", args);

    cfg_select! {
        feature = "cuda" => {
            println!("CUDA enabled");
            run::<burn::backend::Cuda<f32, i32>>(&args, DType::F32);
        }
        feature = "metal" => {
            println!("Metal enabled");
            run::<burn::backend::Metal<f32, i32>>(&args, DType::F32);
        }
        feature = "wgpu" => {
            println!("WGPU enabled");
            run::<burn::backend::Wgpu<f32, i32>>(&args, DType::F32);
        }
        feature = "flex" => {
            println!("Flex enabled");
            run::<burn::backend::Flex>(&args, DType::F32);
        }
        _ => {
            compile_error!("No backend selected");
        }
    }
}

fn run<B: Backend>(
    args: &Args,
    dtype: DType,
) {
    let device = Default::default();

    // Change this to OpenGL::V2_1 if not working.
    let opengl = OpenGL::V3_2;

    let [height, width] = args.grid_shape;

    let background_density = SPEED_OF_SOUND / 100.0;

    let mut world_state: LBMD2Q9State<B> = LBMD2Q9Config::new(args.grid_shape)
        .with_relaxation(RelaxationParam::Tau(args.tau))
        .init(&device, background_density);
    world_state.dist = world_state
        .dist
        .slice_fill(s![50, 20, 1, 1], 5.0 * background_density)
        .slice_fill(s![20, 100, 1, 1], 5.0 * background_density);

    let h6 = (height / 6) as isize;
    let w6 = (width / 6) as isize;
    let stroke = (height / 20) as isize;

    world_state.solid_mask = world_state
        .solid_mask
        .slice_fill(s![2 * h6..2 * h6 + stroke, w6..3 * w6], true)
        .slice_fill(s![2 * h6..2 * h6 + stroke, -2 * w6..-w6], true)
        .slice_fill(s![-h6..-h6 + stroke, w6..2 * w6], true)
        .slice_fill(s![-h6..-h6 + stroke, -3 * w6..-w6], true);

    let mut world_state = world_state.to_dtype(dtype);
    world_state.save_correct_total_mass();

    for _ in 0..args.init_skip_steps {
        world_state.advance_step();
    }

    let solid_mask: Tensor<B, 2, Bool> = world_state.solid_mask.clone();

    let sim_delay = if args.tps > 0.0 {
        Some(Duration::from_secs_f32(1.0 / args.tps))
    } else {
        None
    };

    let vis_cells: Arc<Mutex<TensorData>> =
        Arc::new(Mutex::new(TensorData::zeros::<f32, _>([height, width, 2])));
    let vis_cells_publish = vis_cells.clone();
    let constants: LbmTables<B> = LbmTables::for_dist(&world_state.dist);

    let mut last_export = std::time::Instant::now();
    let export_delay = Duration::from_secs_f32(1.0 / args.fps as f32);

    let sim = Simulation::new(world_state, sim_delay, move |_step_index, state| {
        let now = std::time::Instant::now();
        let dt = now - last_export;

        if dt > export_delay {
            let cells = macroscopic_momentum(state.clone(), constants.e_vec());

            // let scale = cells.clone().max().into_scalar();
            let scale = SPEED_OF_SOUND / 1000.0;

            let cells = ((cells / scale) + 1.0) / 2.0;
            // let cells = cells.mul_scalar(std::f64::consts::PI / 2.0).sin();

            *vis_cells_publish.lock().unwrap() = cells.to_data_convert::<f32>();

            last_export = std::time::Instant::now();
        }
    });

    // Create a Glutin window.
    let mut window: Window = WindowSettings::new(
        format!("lattice-boltzmann-2q9-flow {height}x{width}"),
        [width as f64 * args.zoom, height as f64 * args.zoom],
    )
    .graphics_api(opengl)
    .exit_on_esc(true)
    .build()
    .unwrap();

    // Load the OpenGL function pointers
    gl::load_with(|s| window.get_proc_address(s) as *const _);
    // Create a new game and run it.
    let mut app = FlowVisApp {
        gl: GlGraphics::new(opengl),
        cell_data: vis_cells,
        solid_mask: solid_mask.to_data_convert::<bool>(),
        opacity: args.opacity,
    };

    let mut events = Events::new(EventSettings::new());
    events.set_ups(args.fps);

    while let Some(e) = events.next(&mut window) {
        if let Some(render_args) = e.render_args() {
            app.render(&render_args);
        }
    }

    sim.shutdown();
}

pub struct Simulation {
    handle: Option<JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
}

impl Simulation {
    pub fn new<B: Backend, F>(
        world: LBMD2Q9State<B>,
        step_duration: Option<Duration>,
        mut observer: F,
    ) -> Self
    where
        F: FnMut(usize, Tensor<B, 4>) + Send + 'static,
    {
        let shutdown = Arc::new(AtomicBool::new(false));
        let [height, width] = world.shape();

        let shutdown_clone = shutdown.clone();

        let handle = thread::spawn(move || {
            let progress = ProgressBar::new_spinner();

            let mut world = world;

            let mut stash = 0.0;

            let delay_smoothing = 20;
            let mut avg_delay = std::time::Duration::from_secs_f32(0.0);
            let mut last_time = std::time::Instant::now();

            while !shutdown_clone.load(Ordering::Relaxed) {
                {
                    let now = std::time::Instant::now();
                    let dt = now - last_time;
                    avg_delay = (avg_delay * delay_smoothing + dt) / (delay_smoothing + 1);
                    last_time = now;
                }
                let avg_tps = 1.0 / avg_delay.as_secs_f32();
                progress.set_message(format!("sim:{:.0}tps", avg_tps));
                progress.tick();

                let pre_update_time = std::time::Instant::now();

                let dist = world.dist.clone();

                let drift = width / 4;
                let period = 400.0;

                let offset = ((world.step_count as f32 * std::f32::consts::PI / period).sin()
                    * drift as f32) as isize;

                let start = offset + (width as isize / 2);

                let r = 0.2;
                let outflow_slice = s![-1, start..start + 10, 0, ..];
                let outflow = dist.clone().slice(outflow_slice);

                stash += r * outflow.clone().sum().into_scalar().elem::<f32>();

                let mut dist = dist
                    .clone()
                    .slice_assign(outflow_slice, (1.0 - r) * outflow.clone());

                if world.step_count.is_multiple_of(60) {
                    let (ry, rx) = loop {
                        let ry = rand::rng().random_range(10..height - 10);
                        let rx = rand::rng().random_range(10..width - 10);

                        if world
                            .solid_mask
                            .clone()
                            .slice(s![ry, rx])
                            .into_scalar()
                            .elem::<bool>()
                        {
                            continue;
                        }

                        break (ry, rx);
                    };

                    let existing: f32 = dist.clone().slice(s![ry, rx, 1, 1]).into_scalar().elem();

                    dist = dist.slice_fill(s![ry, rx, 1, 1], existing + stash);
                    stash = 0.0;
                }

                world.dist = dist;

                // Export
                world.advance_step();
                (observer)(world.step_count as usize, world.dist.clone());

                let post_update_time = std::time::Instant::now();
                let update_delay = post_update_time - pre_update_time;

                if let Some(step_duration) = step_duration
                    && step_duration > update_delay
                {
                    let delay = step_duration - update_delay;
                    if delay > std::time::Duration::from_secs_f32(0.0) {
                        thread::sleep(delay);
                    }
                }
            }
        });

        Simulation {
            handle: Some(handle),
            shutdown,
        }
    }

    pub fn shutdown(mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            handle.join().unwrap();
        }
    }
}

pub struct FlowVisApp {
    pub gl: GlGraphics, // OpenGL drawing backend.
    pub cell_data: Arc<Mutex<TensorData>>,
    pub solid_mask: TensorData,
    pub opacity: f32,
}

impl FlowVisApp {
    fn get_cell_data(&self) -> TensorData {
        self.cell_data.lock().unwrap().clone()
    }

    pub fn render(
        &mut self,
        args: &RenderArgs,
    ) {
        use graphics::*;

        let solid_cells: TensorDataView<bool> = self.solid_mask.expect_view();

        let cell_data = self.get_cell_data();
        let vis_cells: TensorDataView<f32> = cell_data.expect_view();
        let [height, width] = cell_data.shape[0..2].try_into().unwrap();

        let [view_width, view_height] = args.viewport().window_size;

        let [x_step, y_step] = [view_width / (width as f64), view_height / (height as f64)];

        self.gl.draw(args.viewport(), |c, gl| {
            for y in 0..height {
                for x in 0..width {
                    let uy: f32 = vis_cells[&[y, x, 0]];
                    let ux: f32 = vis_cells[&[y, x, 1]];
                    let is_solid = solid_cells[&[y, x]];

                    let color = if is_solid {
                        [1., 1., 1., 1.]
                    } else if uy.is_finite() && ux.is_finite() {
                        [0.0, uy, ux, self.opacity]
                    } else {
                        [0., 0., 0., 1.]
                    };

                    let pos = [0., 0., x_step, y_step];

                    let transform = c.transform.trans(x as f64 * x_step, y as f64 * y_step);

                    Rectangle::new(color).draw(pos, &c.draw_state, transform, gl);
                }
            }
        });
    }
}
