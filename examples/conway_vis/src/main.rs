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
    kits::sims::conway::life2d::{
        ConwayLife2DConfig,
        ConwayLife2DState,
    },
    prelude::{
        TensorElemOpExt,
        TensorOpExt,
    },
    support::validators::parse_grid_shape,
    zspace::ravel_dims,
};
use burn::prelude::{
    Backend,
    TensorData,
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

/// Conway's Game of Life demo for Burn.
#[derive(Parser, Debug)]
#[command(long_about = None)]
pub struct Args {
    /// The grid shape as `HEIGHT,WIDTH`, or `SIZE`.
    #[arg(long, value_parser=parse_grid_shape, default_value="1200")]
    pub grid_shape: [usize; 2],

    /// The number of steps to skip on init.
    #[arg(long, default_value_t = 10)]
    pub init_skip_steps: usize,

    /// The initial density of the grid.
    #[arg(long, default_value_t = 0.1)]
    pub initial_density: f64,

    /// The noise to apply to the grid on each step.
    #[arg(long, default_value_t = 0.0001)]
    pub update_noise: f64,

    /// The frames per second.
    #[arg(long, default_value_t = 60)]
    pub fps: u64,

    /// The tics per second.
    #[arg(long, default_value_t = 60.)]
    pub tps: f32,

    /// The initial window zoom.
    #[arg(long, default_value_t = 1.0)]
    pub zoom: f64,

    /// The opacity between frames.
    #[arg(long, default_value_t = 0.8)]
    pub opacity: f32,
}

fn main() {
    let args = Args::parse();
    println!("{:#?}", args);

    cfg_select! {
        feature = "cuda" => {
            println!("CUDA enabled");
            run::<burn::backend::Cuda<burn::tensor::f16, i8>>(&args);
        }
        feature = "metal" => {
            println!("Metal enabled");
            run::<burn::backend::Metal<burn::tensor::f16, i8>>(&args);
        }
        feature = "wgpu" => {
            println!("WGPU enabled");
            run::<burn::backend::Wgpu<burn::tensor::f16>>(&args);
        }
        feature = "flex" => {
            println!("Flex enabled");
            run::<burn::backend::Flex>(&args);
        }
        _ => {
            complie_error!("No backend selected");
        }
    }
}

fn run<B: Backend>(args: &Args) {
    let device = Default::default();

    let mut conway: ConwayLife2DState<B> = ConwayLife2DConfig::new(args.grid_shape).init(&device);
    conway.fuzz(args.initial_density);
    conway.step();

    for _ in 0..args.init_skip_steps {
        conway.fuzz(args.update_noise);
        conway.step();
    }

    let tic_duration = if args.tps == 0.0 {
        None
    } else {
        Some(std::time::Duration::from_secs_f32(1.0 / args.tps))
    };
    let export_duration = std::time::Duration::from_secs_f32(1.0 / args.fps as f32);

    let export_duration = if let Some(tic_duration) = tic_duration {
        std::cmp::max(export_duration, tic_duration)
    } else {
        export_duration
    };

    let sim = Simulation::new(conway, args.update_noise, tic_duration, export_duration);

    // Change this to OpenGL::V2_1 if not working.
    let opengl = OpenGL::V3_2;

    let [height, width] = args.grid_shape;

    // Create a Glutin window.
    let mut window: Window = WindowSettings::new(
        format!("conway's game of life {height}x{width}"),
        [
            args.grid_shape[1] as f64 * args.zoom,
            args.grid_shape[0] as f64 * args.zoom,
        ],
    )
    .graphics_api(opengl)
    .exit_on_esc(true)
    .build()
    .unwrap();

    // Load the OpenGL function pointers
    gl::load_with(|s| window.get_proc_address(s) as *const _);

    // Create a new game and run it.
    let mut app = FishbowlApp {
        gl: GlGraphics::new(opengl),
        last_frame: sim.last_frame.clone(),
        opacity: args.opacity,
    };

    let mut events = Events::new(EventSettings::new());
    events.set_ups(args.fps);

    while let Some(e) = events.next(&mut window) {
        if let Some(args) = e.render_args() {
            app.render(&args);
        }
    }

    sim.shutdown();
}

pub struct FishbowlApp {
    pub gl: GlGraphics, // OpenGL drawing backend.
    pub last_frame: Arc<Mutex<TensorData>>,
    pub opacity: f32,
}

impl FishbowlApp {
    fn get_frame(&self) -> TensorData {
        let lock = self.last_frame.lock().unwrap();
        lock.clone().convert::<bool>()
    }

    pub fn render(
        &mut self,
        args: &RenderArgs,
    ) {
        use graphics::*;

        let frame_data = self.get_frame();
        let frame_slice: &[bool] = frame_data.as_slice().unwrap();

        let h = frame_data.shape[0];
        let w = frame_data.shape[1];

        let [win_w, win_h] = args.viewport().window_size;
        let draw_scale = [win_w / (w as f64), win_h / (h as f64)];

        self.gl.draw(args.viewport(), |c, gl| {
            for h_idx in 0..h {
                for w_idx in 0..w {
                    let is_live: bool = frame_slice[ravel_dims(&[h, w], &[h_idx, w_idx])];

                    let mut color = if is_live {
                        [1.0, 1.0, 1.0, 1.0]
                    } else {
                        [0.0, 0.0, 0.0, 1.0]
                    };

                    color[3] *= self.opacity;

                    let pos = [0., 0., draw_scale[0], draw_scale[1]];

                    let transform = c
                        .transform
                        .trans(w_idx as f64 * draw_scale[0], h_idx as f64 * draw_scale[1]);

                    Rectangle::new(color).draw(pos, &c.draw_state, transform, gl);
                }
            }
        });
    }
}

pub struct Simulation {
    handle: Option<JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
    pub last_frame: Arc<Mutex<TensorData>>,
}

impl Simulation {
    pub fn new<B: Backend>(
        conway: ConwayLife2DState<B>,
        noise: f64,
        tic_duration: Option<Duration>,
        export_duration: Duration,
    ) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let frame_handle_1 = Arc::new(Mutex::new(conway.state.clone().into_data()));
        let frame_handle_2 = frame_handle_1.clone();

        let shutdown_clone = shutdown.clone();

        let handle = thread::spawn(move || {
            let mut conway = conway;

            let progress = ProgressBar::new_spinner();
            let delay_smoothing = 20;
            let mut avg_delay = std::time::Duration::from_secs_f32(0.0);
            let mut last_time = std::time::Instant::now();

            let mut last_export = std::time::Instant::now();

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

                let t0 = std::time::Instant::now();

                // Update simulation
                conway.fuzz(noise);
                conway.step();

                let mut t1 = std::time::Instant::now();

                // Export
                if t1 - last_export > export_duration {
                    last_export = t1;

                    let frame = conway.state.clone().into_data_convert::<bool>();
                    *frame_handle_1.lock().unwrap() = frame;

                    t1 = std::time::Instant::now();
                }

                let update_delay = t1.duration_since(t0);

                if let Some(step_duration) = tic_duration
                    && step_duration > update_delay
                {
                    let sleep_duration = step_duration - update_delay;
                    thread::sleep(sleep_duration);
                }
            }
        });

        Simulation {
            handle: Some(handle),
            shutdown,
            last_frame: frame_handle_2,
        }
    }

    pub fn shutdown(mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            handle.join().unwrap();
        }
    }
}
