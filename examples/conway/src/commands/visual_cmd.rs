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
    errors::BunsenResult,
    kits::sims::conway::life2d::{
        ConwayLife2DConfig,
        ConwayLife2DState,
    },
    prelude::TensorElemOpExt,
    zspace::ravel_dims,
};
use burn::{
    backend::flex::ops::unary::log,
    prelude::{
        Backend,
        TensorData,
    },
};
use clap_common::logging::LogArgs;
use glutin_window::{
    GlutinWindow,
    OpenGL,
};
use indicatif::ProgressBar;
use opengl_graphics::GlGraphics;
use piston::{
    EventLoop,
    EventSettings,
    Events,
    OpenGLWindow,
    RenderArgs,
    RenderEvent,
    WindowSettings,
};

use crate::sim::SimArgs;

/// Visualize Conway's Game of Life.
#[derive(clap::Args, Debug)]
pub struct VisualCmd {
    #[clap(flatten)]
    pub logging: LogArgs,

    #[clap(flatten)]
    pub sim: SimArgs,

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

impl VisualCmd {
    pub fn run<B: Backend>(&self) -> BunsenResult<()> {
        let device = Default::default();

        self.logging.init(None);
        log::info!("Running Conway's Game of Life simulation...");
        log::info!("{self:#?}");

        let mut conway: ConwayLife2DState<B> =
            ConwayLife2DConfig::new(self.sim.grid.grid_shape).init(&device);
        conway.fuzz(self.sim.initial_density);
        conway.step();

        for _ in 0..self.sim.init_skip_steps {
            conway.fuzz(self.sim.update_noise);
            conway.step();
        }

        let tic_duration = if self.tps == 0.0 {
            None
        } else {
            Some(Duration::from_secs_f32(1.0 / self.tps))
        };
        let export_duration = Duration::from_secs_f32(1.0 / self.fps as f32);

        let export_duration = if let Some(tic_duration) = tic_duration {
            std::cmp::max(export_duration, tic_duration)
        } else {
            export_duration
        };

        let sim = Simulation::new(conway, self.sim.update_noise, tic_duration, export_duration);

        // Change this to OpenGL::V2_1 if not working.
        let opengl = OpenGL::V3_2;

        let height = self.sim.grid.grid_shape.height;
        let width = self.sim.grid.grid_shape.width;

        // Create a Glutin window.
        let mut window: GlutinWindow = WindowSettings::new(
            format!("Conway's Game of Life: {width}x{height}"),
            [width as f64 / self.zoom, height as f64 / self.zoom],
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
            opacity: self.opacity,
        };

        let mut events = Events::new(EventSettings::new());
        events.set_ups(self.fps);

        while let Some(e) = events.next(&mut window) {
            if let Some(args) = e.render_args() {
                app.render(&args);
            }
        }

        sim.shutdown();

        Ok(())
    }
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
        let frame_handle_1 = Arc::new(Mutex::new(conway.state.to_data()));
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

                    let frame = conway.state.to_data_as::<bool>();
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
