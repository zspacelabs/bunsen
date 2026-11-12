use bunsen::errors::BunsenResult;

use super::{
    Clock,
    Event,
    Opts,
};

/// What every sink renders an event under: the clock, and whether the
/// statements carry speakers.
#[derive(Clone, Copy, Debug)]
pub(super) struct Cx {
    pub(super) clock: Clock,
    pub(super) diarized: bool,
}

/// One delivery target.
pub(super) trait Sink {
    /// Delivers `event`, rendered under `cx` as `opts` shows it.
    fn push(
        &mut self,
        event: &Event,
        cx: &Cx,
        opts: &Opts,
    ) -> BunsenResult<()>;

    /// Delivers what is pending, and closes.
    fn finish(self: Box<Self>) -> BunsenResult<()>;
}

/// Pushes each event to every target `--publish` configures: built by
/// [`PublishArgs::init`](super::PublishArgs::init), fed by `transcribe` and
/// `live`, and [`finish`](Self::finish)ed once the audio ends.
pub struct Publisher {
    pub(super) sinks: Vec<(Opts, Box<dyn Sink>)>,
    pub(super) cx: Cx,
}

impl Publisher {
    /// Reads the stream's clock as `clock` from here on: `live` sets the
    /// wall clock's origin once the first push says when capture began.
    #[cfg_attr(not(feature = "live"), allow(dead_code))]
    pub fn set_clock(
        &mut self,
        clock: Clock,
    ) {
        self.cx.clock = clock;
    }

    /// Delivers `event` to each target that takes its kind.
    ///
    /// # Errors
    /// As a target's delivery: a local write that fails. A network target
    /// logs its failures and goes on.
    pub fn push(
        &mut self,
        event: &Event,
    ) -> BunsenResult<()> {
        let kind = event.kind();
        for (opts, sink) in &mut self.sinks {
            if opts.events.contains(&kind) {
                sink.push(event, &self.cx, opts)?;
            }
        }
        Ok(())
    }

    /// Delivers what each target has pending, and closes them all.
    ///
    /// # Errors
    /// As the first target that fails to finish; the rest are finished
    /// regardless.
    pub fn finish(self) -> BunsenResult<()> {
        let mut first = Ok(());
        for (_, sink) in self.sinks {
            let result = sink.finish();
            if first.is_ok() {
                first = result;
            }
        }
        first
    }
}
