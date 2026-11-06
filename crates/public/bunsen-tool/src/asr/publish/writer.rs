//! The `stdout` and `file` targets: text, JSON lines or RTTM, a line each.
use std::{
    fs::OpenOptions,
    io::{
        BufWriter,
        IsTerminal,
        Write,
    },
    path::Path,
};

use bunsen::errors::{
    BunsenError,
    BunsenResult,
    sys_at,
};
use serde::Deserialize;

use super::{
    Cx,
    Event,
    Opts,
    Sink,
    record::{
        json_line,
        rttm_line,
        text_line,
    },
};

/// The most of a draft shown on a terminal: its end, where it changes.
const DRAFT_CHARS: usize = 100;

/// What a `stdout` or `file` target writes.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Format {
    /// A line per statement.
    #[default]
    Text,

    /// A JSON object per event.
    #[serde(alias = "json", alias = "jsonlines")]
    Jsonl,

    /// A NIST RTTM `SPEAKER` line per statement with a speaker; needs
    /// `--diarize`.
    Rttm,
}

/// Writes each event as a line of its format to stdout or a file, flushed
/// a line at a time.
///
/// On stdout, text shows drafts on stderr: redrawn in place on a terminal,
/// a line of its own, marked `~`, otherwise; a statement erases the draft
/// before it prints.
pub struct WriterSink {
    out: Box<dyn Write>,
    name: String,
    format: Format,

    /// Whether drafts show on stderr.
    drafts: bool,
    terminal: bool,
    draft_shown: bool,
}

impl WriterSink {
    /// To stdout.
    ///
    /// # Errors
    /// As [`check`].
    pub fn stdout(
        format: Format,
        diarized: bool,
    ) -> BunsenResult<Self> {
        check(format, diarized)?;
        Ok(Self {
            out: Box::new(std::io::stdout()),
            name: "stdout".to_string(),
            format,
            drafts: format == Format::Text,
            terminal: std::io::stderr().is_terminal(),
            draft_shown: false,
        })
    }

    /// To the file at `path`, created or truncated, or appended to.
    ///
    /// # Errors
    /// As [`check`]; as opening the file.
    pub fn file(
        path: &Path,
        format: Format,
        append: bool,
        diarized: bool,
    ) -> BunsenResult<Self> {
        check(format, diarized)?;
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(append)
            .truncate(!append)
            .open(path)
            .map_err(sys_at("open", path))?;
        Ok(Self {
            out: Box::new(BufWriter::new(file)),
            name: path.display().to_string(),
            format,
            drafts: false,
            terminal: false,
            draft_shown: false,
        })
    }

    fn line(
        &mut self,
        line: &str,
    ) -> BunsenResult<()> {
        self.clear_draft();
        writeln!(self.out, "{line}")
            .and_then(|()| self.out.flush())
            .map_err(sys_at("write to", &self.name))
    }

    fn show_draft(
        &mut self,
        text: &str,
    ) {
        if !self.terminal {
            eprintln!("~ {text}");
            return;
        }
        let skip = text.chars().count().saturating_sub(DRAFT_CHARS);
        let tail: String = text.chars().skip(skip).collect();
        eprint!("\r\x1b[2K~ {tail}");
        let _ = std::io::stderr().flush();
        self.draft_shown = true;
    }

    /// Erases a draft shown in place.
    fn clear_draft(&mut self) {
        if self.draft_shown {
            eprint!("\r\x1b[2K");
            let _ = std::io::stderr().flush();
            self.draft_shown = false;
        }
    }
}

/// # Errors
/// [`Policy`](bunsen::errors::BunsenErrorKind::Policy) for RTTM without
/// `--diarize`.
fn check(
    format: Format,
    diarized: bool,
) -> BunsenResult<()> {
    if format == Format::Rttm && !diarized {
        return Err(BunsenError::policy("an RTTM target needs --diarize"));
    }
    Ok(())
}

impl Sink for WriterSink {
    fn push(
        &mut self,
        event: &Event,
        cx: &Cx,
        opts: &Opts,
    ) -> BunsenResult<()> {
        match (self.format, *event) {
            (Format::Jsonl, _) => self.line(&json_line(event, opts, cx)),
            (Format::Text, Event::Statement { statement, .. }) => {
                self.line(&text_line(statement, opts, cx))
            }
            (Format::Text, Event::Draft { text, .. }) if self.drafts => {
                self.show_draft(text.trim());
                Ok(())
            }
            (Format::Rttm, Event::Statement { file, statement }) => {
                match rttm_line(file, statement) {
                    Some(line) => self.line(&line),
                    None => Ok(()),
                }
            }
            _ => Ok(()),
        }
    }

    fn finish(mut self: Box<Self>) -> BunsenResult<()> {
        self.clear_draft();
        self.out.flush().map_err(sys_at("write to", &self.name))
    }
}
