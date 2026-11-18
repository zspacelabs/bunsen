//! A take's generated rows: when generation stops, and the frames a whole
//! decode covers.

use alloc::vec::Vec;

use crate::kits::speech::tts::zonos3::prompt::unshear;

/// Why a take's generation stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Zonos3Stop {
    /// A codebook sampled the end of audio.
    EndOfAudio,

    /// The frame budget ran out first.
    Budget,
}

/// A take's generated rows, as they are sampled: the release's stop rule,
/// and the frames they decode to.
///
/// Generation stops on the budget, or, at the first row where any codebook
/// sampled the end of audio, `n + 1` rows later counting that one: the
/// `n - 1` rows the delay needs to complete the frames before it, and one
/// more. Without an audio row at the prompt's end (no silence prefill, no
/// reference prefix), the first `n - 1` rows are a cold start: each one's
/// codebooks past the row's index are forced to the pad, the delay's
/// triangle.
#[derive(Clone, Debug)]
pub struct Zonos3TakeRows {
    n_codebooks: usize,
    eoa: u32,
    audio_pad: u32,
    cold_start: bool,
    budget: usize,
    rows: Vec<Vec<u32>>,
    eos_frame: Option<usize>,
    countdown: Option<usize>,
}

impl Zonos3TakeRows {
    /// No rows yet.
    ///
    /// # Arguments
    /// - `n_codebooks`: `n`, the codes per row.
    /// - `eoa`, `audio_pad`: the end of audio and the pad.
    /// - `cold_start`: whether the prompt ends without an audio row.
    /// - `budget`: the most rows generated.
    pub fn new(
        n_codebooks: usize,
        eoa: u32,
        audio_pad: u32,
        cold_start: bool,
        budget: usize,
    ) -> Self {
        Self {
            n_codebooks,
            eoa,
            audio_pad,
            cold_start,
            budget,
            rows: Vec::new(),
            eos_frame: None,
            countdown: None,
        }
    }

    /// The rows so far, `[G][n]`, the cold start applied.
    pub fn rows(&self) -> &[Vec<u32>] {
        &self.rows
    }

    /// The end of audio's frame, by the first row that sampled one: the
    /// row's index less its last codebook to sample it.
    pub fn eos_frame(&self) -> Option<usize> {
        self.eos_frame
    }

    /// Records the next sampled row, after its cold start (`codes` is
    /// rewritten to what the model is fed), and says whether generation
    /// stops.
    pub fn push(
        &mut self,
        codes: &mut [u32],
    ) -> Option<Zonos3Stop> {
        let n = self.n_codebooks;
        let generated = self.rows.len();
        if self.cold_start && generated + 1 < n {
            for code in &mut codes[generated + 1..n] {
                *code = self.audio_pad;
            }
        }
        self.rows.push(codes[..n].to_vec());
        let mut finished = self.rows.len() >= self.budget;
        if self.eos_frame.is_none()
            && let Some(last) = codes[..n].iter().rposition(|&c| c == self.eoa)
        {
            self.eos_frame = Some(generated.saturating_sub(last));
            self.countdown = Some(n + 1);
        }
        if let Some(countdown) = &mut self.countdown {
            *countdown -= 1;
            finished |= *countdown == 0;
        }
        finished.then_some(if self.eos_frame.is_some() {
            Zonos3Stop::EndOfAudio
        } else {
            Zonos3Stop::Budget
        })
    }

    /// The frames a whole decode covers, `[F][n]`, and the first one's
    /// index in the stream (the prefill's rows counted).
    ///
    /// The stream is the silence prefill's rows, `prefill` (`[P][n]`), then
    /// the generated rows. It ends at the earliest end of audio, by the
    /// recorded frame and by every row that sampled one, or, without one, at
    /// the last complete frame; it starts `overlap` frames into the prefill
    /// (all of it, for a prefill no longer). Nothing is covered when no
    /// generated frame precedes the end.
    pub fn frames(
        &self,
        prefill: &[Vec<u32>],
        overlap: usize,
    ) -> (Vec<Vec<u32>>, usize) {
        let n = self.n_codebooks;
        let p = prefill.len();
        let mut eos = self.eos_frame.map(|f| f + p);
        for (i, row) in self.rows.iter().enumerate() {
            if let Some(last) = row.iter().rposition(|&c| c == self.eoa) {
                let at = (p + i).saturating_sub(last);
                eos = Some(eos.map_or(at, |e| e.min(at)));
            }
        }
        let buffer: Vec<Vec<u32>> = prefill
            .iter()
            .map(|r| r[..n].to_vec())
            .chain(self.rows.iter().cloned())
            .collect();
        let start = p - overlap.min(p);
        let complete = buffer.len().saturating_sub(n - 1);
        let target = eos.map_or(complete, |e| e.min(complete));
        if target <= p {
            return (Vec::new(), start);
        }
        let end = (target + n - 1).min(buffer.len());
        let mut frames = unshear(&buffer[start..end], self.audio_pad);
        frames.truncate(target - start);
        (frames, start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = 3;
    const EOA: u32 = 8;
    const PAD: u32 = 9;

    /// A generated row of codes, none the end of audio.
    fn row(g: usize) -> Vec<u32> {
        vec![g as u32 % 7; N]
    }

    #[test]
    fn test_stops_n_plus_one_rows_after_the_end_of_audio() {
        let mut take = Zonos3TakeRows::new(N, EOA, PAD, false, 100);
        for g in 0..3 {
            assert_eq!(take.push(&mut row(g)), None);
        }
        // Row 3 samples the end of audio in codebook 1: frame 3 - 1 = 2.
        let mut eoa = row(3);
        eoa[1] = EOA;
        assert_eq!(take.push(&mut eoa), None);
        assert_eq!(take.eos_frame(), Some(2));
        for g in 4..6 {
            assert_eq!(take.push(&mut row(g)), None, "row {g}");
        }
        // The countdown of n + 1 = 4 began on row 3.
        assert_eq!(take.push(&mut row(6)), Some(Zonos3Stop::EndOfAudio));
        assert_eq!(take.rows().len(), 7);
    }

    #[test]
    fn test_budget() {
        let mut take = Zonos3TakeRows::new(N, EOA, PAD, false, 2);
        assert_eq!(take.push(&mut row(0)), None);
        assert_eq!(take.push(&mut row(1)), Some(Zonos3Stop::Budget));
    }

    #[test]
    fn test_cold_start() {
        let mut take = Zonos3TakeRows::new(N, EOA, PAD, true, 100);
        let mut first = vec![1, 2, 3];
        take.push(&mut first);
        assert_eq!(first, [1, PAD, PAD]);
        let mut second = vec![1, 2, 3];
        take.push(&mut second);
        assert_eq!(second, [1, 2, PAD]);
        let mut third = vec![1, 2, 3];
        take.push(&mut third);
        assert_eq!(third, [1, 2, 3], "past the triangle");
    }

    #[test]
    fn test_frames() {
        // Frames f hold `[f, 10 + f, 20 + f]`, sheared; the first two rows
        // are the prefill.
        let frame = |f: usize| vec![f as u32, 10 + f as u32, 20 + f as u32];
        let stream: Vec<Vec<u32>> = crate::kits::speech::tts::zonos3::prompt::shear(
            &(0..8).map(frame).collect::<Vec<_>>(),
            PAD,
        );
        let (prefill, generated) = stream.split_at(2);
        let mut take = Zonos3TakeRows::new(N, EOA, PAD, false, 100);
        for r in generated {
            take.push(&mut r.clone());
        }

        // No end of audio: every complete frame, from the prefill's start.
        let (frames, start) = take.frames(prefill, 4);
        assert_eq!(start, 0);
        assert_eq!(frames, (0..6).map(frame).collect::<Vec<_>>());

        // An overlap shorter than the prefill starts inside it.
        assert_eq!(take.frames(prefill, 1).1, 1);
        assert_eq!(take.frames(prefill, 1).0[0], frame(1));

        // An end of audio at generated row 3 (stream row 5), codebook 2:
        // frames up to 5 - 2 = 3.
        let mut take = Zonos3TakeRows::new(N, EOA, PAD, false, 100);
        for (g, r) in generated.iter().enumerate() {
            let mut r = r.clone();
            if g == 3 {
                r[2] = EOA;
            }
            take.push(&mut r);
        }
        assert_eq!(
            take.frames(prefill, 4).0,
            (0..3).map(frame).collect::<Vec<_>>()
        );

        // An end within the prefill covers nothing.
        let mut take = Zonos3TakeRows::new(N, EOA, PAD, false, 100);
        take.push(&mut [EOA, EOA, EOA]);
        assert_eq!(take.frames(prefill, 4), (Vec::new(), 0));
    }
}
