use std::path::{
    Path,
    PathBuf,
};

use burn::tensor::Device;

use crate::{
    audit::{
        AuditProbe,
        AuditStreamRecorder,
        AuditStreamVerifier,
        reports::{
            BaselineMode,
            ReportsOptions,
            backend_label,
        },
    },
    errors::BunsenResult,
};

/// A test body that runs on any device, emitting audit checkpoints.
///
/// The harness functions [`audit_across`] and [`audit_baseline`] call `run`
/// once per run, each time with an [`AuditProbe`] wired to a recorder
/// ([`AuditStreamRecorder`]) or a verifier ([`AuditStreamVerifier`]). A body
/// must make the same sequence of checkpoints on every device, since events
/// are matched by position.
///
/// A closure `Fn(&mut AuditProbe<'_>, &Device) -> BunsenResult<()>` is a body.
pub trait AuditBody {
    /// Run the body on `device`, sending checkpoints to `probe`.
    ///
    /// # Errors
    /// Errors from `probe` (a checkpoint that does not verify), or from the
    /// body itself.
    fn run(
        &self,
        probe: &mut AuditProbe<'_>,
        device: &Device,
    ) -> BunsenResult<()>;
}

impl<F> AuditBody for F
where
    F: Fn(&mut AuditProbe<'_>, &Device) -> BunsenResult<()>,
{
    fn run(
        &self,
        probe: &mut AuditProbe<'_>,
        device: &Device,
    ) -> BunsenResult<()> {
        self(probe, device)
    }
}

/// Runs `body` on `reference`, recording, then on `target`, verifying
/// against the recording.
///
/// # Errors
/// [`Policy`](crate::errors::BunsenErrorKind::Policy), with a
/// [`ValueMismatch`](crate::errors::ValueMismatch) cause, for the first
/// checkpoint that does not verify, or a mismatched event count. Any error the
/// body itself returns, unchanged.
pub fn audit_across(
    reference: &Device,
    target: &Device,
    body: &impl AuditBody,
) -> BunsenResult<()> {
    let mut recorder = AuditStreamRecorder::default();
    body.run(&mut AuditProbe::new(vec![&mut recorder]), reference)?;

    let mut verifier = recorder.into_verifier();
    body.run(&mut AuditProbe::new(vec![&mut verifier]), target)?;
    verifier.finish()
}

/// What [`audit_baseline`] or [`audit_baseline_at`] did with a stored baseline.
///
/// Which arm is taken depends on the [`BaselineMode`] and on whether the
/// baseline file exists; both arms carry the path of that [`AuditStreamFile`].
///
/// [`AuditStreamFile`]: crate::audit::AuditStreamFile
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BaselineOutcome {
    /// No baseline existed (or recording was forced); one was written here.
    Recorded(PathBuf),

    /// The run matched the baseline stored here.
    Verified(PathBuf),
}

/// Runs `body` on `device` against the stored baseline `name`.
///
/// The baseline lives at
/// `options.report_path(backend_label(device), name + ".cbor")`, and is
/// treated per
/// `options.mode()`; see [`audit_baseline_at`].
///
/// # Errors
/// See [`audit_baseline_at`];
/// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) if `name` escapes the
/// backend directory.
pub fn audit_baseline(
    options: &ReportsOptions,
    device: &Device,
    name: &str,
    body: &impl AuditBody,
) -> BunsenResult<BaselineOutcome> {
    let path = options.report_path(&backend_label(device), &format!("{name}.cbor"))?;
    audit_baseline_at(&path, options.mode(), device, body)
}

/// Runs `body` on `device` against the baseline stream at `path`, under
/// `mode`.
///
/// `Auto` records when the file is absent and verifies when present;
/// `Record` always records; `Verify` requires the file.
///
/// # Errors
/// - [`Policy`](crate::errors::BunsenErrorKind::Policy), with a
///   [`ValueMismatch`](crate::errors::ValueMismatch) cause, for a checkpoint
///   that does not verify, or a mismatched event count;
/// - [`Lookup`](crate::errors::BunsenErrorKind::Lookup) for a missing baseline
///   under `Verify`;
/// - as [`load_audit_stream`] and [`save_audit_stream`] for a baseline that
///   does not read or write;
/// - any error the body itself returns, unchanged.
///
/// [`load_audit_stream`]: crate::audit::load_audit_stream
/// [`save_audit_stream`]: crate::audit::save_audit_stream
pub fn audit_baseline_at(
    path: &Path,
    mode: BaselineMode,
    device: &Device,
    body: &impl AuditBody,
) -> BunsenResult<BaselineOutcome> {
    let path = path.to_path_buf();
    let record = match mode {
        BaselineMode::Auto => !path.exists(),
        BaselineMode::Record => true,
        BaselineMode::Verify => false,
    };

    if record {
        let mut recorder = AuditStreamRecorder::default();
        body.run(&mut AuditProbe::new(vec![&mut recorder]), device)?;
        recorder.save(&path)?;
        Ok(BaselineOutcome::Recorded(path))
    } else {
        let mut verifier = AuditStreamVerifier::load(&path)?;
        body.run(&mut AuditProbe::new(vec![&mut verifier]), device)?;
        verifier.finish()?;
        Ok(BaselineOutcome::Verified(path))
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Distribution;

    use super::*;
    use crate::{
        burner::descriptors::TolerancePolicy,
        errors::{
            BunsenErrorKind,
            LookupError,
            LookupProblem,
            ParseError,
            ValueMismatch,
            testing::{
                ErrorMatcher,
                predicate,
            },
        },
        support::testing::{
            cpu_device,
            performance_device,
            seeded_tensor,
        },
    };

    /// Seeded values, and an arithmetic function of them; `extra` adds a
    /// checkpoint, and `shift` perturbs the values.
    struct Body {
        extra: bool,
        shift: f32,
    }

    impl Body {
        const EXTRA: Body = Body {
            extra: true,
            shift: 0.0,
        };
        const PLAIN: Body = Body {
            extra: false,
            shift: 0.0,
        };
        const SHIFTED: Body = Body {
            extra: false,
            shift: 1e-3,
        };
    }

    impl AuditBody for Body {
        fn run(
            &self,
            probe: &mut AuditProbe<'_>,
            device: &Device,
        ) -> BunsenResult<()> {
            let x = seeded_tensor::<3>(1, [2, 3, 4], Distribution::Uniform(-1.0, 1.0), device)
                + self.shift;

            probe.assert_eq_as::<f32>("x", &x)?;
            probe.assert_approx_eq_as::<f32>(
                "x*x/3 - 1/x",
                &((x.clone() * x.clone()) / 3.0 - x.clone().recip()),
                TolerancePolicy::Balanced,
            )?;
            if self.extra {
                probe.assert_eq_as::<f32>("extra", &x)?;
            }
            Ok(())
        }
    }

    fn record(body: &Body) -> AuditStreamRecorder {
        let mut recorder = AuditStreamRecorder::default();
        body.run(&mut AuditProbe::new(vec![&mut recorder]), &cpu_device())
            .unwrap();
        recorder
    }

    fn verify(
        verifier: &mut AuditStreamVerifier,
        body: &Body,
    ) -> BunsenResult<()> {
        body.run(&mut AuditProbe::new(vec![verifier]), &cpu_device())
    }

    #[test]
    fn test_stream_round_trips_through_a_file() {
        let recorder = record(&Body::PLAIN);
        assert_eq!(recorder.events().len(), 2);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("round_trip").join("stream.cbor");
        recorder.save(&path).unwrap();

        let mut verifier = AuditStreamVerifier::load(&path).unwrap();
        verify(&mut verifier, &Body::PLAIN).unwrap();
        verifier.finish().unwrap();
    }

    #[test]
    fn test_verifier_rejects_a_changed_value() {
        let mut verifier = record(&Body::PLAIN).into_verifier();
        ErrorMatcher::kind(BunsenErrorKind::Policy)
            .frame_contains("audit mismatch at event 0 (\"x\")")
            .assert_err(&verify(&mut verifier, &Body::SHIFTED));
    }

    #[test]
    fn test_verifier_rejects_an_extra_event() {
        let mut verifier = record(&Body::PLAIN).into_verifier();
        ErrorMatcher::kind(BunsenErrorKind::Policy)
            .message_contains("unexpected audit event")
            .has_cause::<ValueMismatch>()
            .assert_err(&verify(&mut verifier, &Body::EXTRA));
    }

    #[test]
    fn test_verifier_rejects_a_missing_event() {
        let mut verifier = record(&Body::EXTRA).into_verifier();
        verify(&mut verifier, &Body::PLAIN).unwrap();
        ErrorMatcher::kind(BunsenErrorKind::Policy)
            .message_contains("saw 2 of 3 events")
            .has_cause::<ValueMismatch>()
            .assert_err(&verifier.finish());
    }

    #[test]
    fn test_load_missing_stream_is_not_found() {
        let dir = tempfile::tempdir().unwrap();
        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .cause(predicate("a missing path", |c: &LookupError| {
                c.problem == LookupProblem::Missing && c.key.ends_with("nope.cbor")
            }))
            .assert_err(&AuditStreamVerifier::load(&dir.path().join("nope.cbor")));
    }

    #[test]
    fn test_load_garbage_stream_is_invalid_resource() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("garbage.cbor");
        std::fs::write(&path, b"\xff\xff not cbor").unwrap();
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .has_cause::<ParseError>()
            .assert_err(&AuditStreamVerifier::load(&path));
    }

    /// The CPU backend against the one selected by feature (`cuda`, `metal`,
    /// `vulkan`, `wgpu`), or the CPU again when none is enabled.
    #[test]
    #[serial_test::serial]
    fn test_audit_across_backends() {
        audit_across(&cpu_device(), &performance_device(), &Body::PLAIN).unwrap();
    }

    #[test]
    fn test_baseline_at_records_then_verifies() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("baseline").join("body.cbor");

        let outcome = audit_baseline_at(&path, BaselineMode::Auto, &cpu_device(), &Body::PLAIN);
        assert_eq!(outcome.unwrap(), BaselineOutcome::Recorded(path.clone()));

        let outcome = audit_baseline_at(&path, BaselineMode::Auto, &cpu_device(), &Body::PLAIN);
        assert_eq!(outcome.unwrap(), BaselineOutcome::Verified(path.clone()));

        assert!(
            audit_baseline_at(&path, BaselineMode::Verify, &cpu_device(), &Body::SHIFTED).is_err()
        );

        let outcome = audit_baseline_at(&path, BaselineMode::Record, &cpu_device(), &Body::SHIFTED);
        assert_eq!(outcome.unwrap(), BaselineOutcome::Recorded(path.clone()));

        let missing = dir.path().join("baseline_missing").join("body.cbor");
        assert!(
            audit_baseline_at(&missing, BaselineMode::Verify, &cpu_device(), &Body::PLAIN).is_err()
        );
    }

    /// A closure is a body.
    #[test]
    fn test_closure_body() {
        let body = |probe: &mut AuditProbe<'_>, device: &Device| -> BunsenResult<()> {
            let x = seeded_tensor::<2>(3, [2, 2], Distribution::Default, device);
            probe.assert_eq_as::<f32>("x", &x)
        };
        audit_across(&cpu_device(), &cpu_device(), &body).unwrap();
    }

    #[test]
    fn test_baseline_uses_the_call_site_location() {
        let dir = tempfile::tempdir().unwrap();
        let options = ReportsOptions::new(dir.path());
        let device = cpu_device();
        let path = options
            .report_path(&backend_label(&device), "group/body.cbor")
            .unwrap();
        assert!(path.starts_with(dir.path()));

        let outcome = audit_baseline(&options, &device, "group/body", &Body::PLAIN);
        assert_eq!(outcome.unwrap(), BaselineOutcome::Recorded(path.clone()));
        assert!(path.exists());

        let outcome = audit_baseline(&options, &device, "group/body", &Body::PLAIN);
        assert_eq!(outcome.unwrap(), BaselineOutcome::Verified(path.clone()));

        let verify = options.with_mode(BaselineMode::Verify);
        assert!(audit_baseline(&verify, &device, "group/body", &Body::SHIFTED).is_err());
        assert!(audit_baseline(&verify, &device, "absent", &Body::PLAIN).is_err());
    }
}
