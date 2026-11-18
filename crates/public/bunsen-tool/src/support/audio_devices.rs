//! The host's audio devices, for capture and playback: finding one by a
//! substring of its id or name, listing them, and the sample formats both
//! convert.

use bunsen::errors::{
    BunsenError,
    BunsenResult,
};
use cpal::{
    Device,
    Host,
    SampleFormat,
    SupportedStreamConfig,
    traits::{
        DeviceTrait,
        HostTrait,
    },
};

/// Which way a device carries audio.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioDirection {
    /// Capture: a microphone.
    Input,

    /// Playback: a speaker.
    Output,
}

impl AudioDirection {
    /// `input` or `output`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::Output => "output",
        }
    }

    /// The flag that names a device of this direction.
    fn device_flag(self) -> &'static str {
        match self {
            Self::Input => "--input-device",
            Self::Output => "--output-device",
        }
    }

    /// The flag that lists the devices of this direction.
    fn list_flag(self) -> &'static str {
        match self {
            Self::Input => "--list-devices",
            Self::Output => "--list-output-devices",
        }
    }

    fn default_device(
        self,
        host: &Host,
    ) -> Option<Device> {
        match self {
            Self::Input => host.default_input_device(),
            Self::Output => host.default_output_device(),
        }
    }

    fn devices(
        self,
        host: &Host,
    ) -> BunsenResult<Vec<Device>> {
        let listed = match self {
            Self::Input => host.input_devices().map(Iterator::collect),
            Self::Output => host.output_devices().map(Iterator::collect),
        };
        listed.map_err(|e| {
            BunsenError::sys(format!("could not list the {} devices", self.name())).with_cause(e)
        })
    }

    /// The device's default configuration in this direction.
    ///
    /// # Errors
    /// [`Sys`](bunsen::errors::BunsenErrorKind::Sys) when the device cannot
    /// say.
    pub fn default_config(
        self,
        device: &Device,
    ) -> BunsenResult<SupportedStreamConfig> {
        let config = match self {
            Self::Input => device.default_input_config(),
            Self::Output => device.default_output_config(),
        };
        config.map_err(|e| {
            BunsenError::sys(format!(
                "could not read the device's default {} configuration",
                self.name()
            ))
            .with_cause(e)
        })
    }
}

/// The device of `direction` `want` names, by a case-insensitive substring
/// of its id or name; the host's default without one.
///
/// # Errors
/// [`Policy`](bunsen::errors::BunsenErrorKind::Policy) when there is no
/// default, or nothing matches; as the host listing its devices.
pub fn find_audio_device(
    host: &Host,
    direction: AudioDirection,
    want: Option<&str>,
) -> BunsenResult<Device> {
    match want {
        None => direction.default_device(host).ok_or_else(|| {
            BunsenError::policy(format!(
                "no default {} device; name one with {}",
                direction.name(),
                direction.device_flag()
            ))
        }),
        Some(want) => {
            let needle = want.to_lowercase();
            direction
                .devices(host)?
                .into_iter()
                .find(|device| {
                    audio_device_id(device).to_lowercase().contains(&needle)
                        || audio_device_name(device).to_lowercase().contains(&needle)
                })
                .ok_or_else(|| {
                    BunsenError::policy(format!(
                        "no {} device matches {want:?}; see {}",
                        direction.name(),
                        direction.list_flag()
                    ))
                })
        }
    }
}

/// Prints the host's default device of `direction`, then each one that
/// opens, as `id  name: default configuration`. One that does not open
/// (ALSA lists plugins with no device behind them) is logged at debug.
///
/// # Errors
/// As the host listing its devices.
pub fn list_audio_devices(
    host: &Host,
    direction: AudioDirection,
) -> BunsenResult<()> {
    match direction.default_device(host) {
        Some(device) => println!(
            "default: {}  {}",
            audio_device_id(&device),
            audio_device_name(&device)
        ),
        None => println!("default: none"),
    }
    for device in direction.devices(host)? {
        let id = audio_device_id(&device);
        let name = audio_device_name(&device);
        match direction.default_config(&device) {
            Ok(config) => println!(
                "  {id}  {name}: {} ch, {} Hz, {}",
                config.channels(),
                config.sample_rate(),
                config.sample_format(),
            ),
            Err(err) => log::debug!("{id} ({name}): {err}"),
        }
    }
    Ok(())
}

/// The device's human-readable name; `?` when the host cannot say.
pub fn audio_device_name(device: &Device) -> String {
    device
        .description()
        .map(|description| description.name().to_string())
        .unwrap_or_else(|_| "?".to_string())
}

/// The device's id within its host (ALSA's PCM name, for one); `?` when
/// the host cannot say.
pub fn audio_device_id(device: &Device) -> String {
    device
        .id()
        .map(|id| id.id().to_string())
        .unwrap_or_else(|_| "?".to_string())
}

/// Preference among the sample formats capture and playback convert;
/// `None` for one they do not.
pub fn sample_format_rank(format: SampleFormat) -> Option<u8> {
    Some(match format {
        SampleFormat::F32 => 0,
        SampleFormat::I16 => 1,
        SampleFormat::I32 => 2,
        SampleFormat::I24 => 3,
        SampleFormat::F64 => 4,
        SampleFormat::U16 => 5,
        SampleFormat::U32 => 6,
        SampleFormat::U24 => 7,
        SampleFormat::I8 => 8,
        SampleFormat::U8 => 9,
        SampleFormat::I64 => 10,
        SampleFormat::U64 => 11,
        _ => return None,
    })
}
