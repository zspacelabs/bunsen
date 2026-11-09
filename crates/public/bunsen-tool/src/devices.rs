//! `bunsen-tool devices`: what this build can run on, and what each device
//! does with each float dtype.

use std::io::Write;

use bunsen::{
    burner::dtypes::{
        DTypeUsage,
        DTypeUsageSet,
        float_dtype_usages,
    },
    errors::{
        BunsenResult,
        sys_at,
    },
};
use bunsen_app::device::{
    BackendArgs,
    DeviceChoice,
};
use burn::tensor::FloatDType;
use clap::ValueEnum;

use crate::{
    card::{
        Value,
        render,
    },
    support::Format,
};

/// Lists each backend this build compiled in, its default device, and what
/// the device does with each float dtype: `storage` (holds and converts it),
/// `arithmetic` (computes in it) and `accelerated` (on tensor cores), as burn
/// reports them. A dtype with none of these is listed as an empty list.
#[derive(clap::Args, Debug)]
pub struct DevicesCommand {
    /// The output's format: YAML, or JSON on one line.
    #[arg(long, value_enum, default_value_t = Format::Yml)]
    format: Format,
}

impl DevicesCommand {
    pub fn run(&self) -> BunsenResult<()> {
        let text = render(&devices()?, self.format)?;
        std::io::stdout()
            .write_all(text.as_bytes())
            .map_err(sys_at("write to", "stdout"))
    }
}

/// Every compiled backend's default device, in [`DeviceChoice`]'s order.
///
/// # Errors
/// As [`BackendArgs::init`].
fn devices() -> BunsenResult<Value> {
    let mut entries = Vec::new();
    for &choice in DeviceChoice::value_variants() {
        if choice == DeviceChoice::Auto {
            continue;
        }
        let device = BackendArgs {
            device: choice,
            device_index: None,
        }
        .init()?;
        let floats = float_dtype_usages(&device)
            .into_iter()
            .map(|(dtype, usage)| (dtype_name(dtype), usages(usage)));
        entries.push(Value::map([
            ("backend", Value::str(choice_name(choice))),
            ("device", Value::str(format!("{device:?}"))),
            ("float", Value::map(floats)),
        ]));
    }
    Ok(Value::map([("devices", Value::List(entries))]))
}

/// The name `--device` takes for `choice`.
fn choice_name(choice: DeviceChoice) -> String {
    choice
        .to_possible_value()
        .expect("no backend is skipped")
        .get_name()
        .to_string()
}

/// A float dtype's short name: `f64`, `f32`, `flex32`, `f16`, `bf16`.
fn dtype_name(dtype: FloatDType) -> &'static str {
    match dtype {
        FloatDType::F64 => "f64",
        FloatDType::F32 => "f32",
        FloatDType::Flex32 => "flex32",
        FloatDType::F16 => "f16",
        FloatDType::BF16 => "bf16",
    }
}

/// A usage set as a list: `storage`, `arithmetic`, `accelerated`, in that
/// order; empty for none. Written out in full, so an empty list is kept.
fn usages(set: DTypeUsageSet) -> Value {
    let names = [
        (DTypeUsage::Storage, "storage"),
        (DTypeUsage::Arithmetic, "arithmetic"),
        (DTypeUsage::Accelerated, "accelerated"),
    ];
    Value::List(
        names
            .into_iter()
            .filter(|(usage, _)| set.contains(*usage))
            .map(|(_, name)| Value::str(name))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every compiled backend is listed, the CPU's last; each lists every
    /// float dtype, and the CPU computes in f32 without accelerating it.
    #[test]
    fn test_devices() {
        let Value::Map(top) = devices().unwrap() else {
            panic!("a map");
        };
        let Value::List(entries) = &top[0].1 else {
            panic!("a list of devices");
        };
        let compiled = DeviceChoice::value_variants().len() - 1;
        assert_eq!(entries.len(), compiled);
        let flex = entries.last().unwrap();
        assert_eq!(flex.get("backend"), Some(&Value::str("flex")));
        let Some(Value::Map(floats)) = flex.get("float") else {
            panic!("the float dtypes");
        };
        let keys: Vec<&str> = floats.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, ["f64", "f32", "flex32", "f16", "bf16"]);
        assert_eq!(
            floats[1].1,
            Value::List(vec![Value::str("storage"), Value::str("arithmetic")])
        );
        let text = render(
            &Value::map([("devices", Value::List(entries.clone()))]),
            Format::Json,
        )
        .unwrap();
        assert!(
            text.contains("\"flex32\":[]"),
            "an empty usage is kept: {text}"
        );
    }
}
