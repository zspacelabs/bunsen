use burn::tensor::{
    BoolStore,
    DType,
    FloatDType,
    quantization::QuantScheme,
};
use serde::{
    Deserialize,
    Serialize,
};

use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// Serde shim for [`FloatDType`], which burn does not derive serde for.
#[derive(Serialize, Deserialize)]
#[serde(remote = "FloatDType")]
#[allow(missing_docs)]
pub enum FloatDTypeShim {
    F64,
    F32,
    Flex32,
    F16,
    BF16,
}

/// Converts a string to a [`DType`].
///
/// Attempts to parse the format produced by [`Display`](std::fmt::Display).
///
/// Currently, no support for [`DType::QFloat`] is implemented.
pub fn dtype_from_str(dtype: &str) -> BunsenResult<DType> {
    let dtype = dtype.trim();
    let lower = dtype.to_ascii_lowercase();
    if let Some(result) = match lower.as_str() {
        "f64" => Some(DType::F64),
        "f32" => Some(DType::F32),
        "flex32" => Some(DType::Flex32),
        "f16" => Some(DType::F16),
        "bf16" => Some(DType::BF16),
        "i64" => Some(DType::I64),
        "i32" => Some(DType::I32),
        "i16" => Some(DType::I16),
        "i8" => Some(DType::I8),
        "u64" => Some(DType::U64),
        "u32" => Some(DType::U32),
        "u16" => Some(DType::U16),
        "u8" => Some(DType::U8),
        "bool" => Some(DType::Bool(BoolStore::Native)),
        "bool(native)" => Some(DType::Bool(BoolStore::Native)),
        "bool(u8)" => Some(DType::Bool(BoolStore::U8)),
        "bool(u32)" => Some(DType::Bool(BoolStore::U32)),
        "qfloat" => Some(DType::QFloat(QuantScheme::default())),
        _ => None,
    } {
        return Ok(result);
    }

    if let Some(buf) = dtype
        .strip_prefix("QFloat")
        .or_else(|| dtype.strip_prefix("qfloat"))
        && let buf = buf.trim()
        && let Some(buf) = buf.strip_prefix("(")
        && let buf = buf.trim()
        && let Some(_buf) = buf.strip_suffix(")")
    {
        return Err(BunsenError::External(format!(
            "bunsen can't parse QFloat QuantScheme yet: \"{dtype:?}\"",
        )));
    }

    Err(BunsenError::External(format!(
        "Unsupported dtype: {}",
        dtype
    )))
}

#[cfg(test)]
mod tests {
    use burn::tensor::quantization::QuantScheme;

    use super::*;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct Example {
        /// The float type the comparison runs at.
        #[serde(with = "FloatDTypeShim")]
        dtype: FloatDType,

        /// String field.
        foo: String,
    }

    #[test]
    fn test_example() {
        let example = Example {
            dtype: FloatDType::F32,
            foo: "bar".to_string(),
        };

        let json = serde_json::to_string(&example).unwrap();
        let deserialized: Example = serde_json::from_str(&json).unwrap();

        assert_eq!(example, deserialized);
    }

    #[test]
    fn test_dtype_from_str() -> BunsenResult<()> {
        for dtype in [
            DType::F64,
            DType::F32,
            DType::Flex32,
            DType::F16,
            DType::BF16,
            DType::I64,
            DType::I32,
            DType::I16,
            DType::I8,
            DType::U64,
            DType::U32,
            DType::U16,
            DType::U8,
            DType::Bool(BoolStore::Native),
            DType::Bool(BoolStore::U8),
            DType::Bool(BoolStore::U32),
        ] {
            let result = dtype_from_str(&format!("{:?}", dtype))?;
            assert_eq!(result, dtype);
        }

        assert_eq!(dtype_from_str("Bool")?, DType::Bool(BoolStore::Native));
        assert_eq!(
            dtype_from_str("QFloat")?,
            DType::QFloat(QuantScheme::default())
        );

        assert_eq!(
            dtype_from_str(format!("{:?}", DType::QFloat(QuantScheme::default())).as_str()),
            Err(BunsenError::External(format!(
                "bunsen can't parse QFloat QuantScheme yet: {:?}",
                DType::QFloat(QuantScheme::default())
            )))
        );

        Ok(())
    }
}
