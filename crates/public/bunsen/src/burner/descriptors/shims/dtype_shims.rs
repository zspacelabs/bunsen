use burn::tensor::FloatDType;
use serde::{
    Deserialize,
    Serialize,
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

#[cfg(test)]
mod tests {
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
}
