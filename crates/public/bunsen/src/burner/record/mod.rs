//! # Burn Record Utilities
//!
//! Debugging aids for `burn` [`Record`]s. [`display_record`] prints a record's
//! layout (its fields, nesting, and tensor shapes) without the tensor data,
//! for checking what a module or checkpoint holds against what a loader
//! expects.

use alloc::{
    string::{
        String,
        ToString,
    },
    vec,
    vec::Vec,
};
use core::iter;

use burn::{
    prelude::Backend,
    record::{
        HalfPrecisionSettings,
        Record,
    },
};
use serde_json::{
    Map,
    Value,
};

/// Prints `record` to stdout as pretty JSON, with its tensor data elided.
///
/// The record is serialized at half precision, then rewritten for reading:
/// every array of numbers becomes `{"_shape": [..]}`, and raw byte payloads
/// and null fields are dropped, so what remains is the record's field
/// structure and each tensor's shape. It is a debugging aid; the output format
/// is not stable.
///
/// # Panics
/// If the record cannot be serialized to JSON.
pub fn display_record<B: Backend, R: Record<B>>(record: R) {
    fn shape_of_numeric_array(arr: &[Value]) -> Option<Vec<usize>> {
        if arr.is_empty() {
            return Some(vec![0]);
        }
        // Check if first element is a number
        if arr[0].is_number() {
            // All numbers — base case
            Some(vec![arr.len()])
        } else if arr[0].is_array() {
            // Recursively get shape of nested array
            let arr = arr[0].as_array().unwrap();
            let inner_shape = shape_of_numeric_array(arr)?;
            Some(iter::once(arr.len()).chain(inner_shape).collect())
        } else {
            None // Mixed or non-numeric
        }
    }

    fn rewrite_value(value: Value) -> Value {
        match value {
            Value::Array(a) => match shape_of_numeric_array(&a) {
                Some(shape) => {
                    let mut obj: Map<String, Value> = Map::new();
                    obj.insert(
                        "_shape".to_string(),
                        Value::Array(shape.into_iter().map(Value::from).collect()),
                    );
                    Value::Object(obj)
                }
                None => Value::Array(a.into_iter().map(rewrite_value).collect()),
            },
            Value::Object(obj) => {
                let mut new_obj: Map<String, Value> = Map::new();
                for (k, v) in obj.iter() {
                    if k == "bytes" || v.is_null() {
                        continue;
                    }
                    if k == "shape" {
                        new_obj.insert(k.clone(), v.clone());
                    } else {
                        new_obj.insert(k.clone(), rewrite_value(v.clone()));
                    }
                }
                Value::Object(new_obj)
            }
            v => v,
        }
    }

    let sr_item = record.into_item::<HalfPrecisionSettings>();
    let value: Value = serde_json::to_value(&sr_item).unwrap();
    let value = rewrite_value(value);
    println!("{}", serde_json::to_string_pretty(&value).unwrap());
}
