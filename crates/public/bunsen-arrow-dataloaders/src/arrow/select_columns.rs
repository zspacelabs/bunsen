use arrow::{
    array::{
        Array,
        RecordBatch,
        StringArray,
    },
    error::ArrowError,
};

/// Selects a column from a `RecordBatch` and returns it as a `StringArray`.
///
/// ## Arguments
/// * `column` - The name of the column to select from the `RecordBatch`.
/// * `iter` - An iterator over `RecordBatch` results.
///
/// ## Returns
/// An iterator over batches of the selected column values as strings,
/// skipping nulls. An `Err` from `iter` is passed along. A batch without
/// `column`, or whose `column` is not Arrow `Utf8` (`LargeUtf8` and
/// `Utf8View` included), yields an [`ArrowError::SchemaError`], converted
/// to `E`.
pub fn select_text_column<'i, I, E>(
    column: &str,
    iter: I,
) -> impl Iterator<Item = Result<Vec<String>, E>> + use<'i, I, E>
where
    I: Iterator<Item = Result<RecordBatch, E>>,
    E: From<ArrowError>,
{
    let column = column.to_string();
    iter.map(move |res| -> Result<Vec<String>, E> {
        let record_batch = res?;

        let array = record_batch
            .column_by_name(&column)
            .ok_or_else(|| ArrowError::SchemaError(format!("text column {column:?} not found")))?;
        let text_column = array
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| {
                ArrowError::SchemaError(format!(
                    "text column {column:?} is {}, not Utf8",
                    array.data_type()
                ))
            })?;

        let samples: Vec<String> = text_column
            .into_iter()
            .flat_map(|x| x.map(|s| s.to_string()))
            .collect::<Vec<String>>();

        Ok(samples)
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::error::ArrowError;

    use super::*;

    #[test]
    fn test_select_text_columns() {
        type E = ArrowError;

        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("text", arrow::datatypes::DataType::Utf8, false),
        ]));

        let batches: Vec<Result<RecordBatch, E>> = vec![
            Ok(RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(StringArray::from(
                    ["hello world", "abc xyz"]
                        .iter()
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>(),
                ))],
            )
            .unwrap()),
            Ok(RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(StringArray::from(
                    ["jkl"].iter().map(|s| s.to_string()).collect::<Vec<_>>(),
                ))],
            )
            .unwrap()),
            Err(ArrowError::ComputeError("error".to_string())),
        ];

        let results: Vec<Result<Vec<String>, E>> =
            select_text_column("text", batches.into_iter()).collect::<Vec<_>>();
        assert_eq!(results.len(), 3);

        assert_eq!(
            results.first().unwrap().as_ref().unwrap(),
            &vec!["hello world".to_string(), "abc xyz".to_string()],
        );

        assert_eq!(
            results.get(1).unwrap().as_ref().unwrap(),
            &vec!["jkl".to_string()],
        );

        assert_eq!(
            results.get(2).unwrap().as_ref().unwrap_err().to_string(),
            "Compute error: error"
        );
    }

    #[test]
    fn test_select_text_column_missing_column_is_err() {
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("body", arrow::datatypes::DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(vec!["hello world"]))],
        )
        .unwrap();

        let results: Vec<Result<Vec<String>, ArrowError>> =
            select_text_column("text", std::iter::once(Ok(batch))).collect();
        assert_eq!(results.len(), 1);

        let err = results[0].as_ref().unwrap_err().to_string();
        assert!(err.contains("\"text\""), "{err}");
    }

    #[test]
    fn test_select_text_column_non_utf8_column_is_err() {
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("text", arrow::datatypes::DataType::LargeUtf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(arrow::array::LargeStringArray::from(vec![
                "hello world",
            ]))],
        )
        .unwrap();

        let results: Vec<Result<Vec<String>, ArrowError>> =
            select_text_column("text", std::iter::once(Ok(batch))).collect();
        assert_eq!(results.len(), 1);

        let err = results[0].as_ref().unwrap_err().to_string();
        assert!(
            err.contains("\"text\"") && err.contains("LargeUtf8"),
            "{err}"
        );
    }
}
