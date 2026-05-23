use arrow::array::{
    Array,
    RecordBatch,
    StringArray,
};

/// Selects a column from a `RecordBatch` and returns it as a `StringArray`.
///
/// ## Arguments
/// * `column` - The name of the column to select from the `RecordBatch`.
/// * `iter` - An iterator over `RecordBatch` results.
///
/// ## Returns
/// An iterator over batches of the selected column values as strings,
/// or an error if the operation fails.
pub fn select_text_column<'i, I, E>(
    column: &str,
    iter: I,
) -> impl Iterator<Item = Result<Vec<String>, E>> + use<'i, I, E>
where
    I: Iterator<Item = Result<RecordBatch, E>>,
{
    let column = column.to_string();
    iter.map(move |res| -> Result<Vec<String>, E> {
        let record_batch = res?;

        let text_column = record_batch
            .column_by_name(&column)
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

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
}
