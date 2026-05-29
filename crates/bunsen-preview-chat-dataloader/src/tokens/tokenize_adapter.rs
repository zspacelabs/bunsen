use std::sync::Arc;

use arrow::error::ArrowError;
use wordchipper::{
    TokenEncoder,
    Tokenizer,
    support::slices::inner_str_view,
};

/// Tokenizes text batches using a given tokenizer.
///
/// ## Arguments
/// * `tokenizer` - the tokenizer to use for tokenization.
/// * `iter` - the iterator of text batches to tokenize.
///
/// ## Returns
/// An iterator over the tokenized text batches, where each batch is a result
/// containing either a vector of tokenized strings or an `ArrowError`.
pub fn tokenize_text_batches<I>(
    tokenizer: Arc<Tokenizer<u32>>,
    iter: I,
) -> impl Iterator<Item = Result<Vec<Vec<u32>>, ArrowError>>
where
    I: Iterator<Item = Result<Vec<String>, ArrowError>>,
{
    iter.map(move |res| -> Result<Vec<Vec<u32>>, ArrowError> {
        let text_batch = res?;

        let tokens = tokenizer
            .try_encode_batch(&inner_str_view(&text_batch), None)
            .map_err(|e| ArrowError::ComputeError(e.to_string()))?;

        Ok(tokens)
    })
}

#[cfg(test)]
mod tests {
    use arrow::error::ArrowError;
    use wordchipper::{
        TokenEncoder,
        UnifiedTokenVocab,
        vocab::utility::testing::build_test_vocab,
    };

    use crate::tokens::tokenize_text_batches;

    #[test]
    fn test_tokenize_text_batches() {
        let vocab: UnifiedTokenVocab<u32> = build_test_vocab(
            Default::default(),
            wordchipper::pretrained::openai::oa_p50k_edit_spanning_config(),
        );

        let tokenizer = wordchipper::TokenizerOptions::default().build(vocab.into());

        let samples: Vec<Result<Vec<String>, ArrowError>> = vec![
            Ok(["hello world", "abc xyz"]
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()),
            Ok(["jkl"].iter().map(|s| s.to_string()).collect::<Vec<_>>()),
            Err(ArrowError::ComputeError("error".to_string())),
        ];

        let results: Vec<Result<Vec<Vec<u32>>, ArrowError>> =
            tokenize_text_batches(tokenizer.clone(), samples.into_iter()).collect::<Vec<_>>();
        assert_eq!(results.len(), 3);

        assert_eq!(
            results.first().unwrap().as_ref().unwrap(),
            &tokenizer
                .try_encode_batch(&["hello world", "abc xyz"], None)
                .unwrap(),
        );

        assert_eq!(
            results.get(1).unwrap().as_ref().unwrap(),
            &tokenizer.try_encode_batch(&["jkl"], None).unwrap(),
        );

        assert_eq!(
            results.get(2).unwrap().as_ref().unwrap_err().to_string(),
            "Compute error: error"
        );
    }
}
