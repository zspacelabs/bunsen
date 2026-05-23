use arrow::{
    array::RecordBatch,
    compute::BatchCoalescer,
    datatypes::SchemaRef,
    error::ArrowError,
};

pub struct Rebatcher<I>
where
    I: Iterator<Item = Result<RecordBatch, ArrowError>>,
{
    inner: I,
    coalescer: BatchCoalescer,
    exhausted: bool,
}

impl<I: Iterator<Item = Result<RecordBatch, ArrowError>>> Rebatcher<I> {
    pub fn new(
        inner: I,
        schema: SchemaRef,
        target_batch_size: usize,
    ) -> Self {
        Self {
            inner,
            coalescer: BatchCoalescer::new(schema, target_batch_size),
            exhausted: false,
        }
    }
}

impl<I: Iterator<Item = Result<RecordBatch, ArrowError>>> Iterator for Rebatcher<I> {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // drain completed batches first
            if let Some(batch) = self.coalescer.next_completed_batch() {
                return Some(Ok(batch));
            }

            if self.exhausted {
                return None;
            }

            // pull from inner
            match self.inner.next() {
                Some(Ok(batch)) => {
                    if let Err(e) = self.coalescer.push_batch(batch) {
                        return Some(Err(e));
                    }
                    // loop back to check for completed batches
                }
                Some(Err(e)) => return Some(Err(e)),
                None => {
                    self.exhausted = true;
                    if let Err(e) = self.coalescer.finish_buffered_batch() {
                        return Some(Err(e));
                    }
                    // loop back to drain the final batch
                }
            }
        }
    }
}
