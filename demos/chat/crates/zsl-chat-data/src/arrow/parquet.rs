use std::{
    fs::File,
    path::PathBuf,
};

use arrow::{
    array::RecordBatch,
    error::Result as ArrowResult,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

pub fn read_parquet_shards<'a, I>(
    paths: I
) -> impl Iterator<Item = ArrowResult<RecordBatch>> + use<'a, I>
where
    I: Iterator<Item = PathBuf> + 'a,
{
    paths
        .into_iter()
        .map(|path| {
            let file = File::open(&path)?;
            log::info!("Loading Parquet: {:?}", path);
            ParquetRecordBatchReaderBuilder::try_new(file).and_then(|b| b.build())
        })
        .flat_map(|res| {
            let iter: Box<dyn Iterator<Item = ArrowResult<RecordBatch>>> = match res {
                Err(e) => Box::new(std::iter::once(Err(e.into()))),
                Ok(reader) => Box::new(reader),
            };
            iter
        })
}
