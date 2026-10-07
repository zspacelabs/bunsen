use std::collections::{
    HashMap,
    HashSet,
};

use bunsen::errors::{
    BunsenError,
    BunsenErrorKind,
    BunsenResult,
    io_error_kind,
};
use burn::data::dataset::InMemDataset;

/// Scan a folder of ``$ROOT/$CLASS/$IMG.{jpg,png}`` into an `InMemDataset`.
///
/// # Errors
///
/// A walk failure, of the kind its `io::Error` sorts into: a missing or
/// unreadable folder is a [`Lookup`](BunsenErrorKind::Lookup); a symlink loop
/// is [`InvalidResource`](BunsenErrorKind::InvalidResource).
pub fn image_dataset_for_folder<P>(root: P) -> BunsenResult<InMemDataset<(String, usize)>>
where
    P: AsRef<std::path::Path>,
{
    // Glob all images with extensions
    let walker = globwalk::GlobWalkerBuilder::from_patterns(root.as_ref(), &["*.{jpg,png}"])
        .follow_links(true)
        .sort_by(|p1, p2| p1.path().cmp(p2.path())) // order by path
        .build()
        .map_err(|e| {
            BunsenError::from_cause(BunsenErrorKind::Internal, e)
                .context(format!("scanning {}", root.as_ref().display()))
        })?;

    // Get all dataset items
    let mut items = Vec::new();
    let mut classes = HashSet::new();
    for img in walker {
        let img = img.map_err(|e| {
            let kind = e
                .io_error()
                .map_or(BunsenErrorKind::InvalidResource, io_error_kind);
            BunsenError::from_cause(kind, e)
                .context(format!("scanning {}", root.as_ref().display()))
        })?;
        let image_path = img.path().to_path_buf();

        // Label name is represented by the parent folder name
        let label = image_path
            .parent()
            .and_then(|parent| parent.file_name())
            .ok_or_else(|| {
                BunsenError::internal(format!(
                    "image path has no parent folder name: {}",
                    image_path.display()
                ))
            })?
            .to_string_lossy()
            .into_owned();

        classes.insert(label.clone());

        items.push((image_path, label))
    }

    let mut classes = classes.into_iter().collect::<Vec<_>>();
    classes.sort();

    let mut class_to_index = HashMap::new();
    class_to_index.extend(
        classes
            .iter()
            .enumerate()
            .map(|(i, class)| (class.clone(), i)),
    );

    let items = items
        .into_iter()
        .map(|(path, label)| {
            let class = class_to_index[&label];
            (path.to_string_lossy().to_string(), class)
        })
        .collect::<Vec<_>>();

    Ok(InMemDataset::new(items))
}
