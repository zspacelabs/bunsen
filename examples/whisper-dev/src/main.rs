use burn_store::{
    ModuleStore,
    PytorchStore,
};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Path to the source model.
    #[arg(long, default_value = "/media/Data/models/whisper/base.en.pt")]
    pub source: String,

    /// Path to the source model.
    #[arg(long, default_value = "model_state_dict")]
    pub top_level_key: String,
}

fn block_layers_from_keys<S: AsRef<str>>(
    kind: &str,
    keys: &[S],
) -> usize {
    keys.iter()
        .filter(|k| {
            let k = k.as_ref();
            k.starts_with(&format!("{kind}.blocks.")) && k.ends_with(".attn.key.weight")
        })
        .count()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    println!("{:#?}", args);

    let mut store = PytorchStore::from_file(args.source).with_top_level_key("model_state_dict");
    /*
    let snapshots = store.get_all_snapshots();
    println!("{:#?}", snapshots);
     */

    let keys = store.keys()?;
    println!("{:#?}", keys);

    let [d_model, n_mels] = store
        .get_snapshot("encoder.conv1.weight")?
        .unwrap()
        .shape
        .dims();

    let [vocab_size, _] = store
        .get_snapshot("decoder.token_embedding.weight")?
        .unwrap()
        .shape
        .dims();

    println!("n_mels: {n_mels}");
    println!("vocab_size: {vocab_size}");
    println!("d_model: {d_model}");

    let encoder_layers = block_layers_from_keys("encoder", &keys);
    let decoder_layers = block_layers_from_keys("decoder", &keys);
    println!("encoder_layers: {encoder_layers}");
    println!("decoder_layers: {decoder_layers}");

    Ok(())
}
