# train

```terminaloutput
$ cargo run --release -p train -- \
  --dataset-dir ~/Data/nanochat/dataset/ --shards 0..10 --cautious-weight-decay --weight-decay 0.02 
```