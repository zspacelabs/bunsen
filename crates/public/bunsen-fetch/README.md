# bunsen-fetch

Bunsen's file fetching, without the rest of bunsen: stream a URL to disk,
hashed as it lands and pinned to a SHA-256, report the transfer to observers,
place the file where it belongs, and lay pinned files out as a bunsen
pretrained cache.

`bunsen`'s disk cache fetches through it at run time, and the
`bind-*-onnx` crates' build scripts fetch through it at build time
(`Asset`), so both share one implementation and one cache layout.

## License

Distributed under the terms of both the MIT license and the Apache License (Version 2.0).
