# bunsen-asset-fetch

Build-script support for bunsen's `bunsen-bundled-*` crates: fetch a file into
`OUT_DIR`, check it against a pinned SHA-256, honour an environment variable
naming a local copy, and lay the file out where the build wants it. A build
dependency, not a runtime one.

## License

Distributed under the terms of both the MIT license and the Apache License (Version 2.0).
