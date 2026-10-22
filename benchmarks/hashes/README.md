# hashes.bzl benchmark

This directory contains the benchmark harness and recorded results for
[`hermeticbuild/hashes.bzl`](https://github.com/hermeticbuild/hashes.bzl).

## Setup

```sh
git clone https://github.com/hermeticbuild/hashes.bzl /tmp/hashes.bzl
cargo build --release
```

The harness also needs Python 3 and the `blake3` package for native BLAKE3
verification. Bazel and `starlark-rs` are optional columns.

## Run

```sh
HASHES_BZL=/tmp/hashes.bzl \
  python3 benchmarks/hashes/run.py --repeat 5 --target-ms 300
```

For a quick compiler-only run:

```sh
HASHES_BZL=/tmp/hashes.bzl \
  python3 benchmarks/hashes/run.py --skip-bazel --skip-starlark-rs \
    --algorithms sha256 --sizes 64,65536 --repeat 3 --target-ms 200
```

The harness compiles each algorithm/size/call-count combination, verifies every
digest against CPython's `hashlib`/`zlib` (and the `blake3` package), subtracts
the zero-call process/module-load baseline, and reports the best run.

Use `--json path/to/results.json` to save raw seconds per hash.

See [`RESULTS.md`](RESULTS.md) for the current tables, machine details, and
caveats.
