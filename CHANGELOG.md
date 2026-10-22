# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-22

### Added

- Native Starlark-to-LLVM lowering with an x86-64 ELF output path.
- Whole-program type inference, including typed integer lists and fixed-length
  list tracking.
- Closures, recursion, keyword/default/varargs parameters, comprehensions,
  f-strings, `struct`, and Bazel-style `load()` labels.
- `starlarkc` CLI with IR, object, assembly, and executable emission.
- `--march`, `--mcpu`, LTO control, and LLVM optimization-level flags.
- `hashes.bzl` benchmark harness with digest verification against CPython.
- Auto-vectorization-friendly byte/word packing and hot-loop unrolling.
- Huge-page arena for large list allocations.

### Performance

- `hashes.bzl` 64 KiB throughput: SHA-256 243 MiB/s, SHA-512 264 MiB/s,
  SHA3-256 38.8 MiB/s, CRC32 394 MiB/s, BLAKE3 6.5 MiB/s on the reference
  machine. See `benchmarks/hashes/RESULTS.md` for full tables and caveats.

### Known limitations

- Integers are 64-bit two's-complement words rather than arbitrary precision.
- Strings are byte-oriented.
- The runtime is leak-friendly and does not reference-count allocations.
