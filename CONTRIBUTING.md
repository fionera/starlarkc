# Contributing

Thanks for helping improve `starlarkc`. Bug reports, reproduction cases,
benchmark results, documentation fixes, and compiler patches are all welcome.

## Prerequisites

- Rust 1.87 or newer
- `clang` with LLVM IR support (the test suite uses the system `clang`)
- Python 3 for the optional `hashes.bzl` benchmark harness

## Build and test

```sh
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo build --release
```

The integration tests compile and execute real Starlark programs, so a working
`clang` is required even for `cargo test`.

To try the CLI directly:

```sh
cargo run --release -- --run examples/hello.star -o target/hello
```

## Benchmarks

The reference workload is
[`hermeticbuild/hashes.bzl`](https://github.com/hermeticbuild/hashes.bzl):

```sh
git clone https://github.com/hermeticbuild/hashes.bzl /tmp/hashes.bzl
cargo build --release
HASHES_BZL=/tmp/hashes.bzl \
  python3 benchmarks/hashes/run.py --skip-bazel --skip-starlark-rs \
    --algorithms sha256 --sizes 64,65536 --repeat 3 --target-ms 200
```

The full run compiles every algorithm/size/call-count combination and takes
several minutes. See [`benchmarks/hashes/RESULTS.md`](benchmarks/hashes/RESULTS.md)
for the methodology and current numbers.

## Code layout

- `src/compiler.rs` — AST discovery, type-directed lowering, and LLVM IR emission
- `src/types.rs` — whole-program type inference
- `src/ir.rs` — textual LLVM IR builder
- `src/main.rs` — CLI and clang invocation
- `runtime/runtime.c` — dynamic values, containers, and built-ins
- `tests/integration.rs` — end-to-end compile-and-run tests

`src/compiler.rs` is intentionally kept as one lowering pass for now. If you
split it into modules, keep the integration tests green and avoid changing the
generated IR shape as part of the same change.

## Style

- Run `cargo fmt` before committing.
- Keep `cargo clippy --all-targets -- -D warnings` clean.
- Prefer small, focused changes with a regression test.
- Do not add native implementations of Starlark library functions to the
  runtime when the function can be lowered from Starlark itself. The point of
  the project is to let LLVM optimize the generated code.
- Avoid machine-specific absolute paths in source and tests.

## Pull requests

A good pull request includes:

1. What changed and why.
2. The Starlark or CLI behavior affected.
3. The commands you ran to verify it.
4. Before/after benchmark numbers when the change targets performance.

Please keep benchmark claims reproducible: include the CPU, clang/LLVM version,
compiler flags, and the exact harness command.

## Reporting bugs

Include the `starlarkc --version` output, host/LLVM details, a minimal `.star`
reproduction, and the expected vs. actual output. If the issue is in generated
code, `--emit=ir` output is very helpful.

## License

By contributing, you agree that your contributions are licensed under the
Apache-2.0 license used by this project.
