# starlarkc

[![CI](https://github.com/fionera/starlarkc/actions/workflows/ci.yml/badge.svg)](https://github.com/fionera/starlarkc/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust 1.87+](https://img.shields.io/badge/rust-1.87%2B-orange.svg)](rust-toolchain.toml)

A native x86-64 compiler for [Starlark](https://github.com/bazelbuild/starlark), built as an LLVM frontend.

It parses Starlark with the AST from
[`facebook/starlark-rust`](https://github.com/facebook/starlark-rust), lowers the
language to textual LLVM IR, and links the result with a small C runtime that
implements dynamic Starlark values, closures, containers, operators, and
built-ins. The result is a normal ELF executable, not an interpreter wrapper.

The implementation also uses [`olivercalder/starlark-rs`](https://github.com/olivercalder/starlark-rs)
as a reference for the language surface and semantics.

## Why starlarkc?

- **A real native compiler.** The final artifact is an ELF executable; there is
  no Starlark interpreter in the hot path.
- **Whole-program type inference.** The frontend discovers integer lists,
  fixed-length lists, and non-escaping comprehensions, then emits unboxed
  `i64` operations where it can prove the representation is safe.
- **LLVM-friendly lowering.** Byte/word packing loops are emitted as widening
  copies and strided gathers so LLVM can auto-vectorize them. Hot loops are
  unrolled by default, and the resulting IR is optimized with `-O2` + LTO.
- **Bazel-oriented.** It understands `load()` labels, repository roots, and the
  top-level statements used by `.bzl` files. It is tested against the real
  [`hermeticbuild/hashes.bzl`](https://github.com/hermeticbuild/hashes.bzl)
  workload.

## Requirements

- Rust 1.87 or newer
- `clang` with LLVM IR support (tested with clang/LLVM 19)
- An x86-64 Linux host for the default target

## Install

From a checkout:

```sh
cargo install --path .
starlarkc --version
```

Or build without installing:

```sh
cargo build --release
./target/release/starlarkc --version
```

## Quick start

```sh
starlarkc examples/hello.star -o hello
./hello
```

Compile and run immediately:

```sh
starlarkc --run examples/fib.star -o fib
```

Emit LLVM IR, an object file, or x86-64 assembly:

```sh
starlarkc --emit=ir  examples/hello.star -o hello.ll
starlarkc --emit=obj examples/hello.star -o hello.o
starlarkc --emit=asm examples/hello.star -o hello.s
```

The default target is `x86_64-unknown-linux-gnu`. Override it with `--target`.

## CLI

```text
starlarkc <INPUT> [OPTIONS]

Options:
  -o, --output <PATH>       Output path
      --emit <KIND>         ir | obj | asm | exe       [default: exe]
  -O <LEVEL>                LLVM optimization level 0-3 [default: 2]
      --target <TRIPLE>     LLVM target triple
      --march <ARCH>        Target CPU architecture (e.g. native, x86-64-v3)
      --mcpu <CPU>          Target CPU model (e.g. native, znver4)
      --no-lto              Disable link-time optimization
      --root <PATH>         Repository root for Bazel-style load labels
      --run                 Run the resulting executable
      --keep-ir             Also write the generated .ll next to the output
      --clang <PATH>        clang executable
```

`STARLARKC_CLANG` can also select the clang executable.  `--march=native`
enables AVX2/AVX-512 auto-vectorization.  The default target is baseline
x86-64 (SSE2), where LLVM's cost model declines to vectorize the widening
and strided loops; pass `--march=x86-64-v3` or `--march=native` to get SIMD.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `STARLARKC_CLANG` | `clang` | clang executable used for lowering/linking |
| `STARLARKC_UNROLL_LIMIT` | `64` | Maximum trip count to unroll |
| `STARLARKC_RANGE_UNROLL_LIMIT` | unroll limit | Separate unroll limit for `range` loops |
| `STARLARKC_CONCAT` | enabled | Fast path for concatenating typed integer lists |
| `STARLARKC_INT_SLICE` | enabled | Fast path for slicing typed integer lists |
| `STARLARKC_FAST_COMPREHENSION` | enabled | Range/comprehension specialization |
| `STARLARKC_COMPREHENSION_CAP` | enabled | Pre-size comprehensions with known bounds |
| `STARLARKC_DUMP_TYPES` | unset | Dump inferred function/global types to stderr |
| `STARLARKC_DUMP_LOCALS` | unset | Dump locals for one function name, or all with `1` |
| `STARLARKC_DUMP_FUNCS` | unset | Dump generated function names to stderr |

Optimization toggles are enabled unless set to `0`. They are primarily useful
for bisecting a miscompile or measuring the effect of an individual fast path.

## Supported language

The compiler implements the core Starlark language:

- `def`, `lambda`, `return`, `if`/`elif`/`else`, `for`, `break`, `continue`,
  `pass`
- lexical closures, nested functions, recursive functions
- positional, keyword, default, `*args`, `**kwargs`, positional-only (`/`),
  and keyword-only (`*`) parameters
- assignments, tuple unpacking, augmented assignment, index and attribute
  assignment targets
- `and`, `or`, `not`, comparisons, `in` / `not in`, arithmetic, bitwise
  operators, shifts, and conditional expressions
- integers, floats, booleans, `None`, strings, lists, tuples, dictionaries,
  ranges, functions, and `struct(...)` values
- list/dict literals and comprehensions, including multiple `for`/`if` clauses
- indexing, negative indexing, and slicing for strings, lists, tuples, and ranges
- f-strings and `%` string formatting
- `load()` of relative `.star`/`.bzl` files, with module caching and aliases
- Bazel-style `load()` labels: `//pkg:file.bzl`, `@repo//pkg:file.bzl`, and
  `@repo`; the repository root is auto-detected from `MODULE.bazel`/`WORKSPACE`
  or set with `--root`
- top-level statements, as used by `.bzl` files

Built-ins include:

`print`, `len`, `str`, `repr`, `type`, `int`, `float`, `bool`, `list`,
`tuple`, `dict`, `range`, `enumerate`, `zip`, `reversed`, `sorted`, `min`,
`max`, `sum`, `abs`, `all`, `any`, `fail`, `getattr`, `hasattr`, `dir`,
`hash`, `ord`, `chr`, `callable`, and `struct`.

Common list, dict, and string methods are implemented, including `append`,
`extend`, `insert`, `remove`, `pop`, `sort`, `get`, `keys`, `values`, `items`,
`update`, `setdefault`, `format`, `join`, `split`, `strip`, `startswith`,
`endswith`, `replace`, `upper`, `lower`, `find`, `count`, and friends.

## Architecture

```text
Starlark source
      |
      v
starlark-rust parser / AST
      |
      v
Rust lowering pass  ->  textual LLVM IR
      |
      v
clang / LLVM  ->  x86-64 object or executable
      |
      v
runtime/runtime.c  (dynamic values, containers, built-ins)
```

### Project layout

| Path | Description |
|---|---|
| `src/compiler.rs` | AST discovery, type-directed lowering, and LLVM IR emission |
| `src/types.rs` | Whole-program type inference |
| `src/ir.rs` | Small textual LLVM IR builder |
| `src/main.rs` | `starlarkc` CLI and clang invocation |
| `runtime/runtime.c` | Dynamic value representation, containers, and built-ins |
| `tests/integration.rs` | End-to-end compile-and-run tests |
| `benchmarks/hashes/` | `hashes.bzl` benchmark harness and results |

### Value representation

Generated IR uses a two-word tagged value:

```llvm
%Value = type { i64, i64 }
```

The tag identifies `None`, bool, int, float, string, list, tuple, dict, range,
function, builtin, or bound method. The payload is either an immediate value or
a pointer to a heap object.

### Closures

Every local variable is stored in a heap cell (`%Value*`). A closure stores an
array of captured cell pointers:

```llvm
%Closure = type { ptr, i64, ptr }
```

This gives nested functions mutable access to the same lexical binding and keeps
the generated function ABI uniform:

```llvm
%Value @fn(ptr %closure, i64 %nargs, ptr %args,
           i64 %nkwargs, ptr %kw_names, ptr %kw_values)
```

Default argument expressions are evaluated at function-definition time and
stored in the closure environment, matching Starlark semantics.

### Runtime

`runtime/runtime.c` is embedded in the `starlarkc` binary and compiled alongside
the generated IR. It is intentionally small and leak-friendly: allocations live
until process exit. This keeps the compiler ABI simple and makes the generated
programs easy to inspect.

## Library API

The compiler is also exposed as a Rust library:

```rust
use starlarkc::Compiler;
use starlark_syntax::syntax::{AstModule, Dialect};

let source = std::fs::read_to_string("hello.star")?;
let ast = AstModule::parse("hello.star", source, &Dialect::AllOptionsInternal)
    .map_err(|e| anyhow::anyhow!("{e}"))?;
let ir = Compiler::with_target(
    "hello.star".into(),
    "x86_64-unknown-linux-gnu".to_owned(),
).compile(&ast)?;
```

`starlarkc::RUNTIME_C` contains the runtime source used by the CLI.

## Examples

```sh
starlarkc --run examples/fib.star -o fib
starlarkc --run examples/modules/main.star -o modules
```

## Benchmarks

`benchmarks/hashes/run.py` benchmarks
[`hermeticbuild/hashes.bzl`](https://github.com/hermeticbuild/hashes.bzl) with
the same workload used by the upstream project. It compares Bazel's Starlark
interpreter, `starlark-rs`, and `starlarkc`, verifies every digest against
CPython's `hashlib`/`zlib` (and the `blake3` package), and subtracts process
startup and module-loading time.

```sh
# Clone the reference workload once.
git clone https://github.com/hermeticbuild/hashes.bzl /tmp/hashes.bzl

# HASHES_BZL points at a hashes.bzl checkout; /tmp/hashes.bzl is the default.
HASHES_BZL=/tmp/hashes.bzl \
  python3 benchmarks/hashes/run.py --repeat 5 --target-ms 300
```

On an AMD EPYC 9555P, 64-byte messages (hashes/s):

| Algorithm | starlarkc | native (CPython) | ratio |
|---|---:|---:|---:|
| md5 | 253,277 | 1,735,548 | 6.9x |
| sha1 | 246,705 | 1,991,411 | 8.1x |
| sha224 | 243,204 | 1,957,299 | 8.0x |
| sha256 | 231,074 | 1,935,909 | 8.4x |
| sha384 | 167,162 | 1,600,789 | 9.6x |
| sha512 | 146,869 | 1,551,658 | 10.6x |
| sha512_224 | 212,018 | 1,638,121 | 7.7x |
| sha512_256 | 214,449 | 1,645,815 | 7.7x |
| sha3_224 | 120,369 | 1,344,174 | 11.2x |
| sha3_256 | 115,940 | 1,339,010 | 11.5x |
| sha3_384 | 111,954 | 1,316,582 | 11.8x |
| sha3_512 | 98,245 | 1,295,063 | 13.2x |
| blake3 | 84,348 | 2,211,881 | 26.2x |
| crc32 | 1,059,639 | 7,046,782 | 6.7x |

65,536-byte messages (throughput):

| Algorithm | starlarkc | native (CPython) | ratio |
|---|---:|---:|---:|
| md5 | 62.5 MiB/s | 924.3 MiB/s | 14.8x |
| sha1 | 63.6 MiB/s | 2143.8 MiB/s | 33.7x |
| sha224 | 245.2 MiB/s | 2036.1 MiB/s | 8.3x |
| sha256 | 243.1 MiB/s | 2034.7 MiB/s | 8.4x |
| sha384 | 265.3 MiB/s | 1189.4 MiB/s | 4.5x |
| sha512 | 264.0 MiB/s | 1190.7 MiB/s | 4.5x |
| sha512_224 | 266.2 MiB/s | 1191.4 MiB/s | 4.5x |
| sha512_256 | 266.8 MiB/s | 1189.4 MiB/s | 4.5x |
| sha3_224 | 40.7 MiB/s | 654.7 MiB/s | 16.1x |
| sha3_256 | 38.8 MiB/s | 620.0 MiB/s | 16.0x |
| sha3_384 | 30.4 MiB/s | 475.3 MiB/s | 15.6x |
| sha3_512 | 21.8 MiB/s | 331.7 MiB/s | 15.2x |
| blake3 | 6.5 MiB/s | 9189.6 MiB/s | 1412.2x |
| crc32 | 394.2 MiB/s | 7041.7 MiB/s | 17.9x |

The native column is CPython's `hashlib`/`zlib`/`blake3`, which use
SHA-NI, AVX-512 and (for BLAKE3) multithreading, so it is a hardware
upper bound rather than a scalar reference.  Against a scalar C SHA-256 the
compiler is within roughly 1.5x.

### Auto-vectorization

The frontend deliberately leaves the byte-conversion and word-packing loops
in a form LLVM can vectorize.  `to_bytes` becomes a widening
`i8 -> i64` copy, and `words_be`/`words_le`/`words64_be`/`words64_le`
become a strided gather plus shift/or.  With `-march=x86-64-v3` (AVX2)
LLVM vectorizes 22 loops in the SHA-256 workload at width 4 (15 of them
interleaved 4x); with `-march=native` (AVX-512) the same loops use width 8:

```sh
starlarkc --root /tmp/hashes.bzl --emit=exe --march=native bench.star -o bench
```

The SHA-2 compression rounds themselves are fully unrolled by the frontend
(default unroll limit 64) and are inherently serial, so they stay scalar;
the vector wins are in the surrounding data movement.  Because the serial
rounds dominate SHA-2, `--march=native` changes the instruction mix but not
the wall-clock time much; the large benchmark gains above come from
eliminating allocations and from fusing the byte/word conversions.

See [`benchmarks/hashes/RESULTS.md`](benchmarks/hashes/RESULTS.md) for the
full table, methodology, and caveats.

## Testing

```sh
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo build --release
```

The integration tests compile and execute real Starlark programs, exercise
closures, comprehensions, keyword arguments, relative and Bazel-style `load`,
`struct`, 64-bit word arithmetic, f-strings, and verify IR, object, and assembly
emission. CI runs the same checks on every push and pull request.

## Limitations

- Integers are 64-bit two's-complement words. Arbitrary-precision Starlark
  integers are not implemented; arithmetic wraps at 64 bits, and values above
  `2**63 - 1` print as negative integers. This is sufficient for
  `hashes.bzl`, which masks every result back to 32 or 64 bits, but it is not
  full Starlark integer semantics.
- Strings are byte strings. Unicode is preserved as UTF-8 bytes, but indexing is
  byte-oriented.
- Bazel-specific values beyond `struct` (`depset`, `select`, providers, etc.)
  are not implemented.
- `struct` fields are stored in a dictionary; unlike Bazel structs, they are
  mutable and are not validated at construction time.
- The runtime is not reference-counted; memory is reclaimed by process exit.
- The implementation targets x86-64 Linux by default. Other LLVM targets can be
  selected with `--target` if a compatible runtime/libc toolchain is available.
- LLVM 19 can hit a jump-threading crash when compiling a standalone call to
  BLAKE3's `_g` helper at `-O2`; the full hash workloads are unaffected. Use
  `-O1`/`-O0` or a newer LLVM if you encounter it.

## References

- [facebook/starlark-rust](https://github.com/facebook/starlark-rust) — parser and AST
- [olivercalder/starlark-rs](https://github.com/olivercalder/starlark-rs) — language reference
- [Starlark specification](https://github.com/bazelbuild/starlark/blob/master/spec.md)

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the
development workflow and [CHANGELOG.md](CHANGELOG.md) for release notes.

## License

Apache-2.0. See [LICENSE](LICENSE).
