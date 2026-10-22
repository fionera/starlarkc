# hashes.bzl benchmark results

Workload: the upstream `benchmark/workload.bzl` from
[`hermeticbuild/hashes.bzl`](https://github.com/hermeticbuild/hashes.bzl),
run through `starlarkc`.  Every digest is checked against CPython's
`hashlib`/`zlib` and the `blake3` package.  Timings subtract a zero-call
baseline (process start + module loading) and take the best of several runs.

Machine: AMD EPYC 9555P (Zen 5), 4.4 GHz, Linux, clang/LLVM 19.1.7.
Compiler flags: `-O2` + LTO, default target `x86_64-unknown-linux-gnu`
(no `-march`).  The `native` column is CPython 3.13 `hashlib`/`zlib`/`blake3`,
which uses SHA-NI, AVX-512 and multithreading; it is a hardware upper bound,
not a scalar reference.

## 64-byte messages (hashes/s)

| Algorithm | starlarkc | native | ratio |
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

## 65,536-byte messages (throughput)

| Algorithm | starlarkc | native | ratio |
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

## Improvement over the previous compiler build

The previous build (before the allocation arena, fixed-length list tracking,
fused `words64_*`, constant-folded Keccak tables and stack-allocated
comprehension results) measured:

| Algorithm (64 KiB) | before | after | speedup |
|---|---:|---:|---:|
| sha256 | 150.6 MiB/s | 243.1 MiB/s | 1.6x |
| sha512 | 33.8 MiB/s | 264.0 MiB/s | 7.8x |
| sha3_256 | 5.5 MiB/s | 38.8 MiB/s | 7.1x |
| md5 | 39.1 MiB/s | 62.5 MiB/s | 1.6x |
| crc32 | 240.9 MiB/s | 394.2 MiB/s | 1.6x |
| blake3 | 5.7 MiB/s | 6.5 MiB/s | 1.1x |

## Reproducing

```sh
# Full harness (compiles each algorithm/size/call-count combination).
HASHES_BZL=/tmp/hashes.bzl \
  python3 benchmarks/hashes/run.py --repeat 3 --target-ms 200

# One algorithm while iterating.
HASHES_BZL=/tmp/hashes.bzl \
  python3 benchmarks/hashes/run.py --skip-bazel --skip-starlark-rs \
    --algorithms sha256 --sizes 64,65536 --repeat 3 --target-ms 200
```

## Notes

* `starlarkc` emits unboxed `i64` list operations and uses the whole-program
  type inference to pick `SkIntList` for every byte/word list.
* The SHA-2 round functions are fully unrolled (`STARLARKC_UNROLL_LIMIT`,
  default 64).  The message schedule and byte packing stay as loops so LLVM
  can vectorize them.
* Large list payloads are carved out of shared 2 MiB-aligned
  `MADV_HUGEPAGE` chunks; this removed most of the kernel page-zeroing cost
  that dominated the earlier measurements.
* `--march=native` enables AVX-512.  It vectorizes the same 22 loops at
  width 8 instead of width 4, but the serial SHA-2 rounds dominate, so the
  wall-clock gain is small.
