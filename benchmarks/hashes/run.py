#!/usr/bin/env python3
"""Benchmark hermeticbuild/hashes.bzl under Bazel, starlark-rs, and starlarkc.

This is the upstream hashes.bzl benchmark harness with an extra `starlarkc`
column. It uses the real `benchmark/workload.bzl`, verifies every digest
against CPython's hashlib/zlib (and the `blake3` package), and reports the
best of several runs after subtracting process/module-load overhead.

Examples:

    # Uses $HASHES_BZL or /tmp/hashes.bzl as the hashes.bzl checkout.
    python3 benchmarks/hashes/run.py

    # Just the native compiler, all algorithms, both sizes.
    python3 benchmarks/hashes/run.py --skip-bazel --skip-starlark-rs

    # One algorithm while iterating on the compiler.
    python3 benchmarks/hashes/run.py --algorithms sha256 --sizes 64,65536

Requirements:
  * a hashes.bzl checkout (default /tmp/hashes.bzl)
  * target/release/starlarkc
  * `starlark` on PATH for the starlark-rs column (optional)
  * `bazel` on PATH for the Bazel column (optional)
  * Python `blake3` for native BLAKE3 verification (optional for other hashes)
"""

import argparse
import hashlib
import json
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import time
import zlib

ALGORITHMS = [
    "md5",
    "sha1",
    "sha224",
    "sha256",
    "sha384",
    "sha512",
    "sha512_224",
    "sha512_256",
    "sha3_224",
    "sha3_256",
    "sha3_384",
    "sha3_512",
    "blake3",
    "crc32",
]

REPO = pathlib.Path(__file__).resolve().parents[2]


def native(algorithm):
    if algorithm == "crc32":
        return lambda data: "%08x" % zlib.crc32(data)
    if algorithm == "blake3":
        import blake3

        return lambda data: blake3.blake3(data).hexdigest()
    return lambda data: hashlib.new(algorithm, data).hexdigest()


def message(size):
    return (b"0123456789abcdef" * (size // 16 + 1))[:size]


def check(interpreter, algorithm, size, digest):
    expected = native(algorithm)(message(size))
    if digest != expected:
        sys.exit(f"{interpreter} {algorithm}({size} bytes) = {digest}, expected {expected}")


def bazel(args):
    """Seconds per call for each (algorithm, size), in Bazel."""
    bench = args.hashes_dir / "benchmark"
    config = json.dumps(
        {
            "algorithms": args.algorithms,
            "repeat": args.repeat,
            "sizes": args.sizes,
            "target_ms": args.target_ms,
        }
    )
    subprocess.run(
        [
            "bazel",
            "fetch",
            "--repo=@hashes_bench_results",
            "--repo_env=BENCH_CONFIG=" + config,
            "--repo_env=BENCH_RUN=%d" % time.time_ns(),
        ],
        cwd=bench,
        check=True,
    )
    output_base = subprocess.run(
        ["bazel", "info", "output_base"], cwd=bench, check=True, capture_output=True, text=True
    ).stdout.strip()
    [path] = pathlib.Path(output_base, "external").glob("*hashes_bench_results/results.json")
    seconds = {}
    for result in json.loads(path.read_text())["results"]:
        check("Bazel", result["algorithm"], result["size"], result["digest"])
        seconds[result["algorithm"], result["size"]] = min(result["ns"]) / result["calls"] / 1e9
    return seconds


_LOAD = re.compile(r'load\("([^"]+)"')


def _path_for(label, package):
    if label == "@hashes.bzl":
        return "hashes.bzl"
    if label.startswith("//"):
        package, _, name = label[2:].partition(":")
    elif label.startswith(":"):
        name = label[1:]
    else:
        raise ValueError(f"can't stage load of {label!r}")
    return "/".join(p for p in [package, name] if p)


def stage(root, dest):
    """Copies the library and workload to dest, with load() labels rewritten to paths."""
    files = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--"]
        + ["hashes.bzl", "private/*.bzl", "benchmark/workload.bzl"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    for name in files:
        package = str(pathlib.PurePosixPath(name).parent).removeprefix(".")
        source = (root / name).read_text(encoding="latin-1")
        source = _LOAD.sub(lambda m: f'load("{_path_for(m.group(1), package)}"', source)
        (dest / name).parent.mkdir(parents=True, exist_ok=True)
        (dest / name).write_text(source, encoding="latin-1")


def starlark_rs(args, stage_dir):
    """Seconds per call for each (algorithm, size), in starlark-rs."""

    def evaluate(algorithm, size, calls):
        script = stage_dir / "bench.star"
        script.write_text(
            'load("benchmark/workload.bzl", "message", "run")\n'
            "def main():\n"
            f'    print(run("{algorithm}", message({size}), {calls}))\n'
            "main()\n"
        )
        start = time.perf_counter_ns()
        result = subprocess.run(
            [args.starlark, "bench.star"], cwd=stage_dir, check=True, capture_output=True, text=True
        )
        ns = time.perf_counter_ns() - start
        # starlark-rs print() writes to stderr.
        return ns, result.stderr.split()[0]

    seconds = {}
    for algorithm in args.algorithms:
        for size in args.sizes:
            baseline_ns = min(evaluate(algorithm, size, 0)[0] for _ in range(args.repeat))
            calls, target_ns = 1, args.target_ms * 1e6
            while True:
                ns, digest = evaluate(algorithm, size, calls)
                if ns - baseline_ns >= target_ns / 8:
                    calls = max(1, int(calls * target_ns / (ns - baseline_ns)))
                    break
                calls *= 4
            check("starlark-rs", algorithm, size, digest)
            ns = min(evaluate(algorithm, size, calls)[0] for _ in range(args.repeat))
            seconds[algorithm, size] = (ns - baseline_ns) / calls / 1e9
    return seconds


def starlarkc(args):
    """Seconds per call for each (algorithm, size), compiled by starlarkc.

    Each measurement is a whole native process. The zero-call process time
    (startup + loading/compiling the module graph) is subtracted, so the
    number is the cost of the hash loop itself.
    """
    build = args.hashes_dir / ".tmp" / "benchmark" / "starlarkc"
    shutil.rmtree(build, ignore_errors=True)
    build.mkdir(parents=True)

    def evaluate(algorithm, size, calls):
        script = build / "bench.star"
        script.write_text(
            'load("@hashes.bzl//benchmark:workload.bzl", "message", "run")\n'
            f'print(run("{algorithm}", message({size}), {calls}))\n'
        )
        exe = build / f"bench-{algorithm}-{size}-{calls}"
        subprocess.run(
            [
                args.starlarkc,
                "--root",
                str(args.hashes_dir),
                "--emit=exe",
                str(script),
                "-o",
                str(exe),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        start = time.perf_counter_ns()
        result = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
        ns = time.perf_counter_ns() - start
        # starlarkc's print() writes to stdout.
        return ns, result.stdout.split()[0]

    seconds = {}
    for algorithm in args.algorithms:
        for size in args.sizes:
            baseline_ns = min(evaluate(algorithm, size, 0)[0] for _ in range(args.repeat))
            calls, target_ns = 1, args.target_ms * 1e6
            while True:
                ns, digest = evaluate(algorithm, size, calls)
                if ns - baseline_ns >= target_ns / 8:
                    calls = max(1, int(calls * target_ns / (ns - baseline_ns)))
                    break
                calls *= 4
            check("starlarkc", algorithm, size, digest)
            ns = min(evaluate(algorithm, size, calls)[0] for _ in range(args.repeat))
            seconds[algorithm, size] = (ns - baseline_ns) / calls / 1e9
            print(
                f"  starlarkc {algorithm} {size}: {calls} calls, "
                f"{seconds[algorithm, size] * 1e6:.1f} us/hash",
                file=sys.stderr,
            )
    return seconds


def native_seconds(args):
    seconds = {}
    for algorithm in args.algorithms:
        fn = native(algorithm)
        for size in args.sizes:
            data = message(size)
            calls, target = 1, args.target_ms / 1e3
            while True:
                start = time.perf_counter()
                for _ in range(calls):
                    fn(data)
                elapsed = time.perf_counter() - start
                if elapsed >= target / 8:
                    calls = max(1, int(calls * target / elapsed))
                    break
                calls *= 4
            best = float("inf")
            for _ in range(args.repeat):
                start = time.perf_counter()
                for _ in range(calls):
                    fn(data)
                best = min(best, time.perf_counter() - start)
            seconds[algorithm, size] = best / calls
    return seconds


def rate(seconds, size):
    if size >= 1024:
        per_second = size / seconds
        for unit in ["B", "KiB", "MiB", "GiB"]:
            if per_second < 1024 or unit == "GiB":
                return (
                    f"{per_second:,.0f} {unit}/s"
                    if per_second >= 10
                    else f"{per_second:.1f} {unit}/s"
                )
            per_second /= 1024
    return f"{1 / seconds:,.0f}/s"


def version(command):
    return subprocess.run(command, capture_output=True, text=True).stdout.strip().splitlines()[0]


def cpu():
    for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.partition(":")[2].strip()
    return platform.processor()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--algorithms", type=lambda s: s.split(","), default=ALGORITHMS)
    parser.add_argument("--sizes", type=lambda s: [int(n) for n in s.split(",")], default=[64, 65536])
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--target-ms", type=int, default=250)
    parser.add_argument(
        "--hashes-dir",
        type=pathlib.Path,
        default=pathlib.Path(os.environ.get("HASHES_BZL", "/tmp/hashes.bzl")),
        help="hashes.bzl checkout (default: $HASHES_BZL or /tmp/hashes.bzl)",
    )
    parser.add_argument("--starlark", default="starlark", help="starlark-rs binary")
    parser.add_argument("--starlarkc", default=str(REPO / "target" / "release" / "starlarkc"))
    parser.add_argument("--skip-bazel", action="store_true")
    parser.add_argument("--skip-starlark-rs", action="store_true")
    parser.add_argument("--json", type=pathlib.Path, help="write raw seconds per hash as JSON")
    args = parser.parse_args()

    args.hashes_dir = args.hashes_dir.resolve()
    args.starlarkc = str(pathlib.Path(shutil.which(args.starlarkc) or args.starlarkc).resolve())
    if not pathlib.Path(args.starlarkc).is_file():
        sys.exit(f"starlarkc not found: {args.starlarkc}")
    if not (args.hashes_dir / "hashes.bzl").is_file():
        sys.exit(
            f"not a hashes.bzl checkout: {args.hashes_dir}\n"
            "clone it with: git clone https://github.com/hermeticbuild/hashes.bzl "
            f"{args.hashes_dir}"
        )

    columns = {}
    if not args.skip_bazel:
        columns["Bazel"] = bazel(args)
    if not args.skip_starlark_rs:
        starlark = shutil.which(args.starlark)
        if starlark is None:
            sys.exit(f"starlark-rs binary not found: {args.starlark}")
        args.starlark = str(pathlib.Path(starlark).resolve())
        stage_dir = args.hashes_dir / ".tmp" / "benchmark" / "stage"
        shutil.rmtree(stage_dir, ignore_errors=True)
        stage_dir.mkdir(parents=True)
        try:
            stage(args.hashes_dir, stage_dir)
            columns["starlark-rs"] = starlark_rs(args, stage_dir)
        finally:
            shutil.rmtree(stage_dir, ignore_errors=True)
    columns["starlarkc"] = starlarkc(args)
    columns["native"] = native_seconds(args)

    print(f"- CPU: {cpu()}")
    if "Bazel" in columns:
        print(f"- Bazel: {version(['bazel', '--version'])}")
    if "starlark-rs" in columns:
        print(f"- starlark-rs: {version([args.starlark, '--version'])}")
    print(f"- starlarkc: {version([args.starlarkc, '--version'])}")
    print(f"- native: CPython {platform.python_version()} hashlib/zlib, blake3 package")
    print(f"- hashes.bzl: {args.hashes_dir}")
    print(f"- repeat: {args.repeat}, target: {args.target_ms} ms")
    for size in args.sizes:
        unit = "throughput" if size >= 1024 else "hashes per second"
        print(f"\n{size:,}-byte messages ({unit}):\n")
        names = list(columns)
        print("| Algorithm | " + " | ".join(names) + " | starlarkc vs native |")
        print("|---|" + "---:|" * (len(names) + 1))
        for algorithm in args.algorithms:
            cells = [rate(columns[name][algorithm, size], size) for name in names]
            slowdown = columns["starlarkc"][algorithm, size] / columns["native"][algorithm, size]
            print(f"| {algorithm} | " + " | ".join(cells) + f" | {slowdown:,.0f}x slower |")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cpu": cpu(),
            "sizes": args.sizes,
            "algorithms": args.algorithms,
            "seconds": {
                name: {f"{a}/{s}": v for (a, s), v in col.items()} for name, col in columns.items()
            },
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
