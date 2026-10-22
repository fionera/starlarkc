use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn starlarkc() -> &'static str {
    env!("CARGO_BIN_EXE_starlarkc")
}

fn write(dir: &Path, name: &str, source: &str) -> PathBuf {
    let path = dir.join(name);
    fs::write(&path, source).unwrap();
    path
}

fn run_program(dir: &Path, source: &str) -> String {
    let input = write(dir, "main.star", source);
    let output = dir.join("main");
    let result = Command::new(starlarkc())
        .arg("--run")
        .arg("-o")
        .arg(&output)
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "compiler failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    String::from_utf8(result.stdout).unwrap()
}

#[test]
fn arithmetic_and_collections() {
    let dir = tempfile::tempdir().unwrap();
    let out = run_program(
        dir.path(),
        r#"
x = [i * 2 for i in range(5) if i > 0]
d = {"a": 1, "b": 2}
print(x, d.get("a"), "a" in d, len(x))
print(7 // 2, -7 // 2, 7 % 3, 1 / 2)
"#,
    );
    assert_eq!(out, "[2, 4, 6, 8] 1 True 4\n3 -4 1 0.5\n");
}

#[test]
fn closures_recursion_and_defaults() {
    let dir = tempfile::tempdir().unwrap();
    let out = run_program(
        dir.path(),
        r#"
def outer(x):
    y = x * 2
    def inner(z):
        return x + y + z
    return inner
f = outer(10)
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)
print(f(5), [fib(i) for i in range(8)])
x = 1
def g(a=x):
    return a
x = 2
print(g())
"#,
    );
    assert_eq!(out, "35 [0, 1, 1, 2, 3, 5, 8, 13]\n1\n");
}

#[test]
fn keyword_varargs_and_positional_only() {
    let dir = tempfile::tempdir().unwrap();
    let out = run_program(
        dir.path(),
        r#"
def f(a, b=2, *args, **kwargs):
    print(a, b, args, kwargs)
    return a + b
print(f(1, 3, 4, x=5))
def g(a, /, b, *, c=7):
    return a + b + c
print(g(1, 2, c=3))
"#,
    );
    assert_eq!(out, "1 3 (4,) {\"x\": 5}\n4\n6\n");
}

#[test]
fn strings_and_fstrings() {
    let dir = tempfile::tempdir().unwrap();
    let out = run_program(
        dir.path(),
        r#"
name = "world"
print(f"hello {name} {{ok}} {1 + 2}")
print("%s=%d" % ("n", 42))
print("a,b,c".split(","), "-".join(["x", "y"]))
print("  hi  ".strip().upper(), "abc"[::-1])
"#,
    );
    assert_eq!(
        out,
        "hello world {ok} 3\nn=42\n[\"a\", \"b\", \"c\"] x-y\nHI cba\n"
    );
}

#[test]
fn cached_load_modules() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "lib.star",
        "print(\"loading lib\")\nBASE = 10\ndef add(x):\n    return x + BASE\n",
    );
    let out = run_program(
        dir.path(),
        r#"
load("lib.star", "BASE", plus="add")
load("lib.star", "add")
print(BASE, plus(5), add(1))
"#,
    );
    assert_eq!(out, "loading lib\n10 15 11\n");
}

#[test]
fn struct_values() {
    let dir = tempfile::tempdir().unwrap();
    let out = run_program(
        dir.path(),
        r#"
s = struct(a = 1, b = "two")
print(type(s), s.a, s.b, getattr(s, "b"), hasattr(s, "a"), hasattr(s, "z"))
print(s == struct(a = 1, b = "two"), "a" in s, dir(s))
"#,
    );
    assert_eq!(
        out,
        "struct 1 two two True False\nTrue True [\"a\", \"b\"]\n"
    );
}

#[test]
fn bazel_load_labels_and_64_bit_words() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("MODULE.bazel"), "").unwrap();
    fs::create_dir_all(dir.path().join("private")).unwrap();
    write(
        &dir.path().join("private"),
        "bits.bzl",
        "VALUE = 0xFFFFFFFFFFFFFFFF\ndef low(x):\n    return x & 0xFFFFFFFF\n",
    );
    let out = run_program(
        dir.path(),
        r#"
load("//private:bits.bzl", "VALUE", "low")
load("@myrepo//private:bits.bzl", wrapped = "VALUE")
print(VALUE == wrapped, low(VALUE), (VALUE + 1) & 0xFFFFFFFFFFFFFFFF)
print(VALUE >> 32)
"#,
    );
    assert_eq!(out, "True 4294967295 0\n4294967295\n");
}

#[test]
fn emits_ir_object_and_assembly() {
    let dir = tempfile::tempdir().unwrap();
    let input = write(dir.path(), "x.star", "print(1 + 2)\n");

    let ir = dir.path().join("x.ll");
    let result = Command::new(starlarkc())
        .arg("--emit=ir")
        .arg("-o")
        .arg(&ir)
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let text = fs::read_to_string(&ir).unwrap();
    assert!(text.contains("target triple = \"x86_64-unknown-linux-gnu\""));
    assert!(text.contains("define void @sk_module_body"));

    let obj = dir.path().join("x.o");
    let result = Command::new(starlarkc())
        .arg("--emit=obj")
        .arg("-o")
        .arg(&obj)
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(fs::metadata(&obj).unwrap().len() > 0);

    let asm = dir.path().join("x.s");
    let result = Command::new(starlarkc())
        .arg("--emit=asm")
        .arg("-o")
        .arg(&asm)
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let asm_text = fs::read_to_string(&asm).unwrap();
    assert!(asm_text.contains(".text"));
}

#[test]
fn comprehension_scope_and_conditional_expressions() {
    let dir = tempfile::tempdir().unwrap();
    let out = run_program(
        dir.path(),
        r#"
def f():
    x = 100
    y = [x for x in range(3)]
    return x, y
print(f())
def choose(a, b, c):
    return (a and b) if c else (a or b)
print(choose(1, 2, True), choose(0, 2, False))
"#,
    );
    assert_eq!(out, "(100, [0, 1, 2])\n2 2\n");
}
