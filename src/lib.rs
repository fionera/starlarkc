//! A native x86-64 compiler for Starlark.
//!
//! The crate parses Starlark with `starlark_syntax`, lowers it to textual LLVM
//! IR, and provides the C runtime source used to link native executables.

pub mod compiler;
pub mod ir;
pub mod types;

pub use compiler::Compiler;

/// The C runtime embedded by the `starlarkc` CLI.
pub const RUNTIME_C: &str = include_str!("../runtime/runtime.c");
