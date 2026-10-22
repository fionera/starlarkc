use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, ValueEnum};
use starlark_syntax::syntax::{AstModule, Dialect};

use starlarkc::{Compiler, RUNTIME_C};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Emit {
    /// Emit textual LLVM IR.
    Ir,
    /// Emit an x86-64 object file (with the runtime linked in).
    Obj,
    /// Emit x86-64 assembly.
    Asm,
    /// Emit a native x86-64 executable.
    Exe,
}

#[derive(Debug, Parser)]
#[command(
    name = "starlarkc",
    version,
    about = "Compile Starlark to native x86-64 through LLVM"
)]
struct Args {
    /// Starlark source file.
    input: PathBuf,

    /// Output path. Defaults based on --emit.
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// What to emit.
    #[arg(long, value_enum, default_value_t = Emit::Exe)]
    emit: Emit,

    /// LLVM optimization level (0-3).
    #[arg(short = 'O', default_value_t = 2)]
    opt: u8,

    /// LLVM target triple.
    #[arg(long, default_value = "x86_64-unknown-linux-gnu")]
    target: String,

    /// Target architecture for clang/LLVM, e.g. `native`, `x86-64-v3`, or
    /// `znver4`.  This is what enables AVX2/AVX-512 and wider auto-vectorization.
    #[arg(long)]
    march: Option<String>,

    /// Target CPU for clang/LLVM, e.g. `native` or `znver4`.
    #[arg(long)]
    mcpu: Option<String>,

    /// Repository root for Bazel-style `//pkg:file.bzl` load labels.
    /// Defaults to the nearest ancestor containing MODULE.bazel or WORKSPACE.
    #[arg(long)]
    root: Option<PathBuf>,

    /// Compile and immediately run the resulting executable.
    #[arg(long)]
    run: bool,

    /// Keep the generated textual IR next to the output.
    #[arg(long)]
    keep_ir: bool,

    /// clang executable to use.
    #[arg(long, env = "STARLARKC_CLANG", default_value = "clang")]
    clang: String,

    /// Disable link-time optimization (LTO is enabled by default).
    #[arg(long = "no-lto", action = clap::ArgAction::SetFalse, default_value_t = true)]
    lto: bool,
}

fn main() {
    if let Err(e) = real_main() {
        eprintln!("starlarkc: {e:#}");
        std::process::exit(1);
    }
}

fn real_main() -> Result<()> {
    let args = Args::parse();
    if args.opt > 3 {
        bail!("optimization level must be between 0 and 3");
    }
    if args.run && !matches!(args.emit, Emit::Exe) {
        bail!("--run requires --emit=exe");
    }
    let source = fs::read_to_string(&args.input)
        .with_context(|| format!("reading {}", args.input.display()))?;
    let ast = AstModule::parse(
        &args.input.display().to_string(),
        source,
        &Dialect::AllOptionsInternal,
    )
    .map_err(|e| anyhow!("{e}"))?;

    let mut compiler = Compiler::with_target(args.input.clone(), args.target.clone());
    if let Some(root) = &args.root {
        compiler = compiler.with_root(root.clone());
    }
    let ir = compiler.compile(&ast)?;
    let output = args.output.clone().unwrap_or_else(|| default_output(&args));
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
    }

    if matches!(args.emit, Emit::Ir) {
        fs::write(&output, ir).with_context(|| format!("writing {}", output.display()))?;
        return Ok(());
    }

    let tmp = tempfile::tempdir().context("creating temporary directory")?;
    let ir_path = tmp.path().join("module.ll");
    fs::write(&ir_path, &ir).context("writing temporary LLVM IR")?;

    if args.keep_ir {
        let keep = output.with_extension("ll");
        fs::write(&keep, &ir).with_context(|| format!("writing {}", keep.display()))?;
    }

    match args.emit {
        Emit::Ir => unreachable!(),
        Emit::Asm => {
            let mut cmd = Command::new(&args.clang);
            cmd.arg("-S")
                .arg(format!("-O{}", args.opt))
                .arg(format!("--target={}", args.target))
                .arg("-o")
                .arg(&output)
                .arg(&ir_path);
            add_codegen_flags(&mut cmd, &args);
            run_command(cmd, "clang (assembly)")?;
        }
        Emit::Exe => {
            let runtime = tmp.path().join("runtime.c");
            fs::write(&runtime, RUNTIME_C).context("writing runtime")?;
            let mut cmd = Command::new(&args.clang);
            cmd.arg(format!("-O{}", args.opt))
                .arg(format!("--target={}", args.target))
                .arg("-o")
                .arg(&output)
                .arg(&ir_path)
                .arg(&runtime)
                .arg("-lm");
            add_codegen_flags(&mut cmd, &args);
            if args.lto {
                cmd.arg("-flto");
            }
            run_command(cmd, "clang (executable)")?;
        }
        Emit::Obj => {
            let runtime = tmp.path().join("runtime.c");
            fs::write(&runtime, RUNTIME_C).context("writing runtime")?;
            let ir_obj = tmp.path().join("module.o");
            let runtime_obj = tmp.path().join("runtime.o");
            let mut c1 = Command::new(&args.clang);
            c1.arg("-c")
                .arg(format!("-O{}", args.opt))
                .arg(format!("--target={}", args.target))
                .arg("-o")
                .arg(&ir_obj)
                .arg(&ir_path);
            add_codegen_flags(&mut c1, &args);
            if args.lto {
                c1.arg("-flto");
            }
            run_command(c1, "clang (IR object)")?;
            let mut c2 = Command::new(&args.clang);
            c2.arg("-c")
                .arg(format!("-O{}", args.opt))
                .arg(format!("--target={}", args.target))
                .arg("-o")
                .arg(&runtime_obj)
                .arg(&runtime);
            add_codegen_flags(&mut c2, &args);
            if args.lto {
                c2.arg("-flto");
            }
            run_command(c2, "clang (runtime object)")?;
            let mut link = Command::new(&args.clang);
            link.arg("-r")
                .arg("-o")
                .arg(&output)
                .arg(&ir_obj)
                .arg(&runtime_obj);
            if args.lto {
                link.arg("-flto");
            }
            run_command(link, "clang (relocatable link)")?;
        }
    }

    if args.run {
        let status = Command::new(&output)
            .status()
            .with_context(|| format!("running {}", output.display()))?;
        if !status.success() {
            bail!("program exited with {status}");
        }
    }
    Ok(())
}

fn default_output(args: &Args) -> PathBuf {
    let stem = args
        .input
        .file_stem()
        .map(|s| s.to_os_string())
        .unwrap_or_else(|| "a.out".into());
    let mut out = args
        .input
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(stem);
    match args.emit {
        Emit::Ir => {
            out.set_extension("ll");
        }
        Emit::Obj => {
            out.set_extension("o");
        }
        Emit::Asm => {
            out.set_extension("s");
        }
        Emit::Exe => {}
    }
    out
}

fn add_codegen_flags(cmd: &mut Command, args: &Args) {
    if let Some(march) = &args.march {
        cmd.arg(format!("-march={march}"));
    }
    if let Some(mcpu) = &args.mcpu {
        cmd.arg(format!("-mcpu={mcpu}"));
    }
}

fn run_command(mut cmd: Command, what: &str) -> Result<()> {
    let output = cmd.output().with_context(|| format!("running {what}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        bail!("{what} failed with {}\n{}{}", output.status, stdout, stderr);
    }
    Ok(())
}
