use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use starlark_syntax::lexer::TokenInt;
use starlark_syntax::syntax::ast::{
    ArgumentP, AssignOp, AssignTarget, AstAssignTarget, AstExpr, AstLiteral, AstStmt, BinOp,
    Clause, DefP, Expr, ForClause, LambdaP, ParameterP, Stmt,
};
use starlark_syntax::syntax::{AstModule, Dialect};

use crate::ir::Builder;

const RUNTIME_DECLS: &str = r#"
%Value = type { i64, i64 }
%SkValue = type { i64, i64 }
%SkList = type { i64, i64, ptr }
%SkString = type { i64, [0 x i8] }
%Closure = type { ptr, i64, ptr }

declare void @sk_runtime_init()
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)
declare %Value @sk_none()
declare %Value @sk_bool(i64)
declare %Value @sk_int(i64)
declare %Value @sk_float(double)
declare %Value @sk_string(ptr, i64)
declare %Value @sk_list_new()
declare %Value @sk_list_new_from(i64, ptr)
declare %Value @sk_int_list_new()
declare %Value @sk_int_list_new_cap(i64)
declare %Value @sk_int_list_new_from(i64, ptr)
declare %Value @sk_int_list_zeros(i64)
declare void @sk_int_list_reserve(%Value, i64)
declare void @sk_int_list_push(%Value, i64)
declare %Value @sk_int_list_concat(i64, ptr)
declare %Value @sk_int_list_slice(%Value, i64, i64, i64, i64, i64, i64)
declare %Value @sk_tuple_new(i64, ptr)
declare %Value @sk_dict_new()
declare %Value @sk_struct_new()
declare %Value @sk_range(i64, i64, i64)
declare %Value @sk_function_new(ptr, i64, ptr)
declare %Value @sk_builtin(i64)
declare %Value @sk_unbound()
declare ptr @sk_cell_new(%Value)
declare ptr @sk_env_new(i64)
declare void @sk_env_set(ptr, i64, ptr)
declare void @sk_list_push(%Value, %Value)
declare %Value @sk_elems(%Value)
declare void @sk_append(%Value, %Value)
declare void @sk_dict_set(%Value, %Value, %Value)
declare %Value @sk_add(%Value, %Value)
declare %Value @sk_sub(%Value, %Value)
declare %Value @sk_mul(%Value, %Value)
declare %Value @sk_div(%Value, %Value)
declare %Value @sk_floordiv(%Value, %Value)
declare %Value @sk_mod(%Value, %Value)
declare %Value @sk_bitand(%Value, %Value)
declare %Value @sk_bitor(%Value, %Value)
declare %Value @sk_bitxor(%Value, %Value)
declare %Value @sk_shl(%Value, %Value)
declare %Value @sk_shr(%Value, %Value)
declare %Value @sk_eq(%Value, %Value)
declare %Value @sk_ne(%Value, %Value)
declare %Value @sk_lt(%Value, %Value)
declare %Value @sk_gt(%Value, %Value)
declare %Value @sk_le(%Value, %Value)
declare %Value @sk_ge(%Value, %Value)
declare %Value @sk_contains(%Value, %Value)
declare %Value @sk_not(%Value)
declare %Value @sk_neg(%Value)
declare %Value @sk_pos(%Value)
declare %Value @sk_bitnot(%Value)
declare i64 @sk_truthy(%Value)
declare %Value @sk_index(%Value, %Value)
declare %Value @sk_slice(%Value, %Value, %Value, %Value, i64, i64, i64)
declare void @sk_set_index(%Value, %Value, %Value)
declare %Value @sk_getattr(%Value, %Value)
declare void @sk_setattr(%Value, %Value, %Value)
declare ptr @sk_callargs_new()
declare void @sk_callargs_arg(ptr, %Value)
declare void @sk_callargs_args(ptr, %Value)
declare void @sk_callargs_kw(ptr, %Value, %Value)
declare void @sk_callargs_kwargs(ptr, %Value)
declare %Value @sk_call(%Value, ptr)
declare i64 @sk_iter_len(%Value)
declare i64 @sk_range_len_i64(i64, i64, i64)
declare %Value @sk_iter_at(%Value, i64)
declare void @sk_unpack_len(%Value, i64)
declare %Value @sk_tuple_slice(ptr, i64, i64)
declare %Value @sk_str(%Value)
declare %Value @sk_repr(%Value)
declare %Value @sk_check_bound(%Value, ptr, i64)
declare %Value @sk_undefined(ptr, i64)
declare void @sk_error_cstr(ptr, i64)
declare i64 @sk_string_eq_cstr(%Value, ptr, i64)
declare i64 @sk_floordiv_i64(i64, i64)
declare i64 @sk_mod_i64(i64, i64)
declare i64 @sk_ord_int(%Value)
declare i64 @sk_int_list_get(%Value, i64)
declare i64 @llvm.fshl.i64(i64, i64, i64)
declare i64 @llvm.fshr.i64(i64, i64, i64)
declare i32 @llvm.fshl.i32(i32, i32, i32)
declare i32 @llvm.fshr.i32(i32, i32, i32)
"#;

#[derive(Clone, Debug)]
pub struct LoadedModule {
    pub init_fn: String,
    pub cache_global: String,
}

pub struct Compiler {
    source_path: PathBuf,
    source_name: String,
    module_root: PathBuf,
    target: String,
    globals: HashMap<String, String>,
    global_order: Vec<String>,
    extra_globals: Vec<String>,
    functions: Vec<String>,
    string_decls: Vec<String>,
    string_map: HashMap<String, String>,
    next_fn: usize,
    next_loaded: usize,
    next_str: usize,
    loaded: HashMap<PathBuf, LoadedModule>,
    loading: HashSet<PathBuf>,
    module_asts: HashMap<PathBuf, AstModule>,
    types: crate::types::ProgramTypes,
    /// Stable LLVM symbol for every top-level Starlark function.  Keeping this
    /// map in the compiler lets call sites bypass the dynamic `sk_call` path
    /// whenever the callee is statically known.
    function_names: HashMap<crate::types::FuncId, String>,
    const_ints: HashMap<(PathBuf, String), i64>,
    const_lists: HashMap<(PathBuf, String), Vec<i64>>,
    const_list_globals: HashMap<(PathBuf, String), String>,
    const_list_decls: Vec<String>,
    next_const_list: usize,
}

impl Compiler {
    pub fn with_target(source_path: PathBuf, target: String) -> Self {
        let source_name = source_path.display().to_string();
        let module_root = detect_module_root(&source_path);
        Self {
            source_path,
            source_name,
            module_root,
            target,
            globals: HashMap::new(),
            global_order: Vec::new(),
            extra_globals: Vec::new(),
            functions: Vec::new(),
            string_decls: Vec::new(),
            string_map: HashMap::new(),
            next_fn: 0,
            next_loaded: 0,
            next_str: 0,
            loaded: HashMap::new(),
            loading: HashSet::new(),
            module_asts: HashMap::new(),
            types: crate::types::ProgramTypes::default(),
            function_names: HashMap::new(),
            const_ints: HashMap::new(),
            const_lists: HashMap::new(),
            const_list_globals: HashMap::new(),
            const_list_decls: Vec::new(),
            next_const_list: 0,
        }
    }

    /// Overrides the repository root used to resolve Bazel-style `//pkg:file.bzl`
    /// load labels. By default the root is the nearest ancestor containing a
    /// `MODULE.bazel`, `WORKSPACE`, or `WORKSPACE.bazel` file.
    pub fn with_root(mut self, root: PathBuf) -> Self {
        self.module_root = root;
        self
    }

    pub fn compile(mut self, ast: &AstModule) -> Result<String> {
        let main_path =
            fs::canonicalize(&self.source_path).unwrap_or_else(|_| self.source_path.clone());
        self.discover_module(&main_path, ast)?;
        let modules: Vec<_> = self
            .module_asts
            .iter()
            .map(|(path, ast)| (path.clone(), ast.clone()))
            .collect();
        for (path, ast) in &modules {
            self.collect_constants(path, ast.statement());
        }
        self.prepare_function_names(&main_path, ast.statement());
        self.types = crate::types::infer_program(&self.module_asts, &self.module_root);
        if std::env::var_os("STARLARKC_DUMP_TYPES").is_some() {
            let mut funcs: Vec<_> = self.types.funcs.iter().collect();
            funcs.sort_by(|a, b| a.0.cmp(b.0));
            for ((module, name), sig) in funcs {
                eprintln!(
                    "{}::{} params={:?} ret={:?}",
                    module.display(),
                    name,
                    sig.params,
                    sig.ret
                );
            }
            let mut globals: Vec<_> = self
                .types
                .modules
                .iter()
                .flat_map(|(m, info)| info.globals.iter().map(move |(n, t)| (m, n, t)))
                .collect();
            globals.sort_by(|a, b| a.0.cmp(b.0).then(a.1.cmp(b.1)));
            for (module, name, ty) in globals {
                eprintln!("{}::{} = {:?}", module.display(), name, ty);
            }
            if let Ok(filter) = std::env::var("STARLARKC_DUMP_LOCALS") {
                let mut locals: Vec<_> = self
                    .types
                    .locals
                    .iter()
                    .filter(|((_, f, _), _)| filter == "1" || f == &filter)
                    .collect();
                locals.sort_by(|a, b| a.0.cmp(b.0));
                for ((module, f, name), ty) in locals {
                    eprintln!("local {}::{}::{} = {:?}", module.display(), f, name, ty);
                }
            }
        }

        let mut names = Vec::new();
        collect_top_level_names(ast.statement(), &mut names);
        for name in &names {
            self.ensure_global(name);
        }

        let source_path = self.source_path.clone();
        let mut fc = FunctionCompiler::new(
            &mut self,
            "sk_module_body".to_owned(),
            Mode::Script,
            true,
            source_path,
            "<module>".to_owned(),
            HashSet::new(),
            Vec::new(),
            HashSet::new(),
            0,
        );
        fc.compile_stmt(ast.statement())?;
        let out = fc.finish();
        self.functions.push(out.text);

        Ok(self.finish())
    }

    fn finish(self) -> String {
        let mut out = String::new();
        out.push_str("; Generated by starlarkc\n");
        let _ = writeln!(
            out,
            "source_filename = \"{}\"",
            escape_ir_string(&self.source_name)
        );
        let _ = writeln!(
            out,
            "target triple = \"{}\"",
            escape_ir_string(&self.target)
        );
        out.push_str(RUNTIME_DECLS);
        out.push('\n');

        for decl in &self.string_decls {
            out.push_str(decl);
            out.push('\n');
        }
        for decl in &self.const_list_decls {
            out.push_str(decl);
            out.push('\n');
        }
        out.push('\n');

        for sym in &self.global_order {
            let _ = writeln!(out, "{sym} = internal global %Value {{ i64 11, i64 0 }}");
        }
        for decl in &self.extra_globals {
            out.push_str(decl);
            out.push('\n');
        }
        out.push('\n');

        for f in &self.functions {
            out.push_str(f);
            out.push('\n');
        }

        out.push_str(
            r#"
define i32 @main() {
entry:
  call void @sk_runtime_init()
  call void @sk_module_body()
  ret i32 0
}
"#,
        );
        out
    }

    fn ensure_global(&mut self, name: &str) -> String {
        if let Some(sym) = self.globals.get(name) {
            return sym.clone();
        }
        let sym = format!("@g{}", self.global_order.len());
        self.globals.insert(name.to_owned(), sym.clone());
        self.global_order.push(sym.clone());
        sym
    }

    fn next_function_name(&mut self) -> String {
        let name = format!("sk_fn_{}", self.next_fn);
        self.next_fn += 1;
        name
    }

    fn function_name_for(&mut self, module: &Path, name: &str) -> String {
        let id = (module.to_path_buf(), name.to_owned());
        if let Some(sym) = self.function_names.get(&id) {
            return sym.clone();
        }
        let sym = self.next_function_name();
        self.function_names.insert(id, sym.clone());
        sym
    }

    /// Pre-assign symbols to module-level functions so a call can be lowered
    /// directly even when the callee is defined later in the same file.
    fn prepare_function_names(&mut self, module: &Path, stmt: &AstStmt) {
        match &stmt.node {
            Stmt::Statements(stmts) => {
                for s in stmts {
                    self.prepare_function_names(module, s);
                }
            }
            Stmt::Def(d) => {
                self.function_name_for(module, &d.name.node.ident);
            }
            Stmt::If(_, body) => self.prepare_function_names(module, body),
            Stmt::IfElse(_, branches) => {
                self.prepare_function_names(module, &branches.0);
                self.prepare_function_names(module, &branches.1);
            }
            Stmt::For(f) => self.prepare_function_names(module, &f.body),
            _ => {}
        }
    }

    fn collect_constants(&mut self, module: &Path, stmt: &AstStmt) {
        match &stmt.node {
            Stmt::Statements(stmts) => {
                for s in stmts {
                    self.collect_constants(module, s);
                }
            }
            Stmt::Assign(a) => {
                let AssignTarget::Identifier(id) = &a.lhs.node else {
                    return;
                };
                let name = id.node.ident.clone();
                match &a.rhs.node {
                    Expr::List(_) | Expr::ListComprehension(..) => {
                        let Some(values) = const_int_list_ast(&a.rhs) else {
                            return;
                        };
                        let key = (module.to_path_buf(), name);
                        let sym = format!("@sk_const_list_{}", self.next_const_list);
                        self.next_const_list += 1;
                        let elems = values
                            .iter()
                            .map(|v| format!("i64 {v}"))
                            .collect::<Vec<_>>()
                            .join(", ");
                        self.const_list_decls.push(format!(
                            "{sym} = private unnamed_addr constant [{} x i64] [{elems}]",
                            values.len()
                        ));
                        self.const_list_globals.insert(key.clone(), sym);
                        self.const_lists.insert(key, values);
                    }
                    _ => {
                        if let Some(v) = const_int_ast(&a.rhs) {
                            self.const_ints.insert((module.to_path_buf(), name), v);
                        }
                    }
                }
            }
            Stmt::If(_, body) => self.collect_constants(module, body),
            Stmt::IfElse(_, branches) => {
                self.collect_constants(module, &branches.0);
                self.collect_constants(module, &branches.1);
            }
            _ => {}
        }
    }

    fn string_const(&mut self, s: &str) -> (String, usize) {
        if let Some(sym) = self.string_map.get(s) {
            return (sym.clone(), s.len());
        }
        let sym = format!("@.str.{}", self.next_str);
        self.next_str += 1;
        let bytes = s.as_bytes();
        let decl = if bytes.is_empty() {
            format!("{sym} = private unnamed_addr constant [0 x i8] zeroinitializer")
        } else {
            format!(
                "{sym} = private unnamed_addr constant [{} x i8] c\"{}\"",
                bytes.len(),
                llvm_escape(bytes)
            )
        };
        self.string_decls.push(decl);
        self.string_map.insert(s.to_owned(), sym.clone());
        (sym, bytes.len())
    }

    fn discover_module(&mut self, path: &Path, ast: &AstModule) -> Result<()> {
        let canonical = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        if self.module_asts.contains_key(&canonical) {
            return Ok(());
        }
        self.module_asts.insert(canonical.clone(), ast.clone());
        self.discover_loads(&canonical, ast.statement())
    }

    fn discover_loads(&mut self, from: &Path, stmt: &AstStmt) -> Result<()> {
        match &stmt.node {
            Stmt::Statements(stmts) => {
                for s in stmts {
                    self.discover_loads(from, s)?;
                }
            }
            Stmt::Load(l) => {
                let path = resolve_load_path(&self.module_root, from, &l.module.node);
                let canonical = fs::canonicalize(&path).unwrap_or(path);
                if !self.module_asts.contains_key(&canonical) {
                    let source = fs::read_to_string(&canonical).with_context(|| {
                        format!("reading loaded module {}", canonical.display())
                    })?;
                    let ast = AstModule::parse(
                        &canonical.display().to_string(),
                        source,
                        &Dialect::AllOptionsInternal,
                    )
                    .map_err(|e| anyhow!("{e}"))?;
                    self.discover_module(&canonical, &ast)?;
                }
            }
            Stmt::If(_, body) => self.discover_loads(from, body)?,
            Stmt::IfElse(_, branches) => {
                self.discover_loads(from, &branches.0)?;
                self.discover_loads(from, &branches.1)?;
            }
            Stmt::For(f) => self.discover_loads(from, &f.body)?,
            _ => {}
        }
        Ok(())
    }

    fn load_module(&mut self, from: &Path, module: &str) -> Result<LoadedModule> {
        let path = resolve_load_path(&self.module_root, from, module);
        let canonical = fs::canonicalize(&path).unwrap_or(path.clone());
        if let Some(m) = self.loaded.get(&canonical) {
            return Ok(m.clone());
        }
        if !self.loading.insert(canonical.clone()) {
            bail!("cyclic load involving {}", canonical.display());
        }
        let ast = if let Some(ast) = self.module_asts.get(&canonical) {
            ast.clone()
        } else {
            let source = fs::read_to_string(&canonical)
                .with_context(|| format!("reading loaded module {}", canonical.display()))?;
            AstModule::parse(
                &canonical.display().to_string(),
                source,
                &Dialect::AllOptionsInternal,
            )
            .map_err(|e| anyhow!("{e}"))?
        };

        let mut names = Vec::new();
        collect_top_level_names(ast.statement(), &mut names);

        self.prepare_function_names(&canonical, ast.statement());

        let init_fn = format!("sk_load_{}", self.next_fn);
        self.next_fn += 1;
        let cache_global = format!("@loaded{}", self.next_loaded);
        self.next_loaded += 1;

        let mut fc = FunctionCompiler::new(
            self,
            init_fn.clone(),
            Mode::Loaded,
            false,
            canonical.clone(),
            "<module>".to_owned(),
            HashSet::new(),
            names.clone(),
            HashSet::new(),
            0,
        );
        fc.compile_stmt(ast.statement())?;

        let dict = fc.b.tmp();
        fc.b.ins(format!("{dict} = call %Value @sk_dict_new()"));
        for name in &names {
            let value = fc.resolve_value(name)?;
            let key = fc.compile_string_value(name)?;
            fc.b.ins(format!(
                "call void @sk_dict_set(%Value {dict}, %Value {key}, %Value {value})"
            ));
        }
        fc.b.ret("%Value", &dict);
        let out = fc.finish();

        self.functions.push(out.text);
        self.extra_globals.push(format!(
            "{cache_global} = internal global %Value {{ i64 11, i64 0 }}"
        ));
        let lm = LoadedModule {
            init_fn,
            cache_global,
        };
        self.loaded.insert(canonical.clone(), lm.clone());
        self.loading.remove(&canonical);
        Ok(lm)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Script,
    Loaded,
    Function,
}

#[derive(Clone)]
enum CaptureKind {
    Name(String),
    Default(usize),
}

struct FunctionOutput {
    text: String,
    captures: Vec<CaptureKind>,
}

struct LoopLabels {
    break_label: String,
    continue_label: String,
}

#[derive(Clone)]
enum Local {
    /// Heap cell containing a `%Value`; required for captured variables.
    Cell(String),
    /// Stack slot containing a `%Value`.
    ValueAlloca(String),
    /// Stack slot containing an unboxed `i64`.
    IntAlloca(String),
}

enum Place {
    Local(Local),
    Global(String),
}

struct FunctionCompiler<'a> {
    c: &'a mut Compiler,
    b: Builder,
    header: String,
    mode: Mode,
    use_script_globals: bool,
    module_path: PathBuf,
    func_id: String,
    captured: HashSet<String>,
    known_bound: HashSet<String>,
    char_locals: HashSet<String>,
    const_locals: HashMap<String, i64>,
    list_views: HashMap<String, (String, String)>,
    /// Fixed-size compact lists whose length is known exactly.  This is used
    /// after loop unrolling turns an index into a constant: the frontend can
    /// then emit a direct load/store instead of a bounds-checked dynamic
    /// access.  LLVM sees the same alloca and can scalar-replace it.
    fixed_list_lens: HashMap<String, i64>,
    /// SSA values produced by a lowering path that knows the list length.
    value_list_lens: HashMap<String, i64>,
    /// Loop variables that are known to be in-bounds for a particular compact
    /// list.  This lets `for i in range(0, len(xs), n): xs[i]` skip the
    /// per-access bounds branch.
    safe_list_indices: Vec<HashMap<String, String>>,
    /// ListInt locals initialized with `[]` and not yet mutated.  Used to
    /// reserve the final capacity before a byte-conversion loop.
    empty_int_lists: HashSet<String>,
    /// ListInt locals whose capacity exactly matches the number of appends in
    /// the current loop.  Appends can then skip the capacity check.
    exact_capacity_lists: HashSet<String>,
    /// For a loop that appends exactly once per iteration to a list that was
    /// empty on entry, the append index is simply the loop counter.  Storing
    /// through that index (instead of load-len/store-len) leaves a clean
    /// counted loop that LLVM can vectorize.
    append_indices: HashMap<String, String>,
    /// Cached `items` pointer for a list that is appended to once per loop
    /// iteration.  Hoisting the header load out of the loop removes an
    /// apparent aliasing edge between the element stores and the list header.
    append_items: HashMap<String, String>,
    /// When set, the next int-list comprehension should build its result in a
    /// caller-provided stack buffer of this length instead of allocating.
    stack_list_hint: Option<i64>,
    /// Local lists that are only indexed/sliced and never escape the function.
    /// Fixed-size instances can live on the stack.
    non_escaping_lists: HashSet<String>,
    locals: HashMap<String, Local>,
    visible_outer: HashSet<String>,
    captures: Vec<CaptureKind>,
    capture_indices: HashMap<String, usize>,
    default_capture_indices: Vec<usize>,
    loops: Vec<LoopLabels>,
    env: Option<String>,
    unbound: String,
    last_closure: Option<String>,
}

impl<'a> FunctionCompiler<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        c: &'a mut Compiler,
        name: String,
        mode: Mode,
        use_script_globals: bool,
        module_path: PathBuf,
        func_id: String,
        captured: HashSet<String>,
        local_names: Vec<String>,
        visible_outer: HashSet<String>,
        num_defaults: usize,
    ) -> Self {
        let mut b = Builder::new();
        let header = match mode {
            Mode::Script => format!("define void @{name}() {{"),
            Mode::Loaded => format!("define internal %Value @{name}() {{"),
            Mode::Function => format!(
                "define internal %Value @{name}(ptr %closure, i64 %nargs, ptr %args, i64 %nkwargs, ptr %kw_names, ptr %kw_values) {{"
            ),
        };
        b.emit_label("entry");

        let env = if mode == Mode::Function {
            let envp = b.tmp();
            b.ins(format!(
                "{envp} = getelementptr %Closure, ptr %closure, i32 0, i32 2"
            ));
            let env = b.tmp();
            b.ins(format!("{env} = load ptr, ptr {envp}"));
            Some(env)
        } else {
            None
        };

        let unbound = b.tmp();
        b.ins(format!("{unbound} = call %Value @sk_unbound()"));

        let mut locals = HashMap::new();
        for local in local_names {
            if locals.contains_key(&local) {
                continue;
            }
            let ty = match mode {
                Mode::Function => c.types.local(&module_path, &func_id, &local),
                Mode::Script | Mode::Loaded => c.types.global(&module_path, &local),
            };
            if mode == Mode::Loaded {
                let cell = b.tmp();
                b.ins(format!("{cell} = call ptr @sk_cell_new(%Value {unbound})"));
                locals.insert(local, Local::Cell(cell));
            } else if mode == Mode::Function
                && ty == crate::types::Ty::Int
                && !captured.contains(&local)
            {
                let slot = b.tmp();
                b.ins(format!("{slot} = alloca i64"));
                b.ins(format!("store i64 0, ptr {slot}"));
                locals.insert(local, Local::IntAlloca(slot));
            } else if mode == Mode::Function && captured.contains(&local) {
                let cell = b.tmp();
                b.ins(format!("{cell} = call ptr @sk_cell_new(%Value {unbound})"));
                locals.insert(local, Local::Cell(cell));
            } else {
                let slot = b.tmp();
                b.ins(format!("{slot} = alloca %Value"));
                b.ins(format!("store %Value {unbound}, ptr {slot}"));
                locals.insert(local, Local::ValueAlloca(slot));
            }
        }

        let mut this = Self {
            c,
            b,
            header,
            mode,
            use_script_globals,
            module_path,
            func_id,
            captured,
            known_bound: HashSet::new(),
            char_locals: HashSet::new(),
            const_locals: HashMap::new(),
            list_views: HashMap::new(),
            fixed_list_lens: HashMap::new(),
            value_list_lens: HashMap::new(),
            safe_list_indices: Vec::new(),
            empty_int_lists: HashSet::new(),
            exact_capacity_lists: HashSet::new(),
            append_indices: HashMap::new(),
            append_items: HashMap::new(),
            stack_list_hint: None,
            non_escaping_lists: HashSet::new(),
            locals,
            visible_outer,
            captures: Vec::new(),
            capture_indices: HashMap::new(),
            default_capture_indices: Vec::new(),
            loops: Vec::new(),
            env,
            unbound,
            last_closure: None,
        };

        if mode == Mode::Function {
            for i in 0..num_defaults {
                let idx = this.add_capture(CaptureKind::Default(i));
                this.default_capture_indices.push(idx);
            }
        }
        this
    }

    fn finish(mut self) -> FunctionOutput {
        if !self.b.terminated() {
            match self.mode {
                Mode::Script => self.b.ret_void(),
                Mode::Loaded | Mode::Function => {
                    let none = self.b.tmp();
                    self.b.ins(format!("{none} = call %Value @sk_none()"));
                    self.b.ret("%Value", &none);
                }
            }
        }
        let text = format!("{}\n{}}}\n", self.header, self.b.text);
        FunctionOutput {
            text,
            captures: self.captures,
        }
    }

    fn add_capture(&mut self, kind: CaptureKind) -> usize {
        if let CaptureKind::Name(name) = &kind {
            if let Some(idx) = self.capture_indices.get(name) {
                return *idx;
            }
        }
        let idx = self.captures.len();
        if let CaptureKind::Name(name) = &kind {
            self.capture_indices.insert(name.clone(), idx);
        }
        self.captures.push(kind);
        idx
    }

    fn capture_cell(&mut self, idx: usize) -> String {
        let env = self
            .env
            .as_ref()
            .expect("capture cell requested outside a closure");
        let cellp = self.b.tmp();
        self.b
            .ins(format!("{cellp} = getelementptr ptr, ptr {env}, i64 {idx}"));
        let cell = self.b.tmp();
        self.b.ins(format!("{cell} = load ptr, ptr {cellp}"));
        cell
    }

    fn capture_for_name(&mut self, name: &str) -> String {
        let idx = if let Some(idx) = self.capture_indices.get(name) {
            *idx
        } else {
            self.add_capture(CaptureKind::Name(name.to_owned()))
        };
        self.capture_cell(idx)
    }

    fn local_ty(&self, name: &str) -> crate::types::Ty {
        match self.mode {
            Mode::Function => {
                let local = self.c.types.local(&self.module_path, &self.func_id, name);
                if local == crate::types::Ty::Unknown {
                    self.c.types.global(&self.module_path, name)
                } else {
                    local
                }
            }
            Mode::Script | Mode::Loaded => self.c.types.global(&self.module_path, name),
        }
    }

    fn resolve_place(&mut self, name: &str) -> Option<Place> {
        if let Some(local) = self.locals.get(name) {
            return Some(Place::Local(local.clone()));
        }
        match self.mode {
            Mode::Script => self.c.globals.get(name).cloned().map(Place::Global),
            Mode::Loaded => None,
            Mode::Function => {
                if self.visible_outer.contains(name) {
                    Some(Place::Local(Local::Cell(self.capture_for_name(name))))
                } else if self.use_script_globals {
                    self.c.globals.get(name).cloned().map(Place::Global)
                } else {
                    None
                }
            }
        }
    }

    fn make_local(&mut self, name: &str) -> Local {
        let ty = self.local_ty(name);
        if self.mode == Mode::Loaded {
            let cell = self.b.tmp();
            self.b.ins(format!(
                "{cell} = call ptr @sk_cell_new(%Value {})",
                self.unbound
            ));
            return Local::Cell(cell);
        }
        if self.mode == Mode::Function
            && ty == crate::types::Ty::Int
            && !self.captured.contains(name)
        {
            let slot = self.b.tmp();
            self.b.ins(format!("{slot} = alloca i64"));
            self.b.ins(format!("store i64 0, ptr {slot}"));
            Local::IntAlloca(slot)
        } else if self.mode == Mode::Function && self.captured.contains(name) {
            let cell = self.b.tmp();
            self.b.ins(format!(
                "{cell} = call ptr @sk_cell_new(%Value {})",
                self.unbound
            ));
            Local::Cell(cell)
        } else {
            let slot = self.b.tmp();
            self.b.ins(format!("{slot} = alloca %Value"));
            self.b
                .ins(format!("store %Value {}, ptr {slot}", self.unbound));
            Local::ValueAlloca(slot)
        }
    }

    fn ensure_local(&mut self, name: &str) -> Local {
        if let Some(local) = self.locals.get(name) {
            return local.clone();
        }
        let local = self.make_local(name);
        self.locals.insert(name.to_owned(), local.clone());
        local
    }

    fn ensure_int_local(&mut self, name: &str) -> Local {
        if let Some(local) = self.locals.get(name) {
            if matches!(local, Local::IntAlloca(_)) {
                return local.clone();
            }
        }
        let slot = self.b.tmp();
        self.b.ins(format!("{slot} = alloca i64"));
        self.b.ins(format!("store i64 0, ptr {slot}"));
        let local = Local::IntAlloca(slot);
        self.locals.insert(name.to_owned(), local.clone());
        local
    }

    fn fresh_local(&mut self, name: &str) -> Local {
        let local = self.make_local(name);
        self.locals.insert(name.to_owned(), local.clone());
        local
    }

    fn load_local_value(&mut self, local: &Local) -> String {
        match local {
            Local::Cell(ptr) | Local::ValueAlloca(ptr) => {
                let v = self.b.tmp();
                self.b.ins(format!("{v} = load %Value, ptr {ptr}"));
                v
            }
            Local::IntAlloca(ptr) => {
                let i = self.b.tmp();
                self.b.ins(format!("{i} = load i64, ptr {ptr}"));
                let v = self.b.tmp();
                self.b.ins(format!("{v} = call %Value @sk_int(i64 {i})"));
                v
            }
        }
    }

    fn load_local_int(&mut self, local: &Local) -> String {
        match local {
            Local::IntAlloca(ptr) => {
                let i = self.b.tmp();
                self.b.ins(format!("{i} = load i64, ptr {ptr}"));
                i
            }
            Local::Cell(ptr) | Local::ValueAlloca(ptr) => {
                let v = self.b.tmp();
                self.b.ins(format!("{v} = load %Value, ptr {ptr}"));
                let i = self.b.tmp();
                self.b.ins(format!("{i} = extractvalue %Value {v}, 1"));
                i
            }
        }
    }

    fn store_local(&mut self, local: &Local, value: &str) {
        match local {
            Local::Cell(ptr) | Local::ValueAlloca(ptr) => {
                self.b.ins(format!("store %Value {value}, ptr {ptr}"));
            }
            Local::IntAlloca(ptr) => {
                let i = self.b.tmp();
                self.b.ins(format!("{i} = extractvalue %Value {value}, 1"));
                self.b.ins(format!("store i64 {i}, ptr {ptr}"));
            }
        }
    }

    fn load_place(&mut self, name: &str, place: Place) -> String {
        let (value, is_int) = match place {
            Place::Local(local) => {
                let is_int = matches!(local, Local::IntAlloca(_));
                (self.load_local_value(&local), is_int)
            }
            Place::Global(sym) => {
                let v = self.b.tmp();
                self.b.ins(format!("{v} = load %Value, ptr {sym}"));
                (v, false)
            }
        };
        if is_int || self.known_bound.contains(name) {
            return value;
        }
        let (msg, len) = self.c.string_const(name);
        let checked = self.b.tmp();
        self.b.ins(format!(
            "{checked} = call %Value @sk_check_bound(%Value {value}, ptr {msg}, i64 {len})"
        ));
        checked
    }

    fn store_place(&mut self, place: Place, value: &str) {
        match place {
            Place::Local(local) => self.store_local(&local, value),
            Place::Global(sym) => self.b.ins(format!("store %Value {value}, ptr {sym}")),
        }
    }

    fn assign_name(&mut self, name: &str, value: &str) -> Result<()> {
        let place = if let Some(local) = self.locals.get(name) {
            Place::Local(local.clone())
        } else {
            match self.mode {
                Mode::Script => Place::Global(self.c.ensure_global(name)),
                Mode::Loaded | Mode::Function => Place::Local(self.ensure_local(name)),
            }
        };
        self.store_place(place, value);
        self.known_bound.insert(name.to_owned());
        self.list_views.remove(name);
        self.empty_int_lists.remove(name);
        if self.local_ty(name) == crate::types::Ty::ListInt {
            self.cache_list_view(name, value);
        }
        if let Some(len) = self.value_list_lens.get(value).copied() {
            self.fixed_list_lens.insert(name.to_owned(), len);
        } else {
            self.fixed_list_lens.remove(name);
        }
        if let Ok(v) = value.parse::<i64>() {
            self.const_locals.insert(name.to_owned(), v);
        } else {
            self.const_locals.remove(name);
        }
        Ok(())
    }

    fn resolve_value(&mut self, name: &str) -> Result<String> {
        match name {
            "True" => return self.compile_bool(true),
            "False" => return self.compile_bool(false),
            "None" => return self.compile_none(),
            _ => {}
        }
        if let Some(place) = self.resolve_place(name) {
            return Ok(self.load_place(name, place));
        }
        if let Some(id) = builtin_id(name) {
            let v = self.b.tmp();
            self.b
                .ins(format!("{v} = call %Value @sk_builtin(i64 {id})"));
            return Ok(v);
        }
        let (sym, len) = self.c.string_const(name);
        let v = self.b.tmp();
        self.b.ins(format!(
            "{v} = call %Value @sk_undefined(ptr {sym}, i64 {len})"
        ));
        Ok(v)
    }

    fn compile_none(&mut self) -> Result<String> {
        // `None` is the zero value of the two-word `%Value` struct.  Emitting
        // it as a constant (rather than a call to `sk_none`) lets LLVM delete
        // the value entirely when, for example, the result of `list.append(x)`
        // is discarded.
        let v = self.b.tmp();
        self.b.ins(format!(
            "{v} = insertvalue %Value zeroinitializer, i64 0, 0"
        ));
        Ok(v)
    }

    fn compile_bool(&mut self, value: bool) -> Result<String> {
        let v = self.b.tmp();
        self.b.ins(format!(
            "{v} = call %Value @sk_bool(i64 {})",
            if value { 1 } else { 0 }
        ));
        Ok(v)
    }

    fn compile_int(&mut self, value: i64) -> Result<String> {
        let v = self.b.tmp();
        self.b
            .ins(format!("{v} = call %Value @sk_int(i64 {value})"));
        Ok(v)
    }

    fn compile_float(&mut self, value: f64) -> Result<String> {
        let v = self.b.tmp();
        self.b.ins(format!(
            "{v} = call %Value @sk_float(double 0x{:016X})",
            value.to_bits()
        ));
        Ok(v)
    }

    fn compile_string_value(&mut self, s: &str) -> Result<String> {
        let (sym, len) = self.c.string_const(s);
        let v = self.b.tmp();
        self.b.ins(format!(
            "{v} = call %Value @sk_string(ptr {sym}, i64 {len})"
        ));
        Ok(v)
    }

    fn compile_truthy(&mut self, value: &str) -> String {
        let t = self.b.tmp();
        self.b
            .ins(format!("{t} = call i64 @sk_truthy(%Value {value})"));
        let c = self.b.tmp();
        self.b.ins(format!("{c} = icmp ne i64 {t}, 0"));
        c
    }

    fn compile_condition(&mut self, expr: &AstExpr) -> Result<String> {
        if let Expr::Op(lhs, op, rhs) = &expr.node {
            if matches!(op, BinOp::Equal | BinOp::NotEqual)
                && ((self.expr_is_none(lhs) && self.expr_is_int(rhs))
                    || (self.expr_is_none(rhs) && self.expr_is_int(lhs)))
            {
                let c = self.b.tmp();
                let truth = *op == BinOp::NotEqual;
                self.b.ins(format!(
                    "{c} = icmp eq i64 0, {}",
                    if truth { 0 } else { 1 }
                ));
                return Ok(c);
            }
        }
        if self.expr_is_none(expr) {
            let c = self.b.tmp();
            self.b.ins(format!("{c} = icmp eq i64 0, 1"));
            return Ok(c);
        }
        let v = self.compile_expr(expr)?;
        Ok(self.compile_truthy(&v))
    }

    fn emit_error(&mut self, msg: &str) {
        let (sym, len) = self.c.string_const(msg);
        self.b
            .ins(format!("call void @sk_error_cstr(ptr {sym}, i64 {len})"));
        self.b.unreachable();
    }

    fn compile_stmt(&mut self, stmt: &AstStmt) -> Result<()> {
        match &stmt.node {
            Stmt::Break => {
                let labels = self
                    .loops
                    .last()
                    .ok_or_else(|| anyhow!("break outside a loop"))?;
                let label = labels.break_label.clone();
                self.b.br(&label);
            }
            Stmt::Continue => {
                let labels = self
                    .loops
                    .last()
                    .ok_or_else(|| anyhow!("continue outside a loop"))?;
                let label = labels.continue_label.clone();
                self.b.br(&label);
            }
            Stmt::Pass => {}
            Stmt::Return(value) => {
                if self.mode != Mode::Function {
                    bail!("return outside a function");
                }
                let v = match value {
                    Some(e) => self.compile_expr(e)?,
                    None => self.compile_none()?,
                };
                self.b.ret("%Value", &v);
            }
            Stmt::Expression(e) => {
                self.compile_expr(e)?;
            }
            Stmt::Assign(a) => {
                // `w[t] = <int expr>` where `w` is a compact int list: store
                // the unboxed i64 directly.  Going through `%Value` here would
                // emit a `sk_int` call and an immediate `extractvalue` for
                // every element, which is very visible in the fully unrolled
                // SHA-2 message schedules.
                if let AssignTarget::Index(pair) = &a.lhs.node {
                    if self.expr_is_list_int(&pair.0) && self.expr_is_int(&a.rhs) {
                        if let Some(v) = self.compile_int_expr(&a.rhs)? {
                            if let Expr::Identifier(obj_id) = &pair.0.node {
                                if let Some(len) =
                                    self.fixed_list_lens.get(&obj_id.node.ident).copied()
                                {
                                    if let Some(idx) = self.const_int_expr(&pair.1) {
                                        let idx = if idx < 0 { idx.wrapping_add(len) } else { idx };
                                        if (0..len).contains(&idx) {
                                            let obj = self.compile_expr(&pair.0)?;
                                            self.compile_int_list_set_unchecked(
                                                &obj,
                                                &idx.to_string(),
                                                &v,
                                            );
                                            return Ok(());
                                        }
                                    }
                                }
                            }
                            let obj = self.compile_expr(&pair.0)?;
                            let idx = self.compile_int_operand(&pair.1)?;
                            self.compile_int_list_set(&obj, &idx, &v);
                            return Ok(());
                        }
                    }
                }
                if let AssignTarget::Identifier(id) = &a.lhs.node {
                    if self.non_escaping_lists.contains(&id.node.ident) {
                        if let Some(value) = self.try_compile_words_padded(&a.rhs, true)? {
                            self.assign_target(&a.lhs, &value)?;
                            return Ok(());
                        }
                    }
                }
                // Keep integer assignments unboxed end-to-end.  Without this,
                // `byte = ord(char)` boxes the result and immediately unboxes
                // it again in store_local.
                if let AssignTarget::Identifier(id) = &a.lhs.node {
                    if self.local_ty(&id.node.ident) == crate::types::Ty::Int {
                        if let Some(v) = self.compile_int_expr(&a.rhs)? {
                            self.assign_int_name(&id.node.ident, &v)?;
                            return Ok(());
                        }
                    }
                }
                // `b = [0] * 25` for a non-escaping local: use a stack
                // buffer and a memset instead of a heap allocation.
                if let AssignTarget::Identifier(id) = &a.lhs.node {
                    if self.non_escaping_lists.contains(&id.node.ident) {
                        if let Expr::Op(l, BinOp::Multiply, r) = &a.rhs.node {
                            if matches!(&l.node, Expr::List(items)
                                if items.len() == 1 && const_int_ast(&items[0]) == Some(0))
                            {
                                if let Some(n) = self.const_int_expr(r) {
                                    if (0..=4096).contains(&n) {
                                        let (v, items) = self.emit_stack_int_list(n);
                                        self.b.ins(format!(
                                            "call void @llvm.memset.p0.i64(ptr {items}, i8 0, i64 {}, i1 false)",
                                            n * 8
                                        ));
                                        self.value_list_lens.insert(v.clone(), n);
                                        self.assign_target(&a.lhs, &v)?;
                                        return Ok(());
                                    }
                                }
                            }
                        }
                    }
                }
                // An empty list literal has no intrinsic element type.  If the
                // assignment target was inferred as ListInt (for example
                // `result = []` followed by `result.append(byte)`), emit the
                // compact unboxed representation directly.
                let value = if let AssignTarget::Identifier(id) = &a.lhs.node {
                    if self.local_ty(&id.node.ident) == crate::types::Ty::ListInt {
                        if let Expr::List(items) = &a.rhs.node {
                            self.compile_sequence(items, true, true)?
                        } else {
                            if self.non_escaping_lists.contains(&id.node.ident) {
                                if let Expr::ListComprehension(body, first, rest) = &a.rhs.node {
                                    if let Some(n) = self.comprehension_const_len(first, rest) {
                                        let _ = body;
                                        self.stack_list_hint = Some(n);
                                    }
                                }
                            }
                            let v = self.compile_expr(&a.rhs)?;
                            self.stack_list_hint = None;
                            v
                        }
                    } else {
                        self.compile_expr(&a.rhs)?
                    }
                } else {
                    self.compile_expr(&a.rhs)?
                };
                self.assign_target(&a.lhs, &value)?;
                if let AssignTarget::Identifier(id) = &a.lhs.node {
                    if self.local_ty(&id.node.ident) == crate::types::Ty::ListInt
                        && matches!(&a.rhs.node, Expr::List(items) if items.is_empty())
                    {
                        self.empty_int_lists.insert(id.node.ident.clone());
                    }
                }
            }
            Stmt::AssignModify(target, op, rhs) => {
                let left = self.read_target(target)?;
                let right = self.compile_expr(rhs)?;
                let value = self.compile_binop_value(*op, &left, &right)?;
                self.assign_target(target, &value)?;
            }
            Stmt::Statements(stmts) => {
                for s in stmts {
                    if self.b.terminated() {
                        break;
                    }
                    self.compile_stmt(s)?;
                }
            }
            Stmt::If(cond, body) => {
                let truth = self.compile_condition(cond)?;
                let then_label = self.b.label();
                let end_label = self.b.label();
                self.b.cond_br(&truth, &then_label, &end_label);
                self.b.emit_label(&then_label);
                self.compile_stmt(body)?;
                if !self.b.terminated() {
                    self.b.br(&end_label);
                }
                self.b.emit_label(&end_label);
            }
            Stmt::IfElse(cond, branches) => {
                let truth = self.compile_condition(cond)?;
                let then_label = self.b.label();
                let else_label = self.b.label();
                let end_label = self.b.label();
                self.b.cond_br(&truth, &then_label, &else_label);
                self.b.emit_label(&then_label);
                self.compile_stmt(&branches.0)?;
                if !self.b.terminated() {
                    self.b.br(&end_label);
                }
                self.b.emit_label(&else_label);
                self.compile_stmt(&branches.1)?;
                if !self.b.terminated() {
                    self.b.br(&end_label);
                }
                self.b.emit_label(&end_label);
            }
            Stmt::For(f) => self.compile_for(f)?,
            Stmt::Def(d) => self.compile_def(d)?,
            Stmt::Load(l) => self.compile_load(l)?,
        }
        Ok(())
    }

    fn assign_int_name(&mut self, name: &str, value: &str) -> Result<()> {
        let local = if let Some(local) = self.locals.get(name) {
            local.clone()
        } else {
            self.ensure_local(name)
        };
        match local {
            Local::IntAlloca(ptr) => {
                self.b.ins(format!("store i64 {value}, ptr {ptr}"));
            }
            _ => {
                let boxed = self.box_int(value);
                self.store_local(&local, &boxed);
            }
        }
        self.known_bound.insert(name.to_owned());
        self.fixed_list_lens.remove(name);
        if let Ok(v) = value.parse::<i64>() {
            self.const_locals.insert(name.to_owned(), v);
        } else {
            self.const_locals.remove(name);
        }
        Ok(())
    }

    fn assign_int_target(&mut self, target: &AstAssignTarget, value: &str) -> Result<()> {
        match &target.node {
            AssignTarget::Identifier(id) => self.assign_int_name(&id.node.ident, value),
            _ => {
                let boxed = self.box_int(value);
                self.assign_target(target, &boxed)
            }
        }
    }

    fn emit_int_loop(
        &mut self,
        f: &starlark_syntax::syntax::ast::ForP<starlark_syntax::syntax::ast::AstNoPayload>,
        start: &str,
        stop: &str,
        step: &str,
    ) -> Result<()> {
        let mut assigned = HashSet::new();
        collect_assigned_names(&f.body, &mut assigned);
        for name in &assigned {
            self.const_locals.remove(name);
        }
        let i_cell = self.b.tmp();
        self.b.ins(format!("{i_cell} = alloca i64"));
        self.b.ins(format!("store i64 {start}, ptr {i_cell}"));

        let loop_label = self.b.label();
        let body_label = self.b.label();
        let inc_label = self.b.label();
        let end_label = self.b.label();
        self.b.br(&loop_label);

        self.b.emit_label(&loop_label);
        let i = self.b.tmp();
        self.b.ins(format!("{i} = load i64, ptr {i_cell}"));
        let step_pos = self.b.tmp();
        self.b.ins(format!("{step_pos} = icmp sgt i64 {step}, 0"));
        let cond_pos = self.b.tmp();
        self.b.ins(format!("{cond_pos} = icmp slt i64 {i}, {stop}"));
        let cond_neg = self.b.tmp();
        self.b.ins(format!("{cond_neg} = icmp sgt i64 {i}, {stop}"));
        let cond = self.b.tmp();
        self.b.ins(format!(
            "{cond} = select i1 {step_pos}, i1 {cond_pos}, i1 {cond_neg}"
        ));
        self.b.cond_br(&cond, &body_label, &end_label);

        self.b.emit_label(&body_label);
        self.assign_int_target(&f.var, &i)?;
        self.loops.push(LoopLabels {
            break_label: end_label.clone(),
            continue_label: inc_label.clone(),
        });
        self.compile_stmt(&f.body)?;
        self.loops.pop();
        if !self.b.terminated() {
            self.b.br(&inc_label);
        }

        self.b.emit_label(&inc_label);
        let next = self.b.tmp();
        self.b.ins(format!("{next} = add i64 {i}, {step}"));
        self.b.ins(format!("store i64 {next}, ptr {i_cell}"));
        self.b.br(&loop_label);

        self.b.emit_label(&end_label);
        for name in &assigned {
            self.const_locals.remove(name);
        }
        Ok(())
    }

    fn emit_int_list_loop(
        &mut self,
        f: &starlark_syntax::syntax::ast::ForP<starlark_syntax::syntax::ast::AstNoPayload>,
        list: &str,
        len: &str,
        items: &str,
    ) -> Result<()> {
        let mut assigned = HashSet::new();
        collect_assigned_names(&f.body, &mut assigned);
        for name in &assigned {
            self.const_locals.remove(name);
        }
        let i_cell = self.b.tmp();
        self.b.ins(format!("{i_cell} = alloca i64"));
        self.b.ins(format!("store i64 0, ptr {i_cell}"));

        let loop_label = self.b.label();
        let body_label = self.b.label();
        let inc_label = self.b.label();
        let end_label = self.b.label();
        self.b.br(&loop_label);

        self.b.emit_label(&loop_label);
        let i = self.b.tmp();
        self.b.ins(format!("{i} = load i64, ptr {i_cell}"));
        let cond = self.b.tmp();
        self.b.ins(format!("{cond} = icmp slt i64 {i}, {len}"));
        self.b.cond_br(&cond, &body_label, &end_label);

        self.b.emit_label(&body_label);
        let elemp = self.b.tmp();
        self.b
            .ins(format!("{elemp} = getelementptr i64, ptr {items}, i64 {i}"));
        let val = self.b.tmp();
        self.b.ins(format!("{val} = load i64, ptr {elemp}"));
        let _ = list;
        self.assign_int_target(&f.var, &val)?;
        self.loops.push(LoopLabels {
            break_label: end_label.clone(),
            continue_label: inc_label.clone(),
        });
        self.compile_stmt(&f.body)?;
        self.loops.pop();
        if !self.b.terminated() {
            self.b.br(&inc_label);
        }

        self.b.emit_label(&inc_label);
        let next = self.b.tmp();
        self.b.ins(format!("{next} = add i64 {i}, 1"));
        self.b.ins(format!("store i64 {next}, ptr {i_cell}"));
        self.b.br(&loop_label);

        self.b.emit_label(&end_label);
        for name in &assigned {
            self.const_locals.remove(name);
        }
        Ok(())
    }

    fn try_compile_string_elems_for(
        &mut self,
        f: &starlark_syntax::syntax::ast::ForP<starlark_syntax::syntax::ast::AstNoPayload>,
    ) -> Result<bool> {
        let Expr::Call(callee, args) = &f.over.node else {
            return Ok(false);
        };
        let Expr::Dot(obj, name) = &callee.node else {
            return Ok(false);
        };
        if name.node.as_str() != "elems" || !args.args.is_empty() {
            return Ok(false);
        }

        let string = self.compile_expr(obj)?;
        let payload = self.b.tmp();
        self.b
            .ins(format!("{payload} = extractvalue %Value {string}, 1"));
        let strp = self.b.tmp();
        self.b
            .ins(format!("{strp} = inttoptr i64 {payload} to ptr"));
        let lenp = self.b.tmp();
        self.b.ins(format!(
            "{lenp} = getelementptr %SkString, ptr {strp}, i32 0, i32 0"
        ));
        let len = self.b.tmp();
        self.b.ins(format!("{len} = load i64, ptr {lenp}"));
        let datap = self.b.tmp();
        self.b.ins(format!(
            "{datap} = getelementptr %SkString, ptr {strp}, i32 0, i32 1, i64 0"
        ));

        // If the loop appends exactly once per iteration to a list that was
        // empty on entry, the append index is the loop counter.  Store through
        // that index and fix up the length once after the loop; this leaves a
        // simple `out[i] = zext(in[i])` loop that LLVM can vectorize.
        let direct_target = find_direct_append_target(&f.body).filter(|target| {
            self.local_ty(target) == crate::types::Ty::ListInt
                && self.empty_int_lists.contains(target)
                && count_ident_stmt(&f.body, target) == 1
                && !stmt_has_break_or_continue(&f.body)
        });
        let mut direct_local = None;
        if let Some(target) = &direct_target {
            if let Some(local) = self.locals.get(target).cloned() {
                let list = self.load_local_value(&local);
                self.b.ins(format!(
                    "call void @sk_int_list_reserve(%Value {list}, i64 {len})"
                ));
                self.empty_int_lists.remove(target);
                self.exact_capacity_lists.insert(target.clone());
                direct_local = Some(local);
            }
        }
        let direct_items = direct_local.as_ref().map(|local| {
            let list = self.load_local_value(local);
            let payload = self.b.tmp();
            self.b
                .ins(format!("{payload} = extractvalue %Value {list}, 1"));
            let lp = self.b.tmp();
            self.b.ins(format!("{lp} = inttoptr i64 {payload} to ptr"));
            let itemsp = self.b.tmp();
            self.b.ins(format!(
                "{itemsp} = getelementptr %SkList, ptr {lp}, i32 0, i32 2"
            ));
            let items = self.b.tmp();
            self.b.ins(format!("{items} = load ptr, ptr {itemsp}"));
            items
        });

        let i_cell = self.b.tmp();
        self.b.ins(format!("{i_cell} = alloca i64"));
        self.b.ins(format!("store i64 0, ptr {i_cell}"));
        let loop_label = self.b.label();
        let body_label = self.b.label();
        let inc_label = self.b.label();
        let end_label = self.b.label();
        self.b.br(&loop_label);

        self.b.emit_label(&loop_label);
        let i = self.b.tmp();
        self.b.ins(format!("{i} = load i64, ptr {i_cell}"));
        let cond = self.b.tmp();
        self.b.ins(format!("{cond} = icmp slt i64 {i}, {len}"));
        self.b.cond_br(&cond, &body_label, &end_label);

        self.b.emit_label(&body_label);
        let bytep = self.b.tmp();
        self.b
            .ins(format!("{bytep} = getelementptr i8, ptr {datap}, i64 {i}"));
        let byte8 = self.b.tmp();
        self.b.ins(format!("{byte8} = load i8, ptr {bytep}"));
        let byte = self.b.tmp();
        self.b.ins(format!("{byte} = zext i8 {byte8} to i64"));
        let var_name = match &f.var.node {
            AssignTarget::Identifier(id) => id.node.ident.clone(),
            _ => {
                let boxed = self.box_int(&byte);
                self.assign_target(&f.var, &boxed)?;
                String::new()
            }
        };
        if !var_name.is_empty() {
            self.ensure_int_local(&var_name);
            self.assign_int_name(&var_name, &byte)?;
            self.char_locals.insert(var_name);
        }
        self.loops.push(LoopLabels {
            break_label: end_label.clone(),
            continue_label: inc_label.clone(),
        });
        if let Some(target) = &direct_target {
            self.append_indices.insert(target.clone(), i.clone());
            if let Some(items) = &direct_items {
                self.append_items.insert(target.clone(), items.clone());
            }
        }
        if let Some(target) = find_append_target(&f.body) {
            self.exact_capacity_lists.insert(target);
        }
        self.compile_stmt(&f.body)?;
        if let Some(target) = find_append_target(&f.body) {
            self.exact_capacity_lists.remove(&target);
        }
        if let Some(target) = &direct_target {
            self.append_indices.remove(target);
            self.append_items.remove(target);
        }
        self.loops.pop();
        if !self.b.terminated() {
            self.b.br(&inc_label);
        }

        self.b.emit_label(&inc_label);
        let next = self.b.tmp();
        self.b.ins(format!("{next} = add i64 {i}, 1"));
        self.b.ins(format!("store i64 {next}, ptr {i_cell}"));
        self.b.br(&loop_label);
        self.b.emit_label(&end_label);
        if let Some(local) = &direct_local {
            let list = self.load_local_value(local);
            let payload = self.b.tmp();
            self.b
                .ins(format!("{payload} = extractvalue %Value {list}, 1"));
            let lp = self.b.tmp();
            self.b.ins(format!("{lp} = inttoptr i64 {payload} to ptr"));
            let lenp = self.b.tmp();
            self.b.ins(format!(
                "{lenp} = getelementptr %SkList, ptr {lp}, i32 0, i32 0"
            ));
            self.b.ins(format!("store i64 {len}, ptr {lenp}"));
        }
        Ok(true)
    }

    fn try_compile_int_for(
        &mut self,
        f: &starlark_syntax::syntax::ast::ForP<starlark_syntax::syntax::ast::AstNoPayload>,
    ) -> Result<bool> {
        if let Expr::Identifier(id) = &f.over.node {
            if !self.locals.contains_key(&id.node.ident) {
                if let Some(values) = self
                    .c
                    .const_lists
                    .get(&(self.module_path.clone(), id.node.ident.clone()))
                    .cloned()
                {
                    if values.len() as i64 <= self.unroll_limit() {
                        self.emit_unrolled_values_loop(f, &values)?;
                        return Ok(true);
                    }
                }
            }
        }
        if let Expr::List(items) = &f.over.node {
            let values: Option<Vec<i64>> = items.iter().map(const_int_ast).collect();
            if let Some(values) = values {
                if values.len() as i64 <= self.unroll_limit() {
                    self.emit_unrolled_values_loop(f, &values)?;
                    return Ok(true);
                }
            }
        }
        if let Expr::Call(callee, args) = &f.over.node {
            if let Expr::Identifier(id) = &callee.node {
                if id.node.ident == "range"
                    && args
                        .args
                        .iter()
                        .all(|a| matches!(a.node, ArgumentP::Positional(_)))
                {
                    let consts: Vec<Option<i64>> = args
                        .args
                        .iter()
                        .map(|a| self.const_int_expr(a.expr()))
                        .collect();
                    let (start_c, stop_c, step_c) = match consts.as_slice() {
                        [Some(stop)] => (Some(0), Some(*stop), Some(1)),
                        [Some(start), Some(stop)] => (Some(*start), Some(*stop), Some(1)),
                        [Some(start), Some(stop), Some(step)] => {
                            (Some(*start), Some(*stop), Some(*step))
                        }
                        _ => (None, None, None),
                    };
                    if let (Some(start_c), Some(stop_c), Some(step_c)) = (start_c, stop_c, step_c) {
                        if let Some(count) = Self::range_trip_count(start_c, stop_c, step_c) {
                            if count <= self.unroll_limit() {
                                self.emit_unrolled_int_loop(f, start_c, step_c, count)?;
                                return Ok(true);
                            }
                        }
                    }
                    let mut vals = Vec::new();
                    for a in &args.args {
                        vals.push(self.compile_int_operand(a.expr())?);
                    }
                    let (start, stop, step) = match vals.len() {
                        1 => ("0".to_owned(), vals[0].clone(), "1".to_owned()),
                        2 => (vals[0].clone(), vals[1].clone(), "1".to_owned()),
                        3 => (vals[0].clone(), vals[1].clone(), vals[2].clone()),
                        _ => return Ok(false),
                    };
                    self.emit_int_loop(f, &start, &stop, &step)?;
                    return Ok(true);
                }
            }
        }
        if self.expr_is_list_int(&f.over) {
            let obj = self.compile_expr(&f.over)?;
            let payload = self.b.tmp();
            self.b
                .ins(format!("{payload} = extractvalue %Value {obj}, 1"));
            let list = self.b.tmp();
            self.b
                .ins(format!("{list} = inttoptr i64 {payload} to ptr"));
            let lenp = self.b.tmp();
            self.b.ins(format!(
                "{lenp} = getelementptr %SkList, ptr {list}, i32 0, i32 0"
            ));
            let len = self.b.tmp();
            self.b.ins(format!("{len} = load i64, ptr {lenp}"));
            let itemsp = self.b.tmp();
            self.b.ins(format!(
                "{itemsp} = getelementptr %SkList, ptr {list}, i32 0, i32 2"
            ));
            let items = self.b.tmp();
            self.b.ins(format!("{items} = load ptr, ptr {itemsp}"));
            self.emit_int_list_loop(f, &obj, &len, &items)?;
            return Ok(true);
        }
        Ok(false)
    }

    fn compile_for(
        &mut self,
        f: &starlark_syntax::syntax::ast::ForP<starlark_syntax::syntax::ast::AstNoPayload>,
    ) -> Result<()> {
        if self.try_compile_string_elems_for(f)? {
            return Ok(());
        }
        if self.try_compile_int_for(f)? {
            return Ok(());
        }
        let iterable = self.compile_expr(&f.over)?;
        let mut assigned = HashSet::new();
        collect_assigned_names(&f.body, &mut assigned);
        for name in &assigned {
            self.const_locals.remove(name);
        }
        let len = self.b.tmp();
        self.b
            .ins(format!("{len} = call i64 @sk_iter_len(%Value {iterable})"));
        let index_cell = self.b.tmp();
        self.b.ins(format!("{index_cell} = alloca i64"));
        self.b.ins(format!("store i64 0, ptr {index_cell}"));

        let loop_label = self.b.label();
        let body_label = self.b.label();
        let inc_label = self.b.label();
        let end_label = self.b.label();
        self.b.br(&loop_label);

        self.b.emit_label(&loop_label);
        let index = self.b.tmp();
        self.b.ins(format!("{index} = load i64, ptr {index_cell}"));
        let cond = self.b.tmp();
        self.b.ins(format!("{cond} = icmp slt i64 {index}, {len}"));
        self.b.cond_br(&cond, &body_label, &end_label);

        self.b.emit_label(&body_label);
        let item = self.b.tmp();
        self.b.ins(format!(
            "{item} = call %Value @sk_iter_at(%Value {iterable}, i64 {index})"
        ));
        self.assign_target(&f.var, &item)?;
        self.loops.push(LoopLabels {
            break_label: end_label.clone(),
            continue_label: inc_label.clone(),
        });
        self.compile_stmt(&f.body)?;
        self.loops.pop();
        if !self.b.terminated() {
            self.b.br(&inc_label);
        }

        self.b.emit_label(&inc_label);
        let next = self.b.tmp();
        self.b.ins(format!("{next} = add i64 {index}, 1"));
        self.b.ins(format!("store i64 {next}, ptr {index_cell}"));
        self.b.br(&loop_label);

        self.b.emit_label(&end_label);
        for name in &assigned {
            self.const_locals.remove(name);
        }
        Ok(())
    }

    fn compile_def(&mut self, d: &DefP<starlark_syntax::syntax::ast::AstNoPayload>) -> Result<()> {
        let mut default_cells = Vec::new();
        for p in &d.params {
            if let ParameterP::Normal(_, _, Some(default)) = &p.node {
                let value = self.compile_expr(default)?;
                let cell = self.b.tmp();
                self.b
                    .ins(format!("{cell} = call ptr @sk_cell_new(%Value {value})"));
                default_cells.push(cell);
            }
        }

        let mut local_names = Vec::new();
        for p in &d.params {
            if let Some(id) = p.node.ident() {
                local_names.push(id.node.ident.clone());
            }
        }
        collect_local_names(&d.body, &mut local_names);
        dedupe(&mut local_names);

        let mut visible_outer: HashSet<String> = self.locals.keys().cloned().collect();
        visible_outer.extend(self.visible_outer.iter().cloned());

        let fn_name = if self.mode == Mode::Function {
            self.c.next_function_name()
        } else {
            self.c
                .function_name_for(&self.module_path, &d.name.node.ident)
        };
        if std::env::var_os("STARLARKC_DUMP_FUNCS").is_some() {
            eprintln!("{} => {}", fn_name, d.name.node.ident);
        }
        let mut child = FunctionCompiler::new(
            self.c,
            fn_name.clone(),
            Mode::Function,
            self.use_script_globals,
            self.module_path.clone(),
            d.name.node.ident.clone(),
            HashSet::new(),
            local_names,
            visible_outer,
            default_cells.len(),
        );
        child.non_escaping_lists = collect_non_escaping_lists(&d.body);
        child.compile_params(&d.params)?;
        if matches!(&d.body.node, Stmt::Return(Some(e)) if matches!(e.node, Expr::ListComprehension(..)))
        {
            child.header = child.header.replacen(" {", " alwaysinline {", 1);
        }
        child.compile_stmt(&d.body)?;
        let out = child.finish();
        self.c.functions.push(out.text);
        self.emit_closure(&fn_name, &out.captures, &default_cells)?;
        let closure = self.last_closure.take().expect("closure value");
        self.assign_name(&d.name.node.ident, &closure)?;
        Ok(())
    }

    fn compile_lambda(
        &mut self,
        l: &LambdaP<starlark_syntax::syntax::ast::AstNoPayload>,
    ) -> Result<String> {
        let mut default_cells = Vec::new();
        for p in &l.params {
            if let ParameterP::Normal(_, _, Some(default)) = &p.node {
                let value = self.compile_expr(default)?;
                let cell = self.b.tmp();
                self.b
                    .ins(format!("{cell} = call ptr @sk_cell_new(%Value {value})"));
                default_cells.push(cell);
            }
        }

        let mut local_names = Vec::new();
        for p in &l.params {
            if let Some(id) = p.node.ident() {
                local_names.push(id.node.ident.clone());
            }
        }
        dedupe(&mut local_names);

        let mut visible_outer: HashSet<String> = self.locals.keys().cloned().collect();
        visible_outer.extend(self.visible_outer.iter().cloned());

        let fn_name = self.c.next_function_name();
        let mut child = FunctionCompiler::new(
            self.c,
            fn_name.clone(),
            Mode::Function,
            self.use_script_globals,
            self.module_path.clone(),
            "<lambda>".to_owned(),
            HashSet::new(),
            local_names,
            visible_outer,
            default_cells.len(),
        );
        child.compile_params(&l.params)?;
        let body = child.compile_expr(&l.body)?;
        child.b.ret("%Value", &body);
        let out = child.finish();
        self.c.functions.push(out.text);
        self.emit_closure(&fn_name, &out.captures, &default_cells)?;
        Ok(self.last_closure.take().expect("closure value"))
    }

    fn emit_closure(
        &mut self,
        fn_name: &str,
        captures: &[CaptureKind],
        default_cells: &[String],
    ) -> Result<()> {
        let env = self.b.tmp();
        self.b.ins(format!(
            "{env} = call ptr @sk_env_new(i64 {})",
            captures.len()
        ));
        for (i, cap) in captures.iter().enumerate() {
            let cell = match cap {
                CaptureKind::Name(name) => self.resolve_capture_cell(name)?,
                CaptureKind::Default(idx) => default_cells
                    .get(*idx)
                    .cloned()
                    .ok_or_else(|| anyhow!("bad default capture index"))?,
            };
            self.b.ins(format!(
                "call void @sk_env_set(ptr {env}, i64 {i}, ptr {cell})"
            ));
        }
        let closure = self.b.tmp();
        self.b.ins(format!(
            "{closure} = call %Value @sk_function_new(ptr @{fn_name}, i64 {}, ptr {env})",
            captures.len()
        ));
        self.last_closure = Some(closure);
        Ok(())
    }

    /// Convert a local into a heap cell on demand.  Type inference may have
    /// chosen an unboxed stack slot before we discover that a nested function
    /// closes over it; moving the current value into a cell preserves the
    /// unboxed fast path for every local that is not actually captured.
    fn ensure_cell_local(&mut self, name: &str) -> Option<String> {
        let local = self.locals.get(name)?.clone();
        match local {
            Local::Cell(cell) => Some(cell),
            Local::ValueAlloca(ptr) => {
                let value = self.b.tmp();
                self.b.ins(format!("{value} = load %Value, ptr {ptr}"));
                let cell = self.b.tmp();
                self.b
                    .ins(format!("{cell} = call ptr @sk_cell_new(%Value {value})"));
                self.locals
                    .insert(name.to_owned(), Local::Cell(cell.clone()));
                Some(cell)
            }
            Local::IntAlloca(ptr) => {
                let int = self.b.tmp();
                self.b.ins(format!("{int} = load i64, ptr {ptr}"));
                let value = self.b.tmp();
                self.b
                    .ins(format!("{value} = call %Value @sk_int(i64 {int})"));
                let cell = self.b.tmp();
                self.b
                    .ins(format!("{cell} = call ptr @sk_cell_new(%Value {value})"));
                self.locals
                    .insert(name.to_owned(), Local::Cell(cell.clone()));
                Some(cell)
            }
        }
    }

    fn resolve_capture_cell(&mut self, name: &str) -> Result<String> {
        if let Some(cell) = self.ensure_cell_local(name) {
            return Ok(cell);
        }
        if self.visible_outer.contains(name) {
            return Ok(self.capture_for_name(name));
        }
        bail!("cannot capture `{name}` from an enclosing scope")
    }

    fn compile_load(
        &mut self,
        l: &starlark_syntax::syntax::ast::LoadP<starlark_syntax::syntax::ast::AstNoPayload>,
    ) -> Result<()> {
        let lm = self.c.load_module(&self.module_path, &l.module.node)?;
        let before = self.b.label();
        self.b.br(&before);
        self.b.emit_label(&before);
        let cached = self.b.tmp();
        self.b
            .ins(format!("{cached} = load %Value, ptr {}", lm.cache_global));
        let tagp = self.b.tmp();
        self.b.ins(format!(
            "{tagp} = getelementptr %Value, ptr {}, i32 0, i32 0",
            lm.cache_global
        ));
        let tag = self.b.tmp();
        self.b.ins(format!("{tag} = load i64, ptr {tagp}"));
        let is_unbound = self.b.tmp();
        self.b.ins(format!("{is_unbound} = icmp eq i64 {tag}, 11"));
        let do_load = self.b.label();
        let ready = self.b.label();
        self.b.cond_br(&is_unbound, &do_load, &ready);

        self.b.emit_label(&do_load);
        let fresh = self.b.tmp();
        self.b
            .ins(format!("{fresh} = call %Value @{}()", lm.init_fn));
        self.b
            .ins(format!("store %Value {fresh}, ptr {}", lm.cache_global));
        self.b.br(&ready);

        self.b.emit_label(&ready);
        let dict = self.b.phi("%Value", &[(cached, before), (fresh, do_load)]);
        for arg in &l.args {
            let key = self.compile_string_value(&arg.their.node)?;
            let value = self.b.tmp();
            self.b.ins(format!(
                "{value} = call %Value @sk_index(%Value {dict}, %Value {key})"
            ));
            self.assign_name(&arg.local.node.ident, &value)?;
        }
        Ok(())
    }

    fn compile_params(
        &mut self,
        params: &[starlark_syntax::syntax::ast::AstParameter],
    ) -> Result<()> {
        let slash_index = params
            .iter()
            .position(|p| matches!(p.node, ParameterP::Slash));

        #[derive(Clone)]
        struct Spec {
            name: String,
            local: Local,
            positional: bool,
            named: bool,
            default_capture: Option<usize>,
            is_args: bool,
            is_kwargs: bool,
        }

        let mut specs = Vec::new();
        let mut seen_star = false;
        let mut default_idx = 0usize;
        for (i, p) in params.iter().enumerate() {
            match &p.node {
                ParameterP::Slash => {}
                ParameterP::NoArgs => seen_star = true,
                ParameterP::Normal(name, _, default) => {
                    let positional = !seen_star;
                    let named = slash_index.is_none_or(|s| i > s);
                    let default_capture = if default.is_some() {
                        let cap = *self
                            .default_capture_indices
                            .get(default_idx)
                            .ok_or_else(|| anyhow!("internal default capture mismatch"))?;
                        default_idx += 1;
                        Some(cap)
                    } else {
                        None
                    };
                    let local = self
                        .locals
                        .get(&name.node.ident)
                        .cloned()
                        .ok_or_else(|| anyhow!("missing local for parameter"))?;
                    specs.push(Spec {
                        name: name.node.ident.clone(),
                        local,
                        positional,
                        named,
                        default_capture,
                        is_args: false,
                        is_kwargs: false,
                    });
                }
                ParameterP::Args(name, _) => {
                    seen_star = true;
                    let local = self
                        .locals
                        .get(&name.node.ident)
                        .cloned()
                        .ok_or_else(|| anyhow!("missing local for *args"))?;
                    specs.push(Spec {
                        name: name.node.ident.clone(),
                        local,
                        positional: false,
                        named: false,
                        default_capture: None,
                        is_args: true,
                        is_kwargs: false,
                    });
                }
                ParameterP::KwArgs(name, _) => {
                    let local = self
                        .locals
                        .get(&name.node.ident)
                        .cloned()
                        .ok_or_else(|| anyhow!("missing local for **kwargs"))?;
                    specs.push(Spec {
                        name: name.node.ident.clone(),
                        local,
                        positional: false,
                        named: false,
                        default_capture: None,
                        is_args: true,
                        is_kwargs: true,
                    });
                }
            }
        }

        // Initialize defaults and *args/**kwargs.
        for spec in &specs {
            if spec.is_args {
                let empty = self.b.tmp();
                if spec.is_kwargs {
                    self.b.ins(format!("{empty} = call %Value @sk_dict_new()"));
                } else {
                    self.b.ins(format!(
                        "{empty} = call %Value @sk_tuple_new(i64 0, ptr null)"
                    ));
                }
                self.store_local(&spec.local, &empty);
            } else if let Some(cap) = spec.default_capture {
                let cell = self.capture_cell(cap);
                let value = self.b.tmp();
                self.b.ins(format!("{value} = load %Value, ptr {cell}"));
                self.store_local(&spec.local, &value);
            }
        }

        let mut bound_flags: HashMap<String, String> = HashMap::new();
        for spec in specs.iter().filter(|s| !s.is_args) {
            let flag = self.b.tmp();
            self.b.ins(format!("{flag} = alloca i1"));
            self.b.ins(format!("store i1 false, ptr {flag}"));
            bound_flags.insert(spec.name.clone(), flag);
        }

        let positional_specs: Vec<&Spec> = specs.iter().filter(|s| s.positional).collect();
        for (i, spec) in positional_specs.iter().enumerate() {
            let has = self.b.tmp();
            self.b.ins(format!("{has} = icmp sgt i64 %nargs, {i}"));
            let bind_label = self.b.label();
            let next_label = self.b.label();
            self.b.cond_br(&has, &bind_label, &next_label);
            self.b.emit_label(&bind_label);
            let argp = self.b.tmp();
            self.b
                .ins(format!("{argp} = getelementptr %Value, ptr %args, i64 {i}"));
            let arg = self.b.tmp();
            self.b.ins(format!("{arg} = load %Value, ptr {argp}"));
            self.store_local(&spec.local, &arg);
            let flag = bound_flags.get(&spec.name).expect("bound flag");
            self.b.ins(format!("store i1 true, ptr {flag}"));
            self.b.br(&next_label);
            self.b.emit_label(&next_label);
        }

        let num_positional = positional_specs.len();
        if let Some(args_spec) = specs.iter().find(|s| s.is_args && !s.is_kwargs) {
            let tuple = self.b.tmp();
            self.b.ins(format!(
                "{tuple} = call %Value @sk_tuple_slice(ptr %args, i64 %nargs, i64 {num_positional})"
            ));
            self.store_local(&args_spec.local, &tuple);
        } else {
            let too_many = self.b.tmp();
            self.b.ins(format!(
                "{too_many} = icmp sgt i64 %nargs, {num_positional}"
            ));
            let err_label = self.b.label();
            let ok_label = self.b.label();
            self.b.cond_br(&too_many, &err_label, &ok_label);
            self.b.emit_label(&err_label);
            self.emit_error("too many positional arguments");
            self.b.emit_label(&ok_label);
        }

        let named_specs: Vec<&Spec> = specs.iter().filter(|s| s.named).collect();
        let kwargs_spec = specs.iter().find(|s| s.is_kwargs);

        let j_cell = self.b.tmp();
        self.b.ins(format!("{j_cell} = alloca i64"));
        self.b.ins(format!("store i64 0, ptr {j_cell}"));
        let loop_label = self.b.label();
        let body_label = self.b.label();
        let end_label = self.b.label();
        self.b.br(&loop_label);
        self.b.emit_label(&loop_label);
        let j = self.b.tmp();
        self.b.ins(format!("{j} = load i64, ptr {j_cell}"));
        let cond = self.b.tmp();
        self.b.ins(format!("{cond} = icmp slt i64 {j}, %nkwargs"));
        self.b.cond_br(&cond, &body_label, &end_label);

        self.b.emit_label(&body_label);
        let keyp = self.b.tmp();
        self.b.ins(format!(
            "{keyp} = getelementptr %Value, ptr %kw_names, i64 {j}"
        ));
        let key = self.b.tmp();
        self.b.ins(format!("{key} = load %Value, ptr {keyp}"));
        let valp = self.b.tmp();
        self.b.ins(format!(
            "{valp} = getelementptr %Value, ptr %kw_values, i64 {j}"
        ));
        let val = self.b.tmp();
        self.b.ins(format!("{val} = load %Value, ptr {valp}"));

        let next_iter = self.b.label();
        for spec in &named_specs {
            let (sym, len) = self.c.string_const(&spec.name);
            let eq = self.b.tmp();
            self.b.ins(format!(
                "{eq} = call i64 @sk_string_eq_cstr(%Value {key}, ptr {sym}, i64 {len})"
            ));
            let is_match = self.b.tmp();
            self.b.ins(format!("{is_match} = icmp ne i64 {eq}, 0"));
            let match_label = self.b.label();
            let next_param = self.b.label();
            self.b.cond_br(&is_match, &match_label, &next_param);

            self.b.emit_label(&match_label);
            let flag = bound_flags.get(&spec.name).expect("bound flag");
            let already = self.b.tmp();
            self.b.ins(format!("{already} = load i1, ptr {flag}"));
            let is_already = self.b.tmp();
            self.b
                .ins(format!("{is_already} = icmp eq i1 {already}, true"));
            let err_label = self.b.label();
            let assign_label = self.b.label();
            self.b.cond_br(&is_already, &err_label, &assign_label);
            self.b.emit_label(&err_label);
            self.emit_error(&format!("multiple values for argument '{}'", spec.name));
            self.b.emit_label(&assign_label);
            self.store_local(&spec.local, &val);
            self.b.ins(format!("store i1 true, ptr {flag}"));
            self.b.br(&next_iter);

            self.b.emit_label(&next_param);
        }

        if let Some(kwargs) = kwargs_spec {
            let kwargs_value = self.load_local_value(&kwargs.local);
            self.b.ins(format!(
                "call void @sk_dict_set(%Value {kwargs_value}, %Value {key}, %Value {val})"
            ));
            self.b.br(&next_iter);
        } else {
            self.emit_error("unexpected keyword argument");
        }

        self.b.emit_label(&next_iter);
        let next = self.b.tmp();
        self.b.ins(format!("{next} = add i64 {j}, 1"));
        self.b.ins(format!("store i64 {next}, ptr {j_cell}"));
        self.b.br(&loop_label);
        self.b.emit_label(&end_label);

        for spec in specs
            .iter()
            .filter(|s| !s.is_args && s.default_capture.is_none())
        {
            let flag = bound_flags.get(&spec.name).expect("bound flag");
            let bound = self.b.tmp();
            self.b.ins(format!("{bound} = load i1, ptr {flag}"));
            let err_label = self.b.label();
            let ok_label = self.b.label();
            self.b.cond_br(&bound, &ok_label, &err_label);
            self.b.emit_label(&err_label);
            self.emit_error(&format!("missing required argument '{}'", spec.name));
            self.b.emit_label(&ok_label);
        }

        // Parameters are known to be bound after the prologue.
        for spec in &specs {
            self.known_bound.insert(spec.name.clone());
        }
        Ok(())
    }

    fn expr_ty(&self, expr: &AstExpr) -> crate::types::Ty {
        self.c.types.expr_ty(&self.module_path, expr.span)
    }

    /// Identifier expression spans are not always the same as the span where
    /// the variable's inferred type was recorded.  Consult the local/global
    /// table for identifiers so `result = []; result.append(x)` still uses the
    /// compact ListInt path.
    fn expr_is_list_int(&self, expr: &AstExpr) -> bool {
        if let Expr::Identifier(id) = &expr.node {
            self.local_ty(&id.node.ident) == crate::types::Ty::ListInt
        } else {
            self.expr_ty(expr) == crate::types::Ty::ListInt
        }
    }

    fn expr_is_int(&self, expr: &AstExpr) -> bool {
        if let Expr::Identifier(id) = &expr.node {
            self.local_ty(&id.node.ident) == crate::types::Ty::Int
        } else {
            self.expr_ty(expr) == crate::types::Ty::Int
        }
    }

    fn expr_is_none(&self, expr: &AstExpr) -> bool {
        matches!(&expr.node, Expr::Identifier(id) if id.node.ident == "None")
    }

    /// Emit a direct length load when the representation is statically known.
    /// This is especially important inside comprehensions, where the generic
    /// `sk_iter_len` call otherwise hides the list header from LLVM.
    fn compile_known_len(&mut self, expr: &AstExpr) -> Result<Option<String>> {
        if self.expr_is_list_int(expr) {
            let value = self.compile_expr(expr)?;
            let payload = self.b.tmp();
            self.b
                .ins(format!("{payload} = extractvalue %Value {value}, 1"));
            let list = self.b.tmp();
            self.b
                .ins(format!("{list} = inttoptr i64 {payload} to ptr"));
            let lenp = self.b.tmp();
            self.b.ins(format!(
                "{lenp} = getelementptr %SkList, ptr {list}, i32 0, i32 0"
            ));
            let len = self.b.tmp();
            self.b.ins(format!("{len} = load i64, ptr {lenp}"));
            return Ok(Some(len));
        }
        if let Expr::Call(callee, args) = &expr.node {
            if let Expr::Dot(obj, name) = &callee.node {
                if name.node.as_str() == "elems" && args.args.is_empty() {
                    let string = self.compile_expr(obj)?;
                    let payload = self.b.tmp();
                    self.b
                        .ins(format!("{payload} = extractvalue %Value {string}, 1"));
                    let strp = self.b.tmp();
                    self.b
                        .ins(format!("{strp} = inttoptr i64 {payload} to ptr"));
                    let lenp = self.b.tmp();
                    self.b.ins(format!(
                        "{lenp} = getelementptr %SkString, ptr {strp}, i32 0, i32 0"
                    ));
                    let len = self.b.tmp();
                    self.b.ins(format!("{len} = load i64, ptr {lenp}"));
                    return Ok(Some(len));
                }
            }
        }
        if let Expr::Literal(AstLiteral::String(s)) = &expr.node {
            return Ok(Some(s.node.len().to_string()));
        }
        Ok(None)
    }

    fn box_int(&mut self, value: &str) -> String {
        let out = self.b.tmp();
        self.b
            .ins(format!("{out} = call %Value @sk_int(i64 {value})"));
        out
    }

    fn unbox_int(&mut self, value: &str) -> String {
        let out = self.b.tmp();
        self.b
            .ins(format!("{out} = extractvalue %Value {value}, 1"));
        out
    }

    /// Evaluate a side-effect-free integer expression at compile time.  This
    /// is deliberately small: it covers literals and arithmetic, which is
    /// enough to discover the fixed trip counts of `range(...)` loops.
    fn const_int_expr(&self, expr: &AstExpr) -> Option<i64> {
        if let Expr::Identifier(id) = &expr.node {
            if let Some(v) = self.const_locals.get(&id.node.ident) {
                return Some(*v);
            }
        }
        // Unrolling makes loop variables constants.  Fold the small linear
        // expressions that show up in hot indexing code (`t - 15`, `i * 64`,
        // `(i + 1) * 64`, ...) so those accesses can use the fixed-list fast
        // path instead of falling back to a dynamic bounds check.
        const_int_ast(expr)
            .or_else(|| self.eval_linear_expr(expr, "", 0))
            .or_else(|| eval_const_int(expr, &self.const_locals))
    }

    fn unroll_limit(&self) -> i64 {
        std::env::var("STARLARKC_UNROLL_LIMIT")
            .ok()
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or(64)
    }

    fn range_trip_count(start: i64, stop: i64, step: i64) -> Option<i64> {
        if step == 0 {
            return None;
        }
        let start = start as i128;
        let stop = stop as i128;
        let step = step as i128;
        let n = if step > 0 {
            if start >= stop {
                0
            } else {
                (stop - start + step - 1) / step
            }
        } else if start <= stop {
            0
        } else {
            (start - stop + (-step) - 1) / (-step)
        };
        i64::try_from(n).ok()
    }

    fn emit_unrolled_values_loop(
        &mut self,
        f: &starlark_syntax::syntax::ast::ForP<starlark_syntax::syntax::ast::AstNoPayload>,
        values: &[i64],
    ) -> Result<()> {
        let end_label = self.b.label();
        for value in values {
            let next_label = self.b.label();
            self.assign_int_target(&f.var, &value.to_string())?;
            if let AssignTarget::Identifier(id) = &f.var.node {
                self.const_locals.insert(id.node.ident.clone(), *value);
            }
            self.loops.push(LoopLabels {
                break_label: end_label.clone(),
                continue_label: next_label.clone(),
            });
            self.compile_stmt(&f.body)?;
            self.loops.pop();
            if !self.b.terminated() {
                self.b.br(&next_label);
            }
            self.b.emit_label(&next_label);
        }
        if !self.b.terminated() {
            self.b.br(&end_label);
        }
        self.b.emit_label(&end_label);
        Ok(())
    }

    fn emit_unrolled_int_loop(
        &mut self,
        f: &starlark_syntax::syntax::ast::ForP<starlark_syntax::syntax::ast::AstNoPayload>,
        start: i64,
        step: i64,
        count: i64,
    ) -> Result<()> {
        let end_label = self.b.label();
        for k in 0..count {
            let next_label = self.b.label();
            let value = start.wrapping_add(k.wrapping_mul(step));
            self.assign_int_target(&f.var, &value.to_string())?;
            if let AssignTarget::Identifier(id) = &f.var.node {
                self.const_locals.insert(id.node.ident.clone(), value);
            }
            self.loops.push(LoopLabels {
                break_label: end_label.clone(),
                continue_label: next_label.clone(),
            });
            self.compile_stmt(&f.body)?;
            self.loops.pop();
            if !self.b.terminated() {
                self.b.br(&next_label);
            }
            self.b.emit_label(&next_label);
        }
        if !self.b.terminated() {
            self.b.br(&end_label);
        }
        self.b.emit_label(&end_label);
        Ok(())
    }

    fn compile_int_operand(&mut self, expr: &AstExpr) -> Result<String> {
        if let Some(v) = self.compile_int_expr(expr)? {
            Ok(v)
        } else {
            let v = self.compile_expr(expr)?;
            Ok(self.unbox_int(&v))
        }
    }

    fn cache_list_view(&mut self, name: &str, value: &str) {
        let payload = self.b.tmp();
        self.b
            .ins(format!("{payload} = extractvalue %Value {value}, 1"));
        let list = self.b.tmp();
        self.b
            .ins(format!("{list} = inttoptr i64 {payload} to ptr"));
        let lenp = self.b.tmp();
        self.b.ins(format!(
            "{lenp} = getelementptr %SkList, ptr {list}, i32 0, i32 0"
        ));
        let len = self.b.tmp();
        self.b.ins(format!("{len} = load i64, ptr {lenp}"));
        let itemsp = self.b.tmp();
        self.b.ins(format!(
            "{itemsp} = getelementptr %SkList, ptr {list}, i32 0, i32 2"
        ));
        let items = self.b.tmp();
        self.b.ins(format!("{items} = load ptr, ptr {itemsp}"));
        self.list_views.insert(name.to_owned(), (len, items));
    }

    fn compile_int_list_get_named(&mut self, name: Option<&str>, obj: &str, idx: &str) -> String {
        let (len, items) = if let Some(name) = name {
            if let Some(view) = self.list_views.get(name) {
                view.clone()
            } else {
                let payload = self.b.tmp();
                self.b
                    .ins(format!("{payload} = extractvalue %Value {obj}, 1"));
                let list = self.b.tmp();
                self.b
                    .ins(format!("{list} = inttoptr i64 {payload} to ptr"));
                let lenp = self.b.tmp();
                self.b.ins(format!(
                    "{lenp} = getelementptr %SkList, ptr {list}, i32 0, i32 0"
                ));
                let len = self.b.tmp();
                self.b.ins(format!("{len} = load i64, ptr {lenp}"));
                let itemsp = self.b.tmp();
                self.b.ins(format!(
                    "{itemsp} = getelementptr %SkList, ptr {list}, i32 0, i32 2"
                ));
                let items = self.b.tmp();
                self.b.ins(format!("{items} = load ptr, ptr {itemsp}"));
                self.list_views
                    .insert(name.to_owned(), (len.clone(), items.clone()));
                (len, items)
            }
        } else {
            let payload = self.b.tmp();
            self.b
                .ins(format!("{payload} = extractvalue %Value {obj}, 1"));
            let list = self.b.tmp();
            self.b
                .ins(format!("{list} = inttoptr i64 {payload} to ptr"));
            let lenp = self.b.tmp();
            self.b.ins(format!(
                "{lenp} = getelementptr %SkList, ptr {list}, i32 0, i32 0"
            ));
            let len = self.b.tmp();
            self.b.ins(format!("{len} = load i64, ptr {lenp}"));
            let itemsp = self.b.tmp();
            self.b.ins(format!(
                "{itemsp} = getelementptr %SkList, ptr {list}, i32 0, i32 2"
            ));
            let items = self.b.tmp();
            self.b.ins(format!("{items} = load ptr, ptr {itemsp}"));
            (len, items)
        };

        let neg = self.b.tmp();
        self.b.ins(format!("{neg} = icmp slt i64 {idx}, 0"));
        let adjusted = self.b.tmp();
        self.b.ins(format!("{adjusted} = add i64 {idx}, {len}"));
        let norm = self.b.tmp();
        self.b.ins(format!(
            "{norm} = select i1 {neg}, i64 {adjusted}, i64 {idx}"
        ));
        let inb = self.b.tmp();
        self.b.ins(format!("{inb} = icmp ult i64 {norm}, {len}"));
        let fast = self.b.label();
        let slow = self.b.label();
        let done = self.b.label();
        self.b.cond_br(&inb, &fast, &slow);
        self.b.emit_label(&fast);
        let elemp = self.b.tmp();
        self.b.ins(format!(
            "{elemp} = getelementptr i64, ptr {items}, i64 {norm}"
        ));
        let fastv = self.b.tmp();
        self.b.ins(format!("{fastv} = load i64, ptr {elemp}"));
        self.b.br(&done);
        self.b.emit_label(&slow);
        let slowv = self.b.tmp();
        self.b.ins(format!(
            "{slowv} = call i64 @sk_int_list_get(%Value {obj}, i64 {idx})"
        ));
        self.b.br(&done);
        self.b.emit_label(&done);
        let out = self.b.tmp();
        self.b.ins(format!(
            "{out} = phi i64 [{fastv}, %{fast}], [{slowv}, %{slow}]"
        ));
        out
    }

    fn compile_int_list_view(&mut self, name: Option<&str>, obj: &str) -> (String, String) {
        if let Some(name) = name {
            if let Some(view) = self.list_views.get(name) {
                return view.clone();
            }
        }
        let payload = self.b.tmp();
        self.b
            .ins(format!("{payload} = extractvalue %Value {obj}, 1"));
        let list = self.b.tmp();
        self.b
            .ins(format!("{list} = inttoptr i64 {payload} to ptr"));
        let lenp = self.b.tmp();
        self.b.ins(format!(
            "{lenp} = getelementptr %SkList, ptr {list}, i32 0, i32 0"
        ));
        let len = self.b.tmp();
        self.b.ins(format!("{len} = load i64, ptr {lenp}"));
        let itemsp = self.b.tmp();
        self.b.ins(format!(
            "{itemsp} = getelementptr %SkList, ptr {list}, i32 0, i32 2"
        ));
        let items = self.b.tmp();
        self.b.ins(format!("{items} = load ptr, ptr {itemsp}"));
        if let Some(name) = name {
            self.list_views
                .insert(name.to_owned(), (len.clone(), items.clone()));
        }
        (len, items)
    }

    fn compile_int_list_get_unchecked(
        &mut self,
        name: Option<&str>,
        obj: &str,
        idx: &str,
    ) -> String {
        let (_, items) = self.compile_int_list_view(name, obj);
        let elemp = self.b.tmp();
        self.b.ins(format!(
            "{elemp} = getelementptr i64, ptr {items}, i64 {idx}"
        ));
        let out = self.b.tmp();
        self.b.ins(format!("{out} = load i64, ptr {elemp}"));
        out
    }

    fn index_is_safe_for_list(&self, list: &str, index: &AstExpr) -> bool {
        let Expr::Identifier(id) = &index.node else {
            return false;
        };
        self.safe_list_indices
            .iter()
            .rev()
            .any(|frame| frame.get(&id.node.ident).is_some_and(|name| name == list))
    }

    fn compile_int_list_push(&mut self, list: &str, value: &str) {
        let payload = self.b.tmp();
        self.b
            .ins(format!("{payload} = extractvalue %Value {list}, 1"));
        let lp = self.b.tmp();
        self.b.ins(format!("{lp} = inttoptr i64 {payload} to ptr"));
        let lenp = self.b.tmp();
        self.b.ins(format!(
            "{lenp} = getelementptr %SkList, ptr {lp}, i32 0, i32 0"
        ));
        let len = self.b.tmp();
        self.b.ins(format!("{len} = load i64, ptr {lenp}"));
        let capp = self.b.tmp();
        self.b.ins(format!(
            "{capp} = getelementptr %SkList, ptr {lp}, i32 0, i32 1"
        ));
        let cap = self.b.tmp();
        self.b.ins(format!("{cap} = load i64, ptr {capp}"));
        let has_space = self.b.tmp();
        self.b
            .ins(format!("{has_space} = icmp slt i64 {len}, {cap}"));
        let fast = self.b.label();
        let slow = self.b.label();
        let done = self.b.label();
        self.b.cond_br(&has_space, &fast, &slow);
        self.b.emit_label(&fast);
        let itemsp = self.b.tmp();
        self.b.ins(format!(
            "{itemsp} = getelementptr %SkList, ptr {lp}, i32 0, i32 2"
        ));
        let items = self.b.tmp();
        self.b.ins(format!("{items} = load ptr, ptr {itemsp}"));
        let elemp = self.b.tmp();
        self.b.ins(format!(
            "{elemp} = getelementptr i64, ptr {items}, i64 {len}"
        ));
        self.b.ins(format!("store i64 {value}, ptr {elemp}"));
        let new_len = self.b.tmp();
        self.b.ins(format!("{new_len} = add i64 {len}, 1"));
        self.b.ins(format!("store i64 {new_len}, ptr {lenp}"));
        self.b.br(&done);
        self.b.emit_label(&slow);
        self.b.ins(format!(
            "call void @sk_int_list_push(%Value {list}, i64 {value})"
        ));
        self.b.br(&done);
        self.b.emit_label(&done);
    }

    /// Push into a compact list whose capacity was reserved for the exact
    /// number of elements the comprehension will produce.  This removes the
    /// capacity check and growth branch from the innermost loop.
    /// Store `value` at `items[idx]` without touching the length field.
    fn compile_int_list_store_at(
        &mut self,
        name: Option<&str>,
        list: &str,
        idx: &str,
        value: &str,
    ) {
        let items = match name.and_then(|n| self.append_items.get(n).cloned()) {
            Some(items) => items,
            None => {
                let payload = self.b.tmp();
                self.b
                    .ins(format!("{payload} = extractvalue %Value {list}, 1"));
                let lp = self.b.tmp();
                self.b.ins(format!("{lp} = inttoptr i64 {payload} to ptr"));
                let itemsp = self.b.tmp();
                self.b.ins(format!(
                    "{itemsp} = getelementptr %SkList, ptr {lp}, i32 0, i32 2"
                ));
                let items = self.b.tmp();
                self.b.ins(format!("{items} = load ptr, ptr {itemsp}"));
                items
            }
        };
        let elemp = self.b.tmp();
        self.b.ins(format!(
            "{elemp} = getelementptr i64, ptr {items}, i64 {idx}"
        ));
        self.b.ins(format!("store i64 {value}, ptr {elemp}"));
    }

    fn compile_int_list_push_unchecked(&mut self, list: &str, value: &str) {
        let payload = self.b.tmp();
        self.b
            .ins(format!("{payload} = extractvalue %Value {list}, 1"));
        let lp = self.b.tmp();
        self.b.ins(format!("{lp} = inttoptr i64 {payload} to ptr"));
        let lenp = self.b.tmp();
        self.b.ins(format!(
            "{lenp} = getelementptr %SkList, ptr {lp}, i32 0, i32 0"
        ));
        let len = self.b.tmp();
        self.b.ins(format!("{len} = load i64, ptr {lenp}"));
        let itemsp = self.b.tmp();
        self.b.ins(format!(
            "{itemsp} = getelementptr %SkList, ptr {lp}, i32 0, i32 2"
        ));
        let items = self.b.tmp();
        self.b.ins(format!("{items} = load ptr, ptr {itemsp}"));
        let elemp = self.b.tmp();
        self.b.ins(format!(
            "{elemp} = getelementptr i64, ptr {items}, i64 {len}"
        ));
        self.b.ins(format!("store i64 {value}, ptr {elemp}"));
        let new_len = self.b.tmp();
        self.b.ins(format!("{new_len} = add i64 {len}, 1"));
        self.b.ins(format!("store i64 {new_len}, ptr {lenp}"));
    }

    fn compile_int_list_set_unchecked(&mut self, obj: &str, idx: &str, value: &str) {
        let payload = self.b.tmp();
        self.b
            .ins(format!("{payload} = extractvalue %Value {obj}, 1"));
        let list = self.b.tmp();
        self.b
            .ins(format!("{list} = inttoptr i64 {payload} to ptr"));
        let itemsp = self.b.tmp();
        self.b.ins(format!(
            "{itemsp} = getelementptr %SkList, ptr {list}, i32 0, i32 2"
        ));
        let items = self.b.tmp();
        self.b.ins(format!("{items} = load ptr, ptr {itemsp}"));
        let elemp = self.b.tmp();
        self.b.ins(format!(
            "{elemp} = getelementptr i64, ptr {items}, i64 {idx}"
        ));
        self.b.ins(format!("store i64 {value}, ptr {elemp}"));
    }

    fn compile_int_list_set(&mut self, obj: &str, idx: &str, value: &str) {
        let payload = self.b.tmp();
        self.b
            .ins(format!("{payload} = extractvalue %Value {obj}, 1"));
        let list = self.b.tmp();
        self.b
            .ins(format!("{list} = inttoptr i64 {payload} to ptr"));
        let lenp = self.b.tmp();
        self.b.ins(format!(
            "{lenp} = getelementptr %SkList, ptr {list}, i32 0, i32 0"
        ));
        let len = self.b.tmp();
        self.b.ins(format!("{len} = load i64, ptr {lenp}"));
        let neg = self.b.tmp();
        self.b.ins(format!("{neg} = icmp slt i64 {idx}, 0"));
        let adjusted = self.b.tmp();
        self.b.ins(format!("{adjusted} = add i64 {idx}, {len}"));
        let norm = self.b.tmp();
        self.b.ins(format!(
            "{norm} = select i1 {neg}, i64 {adjusted}, i64 {idx}"
        ));
        let inb = self.b.tmp();
        self.b.ins(format!("{inb} = icmp ult i64 {norm}, {len}"));
        let fast = self.b.label();
        let slow = self.b.label();
        let done = self.b.label();
        self.b.cond_br(&inb, &fast, &slow);
        self.b.emit_label(&fast);
        let itemsp = self.b.tmp();
        self.b.ins(format!(
            "{itemsp} = getelementptr %SkList, ptr {list}, i32 0, i32 2"
        ));
        let items = self.b.tmp();
        self.b.ins(format!("{items} = load ptr, ptr {itemsp}"));
        let elemp = self.b.tmp();
        self.b.ins(format!(
            "{elemp} = getelementptr i64, ptr {items}, i64 {norm}"
        ));
        self.b.ins(format!("store i64 {value}, ptr {elemp}"));
        self.b.br(&done);
        self.b.emit_label(&slow);
        let idx_box = self.box_int(idx);
        let value_box = self.box_int(value);
        self.b.ins(format!(
            "call void @sk_set_index(%Value {obj}, %Value {idx_box}, %Value {value_box})"
        ));
        self.b.br(&done);
        self.b.emit_label(&done);
    }

    fn compile_rotate(&mut self, x: &str, n: &str, bits: u32, left: bool) -> String {
        // Emit LLVM's funnel-shift intrinsics instead of open-coding
        // `(x << n) | (x >> (bits - n))`.  Besides being clearer to the
        // optimizer, these lower to rotate instructions and let LLVM fold
        // constant rotate amounts after loop unrolling.
        let mask = if bits == 32 { 31 } else { 63 };
        let n_masked = self.b.tmp();
        self.b.ins(format!("{n_masked} = and i64 {n}, {mask}"));
        if bits == 32 {
            let x32 = self.b.tmp();
            self.b.ins(format!("{x32} = trunc i64 {x} to i32"));
            let n32 = self.b.tmp();
            self.b.ins(format!("{n32} = trunc i64 {n_masked} to i32"));
            let rotated = self.b.tmp();
            let intrinsic = if left {
                "llvm.fshl.i32"
            } else {
                "llvm.fshr.i32"
            };
            self.b.ins(format!(
                "{rotated} = call i32 @{intrinsic}(i32 {x32}, i32 {x32}, i32 {n32})"
            ));
            let out = self.b.tmp();
            self.b.ins(format!("{out} = zext i32 {rotated} to i64"));
            out
        } else {
            let rotated = self.b.tmp();
            let intrinsic = if left {
                "llvm.fshl.i64"
            } else {
                "llvm.fshr.i64"
            };
            self.b.ins(format!(
                "{rotated} = call i64 @{intrinsic}(i64 {x}, i64 {x}, i64 {n_masked})"
            ));
            rotated
        }
    }

    /// Compile an expression that is known to produce an integer to an `i64`
    /// SSA value. Returns `None` when the expression needs the dynamic path.
    fn compile_int_expr(&mut self, expr: &AstExpr) -> Result<Option<String>> {
        match &expr.node {
            Expr::Literal(AstLiteral::Int(i)) => {
                let value = match &i.node {
                    TokenInt::I32(v) => *v as i64,
                    TokenInt::BigInt(v) => {
                        let text = v.to_string();
                        text.parse::<i64>()
                            .or_else(|_| text.parse::<u64>().map(|u| u as i64))
                            .map_err(|_| anyhow!("integer literal does not fit in 64 bits: {v}"))?
                    }
                };
                Ok(Some(value.to_string()))
            }
            Expr::Identifier(id) => {
                let name = &id.node.ident;
                if let Some(v) = self.const_locals.get(name) {
                    return Ok(Some(v.to_string()));
                }
                if let Some(local) = self.locals.get(name).cloned() {
                    Ok(Some(self.load_local_int(&local)))
                } else if let Some(place) = self.resolve_place(name) {
                    match place {
                        Place::Local(local) => Ok(Some(self.load_local_int(&local))),
                        Place::Global(sym) => {
                            let v = self.b.tmp();
                            self.b.ins(format!("{v} = load %Value, ptr {sym}"));
                            Ok(Some(self.unbox_int(&v)))
                        }
                    }
                } else {
                    Ok(None)
                }
            }
            Expr::Minus(e) => {
                let v = self.compile_int_operand(e)?;
                let out = self.b.tmp();
                self.b.ins(format!("{out} = sub i64 0, {v}"));
                Ok(Some(out))
            }
            Expr::Plus(e) => Ok(Some(self.compile_int_operand(e)?)),
            Expr::BitNot(e) => {
                let v = self.compile_int_operand(e)?;
                let out = self.b.tmp();
                self.b.ins(format!("{out} = xor i64 {v}, -1"));
                Ok(Some(out))
            }
            Expr::Op(lhs, op, rhs) => {
                use BinOp::*;
                match op {
                    Add | Subtract | Multiply | FloorDivide | Percent | BitAnd | BitOr | BitXor
                    | LeftShift | RightShift => {
                        let l = self.compile_int_operand(lhs)?;
                        let r = self.compile_int_operand(rhs)?;
                        let out = self.b.tmp();
                        let ins = match op {
                            Add => format!("{out} = add i64 {l}, {r}"),
                            Subtract => format!("{out} = sub i64 {l}, {r}"),
                            Multiply => format!("{out} = mul i64 {l}, {r}"),
                            BitAnd => format!("{out} = and i64 {l}, {r}"),
                            BitOr => format!("{out} = or i64 {l}, {r}"),
                            BitXor => format!("{out} = xor i64 {l}, {r}"),
                            LeftShift | RightShift => {
                                let amt = self.b.tmp();
                                self.b.ins(format!("{amt} = and i64 {r}, 63"));
                                let shifted = self.b.tmp();
                                let opcode = if *op == LeftShift { "shl" } else { "lshr" };
                                self.b.ins(format!("{shifted} = {opcode} i64 {l}, {amt}"));
                                let big = self.b.tmp();
                                self.b.ins(format!("{big} = icmp uge i64 {r}, 64"));
                                self.b
                                    .ins(format!("{out} = select i1 {big}, i64 0, i64 {shifted}"));
                                return Ok(Some(out));
                            }
                            FloorDivide => {
                                self.b.ins(format!(
                                    "{out} = call i64 @sk_floordiv_i64(i64 {l}, i64 {r})"
                                ));
                                return Ok(Some(out));
                            }
                            Percent => {
                                self.b
                                    .ins(format!("{out} = call i64 @sk_mod_i64(i64 {l}, i64 {r})"));
                                return Ok(Some(out));
                            }
                            _ => unreachable!(),
                        };
                        self.b.ins(ins);
                        Ok(Some(out))
                    }
                    _ => Ok(None),
                }
            }
            Expr::Index(pair) => {
                if let Expr::Identifier(obj_id) = &pair.0.node {
                    if !self.locals.contains_key(&obj_id.node.ident) {
                        let key = (self.module_path.clone(), obj_id.node.ident.clone());
                        if let Some(values) = self.c.const_lists.get(&key) {
                            if let Some(idx) = self.const_int_expr(&pair.1) {
                                if let Some(v) = values.get(idx as usize) {
                                    return Ok(Some(v.to_string()));
                                }
                            }
                        }
                        if let Some(sym) = self.c.const_list_globals.get(&key).cloned() {
                            let values = self.c.const_lists.get(&key).cloned().unwrap_or_default();
                            let n = values.len() as i64;
                            let idx = self.compile_int_operand(&pair.1)?;
                            let neg = self.b.tmp();
                            self.b.ins(format!("{neg} = icmp slt i64 {idx}, 0"));
                            let adjusted = self.b.tmp();
                            self.b.ins(format!("{adjusted} = add i64 {idx}, {n}"));
                            let norm = self.b.tmp();
                            self.b.ins(format!(
                                "{norm} = select i1 {neg}, i64 {adjusted}, i64 {idx}"
                            ));
                            let inb = self.b.tmp();
                            self.b.ins(format!("{inb} = icmp ult i64 {norm}, {n}"));
                            let fast = self.b.label();
                            let err = self.b.label();
                            self.b.cond_br(&inb, &fast, &err);
                            self.b.emit_label(&err);
                            self.emit_error("list index out of range");
                            self.b.emit_label(&fast);
                            let p = self.b.tmp();
                            self.b.ins(format!(
                                "{p} = getelementptr [{n} x i64], ptr {sym}, i64 0, i64 {norm}"
                            ));
                            let out = self.b.tmp();
                            self.b.ins(format!("{out} = load i64, ptr {p}"));
                            return Ok(Some(out));
                        }
                    }
                }
                if let Expr::Identifier(obj_id) = &pair.0.node {
                    if let Some(len) = self.fixed_list_lens.get(&obj_id.node.ident).copied() {
                        if let Some(idx) = self.const_int_expr(&pair.1) {
                            let idx = if idx < 0 { idx.wrapping_add(len) } else { idx };
                            if (0..len).contains(&idx) {
                                let name = obj_id.node.ident.clone();
                                let obj = self.compile_expr(&pair.0)?;
                                return Ok(Some(self.compile_int_list_get_unchecked(
                                    Some(&name),
                                    &obj,
                                    &idx.to_string(),
                                )));
                            }
                        }
                    }
                }
                if self.expr_is_list_int(&pair.0) {
                    let obj_name = if let Expr::Identifier(id) = &pair.0.node {
                        Some(id.node.ident.clone())
                    } else {
                        None
                    };
                    let obj = self.compile_expr(&pair.0)?;
                    let idx = self.compile_int_operand(&pair.1)?;
                    if let Some(name) = obj_name.as_deref() {
                        if self.index_is_safe_for_list(name, &pair.1) {
                            return Ok(Some(self.compile_int_list_get_unchecked(
                                Some(name),
                                &obj,
                                &idx,
                            )));
                        }
                    }
                    Ok(Some(self.compile_int_list_get_named(
                        obj_name.as_deref(),
                        &obj,
                        &idx,
                    )))
                } else {
                    Ok(None)
                }
            }
            Expr::Call(callee, args) => {
                if let Expr::Identifier(id) = &callee.node {
                    if args.args.len() == 2
                        && args
                            .args
                            .iter()
                            .all(|a| matches!(a.node, ArgumentP::Positional(_)))
                    {
                        let (bits, left) = match id.node.ident.as_str() {
                            "rotl32" => (32, true),
                            "rotr32" => (32, false),
                            "rotr64" => (64, false),
                            _ => (0, false),
                        };
                        if bits != 0 {
                            let x = self.compile_int_operand(args.args[0].expr())?;
                            let n = self.compile_int_operand(args.args[1].expr())?;
                            return Ok(Some(self.compile_rotate(&x, &n, bits, left)));
                        }
                    }
                    match id.node.ident.as_str() {
                        "len" if args.args.len() == 1 => {
                            if let Some(len) = self.compile_known_len(args.args[0].expr())? {
                                return Ok(Some(len));
                            }
                            let arg = self.compile_expr(args.args[0].expr())?;
                            let out = self.b.tmp();
                            self.b
                                .ins(format!("{out} = call i64 @sk_iter_len(%Value {arg})"));
                            return Ok(Some(out));
                        }
                        "ord" if args.args.len() == 1 => {
                            if let Expr::Identifier(arg_id) = &args.args[0].expr().node {
                                if self.char_locals.contains(&arg_id.node.ident) {
                                    let local = self
                                        .locals
                                        .get(&arg_id.node.ident)
                                        .cloned()
                                        .expect("char local");
                                    return Ok(Some(self.load_local_int(&local)));
                                }
                            }
                            if self.expr_ty(args.args[0].expr()) == crate::types::Ty::Int {
                                return Ok(Some(self.compile_int_operand(args.args[0].expr())?));
                            }
                            let arg = self.compile_expr(args.args[0].expr())?;
                            let out = self.b.tmp();
                            self.b
                                .ins(format!("{out} = call i64 @sk_ord_int(%Value {arg})"));
                            return Ok(Some(out));
                        }
                        "int" if args.args.len() == 1 => {
                            if let Some(v) = self.compile_int_expr(args.args[0].expr())? {
                                return Ok(Some(v));
                            }
                        }
                        _ => {}
                    }
                }
                let v = self.compile_call(callee, args)?;
                Ok(Some(self.unbox_int(&v)))
            }
            Expr::If(triple) => {
                let result = self.b.tmp();
                self.b.ins(format!("{result} = alloca i64"));
                let cond = self.compile_expr(&triple.0)?;
                let truth = self.compile_truthy(&cond);
                let then_label = self.b.label();
                let else_label = self.b.label();
                let end_label = self.b.label();
                self.b.cond_br(&truth, &then_label, &else_label);
                self.b.emit_label(&then_label);
                let a = self.compile_int_operand(&triple.1)?;
                self.b.ins(format!("store i64 {a}, ptr {result}"));
                self.b.br(&end_label);
                self.b.emit_label(&else_label);
                let b = self.compile_int_operand(&triple.2)?;
                self.b.ins(format!("store i64 {b}, ptr {result}"));
                self.b.br(&end_label);
                self.b.emit_label(&end_label);
                let out = self.b.tmp();
                self.b.ins(format!("{out} = load i64, ptr {result}"));
                Ok(Some(out))
            }
            _ => Ok(None),
        }
    }

    /// If `callee` names a top-level Starlark function, call its LLVM symbol
    /// directly with a stack argument vector.  This avoids the indirect call
    /// through `sk_call` and lets LLVM inline the callee, constant-fold the
    /// argument-binding prologue, and see through the closure environment.
    fn try_direct_call(
        &mut self,
        callee: &AstExpr,
        args: &starlark_syntax::syntax::ast::CallArgsP<starlark_syntax::syntax::ast::AstNoPayload>,
    ) -> Result<Option<String>> {
        let Expr::Identifier(id) = &callee.node else {
            return Ok(None);
        };
        let Some(func_id) = self.c.types.resolve_func(&self.module_path, &id.node.ident) else {
            return Ok(None);
        };
        let Some(fn_name) = self.c.function_names.get(&func_id).cloned() else {
            return Ok(None);
        };
        let mut positional = Vec::new();
        let mut named = Vec::new();
        for arg in &args.args {
            match &arg.node {
                ArgumentP::Positional(e) => positional.push(e),
                ArgumentP::Named(name, e) => named.push((name, e)),
                _ => return Ok(None),
            }
        }

        // The closure carries the callee's captured environment.  We still
        // load the function value (and its bound check), but the call itself
        // is direct and the argument vector is a plain stack array.
        let callee_value = self.compile_expr(callee)?;
        let payload = self.b.tmp();
        self.b
            .ins(format!("{payload} = extractvalue %Value {callee_value}, 1"));
        let closure = self.b.tmp();
        self.b
            .ins(format!("{closure} = inttoptr i64 {payload} to ptr"));

        let npos = positional.len();
        let arg_ptr = if npos == 0 {
            "null".to_owned()
        } else {
            let arr = self.b.tmp();
            self.b.ins(format!("{arr} = alloca [{npos} x %Value]"));
            for (i, arg) in positional.iter().enumerate() {
                let value = self.compile_expr(arg)?;
                let p = self.b.tmp();
                self.b.ins(format!(
                    "{p} = getelementptr [{npos} x %Value], ptr {arr}, i64 0, i64 {i}"
                ));
                self.b.ins(format!("store %Value {value}, ptr {p}"));
            }
            arr
        };
        let nkw = named.len();
        let (kw_names_ptr, kw_values_ptr) = if nkw == 0 {
            ("null".to_owned(), "null".to_owned())
        } else {
            let names = self.b.tmp();
            let values = self.b.tmp();
            self.b.ins(format!("{names} = alloca [{nkw} x %Value]"));
            self.b.ins(format!("{values} = alloca [{nkw} x %Value]"));
            for (i, (name, arg)) in named.iter().enumerate() {
                let key = self.compile_string_value(&name.node)?;
                let value = self.compile_expr(arg)?;
                let np = self.b.tmp();
                self.b.ins(format!(
                    "{np} = getelementptr [{nkw} x %Value], ptr {names}, i64 0, i64 {i}"
                ));
                self.b.ins(format!("store %Value {key}, ptr {np}"));
                let vp = self.b.tmp();
                self.b.ins(format!(
                    "{vp} = getelementptr [{nkw} x %Value], ptr {values}, i64 0, i64 {i}"
                ));
                self.b.ins(format!("store %Value {value}, ptr {vp}"));
            }
            (names, values)
        };
        let out = self.b.tmp();
        self.b.ins(format!(
            "{out} = call %Value @{fn_name}(ptr {closure}, i64 {npos}, ptr {arg_ptr}, i64 {nkw}, ptr {kw_names_ptr}, ptr {kw_values_ptr})"
        ));
        Ok(Some(out))
    }

    fn compile_call(
        &mut self,
        callee: &AstExpr,
        args: &starlark_syntax::syntax::ast::CallArgsP<starlark_syntax::syntax::ast::AstNoPayload>,
    ) -> Result<String> {
        // Fast paths for the two methods that dominate byte/list-heavy
        // Starlark code.  They avoid the generic getattr + sk_call machinery
        // while preserving normal Starlark behavior.
        if let Expr::Dot(obj, name) = &callee.node {
            let method = name.node.as_str();
            if method == "append"
                && args.args.len() == 1
                && matches!(args.args[0].node, ArgumentP::Positional(_))
            {
                let list = self.compile_expr(obj)?;
                if self.expr_is_list_int(obj) {
                    let item = self.compile_int_operand(args.args[0].expr())?;
                    if let Expr::Identifier(id) = &obj.node {
                        if let Some(idx) = self.append_indices.get(&id.node.ident).cloned() {
                            self.compile_int_list_store_at(
                                Some(&id.node.ident),
                                &list,
                                &idx,
                                &item,
                            );
                            return self.compile_none();
                        }
                    }
                    let unchecked = matches!(&obj.node, Expr::Identifier(id)
                        if self.exact_capacity_lists.contains(&id.node.ident));
                    if unchecked {
                        self.compile_int_list_push_unchecked(&list, &item);
                    } else {
                        self.compile_int_list_push(&list, &item);
                    }
                } else {
                    let item = self.compile_expr(args.args[0].expr())?;
                    self.b.ins(format!(
                        "call void @sk_append(%Value {list}, %Value {item})"
                    ));
                }
                return self.compile_none();
            }
            if method == "elems" && args.args.is_empty() {
                let recv = self.compile_expr(obj)?;
                let out = self.b.tmp();
                self.b
                    .ins(format!("{out} = call %Value @sk_elems(%Value {recv})"));
                return Ok(out);
            }
        }
        if let Some(v) = self.try_direct_call(callee, args)? {
            return Ok(v);
        }
        let callee = self.compile_expr(callee)?;
        let ca = self.b.tmp();
        self.b.ins(format!("{ca} = call ptr @sk_callargs_new()"));
        for arg in &args.args {
            match &arg.node {
                ArgumentP::Positional(e) => {
                    let v = self.compile_expr(e)?;
                    self.b
                        .ins(format!("call void @sk_callargs_arg(ptr {ca}, %Value {v})"));
                }
                ArgumentP::Named(name, e) => {
                    let key = self.compile_string_value(&name.node)?;
                    let v = self.compile_expr(e)?;
                    self.b.ins(format!(
                        "call void @sk_callargs_kw(ptr {ca}, %Value {key}, %Value {v})"
                    ));
                }
                ArgumentP::Args(e) => {
                    let v = self.compile_expr(e)?;
                    self.b
                        .ins(format!("call void @sk_callargs_args(ptr {ca}, %Value {v})"));
                }
                ArgumentP::KwArgs(e) => {
                    let v = self.compile_expr(e)?;
                    self.b.ins(format!(
                        "call void @sk_callargs_kwargs(ptr {ca}, %Value {v})"
                    ));
                }
            }
        }
        let v = self.b.tmp();
        self.b.ins(format!(
            "{v} = call %Value @sk_call(%Value {callee}, ptr {ca})"
        ));
        Ok(v)
    }

    /// Fuse the common `md_pad(to_bytes(s), ...)` pattern into a single
    /// allocation.  `to_bytes` and `md_pad` are ordinary Starlark functions,
    /// but in the hash workloads they are called back-to-back on every input
    /// and otherwise allocate two large intermediate lists.  The generated IR
    /// still does all of the byte copying and padding explicitly; this is just
    /// a representation-aware lowering of the two known list operations.
    fn try_compile_md_pad_from_string(&mut self, expr: &AstExpr) -> Result<Option<String>> {
        let Expr::Call(callee, call_args) = &expr.node else {
            return Ok(None);
        };
        let Expr::Identifier(callee_id) = &callee.node else {
            return Ok(None);
        };
        if callee_id.node.ident != "md_pad" {
            return Ok(None);
        }
        let Some(func_id) = self.c.types.resolve_func(&self.module_path, "md_pad") else {
            return Ok(None);
        };
        if func_id.1 != "md_pad" {
            return Ok(None);
        }

        let mut data = None;
        let mut big_endian = None;
        let mut block_size = None;
        let mut positional = 0usize;
        for arg in &call_args.args {
            match &arg.node {
                ArgumentP::Positional(e) => {
                    match positional {
                        0 => data = Some(e),
                        1 => big_endian = Some(e),
                        2 => block_size = Some(e),
                        _ => return Ok(None),
                    }
                    positional += 1;
                }
                ArgumentP::Named(name, e) => match name.node.as_str() {
                    "big_endian" => big_endian = Some(e),
                    "block_size" => block_size = Some(e),
                    _ => return Ok(None),
                },
                _ => return Ok(None),
            }
        }
        let (Some(data), Some(big_endian)) = (data, big_endian) else {
            return Ok(None);
        };
        let Expr::Call(to_bytes_callee, to_bytes_args) = &data.node else {
            return Ok(None);
        };
        let Expr::Identifier(to_bytes_id) = &to_bytes_callee.node else {
            return Ok(None);
        };
        if to_bytes_id.node.ident != "to_bytes"
            || to_bytes_args.args.len() != 1
            || !matches!(to_bytes_args.args[0].node, ArgumentP::Positional(_))
        {
            return Ok(None);
        }
        let input = to_bytes_args.args[0].expr();

        let string = self.compile_expr(input)?;
        let payload = self.b.tmp();
        self.b
            .ins(format!("{payload} = extractvalue %Value {string}, 1"));
        let strp = self.b.tmp();
        self.b
            .ins(format!("{strp} = inttoptr i64 {payload} to ptr"));
        let lenp = self.b.tmp();
        self.b.ins(format!(
            "{lenp} = getelementptr %SkString, ptr {strp}, i32 0, i32 0"
        ));
        let len = self.b.tmp();
        self.b.ins(format!("{len} = load i64, ptr {lenp}"));
        let datap = self.b.tmp();
        self.b.ins(format!(
            "{datap} = getelementptr %SkString, ptr {strp}, i32 0, i32 1, i64 0"
        ));

        let block = if let Some(expr) = block_size {
            self.compile_int_operand(expr)?
        } else {
            "64".to_owned()
        };
        let be_value = self.compile_expr(big_endian)?;
        let be = self.compile_truthy(&be_value);

        let length_size = self.b.tmp();
        self.b.ins(format!("{length_size} = sdiv i64 {block}, 8"));
        let rem = self.b.tmp();
        self.b.ins(format!("{rem} = sub i64 {block}, 1"));
        let rem = {
            let t = self.b.tmp();
            self.b.ins(format!("{t} = sub i64 {rem}, {length_size}"));
            t
        };
        let rem = {
            let t = self.b.tmp();
            self.b.ins(format!("{t} = sub i64 {rem}, {len}"));
            t
        };
        let zeros = self.b.tmp();
        self.b.ins(format!(
            "{zeros} = call i64 @sk_mod_i64(i64 {rem}, i64 {block})"
        ));
        let total = self.b.tmp();
        self.b.ins(format!("{total} = add i64 {len}, 1"));
        let total = {
            let t = self.b.tmp();
            self.b.ins(format!("{t} = add i64 {total}, {zeros}"));
            t
        };
        let total = {
            let t = self.b.tmp();
            self.b.ins(format!("{t} = add i64 {total}, {length_size}"));
            t
        };
        let out = self.b.tmp();
        self.b.ins(format!(
            "{out} = call %Value @sk_int_list_new_from(i64 {total}, ptr null)"
        ));
        let out_payload = self.b.tmp();
        self.b
            .ins(format!("{out_payload} = extractvalue %Value {out}, 1"));
        let out_list = self.b.tmp();
        self.b
            .ins(format!("{out_list} = inttoptr i64 {out_payload} to ptr"));
        let itemsp = self.b.tmp();
        self.b.ins(format!(
            "{itemsp} = getelementptr %SkList, ptr {out_list}, i32 0, i32 2"
        ));
        let items = self.b.tmp();
        self.b.ins(format!("{items} = load ptr, ptr {itemsp}"));

        // Copy the input bytes into the compact integer list.
        let i_cell = self.b.tmp();
        self.b.ins(format!("{i_cell} = alloca i64"));
        self.b.ins(format!("store i64 0, ptr {i_cell}"));
        let copy_loop = self.b.label();
        let copy_body = self.b.label();
        let copy_inc = self.b.label();
        let copy_end = self.b.label();
        self.b.br(&copy_loop);
        self.b.emit_label(&copy_loop);
        let i = self.b.tmp();
        self.b.ins(format!("{i} = load i64, ptr {i_cell}"));
        let copy_cond = self.b.tmp();
        self.b.ins(format!("{copy_cond} = icmp slt i64 {i}, {len}"));
        self.b.cond_br(&copy_cond, &copy_body, &copy_end);
        self.b.emit_label(&copy_body);
        let bytep = self.b.tmp();
        self.b
            .ins(format!("{bytep} = getelementptr i8, ptr {datap}, i64 {i}"));
        let byte8 = self.b.tmp();
        self.b.ins(format!("{byte8} = load i8, ptr {bytep}"));
        let byte = self.b.tmp();
        self.b.ins(format!("{byte} = zext i8 {byte8} to i64"));
        let destp = self.b.tmp();
        self.b
            .ins(format!("{destp} = getelementptr i64, ptr {items}, i64 {i}"));
        self.b.ins(format!("store i64 {byte}, ptr {destp}"));
        self.b.br(&copy_inc);
        self.b.emit_label(&copy_inc);
        let next = self.b.tmp();
        self.b.ins(format!("{next} = add i64 {i}, 1"));
        self.b.ins(format!("store i64 {next}, ptr {i_cell}"));
        self.b.br(&copy_loop);
        self.b.emit_label(&copy_end);

        // Append the 0x80 terminator.
        let termp = self.b.tmp();
        self.b.ins(format!(
            "{termp} = getelementptr i64, ptr {items}, i64 {len}"
        ));
        self.b.ins(format!("store i64 128, ptr {termp}"));

        // Zero padding.
        let j_cell = self.b.tmp();
        self.b.ins(format!("{j_cell} = alloca i64"));
        self.b.ins(format!("store i64 0, ptr {j_cell}"));
        let zero_loop = self.b.label();
        let zero_body = self.b.label();
        let zero_inc = self.b.label();
        let zero_end = self.b.label();
        self.b.br(&zero_loop);
        self.b.emit_label(&zero_loop);
        let j = self.b.tmp();
        self.b.ins(format!("{j} = load i64, ptr {j_cell}"));
        let zero_cond = self.b.tmp();
        self.b
            .ins(format!("{zero_cond} = icmp slt i64 {j}, {zeros}"));
        self.b.cond_br(&zero_cond, &zero_body, &zero_end);
        self.b.emit_label(&zero_body);
        let zbase = self.b.tmp();
        self.b.ins(format!("{zbase} = add i64 {len}, 1"));
        let zidx = self.b.tmp();
        self.b.ins(format!("{zidx} = add i64 {zbase}, {j}"));
        let zp = self.b.tmp();
        self.b
            .ins(format!("{zp} = getelementptr i64, ptr {items}, i64 {zidx}"));
        self.b.ins(format!("store i64 0, ptr {zp}"));
        self.b.br(&zero_inc);
        self.b.emit_label(&zero_inc);
        let next = self.b.tmp();
        self.b.ins(format!("{next} = add i64 {j}, 1"));
        self.b.ins(format!("store i64 {next}, ptr {j_cell}"));
        self.b.br(&zero_loop);
        self.b.emit_label(&zero_end);

        // Append the bit length.  `length_size` is at most 16, so this loop is
        // tiny and LLVM can fully unroll it.
        let k_cell = self.b.tmp();
        self.b.ins(format!("{k_cell} = alloca i64"));
        self.b.ins(format!("store i64 0, ptr {k_cell}"));
        let len_loop = self.b.label();
        let len_body = self.b.label();
        let len_inc = self.b.label();
        let len_end = self.b.label();
        self.b.br(&len_loop);
        self.b.emit_label(&len_loop);
        let k = self.b.tmp();
        self.b.ins(format!("{k} = load i64, ptr {k_cell}"));
        let len_cond = self.b.tmp();
        self.b
            .ins(format!("{len_cond} = icmp slt i64 {k}, {length_size}"));
        self.b.cond_br(&len_cond, &len_body, &len_end);
        self.b.emit_label(&len_body);
        let bits = self.b.tmp();
        self.b.ins(format!("{bits} = mul i64 {len}, 8"));
        let shift = self.b.tmp();
        self.b.ins(format!("{shift} = mul i64 {k}, 8"));
        let shifted = self.b.tmp();
        self.b.ins(format!("{shifted} = lshr i64 {bits}, {shift}"));
        let byte = self.b.tmp();
        self.b.ins(format!("{byte} = and i64 {shifted}, 255"));
        let rev = self.b.tmp();
        self.b.ins(format!("{rev} = sub i64 {length_size}, 1"));
        let rev = {
            let t = self.b.tmp();
            self.b.ins(format!("{t} = sub i64 {rev}, {k}"));
            t
        };
        let idx = self.b.tmp();
        self.b
            .ins(format!("{idx} = select i1 {be}, i64 {rev}, i64 {k}"));
        let lbase = self.b.tmp();
        self.b.ins(format!("{lbase} = add i64 {len}, 1"));
        let lbase = {
            let t = self.b.tmp();
            self.b.ins(format!("{t} = add i64 {lbase}, {zeros}"));
            t
        };
        let lidx = self.b.tmp();
        self.b.ins(format!("{lidx} = add i64 {lbase}, {idx}"));
        let lp = self.b.tmp();
        self.b
            .ins(format!("{lp} = getelementptr i64, ptr {items}, i64 {lidx}"));
        self.b.ins(format!("store i64 {byte}, ptr {lp}"));
        self.b.br(&len_inc);
        self.b.emit_label(&len_inc);
        let next = self.b.tmp();
        self.b.ins(format!("{next} = add i64 {k}, 1"));
        self.b.ins(format!("store i64 {next}, ptr {k_cell}"));
        self.b.br(&len_loop);
        self.b.emit_label(&len_end);

        Ok(Some(out))
    }

    fn compile_expr(&mut self, expr: &AstExpr) -> Result<String> {
        if let Some(v) = self.try_compile_md_pad_from_string(expr)? {
            return Ok(v);
        }
        if self.expr_ty(expr) == crate::types::Ty::Int {
            if let Some(v) = self.compile_int_expr(expr)? {
                return Ok(self.box_int(&v));
            }
        }
        match &expr.node {
            Expr::Tuple(items) => self.compile_sequence(items, false, false),
            Expr::List(items) => {
                let int_list = self.expr_is_list_int(expr);
                self.compile_sequence(items, true, int_list)
            }
            Expr::Dict(items) => {
                let dict = self.b.tmp();
                self.b.ins(format!("{dict} = call %Value @sk_dict_new()"));
                for (k, v) in items {
                    let key = self.compile_expr(k)?;
                    let val = self.compile_expr(v)?;
                    self.b.ins(format!(
                        "call void @sk_dict_set(%Value {dict}, %Value {key}, %Value {val})"
                    ));
                }
                Ok(dict)
            }
            Expr::Dot(obj, name) => {
                let obj = self.compile_expr(obj)?;
                let name = self.compile_string_value(&name.node)?;
                let v = self.b.tmp();
                self.b.ins(format!(
                    "{v} = call %Value @sk_getattr(%Value {obj}, %Value {name})"
                ));
                Ok(v)
            }
            Expr::Call(callee, args) => self.compile_call(callee, args),
            Expr::Index(pair) => {
                let obj = self.compile_expr(&pair.0)?;
                let idx = self.compile_expr(&pair.1)?;
                let v = self.b.tmp();
                self.b.ins(format!(
                    "{v} = call %Value @sk_index(%Value {obj}, %Value {idx})"
                ));
                Ok(v)
            }
            Expr::Index2(pair) => {
                let obj = self.compile_expr(&pair.0)?;
                let a = self.compile_expr(&pair.1)?;
                let b = self.compile_expr(&pair.2)?;
                let key = self.compile_tuple_values(&[a, b])?;
                let v = self.b.tmp();
                self.b.ins(format!(
                    "{v} = call %Value @sk_index(%Value {obj}, %Value {key})"
                ));
                Ok(v)
            }
            Expr::Slice(obj, start, stop, step) => {
                if self.expr_is_list_int(obj) && opt_enabled("STARLARKC_INT_SLICE") {
                    let obj = self.compile_expr(obj)?;
                    let (start, hs) = self.compile_optional_int_expr(start.as_deref())?;
                    let (stop, he) = self.compile_optional_int_expr(stop.as_deref())?;
                    let (step, hstep) = self.compile_optional_int_expr(step.as_deref())?;
                    let v = self.b.tmp();
                    self.b.ins(format!(
                        "{v} = call %Value @sk_int_list_slice(%Value {obj}, i64 {start}, i64 {stop}, i64 {step}, i64 {hs}, i64 {he}, i64 {hstep})"
                    ));
                    return Ok(v);
                }
                let obj = self.compile_expr(obj)?;
                let (start, hs) = self.compile_optional_expr(start.as_deref())?;
                let (stop, he) = self.compile_optional_expr(stop.as_deref())?;
                let (step, hstep) = self.compile_optional_expr(step.as_deref())?;
                let v = self.b.tmp();
                self.b.ins(format!(
                    "{v} = call %Value @sk_slice(%Value {obj}, %Value {start}, %Value {stop}, %Value {step}, i64 {hs}, i64 {he}, i64 {hstep})"
                ));
                Ok(v)
            }
            Expr::Identifier(id) => self.resolve_value(&id.node.ident),
            Expr::Lambda(l) => self.compile_lambda(l),
            Expr::Literal(lit) => match lit {
                AstLiteral::Int(i) => {
                    let value = match &i.node {
                        TokenInt::I32(v) => *v as i64,
                        TokenInt::BigInt(v) => {
                            let text = v.to_string();
                            text.parse::<i64>()
                                .or_else(|_| text.parse::<u64>().map(|u| u as i64))
                                .map_err(|_| {
                                    anyhow!("integer literal does not fit in 64 bits: {v}")
                                })?
                        }
                    };
                    self.compile_int(value)
                }
                AstLiteral::Float(f) => self.compile_float(f.node),
                AstLiteral::String(s) => self.compile_string_value(&s.node),
                AstLiteral::Bytes(b) => {
                    let s = String::from_utf8_lossy(&b.node).into_owned();
                    self.compile_string_value(&s)
                }
                AstLiteral::Ellipsis => self.compile_string_value("..."),
            },
            Expr::Not(e) => {
                let v = self.compile_expr(e)?;
                let out = self.b.tmp();
                self.b
                    .ins(format!("{out} = call %Value @sk_not(%Value {v})"));
                Ok(out)
            }
            Expr::Minus(e) => {
                let v = self.compile_expr(e)?;
                let out = self.b.tmp();
                self.b
                    .ins(format!("{out} = call %Value @sk_neg(%Value {v})"));
                Ok(out)
            }
            Expr::Plus(e) => {
                let v = self.compile_expr(e)?;
                let out = self.b.tmp();
                self.b
                    .ins(format!("{out} = call %Value @sk_pos(%Value {v})"));
                Ok(out)
            }
            Expr::BitNot(e) => {
                let v = self.compile_expr(e)?;
                let out = self.b.tmp();
                self.b
                    .ins(format!("{out} = call %Value @sk_bitnot(%Value {v})"));
                Ok(out)
            }
            Expr::Op(lhs, op, rhs) => match op {
                BinOp::Multiply
                    if matches!(&lhs.node, Expr::List(items) if items.len() == 1
                        && const_int_ast(&items[0]) == Some(0)) =>
                {
                    let const_n = self.const_int_expr(rhs);
                    let n = self.compile_int_operand(rhs)?;
                    let out = self.b.tmp();
                    self.b
                        .ins(format!("{out} = call %Value @sk_int_list_zeros(i64 {n})"));
                    if let Some(n) = const_n {
                        self.value_list_lens.insert(out.clone(), n);
                    }
                    Ok(out)
                }
                BinOp::Multiply
                    if matches!(&rhs.node, Expr::List(items) if items.len() == 1
                        && const_int_ast(&items[0]) == Some(0)) =>
                {
                    let const_n = self.const_int_expr(lhs);
                    let n = self.compile_int_operand(lhs)?;
                    let out = self.b.tmp();
                    self.b
                        .ins(format!("{out} = call %Value @sk_int_list_zeros(i64 {n})"));
                    if let Some(n) = const_n {
                        self.value_list_lens.insert(out.clone(), n);
                    }
                    Ok(out)
                }
                BinOp::And => self.compile_and(lhs, rhs),
                BinOp::Or => self.compile_or(lhs, rhs),
                BinOp::Add if self.expr_is_list_int(expr) && opt_enabled("STARLARKC_CONCAT") => {
                    self.compile_int_list_add_chain(expr)
                }
                BinOp::Equal | BinOp::NotEqual
                    if (self.expr_is_none(lhs) && self.expr_is_int(rhs))
                        || (self.expr_is_none(rhs) && self.expr_is_int(lhs)) =>
                {
                    self.compile_bool(*op == BinOp::NotEqual)
                }
                _ => {
                    let l = self.compile_expr(lhs)?;
                    let r = self.compile_expr(rhs)?;
                    self.compile_binop(*op, &l, &r)
                }
            },
            Expr::If(pair) => {
                let result = self.b.tmp();
                self.b.ins(format!("{result} = alloca %Value"));
                let cond = self.compile_expr(&pair.0)?;
                let truth = self.compile_truthy(&cond);
                let then_label = self.b.label();
                let else_label = self.b.label();
                let end_label = self.b.label();
                self.b.cond_br(&truth, &then_label, &else_label);
                self.b.emit_label(&then_label);
                let a = self.compile_expr(&pair.1)?;
                self.b.ins(format!("store %Value {a}, ptr {result}"));
                self.b.br(&end_label);
                self.b.emit_label(&else_label);
                let b = self.compile_expr(&pair.2)?;
                self.b.ins(format!("store %Value {b}, ptr {result}"));
                self.b.br(&end_label);
                self.b.emit_label(&end_label);
                let out = self.b.tmp();
                self.b.ins(format!("{out} = load %Value, ptr {result}"));
                Ok(out)
            }
            Expr::ListComprehension(body, first, rest) => {
                let int_list = self.expr_is_list_int(expr);
                let stack_len = if int_list {
                    self.stack_list_hint.take()
                } else {
                    None
                };
                let capacity = if int_list && stack_len.is_none() {
                    self.comprehension_capacity(first, rest)?
                } else {
                    None
                };
                let exact_capacity = capacity.is_some();
                let mut stack_items = None;
                let mut stack_idx = None;
                let list = if let Some(n) = stack_len {
                    let hdr = self.b.tmp();
                    self.b.ins(format!("{hdr} = alloca %SkList"));
                    let arr = self.b.tmp();
                    self.b.ins(format!("{arr} = alloca [{n} x i64]"));
                    let lenp = self.b.tmp();
                    self.b.ins(format!(
                        "{lenp} = getelementptr %SkList, ptr {hdr}, i32 0, i32 0"
                    ));
                    self.b.ins(format!("store i64 {n}, ptr {lenp}"));
                    let capp = self.b.tmp();
                    self.b.ins(format!(
                        "{capp} = getelementptr %SkList, ptr {hdr}, i32 0, i32 1"
                    ));
                    self.b.ins(format!("store i64 {n}, ptr {capp}"));
                    let itemsp = self.b.tmp();
                    self.b.ins(format!(
                        "{itemsp} = getelementptr %SkList, ptr {hdr}, i32 0, i32 2"
                    ));
                    let items0 = self.b.tmp();
                    self.b.ins(format!(
                        "{items0} = getelementptr [{n} x i64], ptr {arr}, i64 0, i64 0"
                    ));
                    self.b.ins(format!("store ptr {items0}, ptr {itemsp}"));
                    let payload = self.b.tmp();
                    self.b.ins(format!("{payload} = ptrtoint ptr {hdr} to i64"));
                    let v0 = self.b.tmp();
                    self.b
                        .ins(format!("{v0} = insertvalue %Value undef, i64 16, 0"));
                    let out = self.b.tmp();
                    self.b
                        .ins(format!("{out} = insertvalue %Value {v0}, i64 {payload}, 1"));
                    let idx_cell = self.b.tmp();
                    self.b.ins(format!("{idx_cell} = alloca i64"));
                    self.b.ins(format!("store i64 0, ptr {idx_cell}"));
                    stack_items = Some(items0);
                    stack_idx = Some(idx_cell);
                    self.value_list_lens.insert(out.clone(), n);
                    out
                } else {
                    let list = self.b.tmp();
                    if int_list {
                        if let Some(capacity) = &capacity {
                            self.b.ins(format!(
                                "{list} = call %Value @sk_int_list_new_cap(i64 {capacity})"
                            ));
                        } else {
                            self.b
                                .ins(format!("{list} = call %Value @sk_int_list_new()"));
                        }
                    } else {
                        self.b.ins(format!("{list} = call %Value @sk_list_new()"));
                    }
                    if let Some(n) = self.comprehension_const_len(first, rest) {
                        self.value_list_lens.insert(list.clone(), n);
                    }
                    list
                };
                let saved = self.locals.clone();
                let clauses = build_clauses(first, rest);
                let mut body_fn = |this: &mut Self| -> Result<()> {
                    if int_list {
                        let v = this.compile_int_operand(body)?;
                        if let (Some(items), Some(idx_cell)) = (&stack_items, &stack_idx) {
                            let idx = this.b.tmp();
                            this.b.ins(format!("{idx} = load i64, ptr {idx_cell}"));
                            let p = this.b.tmp();
                            this.b
                                .ins(format!("{p} = getelementptr i64, ptr {items}, i64 {idx}"));
                            this.b.ins(format!("store i64 {v}, ptr {p}"));
                            let next = this.b.tmp();
                            this.b.ins(format!("{next} = add i64 {idx}, 1"));
                            this.b.ins(format!("store i64 {next}, ptr {idx_cell}"));
                        } else if exact_capacity {
                            this.compile_int_list_push_unchecked(&list, &v);
                        } else {
                            this.compile_int_list_push(&list, &v);
                        }
                    } else {
                        let v = this.compile_expr(body)?;
                        this.b.ins(format!(
                            "call void @sk_list_push(%Value {list}, %Value {v})"
                        ));
                    }
                    Ok(())
                };
                self.compile_clauses(&clauses, &mut body_fn)?;
                self.locals = saved;
                Ok(list)
            }
            Expr::DictComprehension(pair, first, rest) => {
                let dict = self.b.tmp();
                self.b.ins(format!("{dict} = call %Value @sk_dict_new()"));
                let saved = self.locals.clone();
                let clauses = build_clauses(first, rest);
                let mut body_fn = |this: &mut Self| -> Result<()> {
                    let k = this.compile_expr(&pair.0)?;
                    let v = this.compile_expr(&pair.1)?;
                    this.b.ins(format!(
                        "call void @sk_dict_set(%Value {dict}, %Value {k}, %Value {v})"
                    ));
                    Ok(())
                };
                self.compile_clauses(&clauses, &mut body_fn)?;
                self.locals = saved;
                Ok(dict)
            }
            Expr::FString(f) => self.compile_fstring(f),
        }
    }

    fn eval_linear_expr(&self, expr: &AstExpr, var: &str, value: i64) -> Option<i64> {
        match &expr.node {
            Expr::Literal(AstLiteral::Int(i)) => match &i.node {
                TokenInt::I32(v) => Some(*v as i64),
                TokenInt::BigInt(v) => v.to_string().parse::<i64>().ok(),
            },
            Expr::Identifier(id) if id.node.ident == var => Some(value),
            Expr::Identifier(id) => self.const_locals.get(&id.node.ident).copied(),
            Expr::Minus(e) => self.eval_linear_expr(e, var, value).map(|v| -v),
            Expr::Plus(e) => self.eval_linear_expr(e, var, value),
            Expr::Op(lhs, op, rhs) => {
                let l = self.eval_linear_expr(lhs, var, value)?;
                let r = self.eval_linear_expr(rhs, var, value)?;
                match op {
                    BinOp::Add => Some(l.wrapping_add(r)),
                    BinOp::Subtract => Some(l.wrapping_sub(r)),
                    BinOp::Multiply => Some(l.wrapping_mul(r)),
                    BinOp::BitAnd => Some(l & r),
                    BinOp::BitOr => Some(l | r),
                    BinOp::BitXor => Some(l ^ r),
                    BinOp::LeftShift => Some(l.wrapping_shl(r as u32)),
                    BinOp::RightShift => Some(l.wrapping_shr(r as u32)),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn collect_expr_identifiers(&self, expr: &AstExpr, out: &mut HashSet<String>) {
        match &expr.node {
            Expr::Identifier(id) => {
                if !self.const_locals.contains_key(&id.node.ident) {
                    out.insert(id.node.ident.clone());
                }
            }
            Expr::Literal(_) => {}
            Expr::Tuple(items) | Expr::List(items) => {
                for e in items {
                    self.collect_expr_identifiers(e, out);
                }
            }
            Expr::Dict(items) => {
                for (k, v) in items {
                    self.collect_expr_identifiers(k, out);
                    self.collect_expr_identifiers(v, out);
                }
            }
            Expr::Dot(obj, _) => self.collect_expr_identifiers(obj, out),
            Expr::Call(callee, args) => {
                self.collect_expr_identifiers(callee, out);
                for a in &args.args {
                    self.collect_expr_identifiers(a.expr(), out);
                }
            }
            Expr::Index(pair) => {
                self.collect_expr_identifiers(&pair.0, out);
                self.collect_expr_identifiers(&pair.1, out);
            }
            Expr::Index2(pair) => {
                self.collect_expr_identifiers(&pair.0, out);
                self.collect_expr_identifiers(&pair.1, out);
                self.collect_expr_identifiers(&pair.2, out);
            }
            Expr::Slice(obj, start, stop, step) => {
                self.collect_expr_identifiers(obj, out);
                for e in [start, stop, step].into_iter().flatten() {
                    self.collect_expr_identifiers(e, out);
                }
            }
            Expr::Lambda(_) => {}
            Expr::Not(e) | Expr::Minus(e) | Expr::Plus(e) | Expr::BitNot(e) => {
                self.collect_expr_identifiers(e, out);
            }
            Expr::Op(lhs, _, rhs) => {
                self.collect_expr_identifiers(lhs, out);
                self.collect_expr_identifiers(rhs, out);
            }
            Expr::If(triple) => {
                self.collect_expr_identifiers(&triple.0, out);
                self.collect_expr_identifiers(&triple.1, out);
                self.collect_expr_identifiers(&triple.2, out);
            }
            Expr::ListComprehension(body, first, rest) => {
                self.collect_expr_identifiers(body, out);
                self.collect_expr_identifiers(&first.over, out);
                for c in rest {
                    match c {
                        Clause::For(f) => self.collect_expr_identifiers(&f.over, out),
                        Clause::If(e) => self.collect_expr_identifiers(e, out),
                    }
                }
            }
            Expr::DictComprehension(pair, first, rest) => {
                self.collect_expr_identifiers(&pair.0, out);
                self.collect_expr_identifiers(&pair.1, out);
                self.collect_expr_identifiers(&first.over, out);
                for c in rest {
                    match c {
                        Clause::For(f) => self.collect_expr_identifiers(&f.over, out),
                        Clause::If(e) => self.collect_expr_identifiers(e, out),
                    }
                }
            }
            Expr::FString(f) => {
                for e in &f.expressions {
                    self.collect_expr_identifiers(e, out);
                }
            }
        }
    }

    fn static_slice_len(&self, start: Option<&AstExpr>, stop: Option<&AstExpr>) -> Option<i64> {
        let mut ids = HashSet::new();
        if let Some(e) = start {
            self.collect_expr_identifiers(e, &mut ids);
        }
        if let Some(e) = stop {
            self.collect_expr_identifiers(e, &mut ids);
        }
        if ids.len() > 1 {
            return None;
        }
        let var = ids.iter().next().map(String::as_str).unwrap_or("");
        let eval = |e: Option<&AstExpr>, v: i64| -> Option<i64> {
            match e {
                Some(e) => self.eval_linear_expr(e, var, v),
                None => Some(0),
            }
        };
        let s1 = eval(start, 1)?;
        let e1 = eval(stop, 1)?;
        let s0 = eval(start, 0)?;
        let e0 = eval(stop, 0)?;
        if e1.wrapping_sub(s1) == e0.wrapping_sub(s0) {
            Some(e1.wrapping_sub(s1))
        } else {
            None
        }
    }

    fn normalize_slice_bound(&mut self, idx: &str, len: &str) -> String {
        let neg = self.b.tmp();
        self.b.ins(format!("{neg} = icmp slt i64 {idx}, 0"));
        let adjusted = self.b.tmp();
        self.b.ins(format!("{adjusted} = add i64 {idx}, {len}"));
        let norm = self.b.tmp();
        self.b.ins(format!(
            "{norm} = select i1 {neg}, i64 {adjusted}, i64 {idx}"
        ));
        let below = self.b.tmp();
        self.b.ins(format!("{below} = icmp slt i64 {norm}, 0"));
        let above = self.b.tmp();
        self.b.ins(format!("{above} = icmp sgt i64 {norm}, {len}"));
        let low = self.b.tmp();
        self.b
            .ins(format!("{low} = select i1 {below}, i64 0, i64 {norm}"));
        let out = self.b.tmp();
        self.b
            .ins(format!("{out} = select i1 {above}, i64 {len}, i64 {low}"));
        out
    }

    fn compile_slice_bounds(
        &mut self,
        len: &str,
        start: Option<&AstExpr>,
        stop: Option<&AstExpr>,
    ) -> Result<(String, String)> {
        let start = match start {
            Some(e) => {
                let raw = self.compile_int_operand(e)?;
                self.normalize_slice_bound(&raw, len)
            }
            None => "0".to_owned(),
        };
        let stop = match stop {
            Some(e) => {
                let raw = self.compile_int_operand(e)?;
                self.normalize_slice_bound(&raw, len)
            }
            None => len.to_owned(),
        };
        let diff = self.b.tmp();
        self.b.ins(format!("{diff} = sub i64 {stop}, {start}"));
        let neg = self.b.tmp();
        self.b.ins(format!("{neg} = icmp slt i64 {diff}, 0"));
        let count = self.b.tmp();
        self.b
            .ins(format!("{count} = select i1 {neg}, i64 0, i64 {diff}"));
        Ok((start, count))
    }

    /// Recognize `words_be(data) + [0] * n` / `words_le(data) + [0] * n`.
    /// The generic lowering builds four lists (slice, packed words, zero pad,
    /// and the concatenation).  Emitting the packing loop directly lets LLVM
    /// see the byte loads and the output stores, and reduces the hot SHA
    /// compression loop to one allocation per block.
    fn try_compile_words_padded(&mut self, expr: &AstExpr, stack: bool) -> Result<Option<String>> {
        // Accept both `words*(data) + [0] * n` and a bare `words*(data)`.
        let (lhs, pad) = match &expr.node {
            Expr::Op(lhs, BinOp::Add, rhs) => {
                let Some(pad) = self.zero_pad_count(rhs) else {
                    return Ok(None);
                };
                (lhs.as_ref(), pad)
            }
            Expr::Call(_, _) => (expr, 0),
            _ => return Ok(None),
        };
        let Expr::Call(callee, args) = &lhs.node else {
            return Ok(None);
        };
        let Expr::Identifier(callee_id) = &callee.node else {
            return Ok(None);
        };
        let (big_endian, width) = match callee_id.node.ident.as_str() {
            "words_be" => (true, 4i64),
            "words_le" => (false, 4),
            "words64_be" => (true, 8),
            "words64_le" => (false, 8),
            _ => return Ok(None),
        };
        if args.args.len() != 1 || !matches!(args.args[0].node, ArgumentP::Positional(_)) {
            return Ok(None);
        }

        let data_expr = args.args[0].expr();
        let static_len = if let Expr::Slice(_, start, stop, step) = &data_expr.node {
            let simple_step = match step.as_deref() {
                None => true,
                Some(e) => const_int_ast(e) == Some(1),
            };
            if simple_step {
                self.static_slice_len(start.as_deref(), stop.as_deref())
            } else {
                None
            }
        } else {
            None
        };
        let (base, len, items) = if let Expr::Slice(obj, start, stop, step) = &data_expr.node {
            let simple_step = match step.as_deref() {
                None => true,
                Some(e) => const_int_ast(e) == Some(1),
            };
            if simple_step && self.expr_is_list_int(obj) {
                let obj = self.compile_expr(obj)?;
                let (obj_len, obj_items) = self.compile_int_list_view(None, &obj);
                let (start, len) =
                    self.compile_slice_bounds(&obj_len, start.as_deref(), stop.as_deref())?;
                (start, len, obj_items)
            } else {
                let data = self.compile_expr(data_expr)?;
                let (len, items) = self.compile_int_list_view(None, &data);
                ("0".to_owned(), len, items)
            }
        } else {
            let data = self.compile_expr(data_expr)?;
            let (len, items) = self.compile_int_list_view(None, &data);
            ("0".to_owned(), len, items)
        };
        let rem = self.b.tmp();
        self.b.ins(format!("{rem} = srem i64 {len}, {width}"));
        let aligned = self.b.tmp();
        self.b.ins(format!("{aligned} = icmp eq i64 {rem}, 0"));
        let ok = self.b.label();
        let bad = self.b.label();
        self.b.cond_br(&aligned, &ok, &bad);
        self.b.emit_label(&bad);
        self.emit_error("words_be/words_le input length must be a multiple of 4");
        self.b.emit_label(&ok);

        let stack_total = if stack {
            static_len.and_then(|n| {
                if n % width == 0 {
                    let total = n / width + pad;
                    (total <= 4096).then_some(total)
                } else {
                    None
                }
            })
        } else {
            None
        };
        let (words, out, out_items) = if let Some(total) = stack_total {
            let hdr = self.b.tmp();
            self.b.ins(format!("{hdr} = alloca %SkList"));
            let arr = self.b.tmp();
            self.b.ins(format!("{arr} = alloca [{total} x i64]"));
            let lenp = self.b.tmp();
            self.b.ins(format!(
                "{lenp} = getelementptr %SkList, ptr {hdr}, i32 0, i32 0"
            ));
            self.b.ins(format!("store i64 {total}, ptr {lenp}"));
            let capp = self.b.tmp();
            self.b.ins(format!(
                "{capp} = getelementptr %SkList, ptr {hdr}, i32 0, i32 1"
            ));
            self.b.ins(format!("store i64 {total}, ptr {capp}"));
            let itemsp = self.b.tmp();
            self.b.ins(format!(
                "{itemsp} = getelementptr %SkList, ptr {hdr}, i32 0, i32 2"
            ));
            let items0 = self.b.tmp();
            self.b.ins(format!(
                "{items0} = getelementptr [{total} x i64], ptr {arr}, i64 0, i64 0"
            ));
            self.b.ins(format!("store ptr {items0}, ptr {itemsp}"));
            let payload = self.b.tmp();
            self.b.ins(format!("{payload} = ptrtoint ptr {hdr} to i64"));
            let v0 = self.b.tmp();
            self.b
                .ins(format!("{v0} = insertvalue %Value undef, i64 16, 0"));
            let out = self.b.tmp();
            self.b
                .ins(format!("{out} = insertvalue %Value {v0}, i64 {payload}, 1"));
            self.value_list_lens.insert(out.clone(), total);
            ((total - pad).to_string(), out, items0)
        } else {
            let words = self.b.tmp();
            self.b.ins(format!("{words} = sdiv i64 {len}, {width}"));
            let total = self.b.tmp();
            self.b.ins(format!("{total} = add i64 {words}, {pad}"));
            let out = self.b.tmp();
            self.b.ins(format!(
                "{out} = call %Value @sk_int_list_new_from(i64 {total}, ptr null)"
            ));
            let out_payload = self.b.tmp();
            self.b
                .ins(format!("{out_payload} = extractvalue %Value {out}, 1"));
            let out_list = self.b.tmp();
            self.b
                .ins(format!("{out_list} = inttoptr i64 {out_payload} to ptr"));
            let out_itemsp = self.b.tmp();
            self.b.ins(format!(
                "{out_itemsp} = getelementptr %SkList, ptr {out_list}, i32 0, i32 2"
            ));
            let out_items = self.b.tmp();
            self.b
                .ins(format!("{out_items} = load ptr, ptr {out_itemsp}"));
            (words, out, out_items)
        };

        // Iterate over output words rather than over input bytes.  A linear
        // word counter gives the vectorizer a canonical induction variable and
        // a contiguous output store; the four input loads are a stride-4
        // group that LLVM turns into an interleaved gather.
        let k_cell = self.b.tmp();
        self.b.ins(format!("{k_cell} = alloca i64"));
        self.b.ins(format!("store i64 0, ptr {k_cell}"));
        let loop_label = self.b.label();
        let body_label = self.b.label();
        let inc_label = self.b.label();
        let end_label = self.b.label();
        self.b.br(&loop_label);
        self.b.emit_label(&loop_label);
        let k = self.b.tmp();
        self.b.ins(format!("{k} = load i64, ptr {k_cell}"));
        let cond = self.b.tmp();
        self.b.ins(format!("{cond} = icmp slt i64 {k}, {words}"));
        self.b.cond_br(&cond, &body_label, &end_label);
        self.b.emit_label(&body_label);
        let i = self.b.tmp();
        self.b.ins(format!("{i} = mul i64 {k}, {width}"));
        let mut bytes = Vec::with_capacity(width as usize);
        for j in 0..width {
            let idx = self.b.tmp();
            self.b.ins(format!("{idx} = add i64 {base}, {i}"));
            let idx = {
                let t = self.b.tmp();
                self.b.ins(format!("{t} = add i64 {idx}, {j}"));
                t
            };
            let p = self.b.tmp();
            self.b
                .ins(format!("{p} = getelementptr i64, ptr {items}, i64 {idx}"));
            let v = self.b.tmp();
            self.b.ins(format!("{v} = load i64, ptr {p}"));
            bytes.push(v);
        }
        let mut word = if big_endian {
            let first = self.b.tmp();
            self.b.ins(format!(
                "{first} = shl i64 {}, {}",
                bytes[0],
                (width - 1) * 8
            ));
            first
        } else {
            bytes[0].clone()
        };
        for j in 1..width {
            let shift = if big_endian {
                (width - 1 - j) * 8
            } else {
                j * 8
            };
            let shifted = self.b.tmp();
            self.b.ins(format!(
                "{shifted} = shl i64 {}, {shift}",
                bytes[j as usize]
            ));
            let next = self.b.tmp();
            self.b.ins(format!("{next} = or i64 {word}, {shifted}"));
            word = next;
        }
        let wp = self.b.tmp();
        self.b.ins(format!(
            "{wp} = getelementptr i64, ptr {out_items}, i64 {k}"
        ));
        self.b.ins(format!("store i64 {word}, ptr {wp}"));
        self.b.br(&inc_label);
        self.b.emit_label(&inc_label);
        let next = self.b.tmp();
        self.b.ins(format!("{next} = add i64 {k}, 1"));
        self.b.ins(format!("store i64 {next}, ptr {k_cell}"));
        self.b.br(&loop_label);
        self.b.emit_label(&end_label);

        // Zero the padding tail.
        let j_cell = self.b.tmp();
        self.b.ins(format!("{j_cell} = alloca i64"));
        self.b.ins(format!("store i64 0, ptr {j_cell}"));
        let zloop = self.b.label();
        let zbody = self.b.label();
        let zinc = self.b.label();
        let zend = self.b.label();
        self.b.br(&zloop);
        self.b.emit_label(&zloop);
        let j = self.b.tmp();
        self.b.ins(format!("{j} = load i64, ptr {j_cell}"));
        let zcond = self.b.tmp();
        self.b.ins(format!("{zcond} = icmp slt i64 {j}, {pad}"));
        self.b.cond_br(&zcond, &zbody, &zend);
        self.b.emit_label(&zbody);
        let zidx = self.b.tmp();
        self.b.ins(format!("{zidx} = add i64 {words}, {j}"));
        let zp = self.b.tmp();
        self.b.ins(format!(
            "{zp} = getelementptr i64, ptr {out_items}, i64 {zidx}"
        ));
        self.b.ins(format!("store i64 0, ptr {zp}"));
        self.b.br(&zinc);
        self.b.emit_label(&zinc);
        let next = self.b.tmp();
        self.b.ins(format!("{next} = add i64 {j}, 1"));
        self.b.ins(format!("store i64 {next}, ptr {j_cell}"));
        self.b.br(&zloop);
        self.b.emit_label(&zend);
        Ok(Some(out))
    }

    fn zero_pad_count(&self, expr: &AstExpr) -> Option<i64> {
        match &expr.node {
            Expr::List(items) if items.iter().all(|e| const_int_ast(e) == Some(0)) => {
                Some(items.len() as i64)
            }
            Expr::Op(lhs, BinOp::Multiply, rhs) => {
                if matches!(&lhs.node, Expr::List(items) if items.len() == 1 && const_int_ast(&items[0]) == Some(0))
                {
                    self.const_int_expr(rhs)
                } else if matches!(&rhs.node, Expr::List(items) if items.len() == 1 && const_int_ast(&items[0]) == Some(0))
                {
                    self.const_int_expr(lhs)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn collect_int_list_add_leaves<'e>(&self, expr: &'e AstExpr, out: &mut Vec<&'e AstExpr>) {
        if let Expr::Op(lhs, BinOp::Add, rhs) = &expr.node {
            if self.expr_is_list_int(expr) {
                self.collect_int_list_add_leaves(lhs, out);
                self.collect_int_list_add_leaves(rhs, out);
                return;
            }
        }
        out.push(expr);
    }

    fn compile_int_list_add_chain(&mut self, expr: &AstExpr) -> Result<String> {
        if let Some(v) = self.try_compile_words_padded(expr, false)? {
            return Ok(v);
        }
        let mut leaves = Vec::new();
        self.collect_int_list_add_leaves(expr, &mut leaves);
        if leaves.len() <= 1 {
            return self.compile_expr(leaves[0]);
        }
        let arr = self.b.tmp();
        self.b
            .ins(format!("{arr} = alloca [{} x %Value]", leaves.len()));
        for (i, leaf) in leaves.iter().enumerate() {
            // The whole chain is known to be a ListInt, so every literal leaf
            // must use the compact representation.  Mixing a boxed literal
            // with compact `sk_int_list_concat` parts would make the result
            // boxed while callers still treat it as compact.
            let value = if let Expr::List(items) = &leaf.node {
                self.compile_sequence(items, true, true)?
            } else {
                self.compile_expr(leaf)?
            };
            let p = self.b.tmp();
            self.b.ins(format!(
                "{p} = getelementptr [{} x %Value], ptr {arr}, i64 0, i64 {i}",
                leaves.len()
            ));
            self.b.ins(format!("store %Value {value}, ptr {p}"));
        }
        let out = self.b.tmp();
        self.b.ins(format!(
            "{out} = call %Value @sk_int_list_concat(i64 {}, ptr {arr})",
            leaves.len()
        ));
        Ok(out)
    }

    fn compile_and(&mut self, lhs: &AstExpr, rhs: &AstExpr) -> Result<String> {
        let result = self.b.tmp();
        self.b.ins(format!("{result} = alloca %Value"));
        let left = self.compile_expr(lhs)?;
        self.b.ins(format!("store %Value {left}, ptr {result}"));
        let truth = self.compile_truthy(&left);
        let right_label = self.b.label();
        let end_label = self.b.label();
        self.b.cond_br(&truth, &right_label, &end_label);
        self.b.emit_label(&right_label);
        let right = self.compile_expr(rhs)?;
        self.b.ins(format!("store %Value {right}, ptr {result}"));
        self.b.br(&end_label);
        self.b.emit_label(&end_label);
        let out = self.b.tmp();
        self.b.ins(format!("{out} = load %Value, ptr {result}"));
        Ok(out)
    }

    fn compile_or(&mut self, lhs: &AstExpr, rhs: &AstExpr) -> Result<String> {
        let result = self.b.tmp();
        self.b.ins(format!("{result} = alloca %Value"));
        let left = self.compile_expr(lhs)?;
        self.b.ins(format!("store %Value {left}, ptr {result}"));
        let truth = self.compile_truthy(&left);
        let right_label = self.b.label();
        let end_label = self.b.label();
        self.b.cond_br(&truth, &end_label, &right_label);
        self.b.emit_label(&right_label);
        let right = self.compile_expr(rhs)?;
        self.b.ins(format!("store %Value {right}, ptr {result}"));
        self.b.br(&end_label);
        self.b.emit_label(&end_label);
        let out = self.b.tmp();
        self.b.ins(format!("{out} = load %Value, ptr {result}"));
        Ok(out)
    }

    fn compile_binop(&mut self, op: BinOp, left: &str, right: &str) -> Result<String> {
        let function = match op {
            BinOp::Add => "sk_add",
            BinOp::Subtract => "sk_sub",
            BinOp::Multiply => "sk_mul",
            BinOp::Divide => "sk_div",
            BinOp::FloorDivide => "sk_floordiv",
            BinOp::Percent => "sk_mod",
            BinOp::BitAnd => "sk_bitand",
            BinOp::BitOr => "sk_bitor",
            BinOp::BitXor => "sk_bitxor",
            BinOp::LeftShift => "sk_shl",
            BinOp::RightShift => "sk_shr",
            BinOp::Equal => "sk_eq",
            BinOp::NotEqual => "sk_ne",
            BinOp::Less => "sk_lt",
            BinOp::Greater => "sk_gt",
            BinOp::LessOrEqual => "sk_le",
            BinOp::GreaterOrEqual => "sk_ge",
            BinOp::In => "sk_contains",
            BinOp::NotIn => {
                let c = self.b.tmp();
                self.b.ins(format!(
                    "{c} = call %Value @sk_contains(%Value {left}, %Value {right})"
                ));
                let v = self.b.tmp();
                self.b.ins(format!("{v} = call %Value @sk_not(%Value {c})"));
                return Ok(v);
            }
            BinOp::And | BinOp::Or => unreachable!(),
        };
        let v = self.b.tmp();
        self.b.ins(format!(
            "{v} = call %Value @{function}(%Value {left}, %Value {right})"
        ));
        Ok(v)
    }

    fn compile_binop_value(&mut self, op: AssignOp, left: &str, right: &str) -> Result<String> {
        let binop = match op {
            AssignOp::Add => BinOp::Add,
            AssignOp::Subtract => BinOp::Subtract,
            AssignOp::Multiply => BinOp::Multiply,
            AssignOp::Divide => BinOp::Divide,
            AssignOp::FloorDivide => BinOp::FloorDivide,
            AssignOp::Percent => BinOp::Percent,
            AssignOp::BitAnd => BinOp::BitAnd,
            AssignOp::BitOr => BinOp::BitOr,
            AssignOp::BitXor => BinOp::BitXor,
            AssignOp::LeftShift => BinOp::LeftShift,
            AssignOp::RightShift => BinOp::RightShift,
        };
        self.compile_binop(binop, left, right)
    }

    /// Allocate a fixed-length int list on the stack and return its `%Value`
    /// plus the element pointer.
    fn emit_stack_int_list(&mut self, n: i64) -> (String, String) {
        let hdr = self.b.tmp();
        self.b.ins(format!("{hdr} = alloca %SkList"));
        let arr = self.b.tmp();
        self.b.ins(format!("{arr} = alloca [{n} x i64]"));
        let lenp = self.b.tmp();
        self.b.ins(format!(
            "{lenp} = getelementptr %SkList, ptr {hdr}, i32 0, i32 0"
        ));
        self.b.ins(format!("store i64 {n}, ptr {lenp}"));
        let capp = self.b.tmp();
        self.b.ins(format!(
            "{capp} = getelementptr %SkList, ptr {hdr}, i32 0, i32 1"
        ));
        self.b.ins(format!("store i64 {n}, ptr {capp}"));
        let itemsp = self.b.tmp();
        self.b.ins(format!(
            "{itemsp} = getelementptr %SkList, ptr {hdr}, i32 0, i32 2"
        ));
        let items0 = self.b.tmp();
        self.b.ins(format!(
            "{items0} = getelementptr [{n} x i64], ptr {arr}, i64 0, i64 0"
        ));
        self.b.ins(format!("store ptr {items0}, ptr {itemsp}"));
        let payload = self.b.tmp();
        self.b.ins(format!("{payload} = ptrtoint ptr {hdr} to i64"));
        let v0 = self.b.tmp();
        self.b
            .ins(format!("{v0} = insertvalue %Value undef, i64 16, 0"));
        let out = self.b.tmp();
        self.b
            .ins(format!("{out} = insertvalue %Value {v0}, i64 {payload}, 1"));
        (out, items0)
    }

    fn compile_sequence(
        &mut self,
        items: &[AstExpr],
        list: bool,
        int_list: bool,
    ) -> Result<String> {
        if items.is_empty() {
            let v = self.b.tmp();
            if int_list {
                self.b.ins(format!(
                    "{v} = call %Value @sk_int_list_new_from(i64 0, ptr null)"
                ));
            } else if list {
                self.b.ins(format!(
                    "{v} = call %Value @sk_list_new_from(i64 0, ptr null)"
                ));
            } else {
                self.b
                    .ins(format!("{v} = call %Value @sk_tuple_new(i64 0, ptr null)"));
            }
            return Ok(v);
        }
        if int_list {
            let arr = self.b.tmp();
            self.b
                .ins(format!("{arr} = alloca [{} x i64]", items.len()));
            for (i, item) in items.iter().enumerate() {
                let v = self.compile_int_operand(item)?;
                let p = self.b.tmp();
                self.b.ins(format!(
                    "{p} = getelementptr [{} x i64], ptr {arr}, i64 0, i64 {i}",
                    items.len()
                ));
                self.b.ins(format!("store i64 {v}, ptr {p}"));
            }
            let v = self.b.tmp();
            self.b.ins(format!(
                "{v} = call %Value @sk_int_list_new_from(i64 {}, ptr {arr})",
                items.len()
            ));
            self.value_list_lens.insert(v.clone(), items.len() as i64);
            return Ok(v);
        }
        let arr = self.b.tmp();
        self.b
            .ins(format!("{arr} = alloca [{} x %Value]", items.len()));
        for (i, item) in items.iter().enumerate() {
            let v = self.compile_expr(item)?;
            let p = self.b.tmp();
            self.b.ins(format!(
                "{p} = getelementptr [{} x %Value], ptr {arr}, i64 0, i64 {i}",
                items.len()
            ));
            self.b.ins(format!("store %Value {v}, ptr {p}"));
        }
        let v = self.b.tmp();
        if list {
            self.b.ins(format!(
                "{v} = call %Value @sk_list_new_from(i64 {}, ptr {arr})",
                items.len()
            ));
        } else {
            self.b.ins(format!(
                "{v} = call %Value @sk_tuple_new(i64 {}, ptr {arr})",
                items.len()
            ));
        }
        Ok(v)
    }

    fn compile_tuple_values(&mut self, values: &[String]) -> Result<String> {
        let arr = self.b.tmp();
        self.b
            .ins(format!("{arr} = alloca [{} x %Value]", values.len()));
        for (i, value) in values.iter().enumerate() {
            let p = self.b.tmp();
            self.b.ins(format!(
                "{p} = getelementptr [{} x %Value], ptr {arr}, i64 0, i64 {i}",
                values.len()
            ));
            self.b.ins(format!("store %Value {value}, ptr {p}"));
        }
        let v = self.b.tmp();
        self.b.ins(format!(
            "{v} = call %Value @sk_tuple_new(i64 {}, ptr {arr})",
            values.len()
        ));
        Ok(v)
    }

    fn compile_optional_expr(&mut self, expr: Option<&AstExpr>) -> Result<(String, i64)> {
        match expr {
            Some(e) => Ok((self.compile_expr(e)?, 1)),
            None => Ok((self.compile_none()?, 0)),
        }
    }

    fn compile_optional_int_expr(&mut self, expr: Option<&AstExpr>) -> Result<(String, i64)> {
        match expr {
            Some(e) => Ok((self.compile_int_operand(e)?, 1)),
            None => Ok(("0".to_owned(), 0)),
        }
    }

    fn compile_fstring(
        &mut self,
        f: &starlark_syntax::syntax::ast::FStringP<starlark_syntax::syntax::ast::AstNoPayload>,
    ) -> Result<String> {
        let parts = split_fstring(&f.format.node);
        let mut result = self.compile_string_value("")?;
        let mut expr_idx = 0usize;
        for part in parts {
            match part {
                FStringPart::Literal(s) => {
                    if !s.is_empty() {
                        let lit = self.compile_string_value(&s)?;
                        let joined = self.b.tmp();
                        self.b.ins(format!(
                            "{joined} = call %Value @sk_add(%Value {result}, %Value {lit})"
                        ));
                        result = joined;
                    }
                }
                FStringPart::Expr { repr } => {
                    let e = f
                        .expressions
                        .get(expr_idx)
                        .ok_or_else(|| anyhow!("f-string expression count mismatch"))?;
                    expr_idx += 1;
                    let v = self.compile_expr(e)?;
                    let s = self.b.tmp();
                    let func = if repr { "sk_repr" } else { "sk_str" };
                    self.b.ins(format!("{s} = call %Value @{func}(%Value {v})"));
                    let joined = self.b.tmp();
                    self.b.ins(format!(
                        "{joined} = call %Value @sk_add(%Value {result}, %Value {s})"
                    ));
                    result = joined;
                }
            }
        }
        if expr_idx != f.expressions.len() {
            bail!("f-string expression count mismatch");
        }
        Ok(result)
    }

    fn range_unroll_limit(&self) -> i64 {
        std::env::var("STARLARKC_RANGE_UNROLL_LIMIT")
            .ok()
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or_else(|| self.unroll_limit())
    }

    fn range_stop_list_name(&self, expr: &AstExpr) -> Option<String> {
        let Expr::Call(callee, args) = &expr.node else {
            return None;
        };
        let Expr::Identifier(id) = &callee.node else {
            return None;
        };
        if id.node.ident != "len" || args.args.len() != 1 {
            return None;
        }
        let Expr::Identifier(arg) = &args.args[0].expr().node else {
            return None;
        };
        if self.expr_is_list_int(args.args[0].expr()) {
            Some(arg.node.ident.clone())
        } else {
            None
        }
    }

    /// If a comprehension is a single `for i in range(...)` clause, reserve the
    /// exact result capacity up front.  This turns append-heavy comprehensions
    /// into a straight-line store loop and avoids repeated reallocations.
    /// Constant length of a single-clause `range(...)` comprehension, when the
    /// bounds are compile-time constants.
    fn comprehension_const_len(&self, first: &ForClause, rest: &[Clause]) -> Option<i64> {
        if !rest.is_empty() {
            return None;
        }
        let Expr::Call(callee, args) = &first.over.node else {
            return None;
        };
        let Expr::Identifier(id) = &callee.node else {
            return None;
        };
        if id.node.ident != "range"
            || args.args.is_empty()
            || args.args.len() > 3
            || !args
                .args
                .iter()
                .all(|a| matches!(a.node, ArgumentP::Positional(_)))
        {
            return None;
        }
        let vals: Option<Vec<i64>> = args
            .args
            .iter()
            .map(|a| self.const_int_expr(a.expr()))
            .collect();
        let vals = vals?;
        let (start, stop, step) = match vals.len() {
            1 => (0, vals[0], 1),
            2 => (vals[0], vals[1], 1),
            3 => (vals[0], vals[1], vals[2]),
            _ => return None,
        };
        if step == 0 {
            return None;
        }
        let n = if step > 0 {
            (stop - start + step - 1) / step
        } else {
            (start - stop - step - 1) / -step
        };
        (n >= 0).then_some(n)
    }

    fn comprehension_capacity(
        &mut self,
        first: &ForClause,
        rest: &[Clause],
    ) -> Result<Option<String>> {
        if !rest.is_empty() || !opt_enabled("STARLARKC_COMPREHENSION_CAP") {
            return Ok(None);
        }
        let Expr::Call(callee, args) = &first.over.node else {
            return Ok(None);
        };
        let Expr::Identifier(id) = &callee.node else {
            return Ok(None);
        };
        if id.node.ident != "range"
            || args.args.is_empty()
            || args.args.len() > 3
            || !args
                .args
                .iter()
                .all(|a| matches!(a.node, ArgumentP::Positional(_)))
        {
            return Ok(None);
        }
        let mut vals = Vec::new();
        for arg in &args.args {
            vals.push(self.compile_int_operand(arg.expr())?);
        }
        let (start, stop, step) = match vals.len() {
            1 => ("0".to_owned(), vals[0].clone(), "1".to_owned()),
            2 => (vals[0].clone(), vals[1].clone(), "1".to_owned()),
            3 => (vals[0].clone(), vals[1].clone(), vals[2].clone()),
            _ => return Ok(None),
        };
        let count = self.b.tmp();
        self.b.ins(format!(
            "{count} = call i64 @sk_range_len_i64(i64 {start}, i64 {stop}, i64 {step})"
        ));
        Ok(Some(count))
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_range_clause_loop(
        &mut self,
        f: &ForClause,
        rest: &[ClauseRef<'_>],
        body: &mut dyn FnMut(&mut Self) -> Result<()>,
        start: &str,
        stop: &str,
        step: &str,
        safe_list: Option<String>,
    ) -> Result<()> {
        self.declare_fresh_target_locals(&f.var);
        if let AssignTarget::Identifier(id) = &f.var.node {
            self.const_locals.remove(&id.node.ident);
        }
        let i_cell = self.b.tmp();
        self.b.ins(format!("{i_cell} = alloca i64"));
        self.b.ins(format!("store i64 {start}, ptr {i_cell}"));

        let loop_label = self.b.label();
        let body_label = self.b.label();
        let inc_label = self.b.label();
        let end_label = self.b.label();
        self.b.br(&loop_label);

        self.b.emit_label(&loop_label);
        let i = self.b.tmp();
        self.b.ins(format!("{i} = load i64, ptr {i_cell}"));
        let step_pos = self.b.tmp();
        self.b.ins(format!("{step_pos} = icmp sgt i64 {step}, 0"));
        let cond_pos = self.b.tmp();
        self.b.ins(format!("{cond_pos} = icmp slt i64 {i}, {stop}"));
        let cond_neg = self.b.tmp();
        self.b.ins(format!("{cond_neg} = icmp sgt i64 {i}, {stop}"));
        let cond = self.b.tmp();
        self.b.ins(format!(
            "{cond} = select i1 {step_pos}, i1 {cond_pos}, i1 {cond_neg}"
        ));
        self.b.cond_br(&cond, &body_label, &end_label);

        self.b.emit_label(&body_label);
        self.assign_int_target(&f.var, &i)?;
        let safe_frame =
            if let (Some(list), AssignTarget::Identifier(id)) = (safe_list, &f.var.node) {
                let mut frame = HashMap::new();
                frame.insert(id.node.ident.clone(), list);
                self.safe_list_indices.push(frame);
                true
            } else {
                false
            };
        self.compile_clauses(rest, body)?;
        if safe_frame {
            self.safe_list_indices.pop();
        }
        if !self.b.terminated() {
            self.b.br(&inc_label);
        }

        self.b.emit_label(&inc_label);
        let next = self.b.tmp();
        self.b.ins(format!("{next} = add i64 {i}, {step}"));
        self.b.ins(format!("store i64 {next}, ptr {i_cell}"));
        self.b.br(&loop_label);
        self.b.emit_label(&end_label);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_unrolled_range_clause(
        &mut self,
        f: &ForClause,
        rest: &[ClauseRef<'_>],
        body: &mut dyn FnMut(&mut Self) -> Result<()>,
        start: i64,
        step: i64,
        count: i64,
        safe_list: Option<String>,
    ) -> Result<()> {
        self.declare_fresh_target_locals(&f.var);
        let end_label = self.b.label();
        let safe_frame =
            if let (Some(list), AssignTarget::Identifier(id)) = (safe_list, &f.var.node) {
                let mut frame = HashMap::new();
                frame.insert(id.node.ident.clone(), list);
                self.safe_list_indices.push(frame);
                true
            } else {
                false
            };
        for k in 0..count {
            let next_label = self.b.label();
            let value = start.wrapping_add(k.wrapping_mul(step));
            self.assign_int_target(&f.var, &value.to_string())?;
            if let AssignTarget::Identifier(id) = &f.var.node {
                self.const_locals.insert(id.node.ident.clone(), value);
            }
            self.compile_clauses(rest, body)?;
            if !self.b.terminated() {
                self.b.br(&next_label);
            }
            self.b.emit_label(&next_label);
        }
        if safe_frame {
            self.safe_list_indices.pop();
        }
        if let AssignTarget::Identifier(id) = &f.var.node {
            self.const_locals.remove(&id.node.ident);
        }
        if !self.b.terminated() {
            self.b.br(&end_label);
        }
        self.b.emit_label(&end_label);
        Ok(())
    }

    fn try_compile_range_clause(
        &mut self,
        f: &ForClause,
        rest: &[ClauseRef<'_>],
        body: &mut dyn FnMut(&mut Self) -> Result<()>,
    ) -> Result<bool> {
        let Expr::Call(callee, args) = &f.over.node else {
            return Ok(false);
        };
        let Expr::Identifier(id) = &callee.node else {
            return Ok(false);
        };
        if id.node.ident != "range"
            || args.args.is_empty()
            || args.args.len() > 3
            || !args
                .args
                .iter()
                .all(|a| matches!(a.node, ArgumentP::Positional(_)))
        {
            return Ok(false);
        }

        let consts: Vec<Option<i64>> = args
            .args
            .iter()
            .map(|a| self.const_int_expr(a.expr()))
            .collect();
        let (start_c, stop_c, step_c) = match consts.as_slice() {
            [stop] => (Some(0), *stop, Some(1)),
            [start, stop] => (*start, *stop, Some(1)),
            [start, stop, step] => (*start, *stop, *step),
            _ => (None, None, None),
        };
        let stop_expr = match args.args.len() {
            1 => args.args[0].expr(),
            _ => args.args[1].expr(),
        };
        let safe_list = if start_c.is_some_and(|v| v >= 0) && step_c.is_some_and(|v| v > 0) {
            self.range_stop_list_name(stop_expr)
        } else {
            None
        };
        if let (Some(start), Some(stop), Some(step)) = (start_c, stop_c, step_c) {
            if let Some(count) = Self::range_trip_count(start, stop, step) {
                if count <= self.range_unroll_limit() {
                    self.emit_unrolled_range_clause(f, rest, body, start, step, count, safe_list)?;
                    return Ok(true);
                }
            }
        }

        let mut vals = Vec::new();
        for arg in &args.args {
            vals.push(self.compile_int_operand(arg.expr())?);
        }
        let (start, stop, step) = match vals.len() {
            1 => ("0".to_owned(), vals[0].clone(), "1".to_owned()),
            2 => (vals[0].clone(), vals[1].clone(), "1".to_owned()),
            3 => (vals[0].clone(), vals[1].clone(), vals[2].clone()),
            _ => return Ok(false),
        };
        self.emit_range_clause_loop(f, rest, body, &start, &stop, &step, safe_list)?;
        Ok(true)
    }

    fn try_compile_int_list_clause(
        &mut self,
        f: &ForClause,
        rest: &[ClauseRef<'_>],
        body: &mut dyn FnMut(&mut Self) -> Result<()>,
    ) -> Result<bool> {
        if !self.expr_is_list_int(&f.over) {
            return Ok(false);
        }
        let obj = self.compile_expr(&f.over)?;
        let payload = self.b.tmp();
        self.b
            .ins(format!("{payload} = extractvalue %Value {obj}, 1"));
        let list = self.b.tmp();
        self.b
            .ins(format!("{list} = inttoptr i64 {payload} to ptr"));
        let lenp = self.b.tmp();
        self.b.ins(format!(
            "{lenp} = getelementptr %SkList, ptr {list}, i32 0, i32 0"
        ));
        let len = self.b.tmp();
        self.b.ins(format!("{len} = load i64, ptr {lenp}"));
        let itemsp = self.b.tmp();
        self.b.ins(format!(
            "{itemsp} = getelementptr %SkList, ptr {list}, i32 0, i32 2"
        ));
        let items = self.b.tmp();
        self.b.ins(format!("{items} = load ptr, ptr {itemsp}"));

        self.declare_fresh_target_locals(&f.var);
        let i_cell = self.b.tmp();
        self.b.ins(format!("{i_cell} = alloca i64"));
        self.b.ins(format!("store i64 0, ptr {i_cell}"));
        let loop_label = self.b.label();
        let body_label = self.b.label();
        let inc_label = self.b.label();
        let end_label = self.b.label();
        self.b.br(&loop_label);
        self.b.emit_label(&loop_label);
        let i = self.b.tmp();
        self.b.ins(format!("{i} = load i64, ptr {i_cell}"));
        let cond = self.b.tmp();
        self.b.ins(format!("{cond} = icmp slt i64 {i}, {len}"));
        self.b.cond_br(&cond, &body_label, &end_label);
        self.b.emit_label(&body_label);
        let elemp = self.b.tmp();
        self.b
            .ins(format!("{elemp} = getelementptr i64, ptr {items}, i64 {i}"));
        let value = self.b.tmp();
        self.b.ins(format!("{value} = load i64, ptr {elemp}"));
        self.assign_int_target(&f.var, &value)?;
        self.compile_clauses(rest, body)?;
        if !self.b.terminated() {
            self.b.br(&inc_label);
        }
        self.b.emit_label(&inc_label);
        let next = self.b.tmp();
        self.b.ins(format!("{next} = add i64 {i}, 1"));
        self.b.ins(format!("store i64 {next}, ptr {i_cell}"));
        self.b.br(&loop_label);
        self.b.emit_label(&end_label);
        Ok(true)
    }

    fn compile_clauses(
        &mut self,
        clauses: &[ClauseRef<'_>],
        body: &mut dyn FnMut(&mut Self) -> Result<()>,
    ) -> Result<()> {
        if clauses.is_empty() {
            return body(self);
        }
        match &clauses[0] {
            ClauseRef::For(f) => {
                if opt_enabled("STARLARKC_FAST_COMPREHENSION") {
                    if self.try_compile_range_clause(f, &clauses[1..], body)? {
                        return Ok(());
                    }
                    if self.try_compile_int_list_clause(f, &clauses[1..], body)? {
                        return Ok(());
                    }
                }
                let iterable = self.compile_expr(&f.over)?;
                let len = self.b.tmp();
                self.b
                    .ins(format!("{len} = call i64 @sk_iter_len(%Value {iterable})"));
                let index_cell = self.b.tmp();
                self.b.ins(format!("{index_cell} = alloca i64"));
                self.b.ins(format!("store i64 0, ptr {index_cell}"));
                let loop_label = self.b.label();
                let body_label = self.b.label();
                let inc_label = self.b.label();
                let end_label = self.b.label();
                self.b.br(&loop_label);
                self.b.emit_label(&loop_label);
                let index = self.b.tmp();
                self.b.ins(format!("{index} = load i64, ptr {index_cell}"));
                let cond = self.b.tmp();
                self.b.ins(format!("{cond} = icmp slt i64 {index}, {len}"));
                self.b.cond_br(&cond, &body_label, &end_label);
                self.b.emit_label(&body_label);
                let item = self.b.tmp();
                self.b.ins(format!(
                    "{item} = call %Value @sk_iter_at(%Value {iterable}, i64 {index})"
                ));
                self.declare_fresh_target_locals(&f.var);
                self.assign_target(&f.var, &item)?;
                self.compile_clauses(&clauses[1..], body)?;
                if !self.b.terminated() {
                    self.b.br(&inc_label);
                }
                self.b.emit_label(&inc_label);
                let next = self.b.tmp();
                self.b.ins(format!("{next} = add i64 {index}, 1"));
                self.b.ins(format!("store i64 {next}, ptr {index_cell}"));
                self.b.br(&loop_label);
                self.b.emit_label(&end_label);
            }
            ClauseRef::If(cond) => {
                let truth = self.compile_condition(cond)?;
                let then_label = self.b.label();
                let end_label = self.b.label();
                self.b.cond_br(&truth, &then_label, &end_label);
                self.b.emit_label(&then_label);
                self.compile_clauses(&clauses[1..], body)?;
                if !self.b.terminated() {
                    self.b.br(&end_label);
                }
                self.b.emit_label(&end_label);
            }
        }
        Ok(())
    }

    fn declare_fresh_target_locals(&mut self, target: &AstAssignTarget) {
        match &target.node {
            AssignTarget::Identifier(id) => {
                self.fresh_local(&id.node.ident);
                self.const_locals.remove(&id.node.ident);
            }
            AssignTarget::Tuple(items) => {
                for item in items {
                    self.declare_fresh_target_locals(item);
                }
            }
            AssignTarget::Index(_) | AssignTarget::Dot(_, _) => {}
        }
    }

    fn read_target(&mut self, target: &AstAssignTarget) -> Result<String> {
        match &target.node {
            AssignTarget::Identifier(id) => self.resolve_value(&id.node.ident),
            AssignTarget::Index(pair) => {
                let obj = self.compile_expr(&pair.0)?;
                let idx = self.compile_expr(&pair.1)?;
                let v = self.b.tmp();
                self.b.ins(format!(
                    "{v} = call %Value @sk_index(%Value {obj}, %Value {idx})"
                ));
                Ok(v)
            }
            AssignTarget::Dot(obj, name) => {
                let obj = self.compile_expr(obj)?;
                let name = self.compile_string_value(&name.node)?;
                let v = self.b.tmp();
                self.b.ins(format!(
                    "{v} = call %Value @sk_getattr(%Value {obj}, %Value {name})"
                ));
                Ok(v)
            }
            AssignTarget::Tuple(_) => bail!("tuple target is not valid for augmented assignment"),
        }
    }

    fn assign_target(&mut self, target: &AstAssignTarget, value: &str) -> Result<()> {
        match &target.node {
            AssignTarget::Identifier(id) => self.assign_name(&id.node.ident, value),
            AssignTarget::Tuple(items) => {
                self.b.ins(format!(
                    "call void @sk_unpack_len(%Value {value}, i64 {})",
                    items.len()
                ));
                for (i, item) in items.iter().enumerate() {
                    let v = self.b.tmp();
                    self.b.ins(format!(
                        "{v} = call %Value @sk_iter_at(%Value {value}, i64 {i})"
                    ));
                    self.assign_target(item, &v)?;
                }
                Ok(())
            }
            AssignTarget::Index(pair) => {
                if let Expr::Identifier(obj_id) = &pair.0.node {
                    if let Some(len) = self.fixed_list_lens.get(&obj_id.node.ident).copied() {
                        if let Some(idx) = self.const_int_expr(&pair.1) {
                            let idx = if idx < 0 { idx.wrapping_add(len) } else { idx };
                            if (0..len).contains(&idx) {
                                let obj = self.compile_expr(&pair.0)?;
                                let int_value = self.unbox_int(value);
                                self.compile_int_list_set_unchecked(
                                    &obj,
                                    &idx.to_string(),
                                    &int_value,
                                );
                                return Ok(());
                            }
                        }
                    }
                }
                if self.expr_is_list_int(&pair.0) && self.expr_is_int(&pair.1) {
                    let obj = self.compile_expr(&pair.0)?;
                    let idx = self.compile_int_operand(&pair.1)?;
                    let int_value = self.unbox_int(value);
                    self.compile_int_list_set(&obj, &idx, &int_value);
                    return Ok(());
                }
                let obj = self.compile_expr(&pair.0)?;
                let idx = self.compile_expr(&pair.1)?;
                self.b.ins(format!(
                    "call void @sk_set_index(%Value {obj}, %Value {idx}, %Value {value})"
                ));
                Ok(())
            }
            AssignTarget::Dot(obj, name) => {
                let obj = self.compile_expr(obj)?;
                let name = self.compile_string_value(&name.node)?;
                self.b.ins(format!(
                    "call void @sk_setattr(%Value {obj}, %Value {name}, %Value {value})"
                ));
                Ok(())
            }
        }
    }
}

// A small hack: `compile_def`/`compile_lambda` need to return a value from
// `emit_closure` without threading another return value through every call.
// Rust does not allow an extra field on `FunctionCompiler` to be borrowed
// across the recursive call, so we store the most recent closure here via a
// thread-local-free approach: the field is added below by a separate impl.
// (See `last_closure` field in the struct definition.)
//
// The field is declared on the struct by the macro-free definition above; this
// comment is intentionally left as documentation for the lowering strategy.

enum ClauseRef<'a> {
    For(&'a ForClause),
    If(&'a AstExpr),
}

fn build_clauses<'a>(first: &'a ForClause, rest: &'a [Clause]) -> Vec<ClauseRef<'a>> {
    let mut out = Vec::with_capacity(rest.len() + 1);
    out.push(ClauseRef::For(first));
    for c in rest {
        match c {
            Clause::For(f) => out.push(ClauseRef::For(f)),
            Clause::If(e) => out.push(ClauseRef::If(e)),
        }
    }
    out
}

/// Conservatively find locals that are only used as containers: indexed,
/// sliced, iterated, or passed to `len`.  Such lists cannot escape the
/// function, so a fixed-size instance can be allocated on the stack.
fn collect_non_escaping_lists(stmt: &AstStmt) -> HashSet<String> {
    fn walk_expr(expr: &AstExpr, escaping: &mut HashSet<String>) {
        match &expr.node {
            Expr::Identifier(id) => {
                escaping.insert(id.node.ident.clone());
            }
            Expr::Index(pair) => {
                walk_container(&pair.0, escaping);
                walk_expr(&pair.1, escaping);
            }
            Expr::Index2(pair) => {
                walk_container(&pair.0, escaping);
                walk_expr(&pair.1, escaping);
                walk_expr(&pair.2, escaping);
            }
            Expr::Slice(obj, start, stop, step) => {
                walk_container(obj, escaping);
                for e in [start, stop, step].into_iter().flatten() {
                    walk_expr(e, escaping);
                }
            }
            Expr::Call(callee, args) => {
                if matches!(&callee.node, Expr::Identifier(id) if id.node.ident == "len")
                    && args.args.len() == 1
                {
                    walk_container(args.args[0].expr(), escaping);
                } else {
                    walk_expr(callee, escaping);
                    for a in &args.args {
                        walk_expr(a.expr(), escaping);
                    }
                }
            }
            Expr::Tuple(items) | Expr::List(items) => {
                for e in items {
                    walk_expr(e, escaping);
                }
            }
            Expr::Dict(items) => {
                for (k, v) in items {
                    walk_expr(k, escaping);
                    walk_expr(v, escaping);
                }
            }
            Expr::Dot(obj, _) => walk_expr(obj, escaping),
            Expr::Lambda(_) | Expr::Literal(_) => {}
            Expr::Not(e) | Expr::Minus(e) | Expr::Plus(e) | Expr::BitNot(e) => {
                walk_expr(e, escaping);
            }
            Expr::Op(lhs, _, rhs) => {
                walk_expr(lhs, escaping);
                walk_expr(rhs, escaping);
            }
            Expr::If(triple) => {
                walk_expr(&triple.0, escaping);
                walk_expr(&triple.1, escaping);
                walk_expr(&triple.2, escaping);
            }
            Expr::ListComprehension(body, first, rest) => {
                walk_expr(body, escaping);
                walk_expr(&first.over, escaping);
                for c in rest {
                    match c {
                        Clause::For(f) => walk_expr(&f.over, escaping),
                        Clause::If(e) => walk_expr(e, escaping),
                    }
                }
            }
            Expr::DictComprehension(pair, first, rest) => {
                walk_expr(&pair.0, escaping);
                walk_expr(&pair.1, escaping);
                walk_expr(&first.over, escaping);
                for c in rest {
                    match c {
                        Clause::For(f) => walk_expr(&f.over, escaping),
                        Clause::If(e) => walk_expr(e, escaping),
                    }
                }
            }
            Expr::FString(f) => {
                for e in &f.expressions {
                    walk_expr(e, escaping);
                }
            }
        }
    }

    fn walk_container(expr: &AstExpr, escaping: &mut HashSet<String>) {
        if !matches!(&expr.node, Expr::Identifier(_)) {
            walk_expr(expr, escaping);
        }
    }

    fn walk_stmt(stmt: &AstStmt, escaping: &mut HashSet<String>) {
        match &stmt.node {
            Stmt::Statements(stmts) => {
                for s in stmts {
                    walk_stmt(s, escaping);
                }
            }
            Stmt::Assign(a) => {
                match &a.lhs.node {
                    AssignTarget::Identifier(_) => {}
                    AssignTarget::Index(pair) => {
                        walk_container(&pair.0, escaping);
                        walk_expr(&pair.1, escaping);
                    }
                    AssignTarget::Dot(obj, _) => walk_expr(obj, escaping),
                    AssignTarget::Tuple(items) => {
                        for item in items {
                            if let AssignTarget::Index(pair) = &item.node {
                                walk_container(&pair.0, escaping);
                                walk_expr(&pair.1, escaping);
                            }
                        }
                    }
                }
                walk_expr(&a.rhs, escaping);
            }
            Stmt::AssignModify(target, _, rhs) => {
                if let AssignTarget::Index(pair) = &target.node {
                    walk_container(&pair.0, escaping);
                    walk_expr(&pair.1, escaping);
                }
                walk_expr(rhs, escaping);
            }
            Stmt::Return(Some(e)) => walk_expr(e, escaping),
            Stmt::Return(None) | Stmt::Pass | Stmt::Break | Stmt::Continue => {}
            Stmt::Expression(e) => walk_expr(e, escaping),
            Stmt::If(cond, body) => {
                walk_expr(cond, escaping);
                walk_stmt(body, escaping);
            }
            Stmt::IfElse(cond, branches) => {
                walk_expr(cond, escaping);
                walk_stmt(&branches.0, escaping);
                walk_stmt(&branches.1, escaping);
            }
            Stmt::For(f) => {
                walk_container(&f.over, escaping);
                walk_stmt(&f.body, escaping);
            }
            Stmt::Def(_) | Stmt::Load(_) => {}
        }
    }

    fn collect_candidates(stmt: &AstStmt, out: &mut HashSet<String>) {
        match &stmt.node {
            Stmt::Statements(stmts) => {
                for s in stmts {
                    collect_candidates(s, out);
                }
            }
            Stmt::Assign(a) => {
                if let AssignTarget::Identifier(id) = &a.lhs.node {
                    out.insert(id.node.ident.clone());
                }
            }
            Stmt::If(_, body) => collect_candidates(body, out),
            Stmt::IfElse(_, branches) => {
                collect_candidates(&branches.0, out);
                collect_candidates(&branches.1, out);
            }
            Stmt::For(f) => collect_candidates(&f.body, out),
            _ => {}
        }
    }

    let mut escaping = HashSet::new();
    walk_stmt(stmt, &mut escaping);
    let mut assigned = HashSet::new();
    collect_candidates(stmt, &mut assigned);
    assigned.retain(|name| !escaping.contains(name));
    assigned
}

fn collect_assigned_names(stmt: &AstStmt, out: &mut HashSet<String>) {
    fn target(t: &AstAssignTarget, out: &mut HashSet<String>) {
        match &t.node {
            AssignTarget::Identifier(id) => {
                out.insert(id.node.ident.clone());
            }
            AssignTarget::Tuple(items) => {
                for item in items {
                    target(item, out);
                }
            }
            AssignTarget::Index(_) | AssignTarget::Dot(_, _) => {}
        }
    }
    match &stmt.node {
        Stmt::Statements(stmts) => {
            for s in stmts {
                collect_assigned_names(s, out);
            }
        }
        Stmt::Assign(a) => target(&a.lhs, out),
        Stmt::AssignModify(t, _, _) => target(t, out),
        Stmt::If(_, body) => collect_assigned_names(body, out),
        Stmt::IfElse(_, branches) => {
            collect_assigned_names(&branches.0, out);
            collect_assigned_names(&branches.1, out);
        }
        Stmt::For(f) => {
            target(&f.var, out);
            collect_assigned_names(&f.body, out);
        }
        _ => {}
    }
}

/// Like `find_append_target`, but only considers appends that execute on every
/// iteration (top-level statements, not inside `if`/`for`).
fn find_direct_append_target(stmt: &AstStmt) -> Option<String> {
    match &stmt.node {
        Stmt::Statements(stmts) => stmts.iter().find_map(find_direct_append_target),
        Stmt::Expression(expr) => {
            let Expr::Call(callee, args) = &expr.node else {
                return None;
            };
            let Expr::Dot(obj, name) = &callee.node else {
                return None;
            };
            if name.node.as_str() != "append" || args.args.len() != 1 {
                return None;
            }
            let Expr::Identifier(id) = &obj.node else {
                return None;
            };
            Some(id.node.ident.clone())
        }
        _ => None,
    }
}

fn count_ident_expr(expr: &AstExpr, name: &str) -> usize {
    match &expr.node {
        Expr::Identifier(id) => usize::from(id.node.ident == name),
        Expr::Index(pair) => count_ident_expr(&pair.0, name) + count_ident_expr(&pair.1, name),
        Expr::Index2(pair) => {
            count_ident_expr(&pair.0, name)
                + count_ident_expr(&pair.1, name)
                + count_ident_expr(&pair.2, name)
        }
        Expr::Slice(obj, start, stop, step) => {
            count_ident_expr(obj, name)
                + [start, stop, step]
                    .into_iter()
                    .flatten()
                    .map(|e| count_ident_expr(e, name))
                    .sum::<usize>()
        }
        Expr::Call(callee, args) => {
            count_ident_expr(callee, name)
                + args
                    .args
                    .iter()
                    .map(|a| count_ident_expr(a.expr(), name))
                    .sum::<usize>()
        }
        Expr::Tuple(items) | Expr::List(items) => {
            items.iter().map(|e| count_ident_expr(e, name)).sum()
        }
        Expr::Dict(items) => items
            .iter()
            .map(|(k, v)| count_ident_expr(k, name) + count_ident_expr(v, name))
            .sum(),
        Expr::Dot(obj, _) => count_ident_expr(obj, name),
        Expr::Not(e) | Expr::Minus(e) | Expr::Plus(e) | Expr::BitNot(e) => {
            count_ident_expr(e, name)
        }
        Expr::Op(lhs, _, rhs) => count_ident_expr(lhs, name) + count_ident_expr(rhs, name),
        Expr::If(triple) => {
            count_ident_expr(&triple.0, name)
                + count_ident_expr(&triple.1, name)
                + count_ident_expr(&triple.2, name)
        }
        Expr::ListComprehension(body, first, rest) => {
            count_ident_expr(body, name)
                + count_ident_expr(&first.over, name)
                + rest
                    .iter()
                    .map(|c| match c {
                        Clause::For(f) => count_ident_expr(&f.over, name),
                        Clause::If(e) => count_ident_expr(e, name),
                    })
                    .sum::<usize>()
        }
        Expr::DictComprehension(pair, first, rest) => {
            count_ident_expr(&pair.0, name)
                + count_ident_expr(&pair.1, name)
                + count_ident_expr(&first.over, name)
                + rest
                    .iter()
                    .map(|c| match c {
                        Clause::For(f) => count_ident_expr(&f.over, name),
                        Clause::If(e) => count_ident_expr(e, name),
                    })
                    .sum::<usize>()
        }
        Expr::FString(f) => f
            .expressions
            .iter()
            .map(|e| count_ident_expr(e, name))
            .sum(),
        Expr::Lambda(_) | Expr::Literal(_) => 0,
    }
}

fn count_ident_stmt(stmt: &AstStmt, name: &str) -> usize {
    match &stmt.node {
        Stmt::Statements(stmts) => stmts.iter().map(|s| count_ident_stmt(s, name)).sum(),
        Stmt::Assign(a) => {
            let lhs = match &a.lhs.node {
                AssignTarget::Identifier(id) => usize::from(id.node.ident == name),
                AssignTarget::Index(pair) => {
                    count_ident_expr(&pair.0, name) + count_ident_expr(&pair.1, name)
                }
                AssignTarget::Dot(obj, _) => count_ident_expr(obj, name),
                AssignTarget::Tuple(items) => items
                    .iter()
                    .map(|t| match &t.node {
                        AssignTarget::Identifier(id) => usize::from(id.node.ident == name),
                        AssignTarget::Index(pair) => {
                            count_ident_expr(&pair.0, name) + count_ident_expr(&pair.1, name)
                        }
                        _ => 0,
                    })
                    .sum(),
            };
            lhs + count_ident_expr(&a.rhs, name)
        }
        Stmt::AssignModify(target, _, rhs) => {
            let lhs = match &target.node {
                AssignTarget::Identifier(id) => usize::from(id.node.ident == name),
                AssignTarget::Index(pair) => {
                    count_ident_expr(&pair.0, name) + count_ident_expr(&pair.1, name)
                }
                _ => 0,
            };
            lhs + count_ident_expr(rhs, name)
        }
        Stmt::Return(Some(e)) | Stmt::Expression(e) => count_ident_expr(e, name),
        Stmt::Return(None) | Stmt::Pass | Stmt::Break | Stmt::Continue => 0,
        Stmt::If(cond, body) => count_ident_expr(cond, name) + count_ident_stmt(body, name),
        Stmt::IfElse(cond, branches) => {
            count_ident_expr(cond, name)
                + count_ident_stmt(&branches.0, name)
                + count_ident_stmt(&branches.1, name)
        }
        Stmt::For(f) => count_ident_expr(&f.over, name) + count_ident_stmt(&f.body, name),
        Stmt::Def(_) | Stmt::Load(_) => 0,
    }
}

fn stmt_has_break_or_continue(stmt: &AstStmt) -> bool {
    match &stmt.node {
        Stmt::Break | Stmt::Continue => true,
        Stmt::Statements(stmts) => stmts.iter().any(stmt_has_break_or_continue),
        Stmt::If(_, body) => stmt_has_break_or_continue(body),
        Stmt::IfElse(_, branches) => {
            stmt_has_break_or_continue(&branches.0) || stmt_has_break_or_continue(&branches.1)
        }
        // A `break` in a nested loop does not affect the outer iteration count.
        Stmt::For(_) => false,
        _ => false,
    }
}

fn find_append_target(stmt: &AstStmt) -> Option<String> {
    match &stmt.node {
        Stmt::Statements(stmts) => stmts.iter().find_map(find_append_target),
        Stmt::Expression(expr) => {
            let Expr::Call(callee, args) = &expr.node else {
                return None;
            };
            let Expr::Dot(obj, name) = &callee.node else {
                return None;
            };
            if name.node.as_str() != "append" || args.args.len() != 1 {
                return None;
            }
            let Expr::Identifier(id) = &obj.node else {
                return None;
            };
            Some(id.node.ident.clone())
        }
        Stmt::If(_, body) => find_append_target(body),
        Stmt::IfElse(_, branches) => {
            find_append_target(&branches.0).or_else(|| find_append_target(&branches.1))
        }
        Stmt::For(f) => find_append_target(&f.body),
        _ => None,
    }
}

fn opt_enabled(name: &str) -> bool {
    std::env::var(name).map(|v| v != "0").unwrap_or(true)
}

enum FStringPart {
    Literal(String),
    Expr { repr: bool },
}

fn split_fstring(s: &str) -> Vec<FStringPart> {
    let mut parts = Vec::new();
    let mut literal = String::new();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '{' {
            if chars.peek() == Some(&'{') {
                chars.next();
                literal.push('{');
                continue;
            }
            if !literal.is_empty() {
                parts.push(FStringPart::Literal(std::mem::take(&mut literal)));
            }
            let mut field = String::new();
            let mut depth = 1usize;
            for ch in chars.by_ref() {
                if ch == '{' {
                    depth += 1;
                    field.push(ch);
                } else if ch == '}' {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                    field.push(ch);
                } else {
                    field.push(ch);
                }
            }
            let repr = field.contains("!r");
            parts.push(FStringPart::Expr { repr });
            continue;
        }
        if c == '}' && chars.peek() == Some(&'}') {
            chars.next();
            literal.push('}');
            continue;
        }
        literal.push(c);
    }
    if !literal.is_empty() {
        parts.push(FStringPart::Literal(literal));
    }
    parts
}

fn detect_module_root(source: &Path) -> PathBuf {
    let absolute = fs::canonicalize(source).unwrap_or_else(|_| source.to_path_buf());
    let start = absolute.parent().unwrap_or_else(|| Path::new("."));
    for dir in start.ancestors() {
        for marker in ["MODULE.bazel", "WORKSPACE", "WORKSPACE.bazel"] {
            if dir.join(marker).is_file() {
                return dir.to_path_buf();
            }
        }
    }
    start.to_path_buf()
}

/// Resolve a Starlark `load()` string.
///
/// Supports ordinary relative paths as well as Bazel-style labels:
///
/// * `//pkg:file.bzl` resolves against the repository root.
/// * `@repo//pkg:file.bzl` resolves against the repository root (the current
///   checkout is assumed to be that repository).
/// * `@repo` resolves to `<root>/repo.bzl`, matching Bazel's shorthand for
///   `@repo//:repo.bzl`.
fn int_literal_ast(i: &TokenInt) -> Option<i64> {
    match i {
        TokenInt::I32(v) => Some(*v as i64),
        TokenInt::BigInt(v) => {
            let text = v.to_string();
            text.parse::<i64>()
                .or_else(|_| text.parse::<u64>().map(|u| u as i64))
                .ok()
        }
    }
}

/// Compile-time evaluation of an integer expression with a small environment
/// for comprehension loop variables.
fn eval_const_int(expr: &AstExpr, env: &HashMap<String, i64>) -> Option<i64> {
    match &expr.node {
        Expr::Literal(_) => const_int_ast(expr),
        Expr::Identifier(id) => env.get(&id.node.ident).copied(),
        Expr::Minus(e) => Some(eval_const_int(e, env)?.wrapping_neg()),
        Expr::Plus(e) => eval_const_int(e, env),
        Expr::BitNot(e) => Some(!eval_const_int(e, env)?),
        Expr::Op(l, op, r) => {
            let l = eval_const_int(l, env)?;
            let r = eval_const_int(r, env)?;
            use BinOp::*;
            Some(match op {
                Add => l.wrapping_add(r),
                Subtract => l.wrapping_sub(r),
                Multiply => l.wrapping_mul(r),
                FloorDivide => {
                    if r == 0 {
                        return None;
                    }
                    l.div_euclid(r)
                }
                Percent => {
                    if r == 0 {
                        return None;
                    }
                    l.rem_euclid(r)
                }
                BitAnd => l & r,
                BitOr => l | r,
                BitXor => l ^ r,
                LeftShift => l.wrapping_shl(r as u32),
                RightShift => l.wrapping_shr(r as u32),
                _ => return None,
            })
        }
        _ => None,
    }
}

/// Evaluate a list of constants, including a single-clause `range(...)`
/// comprehension with a constant body.  Keccak's `_PI`, `_CHI1` and `_CHI2`
/// tables are written this way; folding them turns every lane lookup into a
/// compile-time constant.
fn const_int_list_ast(expr: &AstExpr) -> Option<Vec<i64>> {
    match &expr.node {
        Expr::List(items) => items.iter().map(const_int_ast).collect(),
        Expr::ListComprehension(body, first, rest) if rest.is_empty() => {
            let Expr::Call(callee, args) = &first.over.node else {
                return None;
            };
            let Expr::Identifier(id) = &callee.node else {
                return None;
            };
            if id.node.ident != "range"
                || args.args.is_empty()
                || args.args.len() > 3
                || !args
                    .args
                    .iter()
                    .all(|a| matches!(a.node, ArgumentP::Positional(_)))
            {
                return None;
            }
            let vals: Option<Vec<i64>> =
                args.args.iter().map(|a| const_int_ast(a.expr())).collect();
            let vals = vals?;
            let (start, stop, step) = match vals.len() {
                1 => (0, vals[0], 1),
                2 => (vals[0], vals[1], 1),
                3 => (vals[0], vals[1], vals[2]),
                _ => return None,
            };
            if step == 0 {
                return None;
            }
            let AssignTarget::Identifier(var) = &first.var.node else {
                return None;
            };
            let mut env = HashMap::new();
            let mut out = Vec::new();
            let mut i = start;
            while (step > 0 && i < stop) || (step < 0 && i > stop) {
                env.insert(var.node.ident.clone(), i);
                out.push(eval_const_int(body, &env)?);
                i = i.checked_add(step)?;
            }
            Some(out)
        }
        _ => None,
    }
}

fn const_int_ast(expr: &AstExpr) -> Option<i64> {
    match &expr.node {
        Expr::Literal(AstLiteral::Int(i)) => int_literal_ast(&i.node),
        Expr::Minus(e) => const_int_ast(e).map(|v| v.wrapping_neg()),
        Expr::Plus(e) => const_int_ast(e),
        Expr::BitNot(e) => const_int_ast(e).map(|v| !v),
        Expr::Op(lhs, op, rhs) => {
            let l = const_int_ast(lhs)?;
            let r = const_int_ast(rhs)?;
            use BinOp::*;
            match op {
                Add => Some(l.wrapping_add(r)),
                Subtract => Some(l.wrapping_sub(r)),
                Multiply => Some(l.wrapping_mul(r)),
                FloorDivide => {
                    if r == 0 {
                        None
                    } else {
                        let q = l.wrapping_div(r);
                        let rem = l.wrapping_rem(r);
                        Some(if rem != 0 && ((rem < 0) != (r < 0)) {
                            q - 1
                        } else {
                            q
                        })
                    }
                }
                Percent => {
                    if r == 0 {
                        None
                    } else {
                        let rem = l.wrapping_rem(r);
                        Some(if rem != 0 && ((rem < 0) != (r < 0)) {
                            rem + r
                        } else {
                            rem
                        })
                    }
                }
                BitAnd => Some(l & r),
                BitOr => Some(l | r),
                BitXor => Some(l ^ r),
                LeftShift => Some(if r >= 64 { 0 } else { l.wrapping_shl(r as u32) }),
                RightShift => Some(if r >= 64 { 0 } else { ((l as u64) >> r) as i64 }),
                _ => None,
            }
        }
        _ => None,
    }
}

pub(crate) fn resolve_load_path_public(root: &Path, from: &Path, module: &str) -> PathBuf {
    resolve_load_path(root, from, module)
}

fn resolve_load_path(root: &Path, from: &Path, module: &str) -> PathBuf {
    if let Some(rest) = module.strip_prefix('@') {
        if let Some((_repo, tail)) = rest.split_once("//") {
            return resolve_bazel_label(root, tail);
        }
        // `@repo` is Bazel shorthand for `@repo//:repo.bzl`.
        return resolve_with_extensions(&root.join(rest));
    }
    if let Some(rest) = module.strip_prefix("//") {
        return resolve_bazel_label(root, rest);
    }

    let p = Path::new(module);
    if p.is_absolute() {
        return p.to_path_buf();
    }
    let base = from.parent().unwrap_or_else(|| Path::new("."));
    resolve_with_extensions(&base.join(p))
}

fn resolve_with_extensions(path: &Path) -> PathBuf {
    if path.exists() {
        return path.to_path_buf();
    }
    if path.extension().is_none() {
        for ext in ["star", "bzl"] {
            let candidate = path.with_extension(ext);
            if candidate.exists() {
                return candidate;
            }
        }
    }
    path.to_path_buf()
}

fn resolve_bazel_label(root: &Path, label: &str) -> PathBuf {
    let (package, name) = match label.split_once(':') {
        Some((pkg, name)) => (pkg, name),
        None => {
            // Bazel's `//pkg` is shorthand for `//pkg:pkg`.
            let pkg = label.trim_matches('/');
            let name = pkg.rsplit('/').next().unwrap_or(pkg);
            (pkg, name)
        }
    };
    let mut path = root.to_path_buf();
    if !package.is_empty() {
        path.push(package);
    }
    if !name.is_empty() {
        path.push(name);
    }
    if path.extension().is_none() {
        for ext in ["bzl", "star"] {
            let candidate = path.with_extension(ext);
            if candidate.exists() {
                return candidate;
            }
        }
    }
    path
}

fn collect_top_level_names(stmt: &AstStmt, out: &mut Vec<String>) {
    match &stmt.node {
        Stmt::Statements(stmts) => {
            for s in stmts {
                collect_top_level_names(s, out);
            }
        }
        Stmt::Assign(a) => collect_target_names(&a.lhs, out),
        Stmt::AssignModify(t, _, _) => collect_target_names(t, out),
        Stmt::Def(d) => out.push(d.name.node.ident.clone()),
        Stmt::For(f) => {
            collect_target_names(&f.var, out);
            collect_top_level_names(&f.body, out);
        }
        Stmt::If(_, body) => collect_top_level_names(body, out),
        Stmt::IfElse(_, branches) => {
            collect_top_level_names(&branches.0, out);
            collect_top_level_names(&branches.1, out);
        }
        Stmt::Load(l) => {
            for arg in &l.args {
                out.push(arg.local.node.ident.clone());
            }
        }
        _ => {}
    }
}

fn collect_local_names(stmt: &AstStmt, out: &mut Vec<String>) {
    match &stmt.node {
        Stmt::Statements(stmts) => {
            for s in stmts {
                collect_local_names(s, out);
            }
        }
        Stmt::Assign(a) => collect_target_names(&a.lhs, out),
        Stmt::AssignModify(t, _, _) => collect_target_names(t, out),
        Stmt::Def(d) => out.push(d.name.node.ident.clone()),
        Stmt::For(f) => {
            collect_target_names(&f.var, out);
            collect_local_names(&f.body, out);
        }
        Stmt::If(_, body) => collect_local_names(body, out),
        Stmt::IfElse(_, branches) => {
            collect_local_names(&branches.0, out);
            collect_local_names(&branches.1, out);
        }
        Stmt::Load(l) => {
            for arg in &l.args {
                out.push(arg.local.node.ident.clone());
            }
        }
        _ => {}
    }
}

fn collect_target_names(target: &AstAssignTarget, out: &mut Vec<String>) {
    match &target.node {
        AssignTarget::Identifier(id) => out.push(id.node.ident.clone()),
        AssignTarget::Tuple(items) => {
            for item in items {
                collect_target_names(item, out);
            }
        }
        AssignTarget::Index(_) | AssignTarget::Dot(_, _) => {}
    }
}

fn dedupe(names: &mut Vec<String>) {
    let mut seen = HashSet::new();
    names.retain(|n| seen.insert(n.clone()));
}

fn builtin_id(name: &str) -> Option<i64> {
    Some(match name {
        "print" => 1,
        "len" => 2,
        "str" => 3,
        "repr" => 4,
        "type" => 5,
        "int" => 6,
        "float" => 7,
        "bool" => 8,
        "list" => 9,
        "tuple" => 10,
        "dict" => 11,
        "range" => 12,
        "enumerate" => 13,
        "zip" => 14,
        "reversed" => 15,
        "sorted" => 16,
        "min" => 17,
        "max" => 18,
        "sum" => 19,
        "abs" => 20,
        "all" => 21,
        "any" => 22,
        "fail" => 23,
        "getattr" => 24,
        "hasattr" => 25,
        "dir" => 26,
        "hash" => 27,
        "ord" => 28,
        "chr" => 29,
        "callable" => 30,
        "struct" => 31,
        _ => return None,
    })
}

fn llvm_escape(bytes: &[u8]) -> String {
    let mut out = String::new();
    for &b in bytes {
        match b {
            b'\\' => out.push_str("\\5C"),
            b'"' => out.push_str("\\22"),
            0x20..=0x7E => out.push(b as char),
            _ => {
                let _ = write!(out, "\\{b:02X}");
            }
        }
    }
    out
}

fn escape_ir_string(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}
