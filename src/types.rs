//! A small whole-program type inference pass used to specialize integer and
//! integer-list code.
//!
//! Starlark is dynamically typed, so this pass is deliberately conservative.
//! It only distinguishes the types that matter for the hot numeric loops in
//! code such as `hashes.bzl`:
//!
//! * `Int`      - a signed 64-bit word
//! * `ListInt`  - a list whose elements are all integers
//! * `DictInt`  - a dictionary whose values are all integers
//! * `Other`    - anything else
//! * `Unknown`  - not known yet
//!
//! Everything else falls back to the existing dynamic `%Value` lowering.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use starlark_syntax::codemap::Span;
use starlark_syntax::syntax::ast::{
    ArgumentP, AssignTarget, AstAssignTarget, AstExpr, AstLiteral, AstStmt, BinOp, Clause, Expr,
    ForClause, ParameterP, Stmt,
};
use starlark_syntax::syntax::AstModule;

/// The inferred type of a value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ty {
    Unknown,
    Int,
    ListInt,
    DictInt,
    Other,
}

impl Ty {
    pub fn is_int(self) -> bool {
        self == Ty::Int
    }

    pub fn is_list_int(self) -> bool {
        self == Ty::ListInt
    }

    pub fn is_dynamic(self) -> bool {
        matches!(self, Ty::Unknown | Ty::Other)
    }
}

/// Join two types at a control-flow merge.
pub fn join(a: Ty, b: Ty) -> Ty {
    match (a, b) {
        (Ty::Unknown, x) | (x, Ty::Unknown) => x,
        (Ty::Int, Ty::Int) => Ty::Int,
        (Ty::ListInt, Ty::ListInt) => Ty::ListInt,
        (Ty::DictInt, Ty::DictInt) => Ty::DictInt,
        _ => Ty::Other,
    }
}

#[derive(Clone, Debug)]
pub struct FuncSig {
    pub params: Vec<Ty>,
    pub ret: Ty,
}

#[derive(Clone, Debug, Default)]
pub struct ModuleInfo {
    /// Local function name -> qualified function id.
    pub functions: HashMap<String, FuncId>,
    /// Local load alias -> qualified function id.
    pub loads: HashMap<String, FuncId>,
    /// Top-level variable types.
    pub globals: HashMap<String, Ty>,
}

pub type FuncId = (PathBuf, String);

#[derive(Clone, Debug, Default)]
pub struct ProgramTypes {
    pub funcs: HashMap<FuncId, FuncSig>,
    pub locals: HashMap<(PathBuf, String, String), Ty>,
    pub modules: HashMap<PathBuf, ModuleInfo>,
    pub expr_types: HashMap<(PathBuf, u32, u32), Ty>,
}

impl ProgramTypes {
    pub fn func(&self, id: &FuncId) -> Option<&FuncSig> {
        self.funcs.get(id)
    }

    pub fn local(&self, module: &Path, func: &str, name: &str) -> Ty {
        self.locals
            .get(&(module.to_path_buf(), func.to_owned(), name.to_owned()))
            .copied()
            .unwrap_or(Ty::Unknown)
    }

    pub fn global(&self, module: &Path, name: &str) -> Ty {
        self.modules
            .get(module)
            .and_then(|m| m.globals.get(name))
            .copied()
            .unwrap_or(Ty::Unknown)
    }

    pub fn expr_ty(&self, module: &Path, span: Span) -> Ty {
        self.expr_types
            .get(&(module.to_path_buf(), span.begin().get(), span.end().get()))
            .copied()
            .unwrap_or(Ty::Unknown)
    }

    pub fn resolve_func(&self, module: &Path, name: &str) -> Option<FuncId> {
        let info = self.modules.get(module)?;
        info.functions
            .get(name)
            .or_else(|| info.loads.get(name))
            .cloned()
    }
}

struct Ctx<'a> {
    types: &'a mut ProgramTypes,
    module: PathBuf,
    func: Option<String>,
    changed: bool,
}

impl<'a> Ctx<'a> {
    fn new(types: &'a mut ProgramTypes, module: PathBuf, func: Option<String>) -> Self {
        Self {
            types,
            module,
            func,
            changed: false,
        }
    }

    fn get(&self, name: &str) -> Ty {
        if let Some(func) = &self.func {
            let key = (self.module.clone(), func.clone(), name.to_owned());
            if let Some(ty) = self.types.locals.get(&key) {
                return *ty;
            }
        }
        self.types.global(&self.module, name)
    }

    fn set(&mut self, name: &str, ty: Ty) {
        if ty == Ty::Unknown {
            return;
        }
        match &self.func {
            Some(func) => {
                let key = (self.module.clone(), func.clone(), name.to_owned());
                let old = self.types.locals.get(&key).copied().unwrap_or(Ty::Unknown);
                let new = join(old, ty);
                if new != old {
                    self.types.locals.insert(key, new);
                    self.changed = true;
                }
            }
            None => {
                let info = self.types.modules.entry(self.module.clone()).or_default();
                let old = info.globals.get(name).copied().unwrap_or(Ty::Unknown);
                let new = join(old, ty);
                if new != old {
                    info.globals.insert(name.to_owned(), new);
                    self.changed = true;
                }
            }
        }
    }

    fn set_ret(&mut self, ty: Ty) {
        let Some(func) = &self.func else { return };
        let id = (self.module.clone(), func.clone());
        let sig = self.types.funcs.entry(id).or_insert_with(|| FuncSig {
            params: Vec::new(),
            ret: Ty::Unknown,
        });
        let new = join(sig.ret, ty);
        if new != sig.ret {
            sig.ret = new;
            self.changed = true;
        }
    }

    fn ret(&self) -> Ty {
        let Some(func) = &self.func else {
            return Ty::Unknown;
        };
        self.types
            .funcs
            .get(&(self.module.clone(), func.clone()))
            .map(|s| s.ret)
            .unwrap_or(Ty::Unknown)
    }

    fn resolve_func(&self, name: &str) -> Option<FuncId> {
        self.types.resolve_func(&self.module, name)
    }
}

/// Infer types for every module in `modules`.
pub fn infer_program(modules: &HashMap<PathBuf, AstModule>, root: &Path) -> ProgramTypes {
    let mut types = ProgramTypes::default();

    // First pass: collect module-level functions, loads, and globals.
    for (path, ast) in modules {
        let mut info = ModuleInfo::default();
        collect_module_info(path, ast.statement(), root, &mut info);
        types.modules.insert(path.clone(), info);
    }

    // Initialize function signatures.
    for (path, ast) in modules {
        collect_function_sigs(path, ast.statement(), &mut types);
    }

    // Fixed point over the whole program.
    for _ in 0..64 {
        let mut changed = false;
        for (path, ast) in modules {
            let mut ctx = Ctx::new(&mut types, path.clone(), None);
            infer_stmt(&mut ctx, ast.statement());
            changed |= ctx.changed;
        }
        if !changed {
            break;
        }
    }

    types
}

fn collect_module_info(path: &Path, stmt: &AstStmt, root: &Path, info: &mut ModuleInfo) {
    match &stmt.node {
        Stmt::Statements(stmts) => {
            for s in stmts {
                collect_module_info(path, s, root, info);
            }
        }
        Stmt::Def(d) => {
            let name = d.name.node.ident.clone();
            info.functions
                .insert(name.clone(), (path.to_path_buf(), name));
        }
        Stmt::Load(l) => {
            for arg in &l.args {
                let local = arg.local.node.ident.clone();
                let remote = arg.their.node.clone();
                let loaded = crate::compiler::resolve_load_path_public(root, path, &l.module.node);
                let loaded = std::fs::canonicalize(&loaded).unwrap_or(loaded);
                info.loads.insert(local, (loaded, remote));
            }
        }
        Stmt::If(_, body) => collect_module_info(path, body, root, info),
        Stmt::IfElse(_, branches) => {
            collect_module_info(path, &branches.0, root, info);
            collect_module_info(path, &branches.1, root, info);
        }
        Stmt::For(f) => collect_module_info(path, &f.body, root, info),
        _ => {}
    }
}

fn collect_function_sigs(path: &Path, stmt: &AstStmt, types: &mut ProgramTypes) {
    match &stmt.node {
        Stmt::Statements(stmts) => {
            for s in stmts {
                collect_function_sigs(path, s, types);
            }
        }
        Stmt::Def(d) => {
            let name = d.name.node.ident.clone();
            let params = d
                .params
                .iter()
                .filter_map(|p| match &p.node {
                    ParameterP::Normal(id, _, _) => Some(id.node.ident.clone()),
                    ParameterP::Args(id, _) | ParameterP::KwArgs(id, _) => {
                        Some(id.node.ident.clone())
                    }
                    ParameterP::Slash | ParameterP::NoArgs => None,
                })
                .collect::<Vec<_>>();
            let sig = FuncSig {
                params: vec![Ty::Unknown; params.len()],
                ret: Ty::Unknown,
            };
            types.funcs.insert((path.to_path_buf(), name.clone()), sig);
            for p in params {
                types
                    .locals
                    .insert((path.to_path_buf(), name.clone(), p), Ty::Unknown);
            }
            collect_function_sigs(path, &d.body, types);
        }
        Stmt::If(_, body) => collect_function_sigs(path, body, types),
        Stmt::IfElse(_, branches) => {
            collect_function_sigs(path, &branches.0, types);
            collect_function_sigs(path, &branches.1, types);
        }
        Stmt::For(f) => collect_function_sigs(path, &f.body, types),
        _ => {}
    }
}

fn infer_stmt(ctx: &mut Ctx<'_>, stmt: &AstStmt) {
    match &stmt.node {
        Stmt::Statements(stmts) => {
            for s in stmts {
                infer_stmt(ctx, s);
            }
        }
        Stmt::Return(Some(e)) => {
            let expected = ctx.ret();
            let t = infer_expr(
                ctx,
                e,
                if expected == Ty::Unknown {
                    None
                } else {
                    Some(expected)
                },
            );
            if std::env::var_os("STARLARKC_DUMP_TYPES").is_some() {
                eprintln!("return in {:?}: {:?}", ctx.func, t);
            }
            ctx.set_ret(t);
        }
        Stmt::Return(None) => ctx.set_ret(Ty::Other),
        Stmt::Expression(e) => {
            infer_expr(ctx, e, None);
        }
        Stmt::Assign(a) => {
            let t = infer_expr(ctx, &a.rhs, None);
            assign_target(ctx, &a.lhs, t);
        }
        Stmt::AssignModify(target, op, rhs) => {
            let rt = infer_expr(ctx, rhs, None);
            let lt = target_ty(ctx, target);
            let result = match op {
                starlark_syntax::syntax::ast::AssignOp::Add => {
                    if lt == Ty::ListInt || rt == Ty::ListInt {
                        Ty::ListInt
                    } else if lt == Ty::Int || rt == Ty::Int {
                        Ty::Int
                    } else {
                        Ty::Unknown
                    }
                }
                starlark_syntax::syntax::ast::AssignOp::Subtract
                | starlark_syntax::syntax::ast::AssignOp::Multiply
                | starlark_syntax::syntax::ast::AssignOp::FloorDivide
                | starlark_syntax::syntax::ast::AssignOp::Percent
                | starlark_syntax::syntax::ast::AssignOp::BitAnd
                | starlark_syntax::syntax::ast::AssignOp::BitOr
                | starlark_syntax::syntax::ast::AssignOp::BitXor
                | starlark_syntax::syntax::ast::AssignOp::LeftShift
                | starlark_syntax::syntax::ast::AssignOp::RightShift => Ty::Int,
                _ => Ty::Unknown,
            };
            assign_target(ctx, target, result);
        }
        Stmt::If(cond, body) => {
            infer_expr(ctx, cond, None);
            infer_stmt(ctx, body);
        }
        Stmt::IfElse(cond, branches) => {
            infer_expr(ctx, cond, None);
            infer_stmt(ctx, &branches.0);
            infer_stmt(ctx, &branches.1);
        }
        Stmt::For(f) => {
            infer_for(ctx, f);
        }
        Stmt::Def(d) => {
            // Nested functions are inferred in their own context. This is
            // enough for the top-level functions used by hashes.bzl.
            let name = d.name.node.ident.clone();
            let mut child = Ctx::new(ctx.types, ctx.module.clone(), Some(name));
            infer_stmt(&mut child, &d.body);
            ctx.changed |= child.changed;
        }
        Stmt::Load(_) | Stmt::Break | Stmt::Continue | Stmt::Pass => {}
    }
}

fn infer_for(
    ctx: &mut Ctx<'_>,
    f: &starlark_syntax::syntax::ast::ForP<starlark_syntax::syntax::ast::AstNoPayload>,
) {
    let over_ty = infer_expr(ctx, &f.over, None);
    let var_ty = match &f.over.node {
        Expr::Call(callee, _) if is_builtin(callee, "range") => Ty::Int,
        Expr::Call(callee, _) if is_builtin(callee, "enumerate") => {
            assign_enumerate_target(ctx, &f.var);
            infer_stmt(ctx, &f.body);
            return;
        }
        Expr::Call(callee, _) if is_builtin(callee, "zip") => {
            assign_zip_target(ctx, &f.var);
            infer_stmt(ctx, &f.body);
            return;
        }
        _ if over_ty == Ty::ListInt => Ty::Int,
        _ => Ty::Unknown,
    };
    assign_target(ctx, &f.var, var_ty);
    infer_stmt(ctx, &f.body);
}

fn assign_enumerate_target(ctx: &mut Ctx<'_>, target: &AstAssignTarget) {
    match &target.node {
        AssignTarget::Tuple(items) if items.len() == 2 => {
            assign_target(ctx, &items[0], Ty::Int);
            assign_target(ctx, &items[1], Ty::Unknown);
        }
        _ => assign_target(ctx, target, Ty::Unknown),
    }
}

fn assign_zip_target(ctx: &mut Ctx<'_>, target: &AstAssignTarget) {
    match &target.node {
        AssignTarget::Tuple(items) => {
            for item in items {
                assign_target(ctx, item, Ty::Unknown);
            }
        }
        _ => assign_target(ctx, target, Ty::Unknown),
    }
}

fn target_ty(ctx: &Ctx<'_>, target: &AstAssignTarget) -> Ty {
    match &target.node {
        AssignTarget::Identifier(id) => ctx.get(&id.node.ident),
        _ => Ty::Unknown,
    }
}

fn assign_target(ctx: &mut Ctx<'_>, target: &AstAssignTarget, ty: Ty) {
    match &target.node {
        AssignTarget::Identifier(id) => ctx.set(&id.node.ident, ty),
        AssignTarget::Tuple(items) => {
            let elem = if ty == Ty::ListInt {
                Ty::Int
            } else {
                Ty::Unknown
            };
            for item in items {
                assign_target(ctx, item, elem);
            }
        }
        AssignTarget::Index(_) | AssignTarget::Dot(_, _) => {}
    }
}

fn is_builtin(expr: &AstExpr, name: &str) -> bool {
    matches!(&expr.node, Expr::Identifier(id) if id.node.ident == name)
}

fn infer_expr(ctx: &mut Ctx<'_>, expr: &AstExpr, expected: Option<Ty>) -> Ty {
    let ty = infer_expr_inner(ctx, expr, expected);
    ctx.types.expr_types.insert(
        (
            ctx.module.clone(),
            expr.span.begin().get(),
            expr.span.end().get(),
        ),
        ty,
    );
    ty
}

fn infer_expr_inner(ctx: &mut Ctx<'_>, expr: &AstExpr, expected: Option<Ty>) -> Ty {
    match &expr.node {
        Expr::Literal(AstLiteral::Int(_)) => Ty::Int,
        Expr::Literal(_) => Ty::Other,
        Expr::Identifier(id) => {
            let name = &id.node.ident;
            let mut ty = ctx.get(name);
            if ty == Ty::Unknown {
                if let Some(exp) = expected {
                    if exp == Ty::Int || exp == Ty::ListInt || exp == Ty::DictInt {
                        ctx.set(name, exp);
                        ty = exp;
                    }
                }
            }
            ty
        }
        Expr::List(items) => {
            if expected == Some(Ty::ListInt) {
                for item in items {
                    infer_expr(ctx, item, Some(Ty::Int));
                }
                return Ty::ListInt;
            }
            let mut all_int = !items.is_empty();
            let mut all_unknown = true;
            for item in items {
                let t = infer_expr(ctx, item, None);
                all_int &= t == Ty::Int;
                all_unknown &= t == Ty::Unknown;
            }
            if all_int {
                Ty::ListInt
            } else if all_unknown {
                Ty::Unknown
            } else {
                Ty::Other
            }
        }
        Expr::Tuple(items) => {
            for item in items {
                infer_expr(ctx, item, None);
            }
            Ty::Other
        }
        Expr::Dict(entries) => {
            let mut all_int = !entries.is_empty();
            for (k, v) in entries {
                infer_expr(ctx, k, None);
                let t = infer_expr(ctx, v, None);
                all_int &= t == Ty::Int;
            }
            if all_int {
                Ty::DictInt
            } else {
                Ty::Other
            }
        }
        Expr::ListComprehension(body, first, clauses) => {
            infer_comprehension(ctx, body, first, clauses, expected);
            if expected == Some(Ty::ListInt) {
                Ty::ListInt
            } else {
                // Re-infer the body with no expectation to see whether it is
                // an integer expression. The clause inference above has
                // already populated the loop-variable types.
                let t = infer_expr(ctx, body, None);
                if t == Ty::Int {
                    Ty::ListInt
                } else if t == Ty::Unknown {
                    Ty::Unknown
                } else {
                    Ty::Other
                }
            }
        }
        Expr::DictComprehension(pair, first, clauses) => {
            infer_comprehension(ctx, &pair.1, first, clauses, None);
            let t = infer_expr(ctx, &pair.1, None);
            if t == Ty::Int {
                Ty::DictInt
            } else if t == Ty::Unknown {
                Ty::Unknown
            } else {
                Ty::Other
            }
        }
        Expr::Call(callee, args) => infer_call(ctx, callee, args, expected),
        Expr::Dot(obj, field) => {
            let _ = infer_expr(ctx, obj, None);
            // Struct fields are not tracked precisely; method results are
            // handled in infer_call.
            let _ = field;
            Ty::Unknown
        }
        Expr::Index(pair) => {
            let (obj, idx) = (&pair.0, &pair.1);
            let obj_expected = if expected == Some(Ty::Int) {
                Some(Ty::ListInt)
            } else {
                None
            };
            let obj_ty = infer_expr(ctx, obj, obj_expected);
            infer_expr(ctx, idx, Some(Ty::Int));
            match obj_ty {
                Ty::ListInt => Ty::Int,
                Ty::DictInt => Ty::Int,
                _ if expected == Some(Ty::Int) => Ty::Int,
                _ => Ty::Unknown,
            }
        }
        Expr::Index2(pair) => {
            infer_expr(ctx, &pair.0, None);
            infer_expr(ctx, &pair.1, Some(Ty::Int));
            infer_expr(ctx, &pair.2, Some(Ty::Int));
            Ty::Other
        }
        Expr::Slice(obj, start, stop, step) => {
            let obj_ty = infer_expr(
                ctx,
                obj,
                if expected == Some(Ty::ListInt) {
                    Some(Ty::ListInt)
                } else {
                    None
                },
            );
            if let Some(e) = start {
                infer_expr(ctx, e, Some(Ty::Int));
            }
            if let Some(e) = stop {
                infer_expr(ctx, e, Some(Ty::Int));
            }
            if let Some(e) = step {
                infer_expr(ctx, e, Some(Ty::Int));
            }
            if obj_ty == Ty::ListInt || expected == Some(Ty::ListInt) {
                Ty::ListInt
            } else if obj_ty == Ty::Unknown {
                Ty::Unknown
            } else {
                Ty::Other
            }
        }
        Expr::Not(e) => {
            infer_expr(ctx, e, None);
            Ty::Other
        }
        Expr::Minus(e) | Expr::Plus(e) => {
            let t = infer_expr(
                ctx,
                e,
                if expected == Some(Ty::Int) {
                    Some(Ty::Int)
                } else {
                    None
                },
            );
            if t == Ty::Int || expected == Some(Ty::Int) {
                Ty::Int
            } else if t == Ty::Unknown {
                Ty::Unknown
            } else {
                Ty::Other
            }
        }
        Expr::BitNot(e) => {
            infer_expr(ctx, e, Some(Ty::Int));
            Ty::Int
        }
        Expr::Op(lhs, op, rhs) => infer_binop(ctx, lhs, *op, rhs, expected),
        Expr::If(triple) => {
            let (cond, then, else_) = (&triple.0, &triple.1, &triple.2);
            infer_expr(ctx, cond, None);
            let a = infer_expr(ctx, then, expected);
            let b = infer_expr(ctx, else_, expected);
            join(a, b)
        }
        Expr::FString(f) => {
            for e in &f.expressions {
                infer_expr(ctx, e, None);
            }
            Ty::Other
        }
        Expr::Lambda(_) => Ty::Other,
    }
}

fn infer_binop(
    ctx: &mut Ctx<'_>,
    lhs: &AstExpr,
    op: BinOp,
    rhs: &AstExpr,
    expected: Option<Ty>,
) -> Ty {
    use BinOp::*;
    match op {
        BitAnd | BitOr | BitXor | LeftShift | RightShift => {
            infer_expr(ctx, lhs, Some(Ty::Int));
            infer_expr(ctx, rhs, Some(Ty::Int));
            Ty::Int
        }
        Add => {
            if expected == Some(Ty::ListInt) {
                infer_expr(ctx, lhs, Some(Ty::ListInt));
                infer_expr(ctx, rhs, Some(Ty::ListInt));
                return Ty::ListInt;
            }
            let l = infer_expr(
                ctx,
                lhs,
                if expected == Some(Ty::Int) {
                    Some(Ty::Int)
                } else {
                    None
                },
            );
            let r = infer_expr(
                ctx,
                rhs,
                if expected == Some(Ty::Int) {
                    Some(Ty::Int)
                } else {
                    None
                },
            );
            if l == Ty::ListInt || r == Ty::ListInt {
                if l == Ty::Unknown {
                    infer_expr(ctx, lhs, Some(Ty::ListInt));
                }
                if r == Ty::Unknown {
                    infer_expr(ctx, rhs, Some(Ty::ListInt));
                }
                return Ty::ListInt;
            }
            if l == Ty::Int && r == Ty::Int {
                return Ty::Int;
            }
            if expected == Some(Ty::Int) && (l == Ty::Unknown || r == Ty::Unknown) {
                infer_expr(ctx, lhs, Some(Ty::Int));
                infer_expr(ctx, rhs, Some(Ty::Int));
                return Ty::Int;
            }
            if l == Ty::Other || r == Ty::Other {
                return Ty::Other;
            }
            Ty::Unknown
        }
        Subtract | FloorDivide | Percent => {
            let l = infer_expr(
                ctx,
                lhs,
                if expected == Some(Ty::Int) {
                    Some(Ty::Int)
                } else {
                    None
                },
            );
            let r = infer_expr(
                ctx,
                rhs,
                if expected == Some(Ty::Int) {
                    Some(Ty::Int)
                } else {
                    None
                },
            );
            if l == Ty::Int && r == Ty::Int {
                return Ty::Int;
            }
            if expected == Some(Ty::Int) && (l == Ty::Unknown || r == Ty::Unknown) {
                infer_expr(ctx, lhs, Some(Ty::Int));
                infer_expr(ctx, rhs, Some(Ty::Int));
                return Ty::Int;
            }
            if l == Ty::Other || r == Ty::Other {
                return Ty::Other;
            }
            Ty::Unknown
        }
        Multiply => {
            let l = infer_expr(
                ctx,
                lhs,
                if expected == Some(Ty::Int) {
                    Some(Ty::Int)
                } else {
                    None
                },
            );
            let r = infer_expr(
                ctx,
                rhs,
                if expected == Some(Ty::Int) {
                    Some(Ty::Int)
                } else {
                    None
                },
            );
            if expected == Some(Ty::ListInt) {
                return Ty::ListInt;
            }
            if (l == Ty::ListInt && r == Ty::Int) || (l == Ty::Int && r == Ty::ListInt) {
                return Ty::ListInt;
            }
            if l == Ty::Int && r == Ty::Int {
                return Ty::Int;
            }
            if expected == Some(Ty::Int) && (l == Ty::Unknown || r == Ty::Unknown) {
                infer_expr(ctx, lhs, Some(Ty::Int));
                infer_expr(ctx, rhs, Some(Ty::Int));
                return Ty::Int;
            }
            if l == Ty::Other || r == Ty::Other {
                return Ty::Other;
            }
            Ty::Unknown
        }
        Divide => {
            infer_expr(ctx, lhs, None);
            infer_expr(ctx, rhs, None);
            Ty::Other
        }
        Equal | NotEqual | Less | Greater | LessOrEqual | GreaterOrEqual | In | NotIn => {
            infer_expr(ctx, lhs, None);
            infer_expr(ctx, rhs, None);
            Ty::Other
        }
        And | Or => {
            let l = infer_expr(ctx, lhs, expected);
            let r = infer_expr(ctx, rhs, expected);
            join(l, r)
        }
    }
}

fn infer_comprehension(
    ctx: &mut Ctx<'_>,
    body: &AstExpr,
    first: &ForClause,
    clauses: &[Clause],
    expected: Option<Ty>,
) {
    let elem_expected = if expected == Some(Ty::ListInt) {
        Some(Ty::Int)
    } else {
        None
    };
    infer_for_clause(ctx, first, elem_expected);
    for clause in clauses {
        match clause {
            Clause::For(f) => infer_for_clause(ctx, f, elem_expected),
            Clause::If(e) => {
                infer_expr(ctx, e, None);
            }
        }
    }
    infer_expr(ctx, body, elem_expected);
}

fn infer_for_clause(ctx: &mut Ctx<'_>, f: &ForClause, elem_expected: Option<Ty>) {
    let over_ty = infer_expr(ctx, &f.over, None);
    let var_ty = match &f.over.node {
        Expr::Call(callee, _) if is_builtin(callee, "range") => Ty::Int,
        Expr::Call(callee, _) if is_builtin(callee, "enumerate") => {
            assign_enumerate_target(ctx, &f.var);
            return;
        }
        Expr::Call(callee, _) if is_builtin(callee, "zip") => {
            assign_zip_target(ctx, &f.var);
            return;
        }
        _ if over_ty == Ty::ListInt => Ty::Int,
        _ if elem_expected == Some(Ty::Int) && over_ty == Ty::Unknown => {
            infer_expr(ctx, &f.over, Some(Ty::ListInt));
            Ty::Int
        }
        _ => Ty::Unknown,
    };
    assign_target(ctx, &f.var, var_ty);
}

fn infer_call(
    ctx: &mut Ctx<'_>,
    callee: &AstExpr,
    args: &starlark_syntax::syntax::ast::CallArgsP<starlark_syntax::syntax::ast::AstNoPayload>,
    expected: Option<Ty>,
) -> Ty {
    // Method call.
    if let Expr::Dot(obj, field) = &callee.node {
        let recv_ty = infer_expr(ctx, obj, None);
        let name = field.node.as_str();
        let arg_tys: Vec<Ty> = args
            .args
            .iter()
            .map(|a| infer_expr(ctx, a.expr(), None))
            .collect();
        return match name {
            "append" => {
                if recv_ty == Ty::Unknown && arg_tys.first() == Some(&Ty::Int) {
                    infer_expr(ctx, obj, Some(Ty::ListInt));
                }
                Ty::Other
            }
            "extend" => {
                if recv_ty == Ty::Unknown && arg_tys.first() == Some(&Ty::ListInt) {
                    infer_expr(ctx, obj, Some(Ty::ListInt));
                }
                Ty::Other
            }
            "get" => {
                if recv_ty == Ty::DictInt {
                    Ty::Int
                } else if expected == Some(Ty::Int) && recv_ty == Ty::Unknown {
                    infer_expr(ctx, obj, Some(Ty::DictInt));
                    Ty::Int
                } else {
                    Ty::Unknown
                }
            }
            "pop" => {
                if recv_ty == Ty::ListInt || recv_ty == Ty::DictInt {
                    Ty::Int
                } else {
                    Ty::Unknown
                }
            }
            "index" | "count" => Ty::Int,
            "elems" | "keys" | "values" | "items" | "split" | "join" | "format" | "strip"
            | "upper" | "lower" | "replace" | "startswith" | "endswith" | "find" | "rfind" => {
                Ty::Other
            }
            _ => Ty::Unknown,
        };
    }

    // Direct call.
    let Expr::Identifier(id) = &callee.node else {
        for a in &args.args {
            infer_expr(ctx, a.expr(), None);
        }
        return Ty::Other;
    };
    let name = id.node.ident.as_str();

    if let Some(func_id) = ctx.resolve_func(name) {
        let sig = ctx.types.funcs.get(&func_id).cloned().unwrap_or(FuncSig {
            params: Vec::new(),
            ret: Ty::Unknown,
        });
        let mut pos = 0;
        for arg in &args.args {
            match &arg.node {
                ArgumentP::Positional(e) => {
                    let expected_param = sig.params.get(pos).copied().unwrap_or(Ty::Unknown);
                    let t = infer_expr(
                        ctx,
                        e,
                        if expected_param == Ty::Unknown {
                            None
                        } else {
                            Some(expected_param)
                        },
                    );
                    if matches!(t, Ty::Int | Ty::ListInt | Ty::DictInt) {
                        if let Some(slot) = ctx
                            .types
                            .funcs
                            .get_mut(&func_id)
                            .and_then(|s| s.params.get_mut(pos))
                        {
                            let new = join(*slot, t);
                            if new != *slot {
                                *slot = new;
                                ctx.changed = true;
                            }
                        }
                    }
                    pos += 1;
                }
                ArgumentP::Named(n, e) => {
                    let t = infer_expr(ctx, e, None);
                    let _ = (n, t);
                }
                ArgumentP::Args(e) | ArgumentP::KwArgs(e) => {
                    infer_expr(ctx, e, None);
                }
            }
        }
        return ctx
            .types
            .funcs
            .get(&func_id)
            .map(|s| s.ret)
            .unwrap_or(Ty::Unknown);
    }

    // Builtins.
    let arg_tys: Vec<Ty> = args
        .args
        .iter()
        .map(|a| infer_expr(ctx, a.expr(), None))
        .collect();
    match name {
        "len" | "ord" | "int" | "hash" => Ty::Int,
        "sum" => match arg_tys.first() {
            Some(Ty::ListInt) => Ty::Int,
            Some(Ty::Unknown) | None => Ty::Unknown,
            _ => Ty::Other,
        },
        "abs" => match arg_tys.first() {
            Some(Ty::Int) => Ty::Int,
            Some(Ty::Unknown) | None => Ty::Unknown,
            _ => Ty::Other,
        },
        "min" | "max" => {
            let all_int = arg_tys.iter().all(|t| *t == Ty::Int) && !arg_tys.is_empty();
            if all_int || arg_tys.first() == Some(&Ty::ListInt) {
                Ty::Int
            } else if arg_tys.contains(&Ty::Unknown) || arg_tys.is_empty() {
                Ty::Unknown
            } else {
                Ty::Other
            }
        }
        "list" => match arg_tys.first() {
            Some(Ty::ListInt) => Ty::ListInt,
            Some(Ty::Unknown) | None => Ty::Unknown,
            _ => Ty::Other,
        },
        "sorted" | "reversed" => match arg_tys.first() {
            Some(Ty::ListInt) => Ty::ListInt,
            Some(Ty::Unknown) | None => Ty::Unknown,
            _ => Ty::Other,
        },
        "getattr" => Ty::Unknown,
        "range" | "enumerate" | "zip" | "str" | "repr" | "chr" | "bool" | "float" | "dict"
        | "tuple" | "hasattr" | "dir" | "print" | "fail" | "struct" | "callable" | "all"
        | "any" => Ty::Other,
        _ => Ty::Unknown,
    }
}
