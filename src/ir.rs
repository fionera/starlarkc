use std::fmt::Write as _;

/// A tiny textual LLVM IR builder.
///
/// We emit textual IR rather than depending on a particular LLVM Rust binding.
/// That keeps the compiler portable across LLVM versions while still producing
/// ordinary LLVM IR that clang/llc can optimize and lower to x86-64.
pub struct Builder {
    pub text: String,
    next_tmp: usize,
    next_label: usize,
    terminated: bool,
}

impl Builder {
    pub fn new() -> Self {
        Self {
            text: String::new(),
            next_tmp: 0,
            next_label: 0,
            terminated: false,
        }
    }

    pub fn tmp(&mut self) -> String {
        let t = format!("%t{}", self.next_tmp);
        self.next_tmp += 1;
        t
    }

    pub fn label(&mut self) -> String {
        let l = format!("bb{}", self.next_label);
        self.next_label += 1;
        l
    }

    pub fn terminated(&self) -> bool {
        self.terminated
    }

    pub fn ins(&mut self, line: impl AsRef<str>) {
        if !self.terminated {
            let _ = writeln!(self.text, "  {}", line.as_ref());
        }
    }

    pub fn emit_label(&mut self, label: &str) {
        let _ = writeln!(self.text, "\n{}:", label);
        self.terminated = false;
    }

    pub fn br(&mut self, label: &str) {
        self.ins(format!("br label %{label}"));
        self.terminated = true;
    }

    pub fn cond_br(&mut self, cond: &str, then_label: &str, else_label: &str) {
        self.ins(format!(
            "br i1 {cond}, label %{then_label}, label %{else_label}"
        ));
        self.terminated = true;
    }

    pub fn ret(&mut self, ty: &str, value: &str) {
        self.ins(format!("ret {ty} {value}"));
        self.terminated = true;
    }

    pub fn ret_void(&mut self) {
        self.ins("ret void");
        self.terminated = true;
    }

    pub fn unreachable(&mut self) {
        self.ins("unreachable");
        self.terminated = true;
    }

    pub fn phi(&mut self, ty: &str, incoming: &[(String, String)]) -> String {
        let t = self.tmp();
        let pairs = incoming
            .iter()
            .map(|(value, label)| format!("[ {value}, %{label} ]"))
            .collect::<Vec<_>>()
            .join(", ");
        self.ins(format!("{t} = phi {ty} {pairs}"));
        t
    }
}

impl Default for Builder {
    fn default() -> Self {
        Self::new()
    }
}
