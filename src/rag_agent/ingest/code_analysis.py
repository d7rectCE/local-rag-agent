"""Static analysis of Python source (ТЗ S2, S3): symbol definitions, call sites,
names a cell defines and uses. Works for modules and notebook code cells."""

from __future__ import annotations

import ast
import builtins
import re
from dataclasses import dataclass, field

from rag_agent.schema import CallSite, Symbol

_MAGIC_RE = re.compile(r"^\s*[%!]")
_BUILTINS = set(dir(builtins))


def strip_magics(source: str) -> str:
    """Blank out IPython magics and shell escapes, keeping line numbers intact."""
    return "\n".join("" if _MAGIC_RE.match(line) else line for line in source.split("\n"))


def first_doc_line(node: ast.AST) -> str:
    doc = ast.get_docstring(node) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)) else None
    return doc.strip().split("\n", 1)[0] if doc else ""


def signature(node: ast.AST) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        ret = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
        return f"def {node.name}({ast.unparse(node.args)}){ret}"
    if isinstance(node, ast.ClassDef):
        bases = ", ".join(ast.unparse(b) for b in node.bases)
        return f"class {node.name}({bases})" if bases else f"class {node.name}"
    return ""


def start_line(node: ast.AST) -> int:
    """First line of a definition including its decorators."""
    decorators = getattr(node, "decorator_list", [])
    return min([node.lineno] + [d.lineno for d in decorators])


@dataclass
class Analysis:
    symbols: list[Symbol] = field(default_factory=list)
    calls: list[CallSite] = field(default_factory=list)
    imports: set[str] = field(default_factory=set)
    defines: set[str] = field(default_factory=set)  # names bound at the top level
    uses: set[str] = field(default_factory=set)  # names read before this source binds them


class _Visitor(ast.NodeVisitor):
    def __init__(self, cell: int | None):
        self.cell = cell
        self.stack: list[tuple[str, str]] = []  # (kind, name)
        self.out = Analysis()

    def _qualname(self, name: str) -> str:
        return ".".join([n for _, n in self.stack] + [name])

    def _define(self, node, kind: str) -> None:
        in_class = bool(self.stack) and self.stack[-1][0] == "class"
        if kind == "function" and in_class:
            kind = "method"
        self.out.symbols.append(
            Symbol(
                name=node.name,
                qualname=self._qualname(node.name),
                kind=kind,
                signature=signature(node),
                doc=first_doc_line(node),
                line_start=start_line(node),
                line_end=node.end_lineno or node.lineno,
                cell=self.cell,
            )
        )
        self.stack.append(("class" if kind == "class" else "function", node.name))
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node):
        self._define(node, "function")

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self._define(node, "class")

    def visit_Import(self, node):
        self.out.imports.update(alias.name for alias in node.names)

    def visit_ImportFrom(self, node):
        if node.module:
            self.out.imports.add(node.module)

    def visit_Call(self, node):
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        if name:
            caller = ".".join(n for _, n in self.stack) or None
            full = ast.unparse(func) if isinstance(func, (ast.Name, ast.Attribute)) else name
            self.out.calls.append(CallSite(name=name, full_name=full[:200], caller=caller, line=node.lineno, cell=self.cell))
        self.generic_visit(node)


_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _names(node: ast.AST, ctx: type) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ctx)}


def _loads_and_stores(node: ast.AST) -> tuple[set[str], set[str]]:
    """Names read and bound in the enclosing (module / cell) scope. Bodies of
    functions, classes and lambdas run later in their own scope and are skipped;
    comprehension variables are local to the comprehension."""
    loads: set[str] = set()
    stores: set[str] = set()
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            stores.add(child.name)
            for expr in [*child.decorator_list, *getattr(child, "bases", [])]:
                loads |= _names(expr, ast.Load)
            continue
        if isinstance(child, ast.Lambda):
            continue
        if isinstance(child, _COMPREHENSIONS):
            local = set().union(*(_names(g.target, ast.Store) for g in child.generators))
            loads |= _names(child, ast.Load) - local
            continue
        if isinstance(child, (ast.Import, ast.ImportFrom)):
            stores.update((a.asname or a.name).split(".")[0] for a in child.names if a.name != "*")
            continue
        if isinstance(child, ast.Name):
            (loads if isinstance(child.ctx, ast.Load) else stores).add(child.id)
            continue
        sub_loads, sub_stores = _loads_and_stores(child)
        loads |= sub_loads
        stores |= sub_stores
    return loads, stores


def def_use(tree: ast.Module) -> tuple[set[str], set[str]]:
    """(defines, uses) of a module or notebook cell, statement by statement: a name
    counts as used when it is read before this source binds it (``df = df.dropna()``
    uses the ``df`` of an earlier cell)."""
    defined: set[str] = set()
    uses: set[str] = set()
    for stmt in tree.body:
        loads, stores = _loads_and_stores(ast.Module(body=[stmt], type_ignores=[]))
        if isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
            loads.add(stmt.target.id)
        uses |= loads - defined - _BUILTINS
        defined |= stores
    return defined, uses


def analyze(source: str, cell: int | None = None) -> Analysis:
    """Raises SyntaxError for unparsable code."""
    tree = ast.parse(source)
    visitor = _Visitor(cell)
    visitor.visit(tree)
    out = visitor.out
    out.defines, out.uses = def_use(tree)
    return out
