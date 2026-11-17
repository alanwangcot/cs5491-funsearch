# CS5491: Reasoning Augmented FunSearch
# Author: 王宇浩 (72655080)
# Responsibility: LLM interface and strategy augmentation

"""Small straight-line numeric language for candidate programs.

This filter narrows the attack surface; execution isolation is still mandatory.
"""

from __future__ import annotations

import ast
import hashlib
from types import SimpleNamespace

import numpy as np


class ValidationError(ValueError):
    pass


SAFE_NUMPY_NAMES = frozenset(
    {
        "abs",
        "absolute",
        "add",
        "all",
        "any",
        "arange",
        "argmax",
        "argmin",
        "argsort",
        "array",
        "asarray",
        "ceil",
        "clip",
        "concatenate",
        "cos",
        "cumsum",
        "diff",
        "divide",
        "dot",
        "exp",
        "expm1",
        "fabs",
        "float64",
        "floor",
        "full_like",
        "greater",
        "greater_equal",
        "isclose",
        "isfinite",
        "less",
        "less_equal",
        "log",
        "log1p",
        "logical_and",
        "logical_not",
        "logical_or",
        "max",
        "maximum",
        "mean",
        "median",
        "min",
        "minimum",
        "mod",
        "multiply",
        "ones_like",
        "percentile",
        "power",
        "quantile",
        "remainder",
        "round",
        "sign",
        "sin",
        "sort",
        "sqrt",
        "square",
        "std",
        "subtract",
        "sum",
        "tanh",
        "var",
        "where",
        "zeros_like",
        "pi",
        "e",
    }
)
SAFE_BUILTINS = {
    "abs": abs,
    "min": min,
    "max": max,
    "len": len,
    "float": float,
    "int": int,
    "round": round,
    "bool": bool,
}
_ALLOWED_NODES = (
    ast.Module,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Assign,
    ast.AugAssign,
    ast.Return,
    ast.If,
    ast.Expr,
    ast.Pass,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.Call,
    ast.keyword,
    ast.Attribute,
    ast.Subscript,
    ast.Slice,
    ast.List,
    ast.Tuple,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.BitAnd,
    ast.BitOr,
    ast.BitXor,
    ast.USub,
    ast.UAdd,
    ast.Not,
    ast.Invert,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Is,
    ast.IsNot,
)


class _Check(ast.NodeVisitor):
    def __init__(self, fn: ast.FunctionDef):
        self.names = {"item", "remaining", "np", *SAFE_BUILTINS}
        self.names.update(
            n.id
            for n in ast.walk(fn)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        )

    def generic_visit(self, node):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValidationError(f"forbidden syntax: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Name(self, node):
        if node.id.startswith("_") or node.id not in self.names:
            raise ValidationError(f"forbidden name: {node.id}")
        if isinstance(node.ctx, ast.Store) and node.id in {"np", *SAFE_BUILTINS}:
            raise ValidationError("cannot overwrite a provided numeric primitive")

    def visit_Constant(self, node):
        if not isinstance(node.value, (int, float, bool, type(None))):
            raise ValidationError("only numeric constants are permitted")
        if isinstance(node.value, int) and node.value.bit_length() > 64:
            raise ValidationError("integer literal exceeds 64 bits")

    def visit_Attribute(self, node):
        if node.attr.startswith("_"):
            raise ValidationError("private attributes are forbidden")
        if isinstance(node.value, ast.Name) and node.value.id == "np":
            if node.attr not in SAFE_NUMPY_NAMES:
                raise ValidationError(f"NumPy operation is not allowed: {node.attr}")
        elif node.attr not in {"shape", "size", "ndim"}:
            raise ValidationError(
                "only numeric array shape/size/ndim attributes are allowed"
            )
        self.visit(node.value)

    def visit_Call(self, node):
        fn = node.func
        permitted = (isinstance(fn, ast.Name) and fn.id in SAFE_BUILTINS) or (
            isinstance(fn, ast.Attribute)
            and isinstance(fn.value, ast.Name)
            and fn.value.id == "np"
            and fn.attr in SAFE_NUMPY_NAMES
            and callable(getattr(np, fn.attr))
        )
        if not permitted:
            raise ValidationError("calls must directly use allowed numeric primitives")
        if any(k.arg is None or k.arg.startswith("_") for k in node.keywords):
            raise ValidationError("unpacked/private keyword arguments are forbidden")
        # NumPy out= can mutate a defensive input; reject it statically too.
        if any(k.arg == "out" for k in node.keywords):
            raise ValidationError("out= mutation is forbidden")
        self.generic_visit(node)

    def visit_Assign(self, node):
        if any(not isinstance(t, ast.Name) for t in node.targets):
            raise ValidationError("assignment targets must be local variable names")
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        # In-place numeric operations could mutate the remaining array.
        raise ValidationError("in-place assignment is forbidden; use a new expression")


def validate_code(code: str) -> str:
    """Validate and canonicalize one function, stripping comments and docstrings."""
    if not isinstance(code, str) or not code.strip() or len(code) > 24000:
        raise ValidationError(
            "code must be a nonempty string of at most 24000 characters"
        )
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise ValidationError("invalid Python syntax") from exc
    if len(list(ast.walk(tree))) > 2500:
        raise ValidationError("candidate AST is too large")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValidationError("exactly one top-level score_bins function is required")
    fn = tree.body[0]
    if (
        sum(
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            for n in ast.walk(tree)
        )
        != 1
    ):
        raise ValidationError("nested functions are forbidden")
    if fn.name != "score_bins" or fn.decorator_list or fn.returns:
        raise ValidationError(
            "function must be undecorated score_bins without annotations"
        )
    args = fn.args
    if (
        args.posonlyargs
        or args.kwonlyargs
        or args.vararg
        or args.kwarg
        or args.defaults
        or args.kw_defaults
        or [a.arg for a in args.args] != ["item", "remaining"]
        or any(a.annotation for a in args.args)
    ):
        raise ValidationError("signature must be score_bins(item, remaining)")
    if (
        fn.body
        and isinstance(fn.body[0], ast.Expr)
        and isinstance(fn.body[0].value, ast.Constant)
        and isinstance(fn.body[0].value.value, str)
    ):
        fn.body.pop(0)
    if not fn.body:
        raise ValidationError("function body is empty")
    checker = _Check(fn)
    checker.visit(tree)
    if not any(isinstance(n, ast.Return) for n in ast.walk(fn)):
        raise ValidationError("a return statement is required")
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def code_hash(code: str) -> str:
    return hashlib.sha256(validate_code(code).encode()).hexdigest()


def safe_namespace() -> dict:
    return {
        "__builtins__": dict(SAFE_BUILTINS),
        "np": SimpleNamespace(**{name: getattr(np, name) for name in SAFE_NUMPY_NAMES}),
    }


def compile_candidate(code: str):
    """Compile in the execution worker only; this is not a security sandbox."""
    canonical = validate_code(code)
    namespace = safe_namespace()
    exec(compile(canonical, "<candidate>", "exec"), namespace)
    return namespace["score_bins"]
