"""Turn root-logger calls (`logging.info(...)`) into calls on a module logger.

    python scripts/codemods/named_loggers.py [PATH ...]   # default: the repo

Each file that calls `logging.<level>(...)` gets `logger = logging.getLogger(__name__)`
after its imports, and its calls become `logger.<level>(...)`. A file that already
has a module logger keeps it and its name. `logging.warn` becomes `.warning`.

Run `ruff check --fix` and `ruff format` on the changed files after it. Files it
cannot change safely are listed at the end with the reason, and left as they were.
Run it again on a branch made before the change to convert the calls that branch
added: it is idempotent.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

LEVELS = {"debug", "info", "warning", "warn", "error", "exception", "critical", "log"}
NAMES = ("logger", "_logger", "_log")
SKIP_DIRS = {".git", "external", "node_modules", "build", "dist", ".venv", "venv"}


def _is_root_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "logging"
        and node.func.attr in LEVELS
    )


def _is_get_logger_name(value: ast.AST) -> bool:
    # logging.getLogger(__name__)
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and isinstance(value.func.value, ast.Name)
        and value.func.value.id == "logging"
        and value.func.attr == "getLogger"
        and len(value.args) == 1
        and isinstance(value.args[0], ast.Name)
        and value.args[0].id == "__name__"
    )


def _existing_logger(tree: ast.Module) -> str | None:
    for stmt in tree.body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and _is_get_logger_name(stmt.value)
        ):
            return stmt.targets[0].id
    return None


def _bindings(tree: ast.Module) -> dict[str, int]:
    """How many times each name is bound or used anywhere in the module."""
    seen: dict[str, int] = {}

    def add(name: str) -> None:
        seen[name] = seen.get(name, 0) + 1

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            add(node.id)
        elif isinstance(node, ast.arg):
            add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            add(node.name)
        elif isinstance(node, ast.alias):
            add((node.asname or node.name).split(".")[0])
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for name in node.names:
                add(name)
    return seen


def _imports_logging_module(tree: ast.Module) -> bool:
    """`logging` is only ever the stdlib module, bound by a plain `import logging`."""
    plain = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "logging" and alias.asname is None:
                    plain = True
                elif (alias.asname or "") == "logging":
                    return False
        elif isinstance(node, ast.ImportFrom):
            if any((a.asname or a.name) == "logging" for a in node.names):
                return False
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            if node.id == "logging":
                return False
        elif isinstance(node, ast.arg) and node.arg == "logging":
            return False
    return plain


def _is_preamble(stmt: ast.stmt, first: bool) -> bool:
    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
        return True
    if first and isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
        return isinstance(stmt.value.value, str)
    # `try: import x / except ImportError: ...` and `if TYPE_CHECKING: import x`
    if isinstance(stmt, (ast.Try, ast.If)):
        body = stmt.body
        return bool(body) and all(
            isinstance(s, (ast.Import, ast.ImportFrom)) for s in body
        )
    return False


def convert(text: str) -> tuple[str | None, str]:
    """The new text, or None and why not."""
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        return None, f"does not parse: {e}"
    calls = [n for n in ast.walk(tree) if _is_root_call(n)]
    if not calls:
        return None, ""
    if not _imports_logging_module(tree):
        return None, "`logging` is not (only) the stdlib module here"

    has_top_import = any(
        isinstance(s, ast.Import)
        and any(a.name == "logging" and a.asname is None for a in s.names)
        for s in tree.body
    )
    name = _existing_logger(tree)
    bindings = _bindings(tree)
    if name is not None:
        # Our own assignment is one binding; any other use of the name must be a
        # reference to it, which a module that already logs through it would have.
        if bindings.get(name, 0) > 1 and any(
            isinstance(n, ast.arg) and n.arg == name for n in ast.walk(tree)
        ):
            return None, f"`{name}` is also a parameter name"
        insert_after = None
    else:
        name = next((n for n in NAMES if n not in bindings), None)
        if name is None:
            return None, "every candidate logger name is taken"
        first_call = min(c.lineno for c in calls)
        last = 0
        logging_at = None
        for i, stmt in enumerate(tree.body):
            # Stop before a preamble block that itself logs (an import in a try).
            if not _is_preamble(stmt, i == 0) or (stmt.end_lineno or 0) >= first_call:
                break
            last = stmt.end_lineno or stmt.lineno
            if isinstance(stmt, ast.Import) and any(
                a.name == "logging" for a in stmt.names
            ):
                logging_at = last
        if has_top_import and logging_at is None:
            return None, f"calls logging before `import logging` (line {first_call})"
        insert_after = last

    lines = text.splitlines(keepends=True)
    # ast column offsets are UTF-8 byte offsets; edit each line as bytes, right to left.
    edits: dict[int, list[tuple[int, int, str]]] = {}
    for call in calls:
        value = call.func.value
        attr = "warning" if call.func.attr == "warn" else call.func.attr
        end = value.end_col_offset + 1 + len(call.func.attr)
        if call.func.end_lineno != value.lineno:
            return None, f"call split over lines at {value.lineno}"
        edits.setdefault(value.lineno, []).append(
            (value.col_offset, end, f"{name}.{attr}")
        )
    for lineno, spans in edits.items():
        raw = lines[lineno - 1].encode("utf-8")
        for start, end, new in sorted(spans, reverse=True):
            if not raw[start:end].startswith(b"logging."):
                return None, f"unexpected call text at line {lineno}"
            raw = raw[:start] + new.encode("utf-8") + raw[end:]
        lines[lineno - 1] = raw.decode("utf-8")

    if insert_after is not None:
        block = f"\n{name} = logging.getLogger(__name__)\n"
        if not has_top_import:
            block = "\nimport logging\n" + block
        if insert_after == 0:
            lines.insert(0, block.lstrip("\n") + "\n")
        else:
            if not lines[insert_after - 1].endswith("\n"):
                lines[insert_after - 1] += "\n"
            lines.insert(insert_after, block)
    return "".join(lines), ""


def _files(paths: list[Path]):
    for path in paths:
        if path.is_file():
            yield path
            continue
        for file in sorted(path.rglob("*.py")):
            if not SKIP_DIRS.intersection(file.parts):
                yield file


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv] or [Path(__file__).resolve().parents[2]]
    changed, skipped = [], []
    for file in _files(paths):
        with open(file, encoding="utf-8", newline="") as f:
            text = f.read()
        new, why = convert(text)
        if new is None:
            if why:
                skipped.append((file, why))
            continue
        if new != text:
            with open(file, "w", encoding="utf-8", newline="") as f:
                f.write(new)
            changed.append(file)
    for file in changed:
        print(file)
    for file, why in skipped:
        print(f"SKIPPED {file}: {why}", file=sys.stderr)
    print(f"{len(changed)} changed, {len(skipped)} skipped", file=sys.stderr)
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
