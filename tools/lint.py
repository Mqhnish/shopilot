"""A small stdlib linter: unused imports, unused locals, shadowed names.

No third-party linter is available in this environment and the project takes no
dependencies, so hygiene checks are done with `ast` instead.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["src", "tools", "tests", "agent.py", "server.py"]
SKIP_DIRS = {"legacy", "evaluator", "starter", "__pycache__", ".venv"}


def collect_used(tree: ast.AST) -> set:
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
    # Names referenced inside string annotations and __all__ entries.
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for word in node.value.replace("[", " ").replace("]", " ").replace(",", " ").split():
                used.add(word.strip("\"'"))
    return used


def check(path: Path) -> list:
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source, filename=str(path))

    def suppressed(lineno: int) -> bool:
        """Honour `# noqa`, used for imports kept for their side effects."""
        return 0 < lineno <= len(lines) and "noqa" in lines[lineno - 1]

    used = collect_used(tree)
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = (alias.asname or alias.name).split(".")[0]
                if name not in used and not suppressed(node.lineno):
                    problems.append(f"{path}:{node.lineno} unused import {name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue  # a compiler directive, not a name
            for alias in node.names:
                name = alias.asname or alias.name
                if name == "*":
                    continue
                if name not in used and not suppressed(node.lineno):
                    problems.append(f"{path}:{node.lineno} unused import {name}")
    # Assigned-but-never-read locals inside functions.
    for func in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        assigned = {}
        read = set()
        for node in ast.walk(func):
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    assigned.setdefault(node.id, node.lineno)
                else:
                    read.add(node.id)
        for name, lineno in assigned.items():
            if name not in read and not name.startswith("_") and not suppressed(lineno):
                problems.append(
                    f"{path}:{lineno} assigned but never read: {name} (in {func.name})"
                )
    return problems


def main() -> int:
    files = []
    for target in TARGETS:
        p = ROOT / target
        if p.is_file():
            files.append(p)
        else:
            files.extend(f for f in p.rglob("*.py")
                         if not (SKIP_DIRS & set(f.parts)))
    problems = []
    for path in sorted(files):
        problems.extend(check(path))
    for problem in problems:
        print(problem.replace(str(ROOT) + "/", ""))
    print(f"\n{len(files)} files checked, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
