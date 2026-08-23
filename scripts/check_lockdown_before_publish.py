#!/usr/bin/env python3
"""Refuse an owner-only lockdown applied to a published path AFTER content.

A secret-bearing file must be locked down BEFORE any content reaches it.
Applying the lockdown once the payload is already at its final path leaves a
window in which the file exists under whatever permissions it inherited -- on
Windows that is the parent directory's DACL, and POSIX mode bits are not
enforced there at all, so ``atomic_write(mode=0o600)`` does not close it.

Issue #5307 converted seven such writers to
``atomic_write(..., restrict_to_owner=True)``, which locks the temp file down
before the first content byte and before the rename. Nothing prevented a NEW
writer from reintroducing the shape; this checker does.

A violation is, within ONE function body, an owner-only lockdown applied to a
path expression that an EARLIER statement in that same function already wrote
content to, where that path is the PUBLISHED one.

Deliberately NOT violations -- flagging these would make the gate worse than no
gate:

* a lockdown on a temp path the same function then renames into place, in
  either spelling (``os.replace(tmp, final)`` or ``tmp.rename(final)``): the
  published file is never exposed, and this is the shape ``atomic_write`` uses;
* the delegated form ``atomic_write(..., restrict_to_owner=True)``, which is
  the fix this checker exists to protect;
* ``chmod``/``chmod_safe`` with a non-restrictive mode -- making a launcher
  executable (``0o755``) is not a lockdown;
* ``fd = os.open(p, O_CREAT...)`` then ``restrict_to_owner(p)`` then writing
  through the descriptor: ``os.open`` creates an EMPTY file, so the lockdown
  lands before any content. The content-write moment for that idiom is
  ``os.fdopen(fd, ...)``, and only a lockdown after THAT is a violation;
* a site carrying ``# lockdown-ok: <reason>`` on the lockdown line, for a
  re-assert or a file that holds no data. The reason is mandatory.

Usage:
    python scripts/check_lockdown_before_publish.py [PATH ...]

Exits non-zero and prints ``file:line`` per violation. With no arguments it
scans ``src/kiro_crew``. Sites tracked by an open conversion issue live in
``KNOWN_UNCONVERTED``; that list is shrink-only.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

#: Unambiguous lockdown helper: always means "owner-only, and nobody else".
RESTRICT_CALL = "restrict_to_owner"

#: ``chmod``-family calls are a lockdown only with an owner-only mode. The same
#: helpers set directory modes and executable bits.
CHMOD_CALLS = frozenset({"chmod_safe", "chmod"})
OWNER_ONLY_MODES = frozenset({0o600, 0o400, 0o700, 0o500})

#: Calls whose Nth positional argument is a path that RECEIVES content.
#: ``replace``/``rename`` are included because the rename is how the published
#: path gets its content in the temp-file idiom -- a lockdown after it is the
#: exact defect (see ``workflows/store.py`` before this checker landed).
WRITE_DESTINATION_ARG = {
    "atomic_write": 0,
    "copy2": 1,  # shutil.copy2(src, dst)
    "copyfile": 1,
    "copy": 1,
    "replace": 1,  # os.replace(tmp, final)
    "rename": 1,  # os.rename(tmp, final)
}

#: ``<path>.write_text(...)`` / ``.write_bytes(...)`` -- receiver is the path.
WRITE_METHODS = frozenset({"write_text", "write_bytes"})

#: ``<tmp>.rename(final)`` / ``<tmp>.replace(final)`` -- receiver is the TEMP.
PUBLISH_CALLS = frozenset({"replace", "rename"})

WRITE_MODE_CHARS = ("w", "a", "x", "+")

#: ``os.fdopen(fd, ...)`` is where content starts flowing for the
#: os.open-then-lock-then-write idiom; the fd is mapped back to its path.
FDOPEN_CALL = "fdopen"

#: ``# lockdown-ok: <reason>`` -- reason is mandatory and must be non-empty.
SUPPRESS_RE = re.compile(r"#\s*lockdown-ok\s*:\s*(?P<reason>\S.*)$")

#: Sites carrying the shape under an OPEN conversion issue. Shrink-only: when a
#: site is converted, delete its entry -- the checker fails if an entry here no
#: longer violates, so the list cannot outlive the debt it tracks.
KNOWN_UNCONVERTED: dict[str, str] = {
    # Enumerated by #5285, still open.
    "src/kiro_crew/aws_consent.py::_preserve_if_unreadable": "#5285",
    "src/kiro_crew/aws_consent.py::_write_all": "#5285",
    "src/kiro_crew/dashboard/handlers/messaging.py::_write_env_updates": "#5285",
    "src/kiro_crew/dashboard/handlers/weixin_qr.py::_atomic_write": "#5285",
    "src/kiro_crew/dashboard/session_transfer.py::_write_layer_b_files": "#5285",
    "src/kiro_crew/tips.py::_save_state": "#5285",
    # Classified by #5346 -- the sites #5307's third acceptance criterion left
    # untriaged. The two snapshot ones are a shutil.copy2 into a final
    # destination, so they need copy-to-temp + restrict + replace rather than a
    # one-line atomic_write swap.
    "src/kiro_crew/snapshot.py::_backup_and_copy": "#5346",
    "src/kiro_crew/snapshot.py::_do_merge": "#5346",
    "src/kiro_crew/workflows/store.py::save": "#5346",
    "src/kiro_crew/sel.py::_append_lines_locked": "#5346",
}


def _norm(expr: str | None) -> str:
    """Normalise a path expression so spellings of one path compare equal."""
    if not expr:
        return ""
    text = " ".join(expr.split())
    changed = True
    while changed:
        changed = False
        for wrapper in ("str(", "Path(", "os.fspath("):
            if text.startswith(wrapper) and text.endswith(")"):
                text = text[len(wrapper) : -1].strip()
                changed = True
    return text


def _callee(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _seg(source: str, node: ast.AST) -> str:
    try:
        return ast.get_source_segment(source, node) or ""
    except Exception:  # pragma: no cover - get_source_segment is lenient
        return ""


def _mode_of(node: ast.Call) -> object:
    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
        return node.args[1].value
    for kw in node.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            return kw.value.value
    return None


def _opens_for_write(node: ast.Call) -> bool:
    """``open(p, "w")`` -- a builtin open in a writable text/binary mode.

    ``os.open`` is deliberately NOT a content write: it creates an empty file
    and the payload arrives through the descriptor, which is why the correct
    idiom locks the empty file down between the two.
    """
    mode = _mode_of(node)
    if not isinstance(mode, str):
        return False
    return any(ch in mode for ch in WRITE_MODE_CHARS)


def _fd_paths(func: ast.AST, source: str) -> dict[str, str]:
    """``fd variable -> path`` for each ``fd = os.open(path, ...)``."""
    out: dict[str, str] = {}
    for node in ast.walk(func):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        value = node.value
        if not isinstance(target, ast.Name) or not isinstance(value, ast.Call):
            continue
        if _callee(value) == "open" and isinstance(value.func, ast.Attribute) and value.args:
            out[target.id] = _norm(_seg(source, value.args[0]))
    return out


def _write_target(
    node: ast.Call, source: str, fd_paths: dict[str, str] | None = None
) -> str | None:
    """The path this call writes content to, if any."""
    name = _callee(node)
    if name is None:
        return None

    # `open(p, "w")` -- but NOT `os.open`, which only creates the file.
    if name == "open" and node.args and isinstance(node.func, ast.Name):
        if _opens_for_write(node):
            return _norm(_seg(source, node.args[0]))
        return None

    # `os.fdopen(fd, ...)` -- content starts flowing into the fd's path here.
    if name == FDOPEN_CALL and node.args and fd_paths:
        first = node.args[0]
        if isinstance(first, ast.Name):
            return fd_paths.get(first.id)
        return None

    if name in WRITE_METHODS and isinstance(node.func, ast.Attribute):
        return _norm(_seg(source, node.func.value))

    index = WRITE_DESTINATION_ARG.get(name)
    if index is None:
        return None
    if name == "atomic_write" and any(kw.arg == "restrict_to_owner" for kw in node.keywords):
        # Locks the temp down before content -- the fix, not the defect.
        return None
    if len(node.args) > index:
        return _norm(_seg(source, node.args[index]))
    return None


def _lockdown_target(node: ast.Call, source: str) -> str | None:
    name = _callee(node)
    if not node.args:
        return None
    if name == RESTRICT_CALL:
        return _norm(_seg(source, node.args[0]))
    if name in CHMOD_CALLS:
        mode = _mode_of(node)
        if isinstance(mode, int) and mode in OWNER_ONLY_MODES:
            return _norm(_seg(source, node.args[0]))
    return None


def _temp_sources(node: ast.Call, source: str) -> list[str]:
    """Paths this call publishes UNDER ANOTHER NAME, in either spelling.

    ``os.replace(tmp, final)`` -- the temp is the first argument.
    ``tmp.rename(final)``      -- the temp is the receiver.
    """
    if _callee(node) not in PUBLISH_CALLS:
        return []
    found = []
    if isinstance(node.func, ast.Attribute) and len(node.args) == 1:
        found.append(_norm(_seg(source, node.func.value)))
    if node.args:
        first = _norm(_seg(source, node.args[0]))
        if len(node.args) >= 2:
            found.append(first)
    return [f for f in found if f]


def _suppressions(source: str) -> dict[int, str]:
    """``lineno -> reason`` for every ``# lockdown-ok:`` marker."""
    out: dict[int, str] = {}
    for i, line in enumerate(source.splitlines(), 1):
        match = SUPPRESS_RE.search(line)
        if match:
            out[i] = match.group("reason").strip()
    return out


def scan_source(source: str) -> list[tuple[int, str, str]]:
    """Return ``(lineno, function, path_expression)`` per violation."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    suppressed = _suppressions(source)
    violations: list[tuple[int, str, str]] = []

    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        writes: list[tuple[int, str]] = []
        lockdowns: list[tuple[int, str]] = []
        temps: set[str] = set()
        fd_paths = _fd_paths(func, source)

        for node in ast.walk(func):
            if not isinstance(node, ast.Call):
                continue
            target = _write_target(node, source, fd_paths)
            if target:
                writes.append((node.lineno, target))
            locked = _lockdown_target(node, source)
            if locked:
                lockdowns.append((node.lineno, locked))
            temps.update(_temp_sources(node, source))

        for line, path in lockdowns:
            if path in temps:
                continue  # published under another name
            if line in suppressed:
                continue  # reasoned re-assert / dataless file
            if any(w_path == path and w_line < line for w_line, w_path in writes):
                violations.append((line, func.name, path))

    return sorted(set(violations))


def scan_path(path: Path, root: Path) -> list[tuple[str, int, str, str]]:
    source = path.read_text(encoding="utf-8", errors="replace")
    rel = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
    return [(rel, line, fn, expr) for line, fn, expr in scan_source(source)]


def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parent.parent
    targets = [Path(a) for a in argv[1:]] or [root / "src" / "kiro_crew"]

    found: list[tuple[str, int, str, str]] = []
    for target in targets:
        files = sorted(target.rglob("*.py")) if target.is_dir() else [target]
        for py in files:
            found.extend(scan_path(py, root))

    new: list[tuple[str, int, str, str]] = []
    seen_known: set[str] = set()
    for rel, line, fn, expr in found:
        key = "%s::%s" % (rel, fn)
        if key in KNOWN_UNCONVERTED:
            seen_known.add(key)
        else:
            new.append((rel, line, fn, expr))

    stale = sorted(set(KNOWN_UNCONVERTED) - seen_known)

    if new:
        print("Lockdown-before-publish check FAILED:\n")
        for rel, line, fn, expr in new:
            print(
                "  %s:%d: `%s` is locked down only AFTER content was written to "
                "it in %s().\n      Use atomic_write(..., restrict_to_owner=True) "
                "so the temp file is locked down before the payload and before "
                "the rename (issue #5307). If this is a load-time re-assert or a "
                "file that holds no data, annotate the lockdown line with "
                "`# lockdown-ok: <reason>`." % (rel, line, expr, fn)
            )
        print("\n%d new violation(s)." % len(new))

    if stale:
        print(
            "\nLockdown-before-publish check FAILED: %d KNOWN_UNCONVERTED entry(ies) "
            "no longer violate -- the debt was paid, so delete the entry.\n"
            "Edit KNOWN_UNCONVERTED in scripts/%s and remove:\n" % (len(stale), Path(__file__).name)
        )
        for key in stale:
            print("  %s   (was tracked by %s)" % (key, KNOWN_UNCONVERTED[key]))

    if new or stale:
        return 1

    if seen_known:
        print(
            "Lockdown-before-publish check passed "
            "(%d known unconverted site(s) tracked by an open issue)." % len(seen_known)
        )
    else:
        print("Lockdown-before-publish check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
