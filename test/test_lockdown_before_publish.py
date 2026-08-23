"""A secret-bearing file must be locked down BEFORE it is published.

Applying the owner-only lockdown after the payload is already at its final path
leaves a window in which the file exists under whatever permissions it
inherited. On Windows that is the parent directory's DACL, and POSIX mode bits
are not enforced there at all, so ``atomic_write(mode=0o600)`` does not close
it. Issue #5307 converted the last seven such writers to
``atomic_write(..., restrict_to_owner=True)``, which locks the temp file down
before the first content byte and before the rename.

Nothing prevented a NEW writer from reintroducing the shape. Two layers here:

* ``scripts/check_lockdown_before_publish.py`` is an AST rule over
  ``src/kiro_crew``, exercised below against fixtures for every shape it must
  catch and every correct shape it must not. Validated against real history:
  run against the tree before #5329 it flags 6/6 of #5307's sites; against
  ``main`` after it, 0/6.
* behavioural probes assert the ORDER at the live writers, rather than the
  final mode -- a final-mode assertion passes just as happily when the payload
  was exposed for the whole write window, and on NTFS reports ``0o666``
  regardless of the DACL. The technique (record whether the FINAL path exists
  at the moment lockdown runs) is from PR #5314 by @leonlaiyc, whose
  production change landed via #5329.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from test_live_target import _make_valid_checkout

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKER_PATH = REPO_ROOT / "scripts" / "check_lockdown_before_publish.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("_lockdown_checker", CHECKER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["_lockdown_checker"] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def _functions(source: str) -> set[str]:
    return {fn for _line, fn, _expr in checker.scan_source(source)}


# ─── shapes the rule MUST catch ─────────────────────────────────────────────


VIOLATIONS = {
    "atomic_write then restrict the final path (#5307 tier 2)": """
def write_overlay(target, spec):
    atomic_write(target, spec, mode=0o600)
    platform_compat.restrict_to_owner(target)
""",
    "temp write, publish, then chmod the published path": """
def save(path, payload):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)
    os.chmod(path, 0o600)
""",
    "write_bytes then restrict": """
def persist(path, blob):
    path.write_bytes(blob)
    platform_compat.restrict_to_owner(path)
""",
    "open for write then restrict": """
def persist(path, secret):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(secret)
    platform_compat.restrict_to_owner(path)
""",
    "copy2 into place then restrict (#5346 snapshot shape)": """
def restore(src, dst):
    shutil.copy2(str(src), str(dst))
    platform_compat.restrict_to_owner(str(dst))
""",
    "content through the fd, then chmod (pre-#5329 spool shape)": """
def write_spool(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    os.chmod(path, 0o600)
""",
    "str() wrapper must not hide the match": """
def persist(outfile, blob):
    outfile.write_bytes(blob)
    platform_compat.restrict_to_owner(str(outfile))
""",
}


# ─── shapes the rule MUST NOT catch ─────────────────────────────────────────


CORRECT = {
    "the fix: atomic_write locks the temp before content": """
def write_overlay(target, spec):
    atomic_write(target, spec, restrict_to_owner=True)
""",
    "restrict the temp, then os.replace it into place": """
def save(path, payload):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(payload, encoding="utf-8")
    platform_compat.restrict_to_owner(tmp)
    os.replace(tmp, path)
""",
    "restrict the temp, then Path.rename it into place (#5317 shape)": """
def snapshot(outfile, stage):
    tmp_tar = outfile.with_suffix(".tar.gz.tmp")
    with tarfile.open(str(tmp_tar), "w:gz") as tar:
        tar.add(str(stage))
    platform_compat.restrict_to_owner(str(tmp_tar))
    tmp_tar.rename(outfile)
""",
    "os.open an EMPTY file, restrict it, then write through the fd": """
def write_secret_file(secret_path, secret):
    fd = os.open(str(secret_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    platform_compat.restrict_to_owner(secret_path)
    with os.fdopen(fd, "w") as handle:
        handle.write(secret)
""",
    "read-side re-assert on a path this function never wrote": """
def load(path):
    data = path.read_bytes()
    platform_compat.restrict_to_owner(path)
    return data
""",
    "chmod with an executable mode is not a lockdown": """
def install_launcher(launcher, body):
    launcher.write_text(body, encoding="utf-8")
    platform_compat.chmod_safe(launcher, 0o755)
""",
    "a directory mode set before anything is written into it": """
def ensure_dir(directory):
    directory.mkdir(parents=True, exist_ok=True)
    platform_compat.chmod_safe(directory, 0o700)
""",
    "an explicit reasoned suppression": """
def append(path, lines):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(lines)
    os.chmod(path, 0o600)  # lockdown-ok: re-assert on an existing audit log
""",
}


class TestTheRuleCatchesTheDefect:
    @pytest.mark.parametrize("label", sorted(VIOLATIONS))
    def test_a_write_then_restrict_shape_is_reported(self, label: str) -> None:
        found = checker.scan_source(VIOLATIONS[label])
        assert found, f"the rule missed a real violation: {label}"


class TestTheRuleLeavesCorrectCodeAlone:
    @pytest.mark.parametrize("label", sorted(CORRECT))
    def test_a_correct_shape_is_not_reported(self, label: str) -> None:
        found = checker.scan_source(CORRECT[label])
        assert not found, f"false positive on a correct shape: {label} -> {found}"

    def test_a_suppression_without_a_reason_does_not_suppress(self) -> None:
        """`# lockdown-ok:` with nothing after the colon must not silence it."""
        source = """
def persist(path, blob):
    path.write_bytes(blob)
    platform_compat.restrict_to_owner(path)  # lockdown-ok:
"""
        assert checker.scan_source(source), "a reasonless marker suppressed the finding"


class TestTheRuleAttributesToTheRightFunction:
    def test_the_reported_function_is_the_enclosing_one(self) -> None:
        source = """
def innocent(path):
    return path.read_text(encoding="utf-8")


def guilty(path, blob):
    path.write_bytes(blob)
    platform_compat.restrict_to_owner(path)
"""
        assert _functions(source) == {"guilty"}

    def test_a_write_in_a_sibling_function_does_not_implicate_a_reassert(self) -> None:
        source = """
def writer(path, blob):
    atomic_write(path, blob, restrict_to_owner=True)


def reasserter(path):
    platform_compat.restrict_to_owner(path)
"""
        assert not checker.scan_source(source)


class TestTheRealTree:
    """The gate the CI job runs."""

    def test_src_has_no_unclassified_violation(self) -> None:
        exit_code = checker.main(["check", str(REPO_ROOT / "src" / "kiro_crew")])
        assert exit_code == 0, (
            "a lockdown-before-publish violation is unclassified. Convert it to "
            "atomic_write(..., restrict_to_owner=True), or annotate a genuine "
            "re-assert with `# lockdown-ok: <reason>`."
        )

    def test_every_known_unconverted_entry_still_violates(self) -> None:
        """KNOWN_UNCONVERTED is shrink-only: a paid-off entry must be deleted.

        Without this the list would quietly outlive the debt it tracks, and a
        future regression at one of those very sites would land unnoticed
        because its entry was already there.
        """
        src = REPO_ROOT / "src" / "kiro_crew"
        live: set[str] = set()
        for py in sorted(src.rglob("*.py")):
            for rel, _line, fn, _expr in checker.scan_path(py, REPO_ROOT):
                live.add(f"{rel}::{fn}")

        stale = sorted(set(checker.KNOWN_UNCONVERTED) - live)
        assert not stale, (
            "these KNOWN_UNCONVERTED entries no longer violate -- delete them "
            f"from scripts/{CHECKER_PATH.name}: {stale}"
        )

    def test_every_known_unconverted_entry_names_an_issue(self) -> None:
        bad = {
            key: ref
            for key, ref in checker.KNOWN_UNCONVERTED.items()
            if not ref.startswith("#") or not ref[1:].isdigit()
        }
        assert not bad, f"KNOWN_UNCONVERTED entries must cite an issue: {bad}"


# ─── behavioural probes: ORDER at the live writers ──────────────────────────
#
# Technique from PR #5314 (@leonlaiyc): patch the lockdown helper and record
# whether the FINAL path already exists when it runs. That is a property both
# platforms must satisfy, unlike a final-mode assertion.


@pytest.fixture()
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
    return tmp_path


def _record_final_path_presence(final_path_of):
    """Patch restrict_to_owner, returning the list it records into."""
    from kiro_crew import platform_compat

    seen: list[bool] = []
    real = platform_compat.restrict_to_owner

    def _recording(path, *args, **kwargs):
        seen.append(Path(final_path_of()).exists())
        return real(path, *args, **kwargs)

    return seen, _recording


class TestTheLiveTargetPointerIsLockedDownBeforePublication:
    def test_write_target_never_publishes_an_unprotected_pointer(self, _home) -> None:
        from kiro_crew.service import live_target

        seen, recording = _record_final_path_presence(live_target.pointer_path)
        with patch("kiro_crew.platform_compat.restrict_to_owner", side_effect=recording):
            live_target.write_target(_make_valid_checkout(_home))

        assert seen, "the pointer was written with no owner-only lockdown at all"
        assert not any(seen), (
            "the lockdown ran while the pointer already existed at its final "
            "path -- the payload was published before it was protected"
        )

    def test_restore_never_publishes_an_unprotected_pointer(self, _home) -> None:
        from kiro_crew.service import live_target

        live_target.write_target(_make_valid_checkout(_home))
        prior = live_target.pointer_path().read_text(encoding="utf-8")
        live_target.pointer_path().unlink()

        seen, recording = _record_final_path_presence(live_target.pointer_path)
        with patch("kiro_crew.platform_compat.restrict_to_owner", side_effect=recording):
            live_target.restore(prior)

        assert seen, "restore rewrote the pointer with no lockdown at all"
        assert not any(seen), "restore locked the pointer down only after republishing it"
