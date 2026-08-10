"""Tests for the git revision lookup.

The overriding requirement for fibsem.versioning is that it can never raise:
it runs inside a desktop application driving microscope hardware, where a
version-string lookup taking down the app is unacceptable. Most of what follows
is therefore failure-mode coverage rather than happy-path coverage.
"""

import logging
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from fibsem import versioning
from fibsem.versioning import get_branch, get_revision, get_version_string

# The repository root as seen from *this file*, not from fibsem.__file__: in CI
# the package is pip-installed into site-packages, so the real lookup correctly
# returns None there while the checkout is still on disk at the workspace root.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Either `v0.5.1-48-g4cd11d9c` from a tagged clone, or a bare short sha from a
# shallow/tagless one (which is what actions/checkout produces by default),
# each optionally suffixed `-dirty`.
DESCRIBE_RE = re.compile(r"^(.+-g)?[0-9a-f]{7,40}(-dirty)?$")


def _clear_caches():
    # Tests that monkeypatch _describe/_branch wholesale replace them with plain
    # functions, and monkeypatch may not have restored them by the time this runs
    # on teardown — so clear only if there is a cache to clear. The setup-side
    # call is the load-bearing one; a replaced function never populated the real
    # cache in the first place.
    for name in ("_describe", "_branch"):
        clear = getattr(getattr(versioning, name), "cache_clear", None)
        if clear is not None:
            clear()


@pytest.fixture(autouse=True)
def _clear_revision_caches():
    """Drop the process-global caches around every test.

    _describe and _branch are cached for the lifetime of the interpreter, and
    `pytest -n auto --dist worksteal` reuses worker processes across tests, so
    without this a stubbed result leaks into whichever test runs next.
    """
    _clear_caches()
    yield
    _clear_caches()


def _completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(
        args=["git"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _git(repo, *args):
    """Run git in `repo`, independently of the developer's own git config.

    identity is passed with -c rather than assumed: CI has no global user.email,
    and signing is forced off so the test does not prompt for a key (or fail) on
    a machine that signs commits and tags by default.
    """
    return subprocess.run(
        [
            "git",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "user.name=test",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "tag.gpgsign=false",
            *args,
        ],
        cwd=str(repo),
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    ).stdout.strip()


# --- happy path ------------------------------------------------------------


@pytest.mark.skipif(
    not (REPO_ROOT / ".git").exists(),
    reason="not running from a source checkout",
)
def test_revision_and_branch_from_real_checkout(monkeypatch):
    """Read a real repository, from a working directory outside it.

    tests/conftest.py chdirs every test into its own tmp_path, so this only
    passes if the lookup honours the explicit cwd= it derives from __file__
    rather than the process cwd.
    """
    monkeypatch.setattr(versioning, "_source_checkout_root", lambda: REPO_ROOT)

    revision = get_revision()
    assert revision is not None
    assert DESCRIBE_RE.match(revision), revision

    # Cross-check against git itself rather than hardcoding a sha.
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    ).stdout.strip()
    assert head in revision

    # Branch is None on a detached HEAD, which is legitimate; when present it
    # must not be the literal "HEAD" that `rev-parse --abbrev-ref` would give.
    branch = get_branch()
    assert branch != "HEAD"


def test_version_string_includes_revision_in_checkout(monkeypatch):
    monkeypatch.setattr(versioning, "_describe", lambda: "v0.5.1-48-g4cd11d9c")

    import fibsem

    assert get_version_string() == f"{fibsem.__version__} (v0.5.1-48-g4cd11d9c)"


# --- not a checkout --------------------------------------------------------


def test_no_checkout_does_not_fork(monkeypatch):
    """A wheel install must take the zero-fork fast path."""
    calls = []

    def _record_and_boom(*args, **kwargs):
        calls.append(args)
        raise AssertionError("subprocess.run must not be reached")

    monkeypatch.setattr(versioning, "_source_checkout_root", lambda: None)
    monkeypatch.setattr(versioning.subprocess, "run", _record_and_boom)

    assert get_revision() is None
    assert get_branch() is None
    assert calls == []


def test_directory_that_is_not_a_repo(monkeypatch, tmp_path):
    """git exits non-zero; we return None rather than propagating."""
    # Keep git from walking up out of tmp_path and finding a real repo, which
    # would otherwise make this depend on where TMPDIR happens to live. This is
    # why GIT_CEILING_DIRECTORIES is excluded from the scrubbed variables.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    monkeypatch.setattr(versioning, "_source_checkout_root", lambda: tmp_path)

    assert get_revision() is None
    assert get_branch() is None


def test_source_checkout_root_requires_both_markers(monkeypatch, tmp_path):
    """.git alone is not enough — an unrelated enclosing repo must not match."""
    # Pretend the package lives at <tmp_path>/fibsem/versioning.py, so the root
    # the module derives from __file__ is tmp_path.
    monkeypatch.setattr(
        versioning, "__file__", str(tmp_path / "fibsem" / "versioning.py")
    )

    assert versioning._source_checkout_root() is None  # neither marker

    (tmp_path / ".git").mkdir()
    assert versioning._source_checkout_root() is None  # .git but no pyproject

    (tmp_path / "pyproject.toml").touch()
    assert versioning._source_checkout_root() == tmp_path  # both


def test_source_checkout_root_accepts_worktree_git_file(monkeypatch, tmp_path):
    """.git is a *file* in a git worktree, so the guard cannot use is_dir()."""
    monkeypatch.setattr(
        versioning, "__file__", str(tmp_path / "fibsem" / "versioning.py")
    )
    (tmp_path / ".git").write_text("gitdir: /elsewhere/.git/worktrees/wt\n")
    (tmp_path / "pyproject.toml").touch()

    assert versioning._source_checkout_root() == tmp_path


# --- failure modes ---------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        FileNotFoundError(2, "No such file or directory: 'git'"),
        subprocess.TimeoutExpired(cmd=["git"], timeout=2.0),
        PermissionError(13, "Permission denied"),
        RuntimeError("kaboom"),
    ],
    ids=["git-missing", "timeout", "spawn-blocked", "arbitrary"],
)
def test_subprocess_failures_return_none(monkeypatch, exc):
    def _boom(*args, **kwargs):
        raise exc

    monkeypatch.setattr(versioning, "_source_checkout_root", lambda: REPO_ROOT)
    monkeypatch.setattr(versioning.subprocess, "run", _boom)

    assert get_revision() is None
    assert get_branch() is None


@pytest.mark.parametrize(
    "completed",
    [
        _completed(returncode=128, stderr="fatal: not a git repository"),
        _completed(returncode=1, stdout=""),
        _completed(returncode=0, stdout="   \n"),
    ],
    ids=["rc-128", "rc-1-empty", "rc-0-whitespace"],
)
def test_unusable_git_output_returns_none(monkeypatch, completed):
    monkeypatch.setattr(versioning, "_source_checkout_root", lambda: REPO_ROOT)
    monkeypatch.setattr(versioning.subprocess, "run", lambda *a, **k: completed)

    assert get_revision() is None
    assert get_branch() is None


def test_root_lookup_survives_permission_error(monkeypatch):
    """Path.exists() propagates EACCES — the guard must be inside the try."""

    def _boom(self):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(versioning.Path, "exists", _boom)

    assert versioning._source_checkout_root() is None
    assert get_revision() is None


# --- command construction --------------------------------------------------


def test_branch_uses_symbolic_ref(monkeypatch):
    """Pins the detached-HEAD decision.

    `rev-parse --abbrev-ref HEAD` prints the literal string "HEAD" when detached;
    `symbolic-ref --quiet` exits 1 with no output, which _run_git already maps to
    None. Assert the argv so nobody silently swaps it back.
    """
    seen = {}

    def _capture(argv, **kwargs):
        seen["argv"] = argv
        return _completed(returncode=1, stdout="")

    monkeypatch.setattr(versioning, "_source_checkout_root", lambda: REPO_ROOT)
    monkeypatch.setattr(versioning.subprocess, "run", _capture)

    assert get_branch() is None
    assert seen["argv"] == ["git", "symbolic-ref", "--quiet", "--short", "HEAD"]


def test_describe_argv(monkeypatch):
    seen = {}

    def _capture(argv, **kwargs):
        seen["argv"] = argv
        return _completed(returncode=0, stdout="v0.5.1-48-g4cd11d9c\n")

    monkeypatch.setattr(versioning, "_source_checkout_root", lambda: REPO_ROOT)
    monkeypatch.setattr(versioning.subprocess, "run", _capture)

    assert get_revision() == "v0.5.1-48-g4cd11d9c"
    assert seen["argv"] == [
        "git",
        "describe",
        "--tags",
        "--always",
        "--dirty",
        "--match",
        "v*",
    ]


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_describe_measures_from_release_tags_only(monkeypatch, tmp_path):
    """A one-off build tag nearer to HEAD must not become the base tag.

    git describe picks the *nearest* reachable tag regardless of what it means,
    so tagging a build for a partner site would silently re-base the revision
    string recorded in every downstream user's experiment metadata.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "commit", "--allow-empty", "--quiet", "-m", "first")
    # Lightweight, as RELEASE.md creates them.
    _git(repo, "tag", "v1.0.0")
    _git(repo, "commit", "--allow-empty", "--quiet", "-m", "second")
    # Annotated, as the real 251111-rosalind is: nearer to HEAD *and* the type
    # plain `git describe` prefers, so it wins on both of git's selection rules.
    _git(repo, "tag", "-a", "251111-rosalind", "-m", "build for a partner site")
    _git(repo, "commit", "--allow-empty", "--quiet", "-m", "third")

    # Control: assert the bug is actually reachable in this repo, so the test
    # below cannot pass simply because the decoy was never a candidate.
    unfiltered = _git(repo, "describe", "--tags", "--always", "--dirty")
    assert unfiltered.startswith("251111-rosalind-"), unfiltered

    monkeypatch.setattr(versioning, "_source_checkout_root", lambda: repo)

    revision = get_revision()
    assert revision.startswith("v1.0.0-"), revision


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_describe_still_falls_back_to_bare_sha(monkeypatch, tmp_path):
    """--match must not cost the --always fallback.

    Filtering to `v*` means a checkout can now have tags and still have no
    candidate. That must degrade to the bare sha, as it does for a tagless
    clone, and not to None — the sha is the half of the revision that matters.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "commit", "--allow-empty", "--quiet", "-m", "first")
    _git(repo, "tag", "-a", "20250616", "-m", "dated build, not a release")

    monkeypatch.setattr(versioning, "_source_checkout_root", lambda: repo)

    revision = get_revision()
    assert revision is not None
    assert not revision.startswith("20250616"), revision
    assert re.fullmatch(r"[0-9a-f]{7,40}", revision), revision


def test_environment_is_scrubbed(monkeypatch):
    """GIT_DIR and friends override cwd=, so they must not reach the child."""
    seen = {}

    def _capture(argv, **kwargs):
        seen.update(kwargs)
        return _completed(returncode=0, stdout="abc1234\n")

    monkeypatch.setenv("GIT_DIR", "/nope/.git")
    monkeypatch.setenv("GIT_WORK_TREE", "/nope")
    monkeypatch.setattr(versioning, "_source_checkout_root", lambda: REPO_ROOT)
    monkeypatch.setattr(versioning.subprocess, "run", _capture)

    assert get_revision() == "abc1234"

    env = seen["env"]
    assert "GIT_DIR" not in env
    assert "GIT_WORK_TREE" not in env
    # Read-only lookup must not take .git/index.lock, which --dirty otherwise does.
    assert env["GIT_OPTIONAL_LOCKS"] == "0"
    # Guards against anyone "simplifying" to a minimal env, which breaks git on
    # Windows (no SystemRoot) and anywhere git is not on the default PATH.
    assert "PATH" in env
    assert seen["timeout"] == versioning._GIT_TIMEOUT_SECONDS
    assert seen["cwd"] == str(REPO_ROOT)


# --- caching ---------------------------------------------------------------


def test_repeated_calls_fork_once(monkeypatch):
    calls = []

    def _count(*args):
        calls.append(args)
        return "v0.5.1-48-g4cd11d9c"

    monkeypatch.setattr(versioning, "_run_git", _count)

    for _ in range(5):
        assert get_revision() == "v0.5.1-48-g4cd11d9c"
    assert len(calls) == 1


def test_metadata_construction_forks_once(monkeypatch):
    """Reading many images must not mean one git fork per image.

    SystemInfo defaults fibsem_revision from get_revision(), and from_dict falls back
    to it for any file that predates the field -- so loading a directory of images
    calls it once per image. Without the cache on _describe that is a subprocess each
    time, which is the regression this test exists to prevent.

    Until FIB-448 the motivating case was FibsemExperimentRef, which carried its own
    copy of the revision and was built per acquired image. That duplicate is gone;
    SystemInfo owns the field now, and the invariant is unchanged.
    """
    from fibsem.structures import SystemInfo

    calls = []

    def _count(*args):
        calls.append(args)
        return "v0.5.1-48-g4cd11d9c"

    monkeypatch.setattr(versioning, "_run_git", _count)

    infos = [SystemInfo.from_dict({}) for _ in range(50)]

    assert len(calls) == 1
    assert all(i.fibsem_revision == "v0.5.1-48-g4cd11d9c" for i in infos)


# --- logging ---------------------------------------------------------------


def test_failure_is_logged_quietly(monkeypatch, caplog):
    """For a wheel install "no git" is the normal case, so it must not warn."""

    def _boom(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory: 'git'")

    monkeypatch.setattr(versioning, "_source_checkout_root", lambda: REPO_ROOT)
    monkeypatch.setattr(versioning.subprocess, "run", _boom)

    with caplog.at_level(logging.WARNING, logger="fibsem.versioning"):
        assert get_revision() is None
    assert caplog.records == []

    caplog.clear()
    _clear_caches()

    # ... but the diagnostics must not be silently discarded either.
    with caplog.at_level(logging.DEBUG, logger="fibsem.versioning"):
        assert get_revision() is None
    assert any("git describe" in r.getMessage() for r in caplog.records)


# --- the hard requirement --------------------------------------------------


@pytest.mark.parametrize(
    "describe",
    [
        pytest.param(lambda: (_ for _ in ()).throw(RuntimeError("kaboom")), id="raises"),
        pytest.param(lambda: None, id="none"),
        pytest.param(lambda: "v0.5.1-48-g4cd11d9c", id="value"),
    ],
)
def test_version_string_never_raises_and_never_returns_none(monkeypatch, describe):
    import fibsem

    monkeypatch.setattr(versioning, "_describe", describe)

    result = get_version_string()

    assert isinstance(result, str)
    assert result
    # A failure in the git half must never cost the caller the package version.
    assert result.startswith(fibsem.__version__)
