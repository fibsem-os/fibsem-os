"""Tests for the bug report bundle builder and its helpers."""

import getpass
import os
import zipfile
from types import SimpleNamespace

import pytest

from fibsem.applications.autolamella.tools import bug_report
from fibsem.applications.autolamella.tools.bug_report import BugReportContent


def _make_experiment(tmp_path):
    """Create a fake experiment directory with typical artifacts on disk."""
    exp_dir = tmp_path / "AutoLamella-test"
    (exp_dir / "01-lam" / "Milling").mkdir(parents=True)

    home = os.path.expanduser("~")
    (exp_dir / "logfile.log").write_text(f"path {home}/secret error boom")
    (exp_dir / "experiment.yaml").write_text(f"path: {home}/secret\nname: exp")
    (exp_dir / "protocol.yaml").write_text("tasks: 4")
    (exp_dir / "01-lam" / "shot.png").write_bytes(b"PNGDATA")
    (exp_dir / "01-lam" / "Milling" / "z.ome.tiff").write_bytes(b"TIFFDATA")

    return SimpleNamespace(name="AutoLamella-test", path=str(exp_dir))


def _names(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        return zf.namelist()


def test_bundle_filename_and_base_contents(tmp_path):
    """A bundle with no experiment still contains the report + system info."""
    content = BugReportContent(title="t", description="d", system_context={"a": "b"})
    zip_path = bug_report.build_bug_report_bundle(
        content, experiment=None, output_dir=str(tmp_path)
    )

    fname = os.path.basename(zip_path)
    assert fname.startswith("bug-report-autolamella-")
    assert fname.endswith(".zip")
    assert "AutoLamella-test" not in fname  # experiment name is not in filename

    names = _names(zip_path)
    assert "report.md" in names
    assert "system_info.json" in names


def test_bundle_includes_selected_text_files_scrubbed(tmp_path):
    """Default selection includes the log/yaml files, scrubbed of the home dir."""
    experiment = _make_experiment(tmp_path)
    content = BugReportContent(title="t", description="d")  # defaults: text on

    zip_path = bug_report.build_bug_report_bundle(
        content, experiment=experiment, output_dir=str(tmp_path)
    )

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        log_name = next(n for n in names if n.endswith("logfile.log"))
        log_text = zf.read(log_name).decode()

    assert any(n.endswith("experiment.yaml") for n in names)
    assert any(n.endswith("protocol.yaml") for n in names)
    # screenshots / images are off by default
    assert not any(n.endswith(".png") for n in names)
    assert not any(n.endswith(".tiff") for n in names)
    # home directory scrubbed out of text artifacts
    assert os.path.expanduser("~") not in log_text
    assert "<redacted>" in log_text


def test_bundle_excludes_text_files_when_unselected(tmp_path):
    experiment = _make_experiment(tmp_path)
    content = BugReportContent(
        include_logfile=False,
        include_experiment_yaml=False,
        include_protocol=False,
    )
    zip_path = bug_report.build_bug_report_bundle(
        content, experiment=experiment, output_dir=str(tmp_path)
    )
    names = _names(zip_path)
    assert not any(n.endswith(".log") for n in names)
    assert not any(n.endswith(".yaml") for n in names)
    # base artifacts always present
    assert "report.md" in names


def test_bundle_includes_images_when_opted_in(tmp_path):
    """Binary artifacts are added verbatim only when explicitly selected."""
    experiment = _make_experiment(tmp_path)
    content = BugReportContent(include_screenshots=True, include_images=True)

    zip_path = bug_report.build_bug_report_bundle(
        content, experiment=experiment, output_dir=str(tmp_path)
    )

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        tiff_name = next(n for n in names if n.endswith(".ome.tiff"))
        tiff_bytes = zf.read(tiff_name)

    assert any(n.endswith(".png") for n in names)
    assert tiff_bytes == b"TIFFDATA"  # binary included verbatim, not scrubbed


def test_estimate_bundle_size(tmp_path):
    experiment = _make_experiment(tmp_path)
    empty = BugReportContent(
        include_logfile=False,
        include_experiment_yaml=False,
        include_protocol=False,
    )
    assert bug_report.estimate_bundle_size(experiment, empty) == 0
    assert bug_report.estimate_bundle_size(None, BugReportContent()) == 0

    full = BugReportContent(include_screenshots=True, include_images=True)
    assert bug_report.estimate_bundle_size(experiment, full) > 0


@pytest.mark.parametrize(
    "text, needle",
    [
        (f"{os.path.expanduser('~')}/x", os.path.expanduser("~")),
        ("/home/someoneelse/data", "/home/someoneelse/"),
        (f"user={getpass.getuser()}", getpass.getuser()),
    ],
)
def test_scrub_text_removes_identifying_info(text, needle):
    scrubbed = bug_report.scrub_text(text)
    assert needle not in scrubbed


def test_scrub_text_handles_empty():
    assert bug_report.scrub_text("") == ""


@pytest.mark.parametrize(
    "text",
    [
        "done for /Users/bob",  # trailing username, no separator
        "path /Users/bob/file.txt",  # username mid-path
        r"see C:\Users\alice",  # windows trailing username
        r"C:\Users\alice\project\x",  # windows mid-path
        "home /home/carol",  # linux trailing username
    ],
)
def test_scrub_text_redacts_other_user_paths(text):
    """Usernames of *other* users are redacted, incl. trailing (no separator)."""
    scrubbed = bug_report.scrub_text(text)
    for name in ("bob", "alice", "carol"):
        assert name not in scrubbed
    assert "<user>" in scrubbed
