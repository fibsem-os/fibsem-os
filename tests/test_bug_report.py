"""Tests for the bug report bundle builder and its helpers."""

import getpass
import json
import os
import zipfile

import pytest

from fibsem.tools import bug_report
from fibsem.tools.bug_report import BugReportContent


def _make_experiment_dir(tmp_path):
    """Create a fake experiment directory with typical artifacts on disk."""
    exp_dir = tmp_path / "AutoLamella-test"
    (exp_dir / "01-lam" / "Milling").mkdir(parents=True)

    home = os.path.expanduser("~")
    (exp_dir / "logfile.log").write_text(f"path {home}/secret error boom")
    (exp_dir / "experiment.yaml").write_text(f"path: {home}/secret\nname: exp")
    (exp_dir / "protocol.yaml").write_text("tasks: 4")
    (exp_dir / "01-lam" / "shot.png").write_bytes(b"PNGDATA")
    (exp_dir / "01-lam" / "Milling" / "z.ome.tiff").write_bytes(b"TIFFDATA")

    return str(exp_dir)


def _names(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        return zf.namelist()


def test_bundle_filename_and_base_contents(tmp_path):
    """A bundle with no experiment still contains the report + system info."""
    content = BugReportContent(title="t", description="d", system_context={"a": "b"})
    zip_path = bug_report.build_bug_report_bundle(
        content, experiment_path=None, output_dir=str(tmp_path)
    )

    fname = os.path.basename(zip_path)
    assert fname.startswith("bug-report-fibsem-")
    assert fname.endswith(".zip")
    assert "AutoLamella-test" not in fname  # experiment name is not in filename

    names = _names(zip_path)
    assert "report.md" in names
    assert "system_info.json" in names


def test_bundle_includes_selected_text_files_scrubbed(tmp_path):
    """Default selection includes the log/yaml files, scrubbed of the home dir."""
    experiment_path = _make_experiment_dir(tmp_path)
    content = BugReportContent(title="t", description="d")  # defaults: text on

    zip_path = bug_report.build_bug_report_bundle(
        content, experiment_path=experiment_path, output_dir=str(tmp_path)
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
    assert "<home>" in log_text


def test_bundle_excludes_text_files_when_unselected(tmp_path):
    experiment_path = _make_experiment_dir(tmp_path)
    content = BugReportContent(
        include_logfile=False,
        include_experiment_yaml=False,
        include_protocol=False,
        include_crash_records=False,
        include_settings=False,
    )
    zip_path = bug_report.build_bug_report_bundle(
        content, experiment_path=experiment_path, output_dir=str(tmp_path)
    )
    names = _names(zip_path)
    assert not any(n.endswith(".log") for n in names)
    assert not any(n.endswith(".yaml") for n in names)
    # base artifacts always present
    assert "report.md" in names


def test_bundle_includes_images_when_opted_in(tmp_path):
    """Binary artifacts are added verbatim only when explicitly selected."""
    experiment_path = _make_experiment_dir(tmp_path)
    content = BugReportContent(include_screenshots=True, include_images=True)

    zip_path = bug_report.build_bug_report_bundle(
        content, experiment_path=experiment_path, output_dir=str(tmp_path)
    )

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        tiff_name = next(n for n in names if n.endswith(".ome.tiff"))
        tiff_bytes = zf.read(tiff_name)

    assert any(n.endswith(".png") for n in names)
    assert tiff_bytes == b"TIFFDATA"  # binary included verbatim, not scrubbed


def test_estimate_bundle_size(tmp_path):
    experiment_path = _make_experiment_dir(tmp_path)
    empty = BugReportContent(
        include_logfile=False,
        include_experiment_yaml=False,
        include_protocol=False,
        include_crash_records=False,
        include_settings=False,
    )
    assert bug_report.estimate_bundle_size(experiment_path, empty) == 0
    assert bug_report.estimate_bundle_size(None, empty) == 0

    full = BugReportContent(include_screenshots=True, include_images=True)
    assert bug_report.estimate_bundle_size(experiment_path, full) > 0


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


def test_the_contact_email_roundtrips():
    """The contact email survives a yaml round-trip, and an older file's moves over."""
    import fibsem.config as cfg

    prefs = cfg.UserPreferences()
    prefs.qt.contact_email = "me@example.org"
    restored = cfg.UserPreferences.from_dict(prefs.to_dict())
    assert restored.qt.contact_email == "me@example.org"
    older = {"reporting": {"contact_email": "me@example.org"}}
    assert cfg.UserPreferences.from_dict(older).qt.contact_email == "me@example.org"


def test_a_preferences_file_naming_the_retired_bug_report_flag_still_loads():
    """The flag is gone; a saved file that still carries it must not fail to load."""
    import fibsem.config as cfg

    prefs = cfg.UserPreferences.from_dict(
        {"features": {"bug_report_enabled": True, "scripts_enabled": False}}
    )
    assert prefs == cfg.UserPreferences()


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


# --- submitting: URL budgets and honest failure reporting ---------------


def _body_of(url):
    """The decoded ``body`` parameter of a prefilled issue URL."""
    from urllib.parse import parse_qs, urlparse

    return parse_qs(urlparse(url).query)["body"][0]


def _capture_url(monkeypatch, opened=True):
    """Record the URL handed to the browser, and control whether it 'opened'."""
    seen = {}

    def fake_open(url, *args, **kwargs):
        seen["url"] = url
        return opened

    monkeypatch.setattr(bug_report.webbrowser, "open", fake_open)
    return seen


def _crash_content():
    """A crash report: the description carries a full traceback."""
    traceback_text = "\n".join(
        f'  File "/opt/fibsem/module_{i}.py", line {i}, in some_function_name_{i}'
        for i in range(200)
    )
    return BugReportContent(
        title="Milling crashed part-way through a trench",
        description=f"An unexpected error occurred:\n\n{traceback_text}",
        steps="1. start a run\n2. wait",
        severity="Crash",
        system_context={"fibsem_version": "0.6.0", "platform": "Windows-10"},
    )


def test_github_url_stays_within_budget(monkeypatch):
    seen = _capture_url(monkeypatch)

    result = bug_report.open_github_issue(_crash_content())

    assert len(seen["url"]) <= bug_report.GITHUB_MAX_URL_LENGTH
    assert result.truncated
    # the untrimmed body is handed back for the clipboard
    assert len(result.full_text) > len(seen["url"])


def test_short_reports_are_not_truncated(monkeypatch):
    seen = _capture_url(monkeypatch)
    content = BugReportContent(title="t", description="it broke", severity="Normal")

    result = bug_report.open_github_issue(content)

    assert not result.truncated
    assert "it broke" in _body_of(seen["url"])


def test_github_reports_a_false_return(monkeypatch):
    """A browser that declines to open is not a submitted report."""
    _capture_url(monkeypatch, opened=False)
    result = bug_report.open_github_issue(BugReportContent(title="t", description="d"))
    assert result.opened is False


def test_github_reports_when_no_browser_opened(monkeypatch):
    monkeypatch.setattr(
        bug_report.webbrowser, "open", lambda *a, **k: (_ for _ in ()).throw(OSError())
    )
    result = bug_report.open_github_issue(BugReportContent(title="t", description="d"))
    assert result.opened is False
    assert "Steps to reproduce" in result.full_text


def test_clip_at_line_prefers_a_line_boundary():
    text = "aaaa\nbbbb\ncccc\ndddd"
    assert bug_report._clip_at_line(text, 12) == "aaaa\nbbbb"
    # a single long line has no boundary to fall back to: hard cut, not empty
    assert bug_report._clip_at_line("x" * 100, 10) == "x" * 10


def test_a_runaway_title_still_leaves_room_for_the_body(monkeypatch):
    """A pasted traceback in the title must not starve the report body."""
    seen = _capture_url(monkeypatch)
    content = BugReportContent(title="T" * 5000, description="the trench collapsed")

    bug_report.open_github_issue(content)

    assert len(seen["url"]) <= bug_report.GITHUB_MAX_URL_LENGTH
    assert "the trench collapsed" in _body_of(seen["url"])


# --- scrubbing: what names the person, the site or the machine ----------


def test_a_login_that_is_an_ordinary_word_is_not_blanked_everywhere(monkeypatch):
    monkeypatch.setattr(bug_report.getpass, "getuser", lambda: "user")

    scrubbed = bug_report.scrub_text("The user moved the stage. user_preferences ok")

    assert scrubbed == "The user moved the stage. user_preferences ok"


def test_the_data_folder_goes_and_the_experiment_folder_stays():
    text = r"saved D:\Data\some-lab\yeast-01\01-lam\ref.tif"

    scrubbed = bug_report.scrub_text(text, r"D:\Data\some-lab\yeast-01")

    assert "some-lab" not in scrubbed
    assert r"<data>\yeast-01\01-lam\ref.tif" in scrubbed


def test_a_path_written_with_doubled_separators_is_scrubbed_too():
    text = r'"path": "D:\\Data\\some-lab\\yeast-01\\ref.tif"'

    scrubbed = bug_report.scrub_text(text, r"D:\Data\some-lab\yeast-01")

    assert "some-lab" not in scrubbed
    assert "ref.tif" in scrubbed


def test_any_other_absolute_path_keeps_only_its_last_two_parts():
    scrubbed = bug_report.scrub_text(r"read E:\facility\some-lab\grids\map.tif")

    assert "some-lab" not in scrubbed and "facility" not in scrubbed
    assert r"<path>\grids\map.tif" in scrubbed


def test_a_traceback_keeps_its_path_inside_the_package():
    scrubbed = bug_report.scrub_text(f'File "{bug_report.__file__}", line 1')

    assert os.path.join("<install>", "fibsem", "tools", "bug_report.py") in scrubbed


def test_a_traceback_keeps_its_path_inside_another_package():
    text = 'File "/opt/env/lib/python3.11/site-packages/numpy/core/numeric.py"'

    scrubbed = bug_report.scrub_text(text)

    assert scrubbed == 'File "<packages>/numpy/core/numeric.py"'


@pytest.mark.parametrize(
    "text",
    [
        "see https://github.com/fibsem-os/fibsem-os/issues/new",
        "in fibsem/ui/widgets/thing.py",
        "ratio 3/4/5 on 2026/11/13",
    ],
)
def test_what_is_not_an_absolute_path_is_left_alone(text):
    """A link, a path inside the repository, a fraction and a date."""
    assert bug_report.scrub_text(text) == text


def test_the_computer_name_is_removed(monkeypatch):
    monkeypatch.setenv("COMPUTERNAME", "FIB-PC-07")

    scrubbed = bug_report.scrub_text("connected from fib-pc-07 to FIB-PC-07:8000")

    assert "pc-07" not in scrubbed.lower()
    assert scrubbed.count("<host>") == 2


def test_a_file_from_the_experiment_loses_who_and_where():
    text = (
        "metadata:\n  user: A Person\n  organisation: Some Lab\n  sample: yeast\n"
        "  project: null\ncontact: a.person@example.org\n"
    )

    scrubbed = bug_report.scrub_file_text(text)

    assert "A Person" not in scrubbed and "Some Lab" not in scrubbed
    assert "example.org" not in scrubbed
    assert "sample: yeast" in scrubbed
    assert "project: null" in scrubbed


def test_a_redacted_json_value_is_still_json():
    scrubbed = bug_report.scrub_file_text('{\n  "user": "A Person",\n  "n": 1\n}')

    assert json.loads(scrubbed) == {"user": "<redacted>", "n": 1}


def test_the_bundle_scrubs_the_experiment_file(tmp_path):
    experiment_path = _make_experiment_dir(tmp_path)
    with open(os.path.join(experiment_path, "experiment.yaml"), "w") as f:
        f.write(f"path: {experiment_path}\nmetadata:\n  user: A Person\n")

    zip_path = bug_report.build_bug_report_bundle(
        BugReportContent(title="t"), experiment_path, output_dir=str(tmp_path)
    )

    with zipfile.ZipFile(zip_path) as zf:
        saved = zf.read("AutoLamella-test/experiment.yaml").decode()
    assert "A Person" not in saved
    assert str(tmp_path) not in saved
    assert "AutoLamella-test" in saved


def test_the_public_issue_is_scrubbed(monkeypatch):
    seen = _capture_url(monkeypatch)
    home = os.path.expanduser("~")
    content = BugReportContent(
        title=f"crash in {home}/data", description=f"open {home}/data/a/b.yaml"
    )

    result = bug_report.open_github_issue(content)

    assert home not in seen["url"] and home not in _body_of(seen["url"])
    assert home not in result.full_text


# --- what the bundle holds beyond the experiment's own files ------------


@pytest.fixture
def this_pc(tmp_path, monkeypatch):
    """A PC with an error record, settings and an agent session, under tmp_path."""
    from fibsem import diagnostics
    from fibsem.ai.agent import manager

    log = tmp_path / "state" / "log"
    (log / "crashes").mkdir(parents=True)
    (log / "crashes" / "2026-01-01-00-00-00-1-01.json").write_text('{"error": "x"}')
    preferences = tmp_path / "state" / "user-preferences.yaml"
    preferences.write_text("contact_email: me@example.org\n")
    configuration = tmp_path / "state" / "microscope-configuration.yaml"
    configuration.write_text("ip_address: 192.168.0.12\nlocal: 127.0.0.1\n")
    session = tmp_path / "sessions" / "2026-01-01-000000-abc"
    session.mkdir(parents=True)
    (session / manager.RECORD).write_text("{}")
    (session / manager.LOG).write_text("the agent said hello")
    monkeypatch.setattr(diagnostics.cfg, "LOG_PATH", str(log))
    monkeypatch.setattr(
        bug_report.fibsem_cfg, "USER_PREFERENCES_PATH", str(preferences)
    )
    monkeypatch.setattr(
        bug_report.fibsem_cfg, "DEFAULT_CONFIGURATION_PATH", str(configuration)
    )
    monkeypatch.setattr(manager, "sessions_dir", lambda: tmp_path / "sessions")
    return tmp_path


def _read(zip_path, name):
    with zipfile.ZipFile(zip_path) as zf:
        return zf.read(name).decode()


def test_a_bundle_without_an_experiment_still_holds_this_pcs_records(this_pc):
    zip_path = bug_report.build_bug_report_bundle(
        BugReportContent(title="t"), None, output_dir=str(this_pc)
    )

    names = _names(zip_path)
    assert "crashes/2026-01-01-00-00-00-1-01.json" in names
    assert "settings/user-preferences.yaml" in names
    assert "settings/microscope-configuration.yaml" in names
    assert not any(name.startswith("agent/") for name in names)
    assert "example.org" not in _read(zip_path, "settings/user-preferences.yaml")
    configuration = _read(zip_path, "settings/microscope-configuration.yaml")
    assert "192.168.0.12" not in configuration
    assert "127.0.0.1" in configuration


def test_the_agent_log_goes_in_only_when_asked(this_pc):
    zip_path = bug_report.build_bug_report_bundle(
        BugReportContent(title="t", include_agent_log=True),
        None,
        output_dir=str(this_pc),
    )

    assert _read(zip_path, "agent/fibsem-agent.log") == "the agent said hello"


def test_this_pcs_records_can_be_left_out(this_pc):
    content = BugReportContent(
        title="t", include_crash_records=False, include_settings=False
    )

    zip_path = bug_report.build_bug_report_bundle(content, None, str(this_pc))

    assert sorted(_names(zip_path)) == ["report.md", "system_info.json"]
    assert bug_report.estimate_bundle_size(None, content) == 0


def test_the_operator_named_on_the_experiment_is_removed_from_its_events(this_pc):
    experiment_path = _make_experiment_dir(this_pc)
    with open(os.path.join(experiment_path, "experiment.yaml"), "w") as f:
        f.write("metadata:\n  user: A Person\n  organisation: Some Lab\n")
    with open(os.path.join(experiment_path, "events.jsonl"), "w") as f:
        f.write('{"actor": "operator:A Person", "kind": "stop"}\n')

    zip_path = bug_report.build_bug_report_bundle(
        BugReportContent(title="t"), experiment_path, output_dir=str(this_pc)
    )

    events = _read(zip_path, "AutoLamella-test/events.jsonl")
    assert "A Person" not in events
    assert json.loads(events) == {"actor": "operator:<redacted>", "kind": "stop"}


def test_a_short_name_is_not_removed_everywhere(tmp_path):
    experiment_path = _make_experiment_dir(tmp_path)
    with open(os.path.join(experiment_path, "experiment.yaml"), "w") as f:
        f.write("metadata:\n  user: Al\n  project: null\n")

    assert bug_report.names_in(experiment_path) == []


def test_the_command_line_makes_a_bundle(this_pc, capsys):
    from fibsem import cli

    experiment_path = _make_experiment_dir(this_pc)
    out = this_pc / "out"

    with pytest.raises(SystemExit) as stopped:
        cli.main(["report", experiment_path, "--title", "it stopped", "-o", str(out)])

    assert stopped.value.code == 0

    (bundle,) = out.glob("bug-report-fibsem-*.zip")
    assert "AutoLamella-test/logfile.log" in _names(bundle)
    assert "it stopped" in _read(bundle, "report.md")
    assert str(bundle) in capsys.readouterr().out


def test_the_command_line_refuses_a_folder_that_is_not_an_experiment(tmp_path, capsys):
    from fibsem import cli

    with pytest.raises(SystemExit) as stopped:
        cli.main(["report", str(tmp_path)])

    assert stopped.value.code == 1
    assert "no experiment" in capsys.readouterr().out
