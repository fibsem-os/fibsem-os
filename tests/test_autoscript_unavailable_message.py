"""The connection error when AutoScript cannot be imported.

One "not installed" string used to cover three different situations: never
installed, too old, or a partial copy in the wrong environment. The message now
quotes the import error and sorts any copies found on disk by whether they sit in
the environment that is running. CI never has AutoScript, so `ThermoMicroscope`
refuses to construct here and the message is what a user would actually see.
"""

import os
import sys

import pytest

from fibsem.microscopes import autoscript


def _fake_site_packages(root, name):
    site = root / name / "Lib" / "site-packages"
    (site / "autoscript_sdb_microscope_client").mkdir(parents=True)
    return str(site)


@pytest.fixture
def searched(tmp_path, monkeypatch):
    """Point the candidate search at a scratch tree and forget any real error."""
    monkeypatch.setattr(
        autoscript,
        "_AUTOSCRIPT_SEARCH_GLOBS",
        [str(tmp_path / "*" / "Lib" / "site-packages")],
    )
    monkeypatch.setattr(autoscript, "THERMO_API_IMPORT_ERROR", None)
    return tmp_path


def test_nothing_found_says_so(searched):
    message = autoscript.autoscript_unavailable_message()

    assert message.startswith("Autoscript (ThermoFisher) is not available.")
    assert "No AutoScript installation was found in 1 common locations" in message
    assert "INSTALLATION.md" in message


def test_the_import_error_is_quoted(searched, monkeypatch):
    monkeypatch.setattr(
        autoscript, "THERMO_API_IMPORT_ERROR", "No module named 'autoscript_core'"
    )

    assert "Reason: No module named 'autoscript_core'." in (
        autoscript.autoscript_unavailable_message()
    )


def test_a_copy_in_the_running_environment_is_called_partial(searched, monkeypatch):
    site = _fake_site_packages(searched, "running")
    monkeypatch.syspath_prepend(site)

    message = autoscript.autoscript_unavailable_message()

    assert "in the currently active environment" in message
    assert site in message
    assert "autoscript_core" in message
    assert "different environment" not in message


def test_a_copy_elsewhere_points_at_activation(searched):
    site = _fake_site_packages(searched, "other-env")
    assert site not in sys.path

    message = autoscript.autoscript_unavailable_message()

    assert "in a different environment than the one currently running" in message
    assert site in message
    assert "currently active environment" not in message


def test_both_kinds_are_reported_separately(searched, monkeypatch):
    active = _fake_site_packages(searched, "running")
    other = _fake_site_packages(searched, "other-env")
    monkeypatch.syspath_prepend(active)

    message = autoscript.autoscript_unavailable_message()

    assert "currently active environment (" + active + ")" in message
    assert "currently running: " + other + "." in message


def test_long_candidate_lists_are_capped(searched):
    sites = [_fake_site_packages(searched, f"env{i}") for i in range(7)]

    message = autoscript.autoscript_unavailable_message()

    assert "; and 2 more" in message
    assert sum(site in message for site in sites) == 5


def test_the_search_never_touches_sys_path(searched):
    _fake_site_packages(searched, "other-env")
    before = list(sys.path)

    autoscript.find_autoscript_install_candidates()
    autoscript.autoscript_unavailable_message()

    assert sys.path == before


@pytest.mark.skipif(autoscript.THERMO_API_AVAILABLE, reason="AutoScript is installed")
def test_constructing_the_backend_raises_the_diagnostic(searched):
    from fibsem import utils

    settings = utils.load_microscope_configuration()

    with pytest.raises(
        Exception, match="Autoscript \\(ThermoFisher\\) is not available"
    ):
        autoscript.ThermoMicroscope(settings.system)


def test_active_environment_check_normalises_the_path(tmp_path, monkeypatch):
    site = tmp_path / "env" / "Lib" / "site-packages"
    site.mkdir(parents=True)
    monkeypatch.syspath_prepend(str(site))

    assert autoscript._is_active_environment(os.path.join(str(site), ".", ""))
    assert not autoscript._is_active_environment(str(tmp_path / "elsewhere"))
