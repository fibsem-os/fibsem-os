"""``fibsem-cli plugins``.

The failure mode this command exists for is "my plugin does not show up", so it
has to answer without a microscope: diagnosing a plugin is usually done away
from the instrument, and every other subcommand connects before it dispatches.
"""

import pytest

from fibsem import cli


def _parse(*argv):
    return cli._build_parser().parse_args(argv)


def test_plugins_does_not_require_a_microscope():
    """The whole reason it is dispatched before setup_session()."""
    assert _parse("plugins").needs_microscope is False
    # ...while everything else still does.
    assert _parse("position").needs_microscope is True


def test_plugins_takes_no_connection_arguments():
    """Connection flags on the subcommand would imply it connects. It must not."""
    with pytest.raises(SystemExit):
        _parse("plugins", "--manufacturer", "Thermo")


def test_plugins_prints_the_report(capsys):
    exit_code = cli.cmd_plugins(_parse("plugins"))
    out = capsys.readouterr().out

    assert exit_code == 0
    for group in ("fibsem.patterns", "fibsem.strategies", "fibsem.tasks"):
        assert group in out
    assert out.rstrip().endswith(cli_environment_line())


def test_all_shows_built_ins_that_are_otherwise_only_counted(capsys):
    cli.cmd_plugins(_parse("plugins"))
    default = capsys.readouterr().out
    cli.cmd_plugins(_parse("plugins", "--all"))
    everything = capsys.readouterr().out

    assert "fibsem.milling.patterning.patterns2.RectanglePattern" not in default
    assert "fibsem.milling.patterning.patterns2.RectanglePattern" in everything
    assert "built-in hidden (use --all)." in default
    assert "built-in hidden" not in everything


def test_broken_plugins_do_not_bury_the_report(monkeypatch, capsys):
    """Loading logs a traceback per broken plugin; the report reports them again.

    Dumping both would push the readable listing off the top of the terminal,
    which defeats the point of a command whose output is meant to be pasted.

    Asserted at the moment of loading rather than by checking stderr afterwards:
    the registries load once per process and cache the result, so by the time
    this test runs there is nothing left to log and an "is stderr clean?" check
    would pass no matter what the command did.
    """
    import logging

    from fibsem.plugins import report

    observed = {}
    real = report.collect_extensions

    def spy():
        observed["disabled_while_loading"] = logging.getLogger().manager.disable
        return real()

    monkeypatch.setattr(report, "collect_extensions", spy)

    cli.cmd_plugins(_parse("plugins"))
    capsys.readouterr()

    assert observed["disabled_while_loading"] >= logging.CRITICAL
    # ...and restored, so this does not silence the rest of the process.
    assert logging.getLogger().manager.disable == 0


def test_debug_keeps_the_tracebacks(monkeypatch):
    """--debug is the escape hatch when the one-line reason is not enough."""
    import logging

    from fibsem.plugins import report

    observed = {}
    real = report.collect_extensions
    monkeypatch.setattr(
        report,
        "collect_extensions",
        lambda: (
            observed.setdefault("disabled", logging.getLogger().manager.disable),
            real(),
        )[1],
    )

    args = _parse("--debug", "plugins")
    assert args.debug is True
    cli.cmd_plugins(args)
    assert observed["disabled"] == 0


def cli_environment_line() -> str:
    from fibsem.plugins.report import environment_line

    return environment_line()
