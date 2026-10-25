"""Connection arguments, on either side of the subcommand.

argparse parses a subcommand into a fresh namespace and then copies every
attribute of it over the root parser's. The connection arguments were one parser
used as a parent of both, so a flag given *before* the subcommand was overwritten
by the subparser that had simply defaulted it -- silently, with nothing logged
and nothing failing: ``fibsem-cli --manufacturer Thermo info`` once connected to
the Demo simulator. The same holds for ``--url`` and ``--token`` now: the wrong
server would be acted on, not merely displayed.

These pin both positions for every connection argument, and sweep every
subcommand so the fix cannot be half-reverted for one of them. See
``_build_connection_parser``.
"""

import argparse

import pytest

from fibsem import cli

# flag argv, namespace attribute, value once given, value when absent
CONNECTION_FLAGS = [
    (["--url", "http://pc:8001"], "url", "http://pc:8001", None),
    (["--token", "secret"], "token", "secret", None),
    (["--debug"], "debug", True, False),
]
_IDS = [dest for _, dest, _, _ in CONNECTION_FLAGS]

# Subcommands with required positionals. Nothing to do with connection
# arguments; they just have to be supplied to reach a successful parse.
POSITIONALS = {
    "microscope": ["get-info"],
    "stage": ["get-position"],
    "beam": ["blank"],
    "device": ["list"],
    "service": ["list"],
    "experiment": ["list-all"],
    "pair-device": ["10.0.0.2:8765", "1234"],
}


def _parse(*argv):
    return cli._build_parser().parse_args(argv)


def _subcommand_names():
    """The registered subcommands, read off the parser so this cannot go stale.

    Private argparse attributes, deliberately: there is no public way to
    enumerate subparsers, and a hardcoded list would silently stop covering
    whatever was added next.
    """
    parser = cli._build_parser()
    return sorted(
        name
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
        for name in action.choices
    )


@pytest.mark.parametrize("flag,dest,given,_absent", CONNECTION_FLAGS, ids=_IDS)
def test_a_connection_flag_is_honoured_from_either_position(flag, dest, given, _absent):
    assert getattr(_parse(*flag, "hardware-check"), dest) == given
    assert getattr(_parse("hardware-check", *flag), dest) == given


@pytest.mark.parametrize("_flag,dest,_given,absent", CONNECTION_FLAGS, ids=_IDS)
def test_the_declared_default_still_applies_when_the_flag_is_absent(
    _flag, dest, _given, absent
):
    """Suppressing the subparsers' defaults must not suppress them everywhere.

    The root parser's copy is the one that guarantees the attribute exists at
    all; without it every ``args.url`` in main() becomes an
    AttributeError.
    """
    assert getattr(_parse("hardware-check"), dest) == absent


def test_the_flag_after_the_subcommand_wins_when_given_in_both_positions():
    """The more specific position, and the one that already worked."""
    args = _parse("--url", "http://a", "hardware-check", "--url", "http://b")
    assert args.url == "http://b"


@pytest.mark.parametrize("subcommand", _subcommand_names())
def test_every_subcommand_honours_a_flag_given_before_it(subcommand):
    """Swept rather than spot-checked: this broke for all of them at once.

    A subcommand built with its own defaulted copy of the connection parser --
    ``_build_connection_parser()`` instead of the shared suppressed ``conn`` --
    would reintroduce the bug for that one subcommand only.
    """
    argv = ["--url", "http://pc", subcommand, *POSITIONALS.get(subcommand, [])]
    assert _parse(*argv).url == "http://pc"


def test_the_sweep_covers_every_subcommand():
    """Guard the guard: a new subcommand needing positionals must be added above.

    Otherwise its parse fails, the sweep cannot reach its assertion, and the
    coverage quietly stops one subcommand short.
    """
    for subcommand in _subcommand_names():
        argv = [
            "--url",
            "http://pc",
            subcommand,
            *POSITIONALS.get(subcommand, []),
        ]
        try:
            _parse(*argv)
        except SystemExit:
            pytest.fail(
                f"'{subcommand}' cannot be parsed as {argv!r}: add its required "
                f"positionals to POSITIONALS so the sweep actually covers it."
            )
