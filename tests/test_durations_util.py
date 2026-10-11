"""fibsem.util.durations: five shapes of duration, one per purpose (FIB-1191)."""

import pytest

from fibsem.util import durations

# Values either side of each boundary, and the awkward ones.
SECONDS = [0, 0.4, 0.5, 1, 44.6, 59.4, 59.6, 60, 61, 134, 599.5, 3599.4, 3599.6, 3600]
SECONDS += [3661, 3725.25, 13200, 86399, 86400, 90061.5]


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (45, "45.00s"),
        (134, "2m 14.00s"),
        (3785, "1h 3m 5.00s"),
        (59.996, "60.00s"),  # two decimals, as it always was
    ],
)
def test_precise(seconds, expected):
    assert durations.format_duration_precise(seconds) == expected


@pytest.mark.parametrize(
    "seconds, expected, padded",
    [
        (5, "5s", "5s"),
        (59.6, "1m 0s", "1m 00s"),  # rounded before the units are chosen
        (252, "4m 12s", "4m 12s"),
        (242, "4m 2s", "4m 02s"),
        (3660, "1h 1m", "1h 01m"),
    ],
)
def test_approximate(seconds, expected, padded):
    assert durations.format_duration_rounded(seconds) == expected
    assert durations.format_duration_rounded(seconds, pad=True) == padded


@pytest.mark.parametrize(
    "seconds, expected",
    [(0, "0:00"), (245, "4:05"), (244.6, "4:05"), (3725, "1:02:05"), (59.6, "1:00")],
)
def test_as_clock(seconds, expected):
    assert durations.format_duration_as_clock(seconds) == expected


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (45, "45 s"),
        (59.6, "1 min"),
        (720, "12 min"),
        (13200, "3 h 40 min"),
        (7200, "2 h"),
        (-3, "0 s"),
    ],
)
def test_in_words(seconds, expected):
    assert durations.format_duration_in_words(seconds) == expected


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (5, "just now"),
        (230, "3 min ago"),  # floored: at least that old
        (7300, "2 h ago"),
        (90000, "1 d ago"),
        (-10, "just now"),
    ],
)
def test_ago(seconds, expected):
    assert durations.format_time_ago(seconds) == expected


@pytest.mark.parametrize(
    "shape",
    [
        durations.format_duration_precise,
        durations.format_duration_rounded,
        durations.format_duration_as_clock,
        durations.format_duration_in_words,
        durations.format_time_ago,
    ],
)
@pytest.mark.parametrize("missing", [None, float("nan"), float("inf"), "", "n/a", True])
def test_a_missing_value_is_blank(shape, missing):
    assert shape(missing) == ""


# -- the formatters these replaced (FIB-1191): same output ------------------------


def _old_format_duration(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes}m {seconds:.2f}s"
    elif minutes > 0:
        return f"{minutes}m {seconds:.2f}s"
    else:
        return f"{seconds:.2f}s"


def _old_format_time_remaining(seconds, pad=False):
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    width = "02d" if pad else "d"
    if hours:
        return f"{hours}h {minutes:{width}}m"
    if minutes:
        return f"{minutes}m {secs:{width}}s"
    return f"{secs}s"


@pytest.mark.parametrize("seconds", SECONDS)
def test_precise_and_rounded_read_as_the_formatters_they_replaced(seconds):
    """utils.format_duration and utils.format_time_remaining, removed in FIB-1251."""
    assert durations.format_duration_precise(seconds) == _old_format_duration(seconds)
    for pad in (False, True):
        assert durations.format_duration_rounded(
            seconds, pad=pad
        ) == _old_format_time_remaining(seconds, pad=pad)


@pytest.mark.parametrize("seconds", SECONDS)
def test_the_report_clock_is_format_duration_as_clock(seconds):
    """With its dash for a run that has no length."""
    from fibsem.applications.autolamella.tools import report_v2

    assert durations.format_duration_as_clock(seconds) == report_v2._clock(seconds)
    assert report_v2._clock(None) == "—"


def test_seconds_are_read_from_strings_and_numpy():
    np = pytest.importorskip("numpy")
    assert durations.format_duration_rounded("252") == "4m 12s"
    assert durations.format_duration_as_clock(np.float64(245.0)) == "4:05"
