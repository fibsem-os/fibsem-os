"""The LUT's failure modes (FIB-592).

Deliberately not in test_refractive_index.py: that module skips wholesale when
the real 9 MB CSV is absent, and these are exactly the cases where it is absent
or broken. Everything here runs against a synthetic LUT in tmp_path, so it holds
on a machine that has never downloaded the real one.
"""
import itertools
from pathlib import Path

import pytest

import fibsem.correlation.refractive_index as ri


def _write_valid_lut(path: Path) -> None:
    """A complete, if tiny, grid over the five axes the interpolator expects."""
    axes = {
        "tilt_deg": [0.0, 30.0],
        "depth_um": [0.5, 14.5],
        "n2": [1.22, 1.46],
        "wavelength_um": [0.42, 0.72],
        "NA": [0.2, 0.9],
    }
    names = list(axes)
    rows = ["tilt_deg,depth_um,n2,wavelength_um,NA,zeta"]
    for i, combo in enumerate(itertools.product(*(axes[n] for n in names))):
        rows.append(",".join(str(v) for v in combo) + f",{1.0 + i * 0.01}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n")


@pytest.fixture(autouse=True)
def lut_path(tmp_path, monkeypatch):
    """Point the module at a scratch LUT and clear every cached verdict."""
    path = tmp_path / "data" / "table_refractive_index_lookup.csv"
    monkeypatch.setattr(ri, "_LUT_PATH", path)
    monkeypatch.setattr(ri, "_download_attempted", False)
    ri.reset_lut_cache()
    yield path
    ri.reset_lut_cache()


# ---------------------------------------------------------------------------
# lut_error — "the file exists" is not "the file works"
# ---------------------------------------------------------------------------


def test_valid_lut_reports_no_error(lut_path):
    _write_valid_lut(lut_path)
    assert ri.lut_error() is None
    assert ri.get_lut() is not None


def test_missing_lut_says_so(lut_path):
    assert "not found" in ri.lut_error()


@pytest.mark.parametrize(
    "content, expected",
    [
        pytest.param(None, "could not be read", id="truncated-download"),
        pytest.param(
            "<html><body>403 Forbidden</body></html>\n",
            "could not be read",
            id="proxy-error-page",
        ),
        pytest.param("", "could not be read", id="empty-file"),
    ],
)
def test_a_present_but_unusable_lut_is_an_error_not_availability(
    lut_path, content, expected
):
    """The FIB-592 core: a file that exists but will not parse must report as
    unavailable, or the widget enables five spinboxes over a dead calculator."""
    if content is None:  # half a real transfer
        _write_valid_lut(lut_path)
        raw = lut_path.read_text()
        lut_path.write_text(raw[: len(raw) // 3])
    else:
        lut_path.parent.mkdir(parents=True, exist_ok=True)
        lut_path.write_text(content)

    error = ri.lut_error()
    assert error is not None and expected in error
    assert lut_path.exists(), "the precondition is that the file IS there"

    with pytest.raises(RuntimeError, match="unavailable"):
        ri.get_lut()


def test_wrong_columns_name_what_is_missing(lut_path):
    lut_path.parent.mkdir(parents=True, exist_ok=True)
    lut_path.write_text("a,b,c\n1,2,3\n")
    assert "missing column" in ri.lut_error()


def test_the_verdict_is_cached(lut_path, monkeypatch):
    """Re-parsing a 9 MB CSV on every spinbox keystroke is the cost of not
    caching, so the failure has to be remembered as firmly as the success."""
    _write_valid_lut(lut_path)
    assert ri.lut_error() is None

    reads = []
    real = ri.pd.read_csv
    monkeypatch.setattr(ri.pd, "read_csv", lambda *a, **k: (reads.append(1), real(*a, **k))[1])
    for _ in range(5):
        ri.lut_error()
        ri.get_lut()
    assert reads == []


# ---------------------------------------------------------------------------
# _download_lut — a failed transfer must not land at the real path
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload: bytes, fail_after: int = -1):
        self._payload, self._fail_after, self._sent = payload, fail_after, 0

    def read(self, n=-1):
        if self._fail_after >= 0 and self._sent >= self._fail_after:
            raise ConnectionResetError("connection dropped mid-transfer")
        chunk = self._payload[self._sent : self._sent + 8]
        self._sent += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_download_is_atomic_and_leaves_nothing_behind(lut_path, monkeypatch):
    """A transfer that dies half way must leave NO file at the real path — one
    that exists is never re-fetched, so it breaks the install permanently."""
    _write_valid_lut(lut_path)
    payload = lut_path.read_bytes()
    lut_path.unlink()

    monkeypatch.setattr(
        ri.urllib.request,
        "urlopen",
        lambda url, timeout=None: _FakeResponse(payload, fail_after=len(payload) // 2),
    )
    with pytest.raises(ConnectionResetError):
        ri._download_lut()

    assert not lut_path.exists()
    assert list(lut_path.parent.glob("*.part")) == [], "temp file left behind"


def test_download_passes_a_timeout(lut_path, monkeypatch):
    """Microscope PCs sit behind proxies that blackhole rather than refuse; with
    no timeout the correlation window hangs until the OS gives up."""
    seen = {}

    def _urlopen(url, timeout=None):
        seen["timeout"] = timeout
        return _FakeResponse(b"tilt_deg\n")

    monkeypatch.setattr(ri.urllib.request, "urlopen", _urlopen)
    ri._download_lut()
    assert seen["timeout"] == ri._LUT_DOWNLOAD_TIMEOUT_S
    assert seen["timeout"] is not None


def test_successful_download_lands_at_the_real_path(lut_path, monkeypatch):
    _write_valid_lut(lut_path)
    payload = lut_path.read_bytes()
    lut_path.unlink()

    monkeypatch.setattr(
        ri.urllib.request, "urlopen", lambda url, timeout=None: _FakeResponse(payload)
    )
    ri._download_lut()
    assert lut_path.read_bytes() == payload
    assert list(lut_path.parent.glob("*.part")) == []


# ---------------------------------------------------------------------------
# _ensure_lut — self-healing, and bounded
# ---------------------------------------------------------------------------


def test_ensure_lut_is_a_noop_when_the_lut_works(lut_path, monkeypatch):
    _write_valid_lut(lut_path)
    monkeypatch.setattr(
        ri.urllib.request,
        "urlopen",
        lambda *a, **k: pytest.fail("must not touch the network"),
    )
    ri._ensure_lut()


def test_ensure_lut_refetches_a_corrupt_file(lut_path, monkeypatch):
    """An install that picked up a truncated CSV before the atomic write existed
    would otherwise stay broken forever, with no way out but deleting it by hand."""
    _write_valid_lut(lut_path)
    payload = lut_path.read_bytes()
    lut_path.write_bytes(payload[: len(payload) // 3])
    assert ri.lut_error() is not None

    monkeypatch.setattr(
        ri.urllib.request, "urlopen", lambda url, timeout=None: _FakeResponse(payload)
    )
    ri._ensure_lut()
    assert ri.lut_error() is None


def test_ensure_lut_attempts_the_download_once_per_process(lut_path, monkeypatch):
    """Bounded, because it runs on the GUI thread every time the correlation
    window opens and an offline machine will not come back mid-session."""
    calls = []

    def _urlopen(url, timeout=None):
        calls.append(url)
        raise OSError("offline")

    monkeypatch.setattr(ri.urllib.request, "urlopen", _urlopen)
    with pytest.raises(OSError):
        ri._ensure_lut()
    for _ in range(3):
        ri._ensure_lut()  # must not raise, must not retry
    assert len(calls) == 1


def test_ensure_lut_raises_when_what_arrived_is_still_unusable(lut_path, monkeypatch):
    monkeypatch.setattr(
        ri.urllib.request,
        "urlopen",
        lambda url, timeout=None: _FakeResponse(b"<html>404</html>\n"),
    )
    with pytest.raises(RuntimeError, match="unusable"):
        ri._ensure_lut()
