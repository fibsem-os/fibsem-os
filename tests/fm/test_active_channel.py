"""The FM and the beams share one connection, so an FM read must hand it back (FIB-517).

Setting the FM's channel points the shared connection at the FM. A property getter that
did it and walked away left the microscope there -- and since the FM overview's info bar
read the objective on every stage poll, a beam acquisition that had set its own channel
found the FM instead. It stopped a workflow task.

The Thermo FM devices' channel (`AutoscriptFMChannel` in
`fibsem/devices/drivers/autoscript_fm.py`) cannot be imported without the AutoScript SDK,
which is not installed off the microscope, so the scope is exercised against a stub
connection through a stub of it rather than the real import. What is under test is the contract --
capture, set, restore, restore-on-failure -- not the SDK.
"""

from contextlib import contextmanager

import pytest


class _Imaging:
    """The half of the connection that owns the active view."""

    def __init__(self, view: int = 1):
        self.view = view
        self.history: list = []

    def get_active_view(self) -> int:
        return self.view

    def set_active_view(self, view: int) -> None:
        self.view = view
        self.history.append(("view", view))

    def set_active_device(self, device) -> None:
        self.history.append(("device", device))


class _Connection:
    def __init__(self, view: int = 1):
        self.imaging = _Imaging(view)


class _FM:
    """`AutoscriptFMChannel`'s channel handling, without the SDK import.

    Copied rather than imported: `fibsem/devices/drivers/autoscript_fm.py` imports
    `autoscript_sdb_microscope_client` at module scope, which is absent off the
    microscope. `TestTheDeviceChannelMatches` pins the real source's shape so the two
    cannot drift.
    """

    FM_VIEW = 3
    FM_DEVICE = "FLM"

    def __init__(self, view: int = 1):
        import threading

        self.connection = _Connection(view)
        self._active_view = self.FM_VIEW
        self._active_device = self.FM_DEVICE
        self._channel_lock = threading.RLock()
        self._channel_depth = 0
        self._restore_view = None

    def set_active_channel(self):
        self.connection.imaging.set_active_view(self._active_view)
        self.connection.imaging.set_active_device(self._active_device)

    @contextmanager
    def active_channel(self):
        with self._channel_lock:
            if self._channel_depth == 0:
                self._restore_view = self.connection.imaging.get_active_view()
            self.set_active_channel()
            self._channel_depth += 1
        try:
            yield
        finally:
            with self._channel_lock:
                self._channel_depth -= 1
                if self._channel_depth == 0:
                    self.connection.imaging.set_active_view(self._restore_view)


class TestTheContract:
    def test_the_block_runs_on_the_fm(
        self,
    ):
        fm = _FM(view=1)  # the beam side owns it

        with fm.active_channel():
            assert fm.connection.imaging.view == _FM.FM_VIEW

    def test_it_hands_the_view_back(self):
        """The whole point. Leaving it on the FM is what broke a running beam task."""
        fm = _FM(view=1)

        with fm.active_channel():
            pass

        assert fm.connection.imaging.view == 1

    def test_it_hands_it_back_when_the_read_fails(self):
        """A getter that raises must not leave the microscope pointed elsewhere -- an
        FM that is off, or answering slowly, would otherwise strand the beam side."""
        fm = _FM(view=1)

        with pytest.raises(RuntimeError):
            with fm.active_channel():
                raise RuntimeError("detector did not answer")

        assert fm.connection.imaging.view == 1

    def test_it_restores_whatever_was_there(self):
        """Not a hardcoded beam view: it puts back what it found."""
        for view in (0, 1, 2, 7):
            fm = _FM(view=view)
            with fm.active_channel():
                pass
            assert fm.connection.imaging.view == view

    def test_only_the_outermost_scope_restores(self):
        """A tileset holds the channel for the whole run and each tile opens a scope
        inside it. An inner scope that restored would put the beam view back between
        every pair of tiles -- the flicker the run-level scope exists to avoid."""
        fm = _FM(view=1)

        with fm.active_channel():
            with fm.active_channel():
                assert fm.connection.imaging.view == _FM.FM_VIEW
            assert fm.connection.imaging.view == _FM.FM_VIEW, "inner scope restored"

        assert fm.connection.imaging.view == 1

    def test_the_lock_is_not_held_across_the_body(self):
        """The scope can span a whole tileset, and the lock it takes is the
        microscope's `_threading_lock`, shared by every caller on the microscope and by
        its devices as `imaging_channel`. Holding it for minutes would block them all;
        the known ones are a Pause/Resume click and the milling monitor loop, neither of
        which should overlap an FM run, but the point is that a shared lock held that
        long makes any future caller a hostage.
        """
        import threading

        fm = _FM(view=1)
        acquired = threading.Event()

        def take_the_lock():
            with fm._channel_lock:
                acquired.set()

        with fm.active_channel():
            other = threading.Thread(target=take_the_lock)
            other.start()
            other.join(timeout=2)

        assert acquired.is_set(), "another thread could not take the lock mid-scope"

    def test_a_failed_entry_does_not_poison_every_later_scope(self):
        """Entering is two RPCs, and a dropped connection can fail the second.

        That raises out of `__enter__`, so the caller's block never runs and the
        `finally` never runs either. Counting the scope before the channel was actually
        taken left the depth permanently too high -- every later scope then looked
        nested, nothing ever restored the view again, and FIB-517 was back for the rest
        of the session with nothing on screen to say so.
        """
        fm = _FM(view=1)

        def connection_reset(device):
            raise RuntimeError("connection reset")

        fm.connection.imaging.set_active_device = connection_reset
        with pytest.raises(RuntimeError):
            with fm.active_channel():
                pass

        assert fm._channel_depth == 0, "a failed entry left a scope counted as open"

        # And the next healthy scope still hands the view back.
        fm.connection.imaging.set_active_device = lambda device: None
        fm.connection.imaging.view = 1
        with fm.active_channel():
            pass

        assert fm.connection.imaging.view == 1

    def test_an_inner_scope_that_fails_leaves_the_outer_one_intact(self):
        """The same failure one level down must not decrement the run's own scope: a
        tileset still owns the channel, and its restore is still the outermost one."""
        fm = _FM(view=1)

        with fm.active_channel():
            fm.connection.imaging.set_active_device = self._raises
            with pytest.raises(RuntimeError):
                with fm.active_channel():
                    pass
            assert fm._channel_depth == 1
            fm.connection.imaging.set_active_device = lambda device: None

        assert fm.connection.imaging.view == 1

    @staticmethod
    def _raises(device):
        raise RuntimeError("connection reset")

    def test_set_active_channel_alone_does_not_restore(self):
        """The unscoped call still exists, for the live stream that owns the channel for
        its duration. This is what every other caller must not use."""
        fm = _FM(view=1)

        fm.set_active_channel()

        assert fm.connection.imaging.view == _FM.FM_VIEW


class TestTheDeviceChannelMatches:
    """Structural, since the SDK import cannot be satisfied here: the devices' FM
    channel (`AutoscriptFMChannel.scope`) is the scope the stub above runs. Thermo's FM
    API runs on it."""

    @staticmethod
    def _scope():
        import ast
        from pathlib import Path

        import fibsem.devices.drivers as drivers

        source = (Path(drivers.__file__).parent / "autoscript_fm.py").read_text(
            encoding="utf-8"
        )
        cls = next(
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ClassDef) and node.name == "AutoscriptFMChannel"
        )
        return next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "scope"
        )

    def test_it_captures_and_restores_the_view_in_a_finally(self):
        import ast

        node = self._scope()
        body = ast.dump(node)
        assert "get_active_view" in body and "set_active_view" in body
        assert any(isinstance(n, ast.Try) and n.finalbody for n in ast.walk(node))

    def test_the_lock_is_not_held_across_the_body(self):
        import ast

        locked = [
            child
            for child in ast.walk(self._scope())
            if isinstance(child, ast.With) and "lock" in ast.dump(child.items[0])
        ]
        assert locked
        assert not any(
            isinstance(inner, ast.Expr) and isinstance(inner.value, ast.Yield)
            for block in locked
            for inner in ast.walk(block)
        )

    def test_the_scope_is_counted_only_after_the_channel_is_taken(self):
        import ast

        entry = next(
            child
            for child in ast.walk(self._scope())
            if isinstance(child, ast.With) and "set_active_channel" in ast.dump(child)
        )
        calls = [ast.dump(stmt) for stmt in entry.body]
        took_it = next(i for i, d in enumerate(calls) if "set_active_channel" in d)
        counted_it = next(
            i
            for i, stmt in enumerate(entry.body)
            if isinstance(stmt, ast.AugAssign) and "_depth" in ast.dump(stmt.target)
        )
        assert took_it < counted_it
