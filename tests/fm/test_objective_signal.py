"""The objective announces its moves, so displays stop polling for them (FIB-534).

Around twenty call sites outside the drivers read `fm.objective.position` or `.state`.
On a TFS system every one of those takes the shared imaging channel, so a label that
refreshes on each stage poll moves the microscope's own active view to the FM and back
to fetch a millimetre reading. That is the mechanism behind FIB-517.

`ObjectiveLens.position_changed` replaces the poll with a notification carrying position and
state, as `stage_position_changed` does. Read once at the source rather than by each
subscriber: three displays each re-reading would turn one move into six channel takes.

Two things are pinned here, and the second is the one that will decay:

1. the simulated objective announces every kind of move, and stays quiet when a call
   does not actually move anything;
2. **every** write method in **every** driver announces -- structurally, because
   `autoscript.py` and `odemis.py` need SDKs that are absent off the microscope, and
   because there is no single chokepoint to rely on. Delegation differs per driver:
   the simulated `move_relative` adjusts its own field, the TFS one delegates to
   `move_absolute`, and `insert`/`retract` delegate on the simulator but not on TFS or
   odemis.
"""

import ast
from pathlib import Path

import pytest

import fibsem
from fibsem.fm.microscope import FluorescenceMicroscope


@pytest.fixture
def fm():
    return FluorescenceMicroscope()


@pytest.fixture
def moves(fm):
    """Every `(position, state)` emitted."""
    seen = []
    fm.objective.position_changed.connect(
        lambda position, state: seen.append((position, state))
    )
    return seen


class TestTheSimulatedObjectiveAnnounces:
    def test_move_absolute(self, fm, moves):
        fm.objective.move_absolute(fm.objective.position + 1e-4)
        assert len(moves) == 1

    def test_move_relative(self, fm, moves):
        """Not covered by `move_absolute`: this implementation adjusts its own field."""
        fm.objective.move_relative(1e-4)
        assert len(moves) == 1

    def test_insert(self, fm, moves):
        fm.objective.retract()
        moves.clear()

        fm.objective.insert()

        assert moves, "an insert that moved the objective announced nothing"

    def test_retract(self, fm, moves):
        fm.objective.insert()
        moves.clear()

        fm.objective.retract()

        assert moves, "a retract that moved the objective announced nothing"

    def test_a_parentless_objective_announces_its_own_moves(self):
        """The signal lives on the objective, so it does not need a microscope.

        The earlier design emitted through `self.parent` and so went quiet on an
        objective built alone -- silently, which is the worst way for a notification to
        fail.
        """
        from fibsem.fm.microscope import ObjectiveLens

        objective = ObjectiveLens()
        seen = []
        objective.position_changed.connect(
            lambda position, state: seen.append(position)
        )

        objective.move_absolute(1e-3)

        assert seen == [pytest.approx(1e-3)]


class TestThePayload:
    def test_it_carries_where_the_objective_actually_ended_up(self, fm, moves):
        target = fm.objective.position + 2e-4

        fm.objective.move_absolute(target)

        position, state = moves[-1]
        assert position == pytest.approx(fm.objective.position)
        assert position == pytest.approx(target)
        assert state == fm.objective.state

    def test_the_state_travels_with_it(self, fm, moves):
        """The info bar renders both from one line, so both have to arrive."""
        fm.objective.insert()
        assert moves[-1][1] == "Inserted"

        fm.objective.retract()
        assert moves[-1][1] == "Retracted"

    def test_both_reads_share_one_channel_scope(self, fm, monkeypatch):
        """What keeps the payload cheap on a TFS system.

        Position and state are each a device read that takes the shared imaging channel.
        Under one scope a move costs a single change of the microscope's active view;
        read separately it costs two, on every move, forever.

        One rather than none: this runs after the move's own scope has closed, so that
        subscribers do not execute while the channel is held. The saving is against the
        per-poll reads it replaces, not against zero.
        """
        from contextlib import contextmanager

        entries = []

        @contextmanager
        def counting_scope():
            entries.append(1)
            yield

        monkeypatch.setattr(fm, "active_channel", counting_scope)

        fm.objective._notify_moved()

        assert len(entries) == 1, (
            f"the objective was read under {len(entries)} channel scopes; both reads "
            f"belong in one"
        )

    def test_a_read_that_fails_does_not_break_the_move(self, fm, moves, monkeypatch):
        """A move that worked must not report failure because the readback did not.

        A stale label until the next move is the cheaper failure.

        Patched through `monkeypatch`, not by assigning to the class: `state` lives on
        `ObjectiveLens` itself, so a hand-rolled set-then-`del` removes the real property
        rather than an override and leaves every later reader with an AttributeError --
        which surfaces as a *process abort* rather than a failure once a Qt slot reads it
        (FIB-329).
        """

        def raises(self):
            raise RuntimeError("detector did not answer")

        monkeypatch.setattr(type(fm.objective), "state", property(raises))

        fm.objective.move_absolute(fm.objective.position + 1e-4)

        assert moves == [], "a failed readback still emitted"


class TestEveryDriverAnnounces:
    """Structural, over the source of all three implementations.

    A driver that silently stops announcing is a display that silently goes stale, and
    nothing else would catch it: the TFS and odemis classes cannot be constructed here.
    """

    WRITE_METHODS = ["move_absolute", "move_relative", "insert", "retract", "home"]

    @staticmethod
    def _objective_classes() -> dict:
        """Every `ObjectiveLens` implementation, by qualified name."""
        found = {}
        for module in ("fm/microscope.py", "fm/autoscript.py", "fm/odemis.py"):
            source = (Path(fibsem.__file__).parent / module).read_text(encoding="utf-8")
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.ClassDef) and "Objective" in node.name:
                    found[f"{module}:{node.name}"] = node
        return found

    @staticmethod
    def _announces(method: ast.FunctionDef) -> bool:
        """Either it notifies, or it delegates to something that does."""
        for child in ast.walk(method):
            if not isinstance(child, ast.Call) or not isinstance(
                child.func, ast.Attribute
            ):
                continue
            if child.func.attr in ("_notify_moved", "move_absolute", "move_relative"):
                return True
        return False

    def test_all_three_implementations_are_found(self):
        """Guard against the probe silently matching nothing."""
        classes = self._objective_classes()
        assert len(classes) == 3, (
            f"expected three ObjectiveLens classes, found {sorted(classes)}"
        )

    @pytest.mark.parametrize("write_method", WRITE_METHODS)
    def test_every_write_method_announces(self, write_method):
        offenders = []
        for name, cls in self._objective_classes().items():
            method = next(
                (
                    node
                    for node in cls.body
                    if isinstance(node, ast.FunctionDef) and node.name == write_method
                ),
                None,
            )
            if method is None:
                continue  # not every driver implements every one -- `home` is TFS only
            if not self._announces(method):
                offenders.append(f"{name}.{write_method}")

        assert offenders == [], (
            f"{offenders} move the objective without calling `_notify_moved` or "
            f"delegating to something that does, so any display subscribed to "
            f"`position_changed` goes stale after that call"
        )


class TestTheGuardsStillReadTheDevice:
    """The boundary this design depends on, and the one most likely to erode.

    `ObjectiveLens.position_changed` is for *displays*. Guards read the device, every time, because one
    of them is a collision guard: `ThermoMicroscope.move_stage_absolute` suppresses z and
    r from a stage move while the objective is inserted, and a stale `"Retracted"` there
    means the stage changes height and rotates with the objective in the chamber.

    That is affordable because the guards were never the polling load -- this one runs
    once per stage move, not once per poll tick. But routing it through a cached value is
    exactly the sort of tidy-up that reads as an optimisation, so it is pinned.

    Structural, since `ThermoMicroscope` needs an SDK that is absent here.
    """

    @staticmethod
    def _method(class_name: str, method_name: str) -> ast.FunctionDef:
        source = (Path(fibsem.__file__).parent / "microscope.py").read_text(
            encoding="utf-8"
        )
        cls = next(
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        return next(
            node
            for node in ast.walk(cls)
            if isinstance(node, ast.FunctionDef) and node.name == method_name
        )

    def test_the_stage_move_reads_the_objective_state_directly(self):
        guard = self._method("ThermoMicroscope", "move_stage_absolute")

        reads = [
            node
            for node in ast.walk(guard)
            if isinstance(node, ast.Attribute)
            and node.attr == "state"
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "objective"
        ]

        assert reads, (
            "move_stage_absolute no longer reads `objective.state` off the device. If "
            "that is now a cached or pushed value, the collision guard can be wrong: a "
            "stale 'Retracted' lets the stage move in z and rotate with the objective "
            "inserted (FIB-534)"
        )
