"""The correlation point store: one owner for the points and the selection (FIB-973).

The list widget, the canvas overlay and the tab widget each held a copy of the
points, and three bugs were the copies disagreeing (FIB-958, FIB-965, FIB-972).
These pin what a single owner guarantees: removal by identity, the neighbour
selection, the state a signal announces already being in place when a slot runs,
and one emit per bulk operation.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_correlation_point_store.py
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

from fibsem.correlation.structures import (
    Coordinate,
    PointProvenance,
    PointStatus,
    PointType,
    PointXYZ,
)
from fibsem.ui.correlation.point_store import POINT_RULES, CorrelationPointStore

_app = QApplication.instance() or QApplication(sys.argv)


def _coord(pt: PointType = PointType.FIB, x: float = 0.0, **kwargs) -> Coordinate:
    return Coordinate(PointXYZ(x, x, 0), pt, **kwargs)


def _store(n: int = 4, pt: PointType = PointType.FIB):
    store = CorrelationPointStore()
    coords = [_coord(pt, 10.0 * i) for i in range(n)]
    store.replace_all(coords)
    return store, coords


class _Log:
    """Records each emit together with the store's state when the slot ran."""

    def __init__(self, store: CorrelationPointStore) -> None:
        self.events = []
        store.structure_changed.connect(
            lambda: self.events.append(("structure", store.coordinates))
        )
        store.points_changed.connect(lambda cs: self.events.append(("points", cs)))
        store.selection_changed.connect(
            lambda: self.events.append(("selection", store.selection))
        )

    @property
    def names(self):
        return [name for name, _ in self.events]


def _same(a, b) -> bool:
    return len(a) == len(b) and all(x is y for x, y in zip(a, b))


# --------------------------------------------------------------------------
# Queries
# --------------------------------------------------------------------------


def test_points_are_grouped_by_type_and_by_canvas_side():
    store = CorrelationPointStore()
    fib, surface = _coord(PointType.FIB), _coord(PointType.SURFACE)
    fm, poi = _coord(PointType.FM), _coord(PointType.POI)
    store.replace_all([fm, fib, poi, surface])

    assert _same(store.of_type(PointType.FM), [fm])
    assert _same(store.on_side("fib"), [fib, surface])
    assert _same(store.on_side("fm"), [fm, poi])
    assert len(store) == 4


def test_the_sides_match_the_tab_widgets_table():
    # POINT_RULES repeats _POINT_TYPE_SIDES until the tab widget reads the store.
    from fibsem.ui.correlation.widgets.correlation_tab_widget import (
        _POINT_TYPE_SIDES,
    )

    assert {pt: rule.side for pt, rule in POINT_RULES.items()} == _POINT_TYPE_SIDES


def test_membership_is_by_identity_not_equality():
    store, coords = _store(2)
    twin = _coord(PointType.FIB, 0.0)
    assert twin == coords[0]
    assert coords[0] in store
    assert twin not in store


def test_a_returned_list_is_a_copy():
    store, coords = _store(2)
    store.of_type(PointType.FIB).clear()
    assert _same(store.of_type(PointType.FIB), coords)


# --------------------------------------------------------------------------
# Removal
# --------------------------------------------------------------------------


def test_remove_takes_the_point_given_not_its_equal_twin():
    store = CorrelationPointStore()
    a, b = _coord(PointType.FIB, 5.0), _coord(PointType.FIB, 5.0)
    store.replace_all([a, b])

    assert store.remove(b) is True
    assert _same(store.coordinates, [a])


def test_remove_of_a_point_that_is_not_here_changes_nothing():
    store, coords = _store(2)
    log = _Log(store)
    assert store.remove(_coord()) is False
    assert log.events == []
    assert _same(store.coordinates, coords)


def test_remove_selects_the_row_that_followed():
    store, coords = _store(4)
    store.remove(coords[1])
    assert store.current is coords[2]


def test_remove_of_the_last_row_selects_the_new_last():
    store, coords = _store(4)
    store.remove(coords[3])
    assert store.current is coords[2]


def test_remove_of_the_only_point_leaves_nothing_selected():
    store, coords = _store(1)
    store.select(coords[0])
    store.remove(coords[0])
    assert store.selection == ()


def test_remove_every_point_one_at_a_time():
    store, coords = _store(5)
    for coord in list(coords):
        store.remove(coord)
        assert coord not in store
    assert len(store) == 0


def test_the_removal_is_announced_before_the_new_selection():
    store, coords = _store(3)
    log = _Log(store)
    store.remove(coords[0])
    assert log.names == ["structure", "selection"]


def test_a_slot_sees_the_removal_already_made():
    store, coords = _store(3)
    log = _Log(store)
    store.remove(coords[0])
    name, seen = log.events[0]
    assert name == "structure"
    assert _same(seen, coords[1:])


def test_remove_many_emits_once_and_selects_at_the_lowest_removed_row():
    store, coords = _store(5)
    log = _Log(store)
    assert store.remove_many([coords[3], coords[1]]) == 2
    assert log.names == ["structure", "selection"]
    assert _same(store.coordinates, [coords[0], coords[2], coords[4]])
    assert store.current is coords[2]


def test_remove_many_across_types_anchors_on_the_current_selection():
    store = CorrelationPointStore()
    fm = [_coord(PointType.FM, float(i)) for i in range(3)]
    poi = [_coord(PointType.POI, float(i)) for i in range(3)]
    store.replace_all(fm + poi)
    store.select(poi[0])

    store.remove_many([fm[0], poi[0]])
    assert store.current is poi[1]


# --------------------------------------------------------------------------
# Adding, and the per-type rules
# --------------------------------------------------------------------------


def test_add_appends_and_selects_the_new_point():
    store, coords = _store(2)
    log = _Log(store)
    new = _coord(PointType.FIB, 99.0)
    store.add(new)
    assert _same(store.of_type(PointType.FIB), coords + [new])
    assert store.current is new
    assert log.names == ["structure", "selection"]


def test_adding_a_point_twice_is_an_error():
    store, coords = _store(1)
    with pytest.raises(ValueError):
        store.add(coords[0])


def test_a_surface_point_replaces_the_one_before_it():
    store = CorrelationPointStore()
    first, second = _coord(PointType.SURFACE, 1.0), _coord(PointType.SURFACE, 2.0)
    store.add(first)
    store.add(second)
    assert _same(store.of_type(PointType.SURFACE), [second])


def test_the_two_surface_types_are_mutually_exclusive_in_one_emit():
    store = CorrelationPointStore()
    fib_surface = _coord(PointType.SURFACE)
    fm_surface = _coord(PointType.SURFACE_FM)
    store.add(fib_surface)
    log = _Log(store)

    store.add(fm_surface)
    assert store.of_type(PointType.SURFACE) == []
    assert _same(store.of_type(PointType.SURFACE_FM), [fm_surface])
    assert log.names.count("structure") == 1


def test_a_surface_point_leaves_the_fiducials_alone():
    store, coords = _store(3, PointType.FM)
    store.add(_coord(PointType.SURFACE_FM))
    assert _same(store.of_type(PointType.FM), coords)


def test_add_many_appends_in_one_emit_and_keeps_the_selection():
    store, coords = _store(2, PointType.FM)
    store.select(coords[0])
    log = _Log(store)
    new = [_coord(PointType.FM, 50.0), _coord(PointType.FM, 60.0)]
    store.add_many(new)
    assert _same(store.of_type(PointType.FM), coords + new)
    assert store.current is coords[0]
    assert log.names == ["structure"]


def test_add_many_refuses_a_second_surface_point():
    store = CorrelationPointStore()
    store.add(_coord(PointType.SURFACE))
    with pytest.raises(ValueError):
        store.add_many([_coord(PointType.SURFACE)])
    assert len(store) == 1


def test_replace_all_clears_the_selection_rather_than_selecting_row_1():
    store, coords = _store(3)
    store.select(coords[1])
    log = _Log(store)
    store.replace_all([_coord(PointType.FIB, 1.0), _coord(PointType.FIB, 2.0)])
    assert store.selection == ()
    assert log.names == ["structure", "selection"]


def test_reorder_keeps_the_selection_and_the_objects():
    store, coords = _store(3)
    store.select(coords[0])
    log = _Log(store)
    store.reorder(PointType.FIB, coords[::-1])
    assert _same(store.of_type(PointType.FIB), coords[::-1])
    assert store.current is coords[0]
    assert log.names == ["structure"]


def test_reorder_must_keep_the_same_points():
    store, coords = _store(3)
    with pytest.raises(ValueError):
        store.reorder(PointType.FIB, coords[:2] + [_coord()])
    assert _same(store.of_type(PointType.FIB), coords)


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


def test_selection_is_one_value_across_every_type():
    store = CorrelationPointStore()
    fib, fm = _coord(PointType.FIB), _coord(PointType.FM)
    store.replace_all([fib, fm])
    store.select(fib)
    store.select(fm)
    assert _same(store.selection, [fm])


def test_selecting_the_selected_point_emits_nothing():
    store, coords = _store(2)
    store.select(coords[0])
    log = _Log(store)
    store.select(coords[0])
    assert log.events == []


def test_select_none_clears():
    store, coords = _store(2)
    store.select(coords[0])
    store.select(None)
    assert store.selection == () and store.current is None


def test_selection_is_capped_at_one_even_when_extended():
    store, coords = _store(3)
    store.select(coords[0])
    store.select(coords[1], extend=True)
    assert _same(store.selection, [coords[1]])


def test_selecting_a_point_that_is_not_here_is_an_error():
    store, _ = _store(1)
    with pytest.raises(ValueError):
        store.select(_coord())


# --------------------------------------------------------------------------
# Values and status
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status", [PointStatus.PREDICTED, PointStatus.FITTED, PointStatus.ACCEPTED]
)
def test_a_move_makes_the_point_placed(status):
    store = CorrelationPointStore()
    coord = _coord(PointType.FM, status=status, fitted=True)
    store.replace_all([coord])
    log = _Log(store)

    store.move(coord, 7.0, 8.0)
    assert (coord.point.x, coord.point.y) == (7.0, 8.0)
    assert coord.status == PointStatus.PLACED
    assert coord.fitted is False
    assert log.events == [("points", (coord,))]


def test_a_typed_value_makes_the_point_placed():
    store = CorrelationPointStore()
    coord = _coord(PointType.FM, status=PointStatus.PREDICTED)
    store.replace_all([coord])
    store.set_field(coord, "z", 3.0)
    assert coord.point.z == 3.0
    assert coord.status == PointStatus.PLACED


def test_a_moved_rejected_point_stays_rejected():
    store = CorrelationPointStore()
    coord = _coord(PointType.FM, status=PointStatus.REJECTED)
    store.replace_all([coord])
    store.move(coord, 1.0, 1.0)
    assert coord.status == PointStatus.REJECTED


def test_an_unknown_field_is_an_error():
    store, coords = _store(1)
    with pytest.raises(ValueError):
        store.set_field(coords[0], "w", 1.0)


def test_move_many_emits_once():
    store, coords = _store(3)
    log = _Log(store)
    store.move_many(coords[:2], 1.0, -1.0)
    assert log.names == ["points"]
    assert _same(log.events[0][1], coords[:2])
    assert (coords[0].point.x, coords[0].point.y) == (1.0, -1.0)
    assert coords[2].point.x == 20.0


def test_notify_changed_announces_without_touching_status():
    store = CorrelationPointStore()
    coord = _coord(PointType.FM, status=PointStatus.PREDICTED)
    store.replace_all([coord])
    log = _Log(store)
    coord.point.x = 42.0  # place_predictions moves points in place
    store.notify_changed([coord])
    assert coord.status == PointStatus.PREDICTED
    assert log.events == [("points", (coord,))]


def test_apply_fit_moves_the_point_and_marks_it_fitted():
    store, coords = _store(1)
    store.apply_fit(coords[0], 1.0, 2.0, 3.0)
    p = coords[0].point
    assert (p.x, p.y, p.z) == (1.0, 2.0, 3.0)
    assert coords[0].fitted is True
    assert coords[0].status == PointStatus.FITTED


def test_accept_takes_only_the_predictions_and_emits_once():
    store = CorrelationPointStore()
    predicted = [_coord(PointType.FM, status=PointStatus.PREDICTED) for _ in range(2)]
    placed = _coord(PointType.FM, status=PointStatus.PLACED)
    store.replace_all(predicted + [placed])
    log = _Log(store)

    accepted = store.accept(store.of_type(PointType.FM))
    assert _same(accepted, predicted)
    assert all(c.status == PointStatus.ACCEPTED for c in predicted)
    assert placed.status == PointStatus.PLACED
    assert log.names == ["points"]


def test_accept_with_no_predictions_emits_nothing():
    store, coords = _store(2)
    log = _Log(store)
    assert store.accept(coords) == []
    assert log.events == []


def test_reject_and_bring_back():
    store = CorrelationPointStore()
    placed = _coord(PointType.FM, status=PointStatus.PLACED)
    fitted = _coord(PointType.FM, status=PointStatus.FITTED, fitted=True)
    store.replace_all([placed, fitted])

    for coord, back in ((placed, PointStatus.PLACED), (fitted, PointStatus.FITTED)):
        assert store.toggle_rejected(coord) is True
        assert coord.status == PointStatus.REJECTED
        assert store.toggle_rejected(coord) is True
        assert coord.status == back


def test_a_prediction_cannot_be_rejected():
    store = CorrelationPointStore()
    coord = _coord(PointType.FM, status=PointStatus.PREDICTED)
    store.replace_all([coord])
    log = _Log(store)
    assert store.toggle_rejected(coord) is False
    assert coord.status == PointStatus.PREDICTED
    assert log.events == []


def test_reset_makes_a_projected_point_a_prediction_again():
    store = CorrelationPointStore()
    coord = _coord(
        PointType.FM,
        status=PointStatus.FITTED,
        fitted=True,
        provenance=PointProvenance.PROJECTED,
    )
    store.replace_all([coord])
    assert store.reset_to_predicted(coord) is True
    assert coord.status == PointStatus.PREDICTED
    assert coord.fitted is False


def test_reset_refuses_a_point_the_projection_did_not_make():
    store = CorrelationPointStore()
    coord = _coord(
        PointType.FM, status=PointStatus.PLACED, provenance=PointProvenance.USER
    )
    store.replace_all([coord])
    log = _Log(store)
    assert store.reset_to_predicted(coord) is False
    assert coord.status == PointStatus.PLACED
    assert log.events == []


def test_no_transition_writes_provenance():
    store = CorrelationPointStore()
    coord = _coord(
        PointType.FM,
        status=PointStatus.PREDICTED,
        provenance=PointProvenance.PROJECTED,
    )
    store.replace_all([coord])
    store.accept(coord)
    store.move(coord, 1.0, 1.0)
    store.apply_fit(coord, 2.0, 2.0, 0.0)
    store.toggle_rejected(coord)
    store.toggle_rejected(coord)
    store.reset_to_predicted(coord)
    assert coord.provenance == PointProvenance.PROJECTED


# --------------------------------------------------------------------------
# replace_type: what assigning a list widget's coordinates does
# --------------------------------------------------------------------------


def test_replace_type_leaves_the_other_types_and_selects_nothing():
    store = CorrelationPointStore()
    fm, poi = _coord(PointType.FM), _coord(PointType.POI)
    store.replace_all([fm, poi])
    log = _Log(store)
    new = [_coord(PointType.FM, 1.0), _coord(PointType.FM, 2.0)]

    store.replace_type(PointType.FM, new)
    assert _same(store.of_type(PointType.FM), new)
    assert _same(store.of_type(PointType.POI), [poi])
    assert store.selection == ()
    assert log.names == ["structure"]


def test_replace_type_keeps_a_selection_that_is_still_here():
    store = CorrelationPointStore()
    fm, poi = _coord(PointType.FM), _coord(PointType.POI)
    store.replace_all([fm, poi])
    store.select(poi)
    store.replace_type(PointType.FM, [])
    assert store.current is poi


def test_replace_type_drops_a_selection_that_is_gone():
    store, coords = _store(2)
    store.select(coords[0])
    log = _Log(store)
    store.replace_type(PointType.FIB, [coords[1]])
    assert store.selection == ()
    assert log.names == ["structure", "selection"]


def test_replace_type_refuses_a_point_of_another_type():
    store, coords = _store(1)
    with pytest.raises(ValueError):
        store.replace_type(PointType.FIB, [_coord(PointType.FM)])
    assert _same(store.of_type(PointType.FIB), coords)


# --------------------------------------------------------------------------
# One selected point per canvas: a verdict's pair link selects both partners
# --------------------------------------------------------------------------


def test_a_pair_is_one_selected_point_on_each_canvas():
    store = CorrelationPointStore()
    fib, fm = _coord(PointType.FIB), _coord(PointType.FM)
    store.replace_all([fib, fm])
    store.select(fm)
    store.select(fib, extend=True)

    assert _same(store.selection, [fm, fib])
    assert store.selected_on("fm") is fm
    assert store.selected_on("fib") is fib
    assert store.selected_of_type(PointType.FM) is fm
    assert store.selected_of_type(PointType.POI) is None


def test_extending_replaces_the_selection_on_the_same_canvas_only():
    store = CorrelationPointStore()
    fib, fm, poi = _coord(PointType.FIB), _coord(PointType.FM), _coord(PointType.POI)
    store.replace_all([fib, fm, poi])
    store.select([fib, fm])

    store.select(poi, extend=True)  # POI is drawn on the FM canvas
    assert _same(store.selection, [fib, poi])


def test_a_plain_select_clears_the_other_canvas():
    store = CorrelationPointStore()
    fib, fm = _coord(PointType.FIB), _coord(PointType.FM)
    store.replace_all([fib, fm])
    store.select([fib, fm])
    store.select(fm)
    assert _same(store.selection, [fm])


def test_deselect_leaves_the_rest_of_the_selection():
    store = CorrelationPointStore()
    fib, fm = _coord(PointType.FIB), _coord(PointType.FM)
    store.replace_all([fib, fm])
    store.select([fib, fm])
    log = _Log(store)

    store.deselect(fib)
    assert _same(store.selection, [fm])
    store.deselect(None)
    assert log.names == ["selection"]
