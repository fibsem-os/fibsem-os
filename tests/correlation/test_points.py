"""The rules for correlation points, without Qt: which canvas a type is drawn
on, which types replace and clear each other, fiducial pairs, names, and how an
edit changes a point's status.

The store tests (tests/ui/test_correlation_point_store.py) pin the same rules
through the store, with its signals; these pin them on the coordinates alone.
"""

import pytest

from fibsem.correlation.config import (
    DELETE_FIDUCIAL_ASK,
    DELETE_FIDUCIAL_ONE,
    DELETE_FIDUCIAL_PAIR,
)
from fibsem.correlation.points import (
    POINT_RULES,
    accept,
    apply_refinement,
    check_additions,
    cleared_by,
    fiducial_partner,
    partner_type,
    place_by_hand,
    point_name,
    point_names,
    removal_for,
    reset_to_predicted,
    set_field,
    side_of,
    toggle_rejected,
    types_on,
)
from fibsem.correlation.structures import (
    Coordinate,
    PointProvenance,
    PointStatus,
    PointType,
    PointXYZ,
)


def _coord(pt: PointType = PointType.FIB, x: float = 0.0, **kwargs) -> Coordinate:
    return Coordinate(PointXYZ(x, x, 0), pt, **kwargs)


def _same(a, b) -> bool:
    return len(a) == len(b) and all(x is y for x, y in zip(a, b))


# --------------------------------------------------------------------------
# Point types
# --------------------------------------------------------------------------


def test_the_sides():
    assert {pt: rule.side for pt, rule in POINT_RULES.items()} == {
        PointType.FIB: "fib",
        PointType.SURFACE: "fib",
        PointType.FM: "fm",
        PointType.POI: "fm",
        PointType.SURFACE_FM: "fm",
    }
    assert side_of(PointType.POI) == "fm"


def test_types_on_a_side_keep_the_rules_order():
    assert types_on("fib") == [PointType.FIB, PointType.SURFACE]
    assert types_on("fm") == [PointType.FM, PointType.POI, PointType.SURFACE_FM]


def test_every_point_type_has_a_rule():
    assert set(POINT_RULES) == set(PointType)


def test_a_surface_point_clears_itself_and_the_other_surface():
    assert cleared_by(PointType.SURFACE) == [PointType.SURFACE, PointType.SURFACE_FM]
    assert cleared_by(PointType.SURFACE_FM) == [PointType.SURFACE_FM, PointType.SURFACE]


@pytest.mark.parametrize("pt", [PointType.FIB, PointType.FM, PointType.POI])
def test_a_fiducial_or_target_clears_nothing(pt):
    assert cleared_by(pt) == []


def test_check_additions_refuses_a_second_surface_point():
    held = {PointType.SURFACE: [_coord(PointType.SURFACE)]}
    with pytest.raises(ValueError):
        check_additions(held, [_coord(PointType.SURFACE)])
    with pytest.raises(ValueError):
        check_additions({}, [_coord(PointType.SURFACE_FM)] * 2)


def test_check_additions_lets_fiducials_and_one_surface_through():
    held = {PointType.FIB: [_coord(), _coord()]}
    check_additions(held, [_coord(), _coord(PointType.SURFACE)])


# --------------------------------------------------------------------------
# Fiducial pairs
# --------------------------------------------------------------------------


def test_fib_and_fm_pair_and_nothing_else_does():
    assert partner_type(PointType.FIB) is PointType.FM
    assert partner_type(PointType.FM) is PointType.FIB
    for pt in (PointType.POI, PointType.SURFACE, PointType.SURFACE_FM):
        assert partner_type(pt) is None


def test_the_partner_is_the_point_at_the_same_place_on_the_other_side():
    fib = [_coord(PointType.FIB, float(i)) for i in range(3)]
    fm = [_coord(PointType.FM, float(i)) for i in range(3)]
    assert fiducial_partner(fib[1], fib, fm) is fm[1]
    assert fiducial_partner(fm[2], fib, fm) is fib[2]


def test_the_partner_is_found_by_identity_not_equality():
    fib = [_coord(PointType.FIB, 5.0), _coord(PointType.FIB, 5.0)]
    fm = [_coord(PointType.FM, 0.0), _coord(PointType.FM, 1.0)]
    assert fiducial_partner(fib[1], fib, fm) is fm[1]


def test_no_partner_past_the_end_off_the_lists_or_for_a_type_that_does_not_pair():
    fib = [_coord(PointType.FIB, float(i)) for i in range(3)]
    fm = [_coord(PointType.FM, 0.0)]
    assert fiducial_partner(fib[2], fib, fm) is None
    assert fiducial_partner(_coord(PointType.FIB), fib, fm) is None
    assert fiducial_partner(_coord(PointType.POI), fib, fm) is None


def test_removal_for_each_delete_fiducial_choice():
    coord, partner = _coord(PointType.FIB), _coord(PointType.FM)
    assert _same(removal_for(coord, partner, DELETE_FIDUCIAL_PAIR), [coord, partner])
    assert _same(removal_for(coord, partner, DELETE_FIDUCIAL_ONE), [coord])
    assert removal_for(coord, partner, "") is None  # cancelled
    # asking is the caller's: an unanswered question removes nothing
    assert removal_for(coord, partner, DELETE_FIDUCIAL_ASK) is None


# --------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------


def test_names_are_numbered_within_a_type():
    coords = [
        _coord(PointType.FIB),
        _coord(PointType.POI),
        _coord(PointType.FIB),
    ]
    assert point_names(coords) == ["FIB 1", "POI 1", "FIB 2"]


def test_a_name_is_the_type_and_the_1_based_row():
    assert point_name(PointType.FM, 3) == "FM 4"
    assert point_name(PointType.SURFACE_FM, 0) == "FM-SURFACE 1"


# --------------------------------------------------------------------------
# Status
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status", [PointStatus.PREDICTED, PointStatus.FITTED, PointStatus.ACCEPTED]
)
def test_a_move_by_hand_makes_the_point_placed(status):
    coord = _coord(PointType.FM, status=status, fitted=True)
    place_by_hand(coord)
    assert coord.status == PointStatus.PLACED
    assert coord.fitted is False


def test_a_point_moved_by_hand_that_was_rejected_stays_rejected():
    coord = _coord(PointType.FM, status=PointStatus.REJECTED)
    place_by_hand(coord)
    assert coord.status == PointStatus.REJECTED


def test_a_typed_value_makes_the_point_placed():
    coord = _coord(PointType.FM, status=PointStatus.PREDICTED)
    set_field(coord, "z", 3.0)
    assert coord.point.z == 3.0
    assert coord.status == PointStatus.PLACED


def test_an_unknown_field_is_an_error_and_changes_nothing():
    coord = _coord(PointType.FM, status=PointStatus.PREDICTED)
    with pytest.raises(ValueError):
        set_field(coord, "w", 1.0)
    assert coord.status == PointStatus.PREDICTED


def test_apply_refinement_moves_the_point_and_marks_it_fitted():
    coord = _coord()
    apply_refinement(coord, 1.0, 2.0, 3.0)
    assert (coord.point.x, coord.point.y, coord.point.z) == (1.0, 2.0, 3.0)
    assert coord.fitted is True
    assert coord.status == PointStatus.FITTED


def test_accept_takes_only_the_predictions():
    predicted = [_coord(PointType.FM, status=PointStatus.PREDICTED) for _ in range(2)]
    placed = _coord(PointType.FM, status=PointStatus.PLACED)
    assert _same(accept(predicted + [placed]), predicted)
    assert all(c.status == PointStatus.ACCEPTED for c in predicted)
    assert placed.status == PointStatus.PLACED
    assert accept([placed]) == []


def test_reject_and_bring_back():
    placed = _coord(PointType.FM, status=PointStatus.PLACED)
    fitted = _coord(PointType.FM, status=PointStatus.FITTED, fitted=True)
    for coord, back in ((placed, PointStatus.PLACED), (fitted, PointStatus.FITTED)):
        assert toggle_rejected(coord) is True
        assert coord.status == PointStatus.REJECTED
        assert toggle_rejected(coord) is True
        assert coord.status == back


def test_a_prediction_cannot_be_rejected():
    coord = _coord(PointType.FM, status=PointStatus.PREDICTED)
    assert toggle_rejected(coord) is False
    assert coord.status == PointStatus.PREDICTED


def test_reset_makes_a_projected_point_a_prediction_again():
    coord = _coord(
        PointType.FM,
        status=PointStatus.FITTED,
        fitted=True,
        provenance=PointProvenance.PROJECTED,
    )
    assert reset_to_predicted(coord) is True
    assert coord.status == PointStatus.PREDICTED
    assert coord.fitted is False


def test_reset_refuses_a_point_the_projection_did_not_make():
    coord = _coord(
        PointType.FM, status=PointStatus.PLACED, provenance=PointProvenance.USER
    )
    assert reset_to_predicted(coord) is False
    assert coord.status == PointStatus.PLACED


def test_no_transition_writes_provenance():
    coord = _coord(
        PointType.FM,
        status=PointStatus.PREDICTED,
        provenance=PointProvenance.PROJECTED,
    )
    accept([coord])
    place_by_hand(coord)
    apply_refinement(coord, 2.0, 2.0, 0.0)
    toggle_rejected(coord)
    toggle_rejected(coord)
    reset_to_predicted(coord)
    assert coord.provenance == PointProvenance.PROJECTED
