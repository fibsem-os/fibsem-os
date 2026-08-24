"""The typed contract carried by ``tiled_acquisition_signal``."""

from __future__ import annotations

import ast
from dataclasses import MISSING, FrozenInstanceError, fields
from pathlib import Path

import pytest

from fibsem import utils
from fibsem.imaging.tiled import TiledAcquisitionRunner
from fibsem.imaging.tiling.progress import (
    MODALITY_BEAM,
    MODALITY_FLUORESCENCE,
    BeamTileCompletedEvent,
    CountedTiledPhaseEvent,
    CountedTiledTerminalEvent,
    FluorescenceTileCompletedEvent,
    FluorescenceTileCountEvent,
    TiledAcquisitionEvent,
    TiledEventType,
    TiledOutcome,
    TiledPhase,
    TiledPhaseEvent,
    TiledTerminalEvent,
    TileStartedEvent,
    is_modality,
    modality_of,
)
from fibsem.structures import BeamType, ImageSettings, OverviewAcquisitionSettings

FIBSEM_ROOT = Path(__import__("fibsem").__file__).parent


@pytest.fixture(scope="module")
def microscope():
    scope, _ = utils.setup_session(manufacturer="Demo")
    return scope


@pytest.fixture(scope="module")
def events(microscope, tmp_path_factory):
    collected = []
    microscope.tiled_acquisition_signal.connect(collected.append)
    try:
        TiledAcquisitionRunner(
            microscope,
            OverviewAcquisitionSettings(
                image_settings=ImageSettings(
                    hfw=100e-6,
                    resolution=(64, 64),
                    beam_type=BeamType.ELECTRON,
                    save=False,
                    path=str(tmp_path_factory.mktemp("tiles")),
                    filename="t",
                ),
                nrows=1,
                ncols=2,
            ),
        ).run()
    finally:
        microscope.tiled_acquisition_signal.disconnect(collected.append)
    return collected


def _per_tile(events):
    return [event for event in events if isinstance(event, BeamTileCompletedEvent)]


def _all_event_variants():
    image = object()
    common_fm = dict(modality=MODALITY_FLUORESCENCE)
    return [
        TiledPhaseEvent(phase=TiledPhase.MOVING),
        CountedTiledPhaseEvent(
            phase=TiledPhase.COMPUTING_POSITIONS,
            completed=0,
            total=2,
            message="Computing Tile Positions",
        ),
        TileStartedEvent(
            **common_fm, row_index=0, column_index=0, rows=1, columns=2
        ),
        FluorescenceTileCountEvent(
            **common_fm,
            completed=0,
            total=2,
            estimated_total_seconds=10.0,
            estimated_remaining_seconds=10.0,
            elapsed_seconds=0.0,
        ),
        BeamTileCompletedEvent(
            completed=1,
            total=2,
            row_index=0,
            column_index=0,
            rows=1,
            columns=2,
            image=image,
            preview=image,
            message="Tile Collected",
        ),
        FluorescenceTileCompletedEvent(
            **common_fm,
            completed=1,
            total=2,
            row_index=0,
            column_index=0,
            rows=1,
            columns=2,
            image=image,
            preview_stride=2,
            estimated_total_seconds=10.0,
            estimated_remaining_seconds=5.0,
            elapsed_seconds=5.0,
        ),
        TiledTerminalEvent(
            **common_fm, outcome=TiledOutcome.FINISHED, message="Overview Complete"
        ),
        CountedTiledTerminalEvent(
            outcome=TiledOutcome.FINISHED,
            message="Acquisition Complete",
            completed=2,
            total=2,
        ),
    ]


def test_every_variant_is_frozen_and_tagged():
    variants = _all_event_variants()
    assert {event.event_type for event in variants} == set(TiledEventType)
    for event in variants:
        with pytest.raises(FrozenInstanceError):
            event.modality = "changed"


@pytest.mark.parametrize(
    "event_class, required",
    [
        (TiledPhaseEvent, {"phase"}),
        (
            CountedTiledPhaseEvent,
            {"phase", "completed", "total", "message"},
        ),
        (TileStartedEvent, {"row_index", "column_index", "rows", "columns"}),
        (
            FluorescenceTileCountEvent,
            {
                "completed",
                "total",
                "estimated_total_seconds",
                "estimated_remaining_seconds",
                "elapsed_seconds",
            },
        ),
        (
            BeamTileCompletedEvent,
            {
                "completed",
                "total",
                "row_index",
                "column_index",
                "rows",
                "columns",
                "image",
                "preview",
                "message",
            },
        ),
        (
            FluorescenceTileCompletedEvent,
            {
                "completed",
                "total",
                "row_index",
                "column_index",
                "rows",
                "columns",
                "image",
                "preview_stride",
                "estimated_total_seconds",
                "estimated_remaining_seconds",
                "elapsed_seconds",
            },
        ),
        (TiledTerminalEvent, {"outcome", "message"}),
        (
            CountedTiledTerminalEvent,
            {"outcome", "message", "completed", "total"},
        ),
    ],
)
def test_each_event_shape_declares_its_required_fields(event_class, required):
    actual = {
        item.name
        for item in fields(event_class)
        if item.init and item.default is MISSING and item.default_factory is MISSING
    }
    assert actual == required
    with pytest.raises(TypeError):
        event_class()


def test_phase_and_outcome_vocabularies_are_closed():
    assert set(TiledPhase) == {
        TiledPhase.COMPUTING_POSITIONS,
        TiledPhase.MOVING,
        TiledPhase.ACQUIRING,
        TiledPhase.TILES_ACQUIRED,
        TiledPhase.STITCHING,
        TiledPhase.SAVING,
    }
    assert set(TiledOutcome) == {
        TiledOutcome.FINISHED,
        TiledOutcome.CANCELLED,
        TiledOutcome.FAILED,
    }


def test_modality_defaults_to_beam_and_unknown_values_are_retained():
    beam = TiledPhaseEvent(phase=TiledPhase.MOVING)
    future = TiledPhaseEvent(modality="future", phase=TiledPhase.MOVING)
    assert modality_of(beam) == MODALITY_BEAM
    assert is_modality(beam, MODALITY_BEAM)
    assert modality_of(future) == "future"
    assert not is_modality(future, MODALITY_BEAM)


def test_every_beam_emit_is_a_typed_beam_event(events):
    assert events
    assert all(modality_of(event) == MODALITY_BEAM for event in events)
    assert isinstance(events[0], CountedTiledPhaseEvent)
    assert events[0].phase is TiledPhase.COMPUTING_POSITIONS
    assert isinstance(events[-1], CountedTiledTerminalEvent)
    assert events[-1].outcome is TiledOutcome.FINISHED


def test_every_completed_tile_has_the_full_contract(events):
    updates = _per_tile(events)
    assert len(updates) == 2
    assert [event.completed for event in updates] == [1, 2]
    assert all(event.total == 2 for event in updates)
    assert all(event.message == "Tile Collected" for event in updates)
    assert [(event.row_index, event.column_index) for event in updates] == [(0, 0), (0, 1)]


def test_a_tile_update_carries_a_placeable_preview(events):
    for event in _per_tile(events):
        preview = event.preview
        assert preview.metadata.stage_position is not None
        assert preview.metadata.pixel_size.x
        assert preview.metadata.hardware_geometry is not None
        assert preview.metadata.image_settings.beam_type is not None


def test_the_preview_fills_without_changing_shape(events):
    previews = [event.preview for event in _per_tile(events)]
    filled = [int((preview.data > 0).sum()) for preview in previews]
    assert filled[-1] > filled[0]
    assert len({preview.data.shape for preview in previews}) == 1


def test_the_signal_declaration_names_the_event_contract():
    from fibsem.microscope import FibsemMicroscope

    source = (FIBSEM_ROOT / "microscope.py").read_text(encoding="utf-8")
    assert "tiled_acquisition_signal = Signal(TiledAcquisitionEvent)" in source
    parameter = next(iter(FibsemMicroscope.tiled_acquisition_signal.signature.parameters.values()))
    assert parameter.annotation is TiledAcquisitionEvent


def test_producers_no_longer_emit_dicts_or_the_vestigial_task_tag():
    for relative in (
        "imaging/tiled.py",
        "fm/acquisition.py",
        "ui/fm/widgets/fm_overview_widget.py",
    ):
        source = (FIBSEM_ROOT / relative).read_text(encoding="utf-8")
        assert '"task": "tileset"' not in source
        tree = ast.parse(source)
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            function = call.func
            if not (
                isinstance(function, ast.Attribute)
                and function.attr == "emit"
                and isinstance(function.value, ast.Attribute)
                and function.value.attr == "tiled_acquisition_signal"
            ):
                continue
            assert not call.args or not isinstance(call.args[0], ast.Dict), relative


def test_tiled_consumers_use_typed_attributes_not_mapping_access():
    consumers = {
        "ui/FibsemMinimapWidget.py": "handle_tile_acquisition_progress",
        "ui/widgets/overview_widget.py": "_apply_progress",
        "ui/fm/widgets/fm_overview_widget.py": "_apply_tile_progress",
        "applications/autolamella/ui/AutoLamellaMainUI.py": (
            "_on_tile_acquisition_progress"
        ),
    }
    for relative, method_name in consumers.items():
        tree = ast.parse((FIBSEM_ROOT / relative).read_text(encoding="utf-8"))
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == method_name
        )
        assert not any(isinstance(node, ast.Subscript) for node in ast.walk(method))
        assert not any(
            isinstance(node, ast.Attribute)
            and node.attr == "get"
            and isinstance(node.value, ast.Name)
            and node.value.id == "event"
            for node in ast.walk(method)
        )
