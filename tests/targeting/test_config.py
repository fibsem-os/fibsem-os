"""Serialisation / deserialisation tests for fibsem.targeting.config dataclasses."""

import pytest

from fibsem.structures import BeamType, FibsemStagePosition, OverviewAcquisitionSettings
from fibsem.targeting.config import (
    GridPreparationParameters,
    GridScreeningParameters,
    ModelParameters,
    PostProcessingParameters,
    SampleGridParameters,
    TargetScoringWeights,
)

# ---------------------------------------------------------------------------
# ModelParameters
# ---------------------------------------------------------------------------


class TestModelParameters:
    def test_roundtrip(self):
        original = ModelParameters(checkpoint_path="/some/model.pth")
        restored = ModelParameters.from_dict(original.to_dict())
        assert restored.checkpoint_path == original.checkpoint_path

    def test_empty_path_roundtrip(self):
        original = ModelParameters(checkpoint_path="")
        restored = ModelParameters.from_dict(original.to_dict())
        assert restored.checkpoint_path == ""

    def test_from_dict_missing_key_defaults_to_empty(self):
        restored = ModelParameters.from_dict({})
        assert restored.checkpoint_path == ""

    def test_to_dict_contains_checkpoint_path(self):
        d = ModelParameters(checkpoint_path="foo.pth").to_dict()
        assert "checkpoint_path" in d
        assert d["checkpoint_path"] == "foo.pth"


# ---------------------------------------------------------------------------
# SampleGridParameters
# ---------------------------------------------------------------------------


class TestSampleGridParameters:
    def test_roundtrip(self):
        original = SampleGridParameters(
            lamella_width=12e-6,
            lamella_height=9e-6,
            hole_width=95e-6,
            pitch=130e-6,
            bar_width=40e-6,
            milling_angle=12.5,
            grid_type="Hexagonal",
        )
        restored = SampleGridParameters.from_dict(original.to_dict())
        assert restored.lamella_width == pytest.approx(original.lamella_width)
        assert restored.lamella_height == pytest.approx(original.lamella_height)
        assert restored.hole_width == pytest.approx(original.hole_width)
        assert restored.pitch == pytest.approx(original.pitch)
        assert restored.bar_width == pytest.approx(original.bar_width)
        assert restored.milling_angle == pytest.approx(original.milling_angle)
        assert restored.grid_type == original.grid_type

    def test_from_dict_empty_uses_defaults(self):
        p = SampleGridParameters.from_dict({})
        assert p.lamella_width == pytest.approx(11e-6)
        assert p.lamella_height == pytest.approx(8e-6)
        assert p.hole_width == pytest.approx(90e-6)
        assert p.pitch == pytest.approx(125e-6)
        assert p.bar_width == pytest.approx(35e-6)
        assert p.milling_angle == pytest.approx(11.0)
        assert p.grid_type == "Square"

    def test_to_dict_keys(self):
        d = SampleGridParameters().to_dict()
        expected_keys = {
            "lamella_width",
            "lamella_height",
            "hole_width",
            "pitch",
            "bar_width",
            "milling_angle",
            "grid_type",
        }
        assert set(d.keys()) == expected_keys


# ---------------------------------------------------------------------------
# TargetScoringWeights
# ---------------------------------------------------------------------------


class TestTargetScoringWeights:
    def test_roundtrip(self):
        original = TargetScoringWeights(
            cell_size=0.5,
            centrality=1.5,
            contamination_free=3.0,
            crack_free=2.5,
            contains_target=0.8,
            max_targets_per_grid_square=2,
        )
        restored = TargetScoringWeights.from_dict(original.to_dict())
        assert restored.cell_size == pytest.approx(original.cell_size)
        assert restored.centrality == pytest.approx(original.centrality)
        assert restored.contamination_free == pytest.approx(original.contamination_free)
        assert restored.crack_free == pytest.approx(original.crack_free)
        assert restored.contains_target == pytest.approx(original.contains_target)
        assert (
            restored.max_targets_per_grid_square == original.max_targets_per_grid_square
        )

    def test_from_dict_empty_uses_defaults(self):
        w = TargetScoringWeights.from_dict({})
        assert w.cell_size == pytest.approx(1.0)
        assert w.centrality == pytest.approx(1.0)
        assert w.contamination_free == pytest.approx(2.0)
        assert w.crack_free == pytest.approx(2.0)
        assert w.contains_target == pytest.approx(1.0)
        assert w.max_targets_per_grid_square == 1

    def test_backward_compat_legacy_bool_true(self):
        """Old configs stored select_best_per_grid_square as a bool."""
        w = TargetScoringWeights.from_dict({"select_best_per_grid_square": True})
        assert w.max_targets_per_grid_square == 1

    def test_backward_compat_legacy_bool_false(self):
        w = TargetScoringWeights.from_dict({"select_best_per_grid_square": False})
        assert w.max_targets_per_grid_square == 0

    def test_explicit_max_targets_takes_precedence(self):
        d = {"max_targets_per_grid_square": 3, "select_best_per_grid_square": True}
        w = TargetScoringWeights.from_dict(d)
        assert w.max_targets_per_grid_square == 3


# ---------------------------------------------------------------------------
# PostProcessingParameters
# ---------------------------------------------------------------------------


class TestPostProcessingParameters:
    def test_roundtrip_no_milling_stages(self):
        original = PostProcessingParameters(
            min_cell_diameter_um=6.0,
            max_cell_diameter_um=25.0,
            reject_threshold=0.3,
            edge_mask_padding=60,
            scoring_weights=TargetScoringWeights(cell_size=0.5),
            grid=SampleGridParameters(pitch=150e-6),
        )
        restored = PostProcessingParameters.from_dict(original.to_dict())
        assert restored.min_cell_diameter_um == pytest.approx(
            original.min_cell_diameter_um
        )
        assert restored.max_cell_diameter_um == pytest.approx(
            original.max_cell_diameter_um
        )
        assert restored.reject_threshold == pytest.approx(original.reject_threshold)
        assert restored.edge_mask_padding == original.edge_mask_padding
        assert restored.milling_stages == []
        assert restored.scoring_weights.cell_size == pytest.approx(0.5)
        assert restored.grid.pitch == pytest.approx(150e-6)

    def test_from_dict_empty_uses_defaults(self):
        pp = PostProcessingParameters.from_dict({})
        assert pp.min_cell_diameter_um == pytest.approx(5.0)
        assert pp.max_cell_diameter_um == pytest.approx(30.0)
        assert pp.reject_threshold == pytest.approx(0.2)
        assert pp.edge_mask_padding == 50
        assert pp.milling_stages == []

    def test_backward_compat_scoring_key(self):
        """Old configs stored grid geometry under top-level 'scoring' key."""
        d = {
            "scoring": {"pitch": 200e-6, "bar_width": 40e-6},
        }
        pp = PostProcessingParameters.from_dict(d)
        assert pp.grid.pitch == pytest.approx(200e-6)
        assert pp.grid.bar_width == pytest.approx(40e-6)

    def test_grid_key_takes_precedence_over_scoring(self):
        d = {
            "grid": {"pitch": 300e-6},
            "scoring": {"pitch": 999e-6},
        }
        pp = PostProcessingParameters.from_dict(d)
        assert pp.grid.pitch == pytest.approx(300e-6)


# ---------------------------------------------------------------------------
# GridScreeningParameters
# ---------------------------------------------------------------------------


class TestGridScreeningParameters:
    def _make_full(self) -> GridScreeningParameters:
        return GridScreeningParameters(
            overview=OverviewAcquisitionSettings(nrows=4, ncols=5),
            model=ModelParameters(checkpoint_path="/model.pth"),
            post_processing=PostProcessingParameters(
                min_cell_diameter_um=7.0,
                grid=SampleGridParameters(pitch=200e-6),
                scoring_weights=TargetScoringWeights(centrality=0.5),
            ),
            stage_positions=[
                FibsemStagePosition(x=1e-3, y=2e-3, z=3e-3),
                FibsemStagePosition(x=4e-3, y=5e-3, z=6e-3),
            ],
        )

    def test_roundtrip(self):
        original = self._make_full()
        restored = GridScreeningParameters.from_dict(original.to_dict())

        assert restored.overview.nrows == original.overview.nrows
        assert restored.overview.ncols == original.overview.ncols
        assert restored.model.checkpoint_path == original.model.checkpoint_path
        assert restored.post_processing.min_cell_diameter_um == pytest.approx(
            original.post_processing.min_cell_diameter_um
        )
        assert restored.post_processing.grid.pitch == pytest.approx(
            original.post_processing.grid.pitch
        )
        assert restored.post_processing.scoring_weights.centrality == pytest.approx(0.5)
        assert len(restored.stage_positions) == 2
        assert restored.stage_positions[0].x == pytest.approx(1e-3)
        assert restored.stage_positions[1].y == pytest.approx(5e-3)

    def test_from_dict_empty_uses_defaults(self):
        params = GridScreeningParameters.from_dict({})
        assert params.overview.nrows == 3
        assert params.overview.ncols == 3
        assert params.model.checkpoint_path == ""
        assert params.stage_positions == []

    def test_to_dict_top_level_keys(self):
        d = GridScreeningParameters().to_dict()
        assert set(d.keys()) == {
            "overview",
            "model",
            "post_processing",
            "grid_preparation",
            "protocol_path",
            "stage_positions",
        }

    def test_protocol_path_roundtrip(self):
        params = GridScreeningParameters(protocol_path="/some/protocol.yaml")
        restored = GridScreeningParameters.from_dict(params.to_dict())
        assert restored.protocol_path == "/some/protocol.yaml"

    def test_stage_positions_empty_list_roundtrip(self):
        params = GridScreeningParameters(stage_positions=[])
        restored = GridScreeningParameters.from_dict(params.to_dict())
        assert restored.stage_positions == []

    def test_grid_preparation_roundtrip(self):
        params = GridScreeningParameters()
        params.grid_preparation.enabled = True
        params.grid_preparation.deposition_position = FibsemStagePosition(
            x=1e-3, y=2e-3, z=3e-3, name="dep"
        )
        restored = GridScreeningParameters.from_dict(params.to_dict())
        assert restored.grid_preparation.enabled is True
        assert restored.grid_preparation.deposition_position.x == pytest.approx(1e-3)
        assert restored.grid_preparation.center_position is None


# ---------------------------------------------------------------------------
# GridPreparationParameters
# ---------------------------------------------------------------------------


class TestGridPreparationParameters:
    def test_defaults(self):
        p = GridPreparationParameters()
        assert p.enabled is False
        assert p.deposition_position is None
        assert p.center_position is None
        assert p.purge_duration == pytest.approx(5.0)
        assert p.deposition_duration == pytest.approx(25.0)
        assert p.heat_needle is True
        assert p.autofocus is True

    def test_from_dict_empty_uses_defaults(self):
        p = GridPreparationParameters.from_dict({})
        assert p.enabled is False
        assert p.deposition_position is None
        assert p.purge_duration == pytest.approx(5.0)

    def test_roundtrip_with_positions(self):
        original = GridPreparationParameters(
            enabled=True,
            deposition_position=FibsemStagePosition(x=1e-3, y=2e-3, z=3e-3, name="dep"),
            center_position=FibsemStagePosition(x=4e-3, y=5e-3, z=6e-3, name="center"),
            purge_duration=7.0,
            deposition_duration=30.0,
            heat_needle=False,
            autofocus=False,
            port="Pt Dep Cryo",
        )
        restored = GridPreparationParameters.from_dict(original.to_dict())
        assert restored.enabled is True
        assert restored.deposition_position.name == "dep"
        assert restored.center_position.y == pytest.approx(5e-3)
        assert restored.purge_duration == pytest.approx(7.0)
        assert restored.deposition_duration == pytest.approx(30.0)
        assert restored.heat_needle is False
        assert restored.autofocus is False
        assert restored.port == "Pt Dep Cryo"

    def test_none_positions_serialise_to_none(self):
        d = GridPreparationParameters().to_dict()
        assert d["deposition_position"] is None
        assert d["center_position"] is None

    def test_backward_compat_top_level_scoring_key(self):
        """Old configs had a top-level 'scoring' key for grid geometry."""
        d = {
            "scoring": {"pitch": 250e-6, "bar_width": 30e-6},
        }
        params = GridScreeningParameters.from_dict(d)
        assert params.post_processing.grid.pitch == pytest.approx(250e-6)

    def test_image_settings_beam_type_roundtrip(self):
        from fibsem.structures import ImageSettings

        params = GridScreeningParameters()
        params.overview.image_settings.beam_type = BeamType.ION
        params.overview.image_settings.hfw = 100e-6
        params.overview.image_settings.resolution = (1024, 1024)
        params.overview.image_settings.filename = "overview-image"

        restored = GridScreeningParameters.from_dict(params.to_dict())
        assert restored.overview.image_settings.beam_type == BeamType.ION
        assert restored.overview.image_settings.hfw == pytest.approx(100e-6)
        assert tuple(restored.overview.image_settings.resolution) == (1024, 1024)
        assert restored.overview.image_settings.filename == "overview-image"

    def test_milling_stages_not_serialised_by_default(self):
        """milling_stages are runtime-only; from_dict({}) must produce an empty list."""
        params = GridScreeningParameters.from_dict({})
        assert params.post_processing.milling_stages == []


# ---------------------------------------------------------------------------
# GridScreeningParameters.load()
# ---------------------------------------------------------------------------


class TestGridScreeningParametersLoad:
    def test_load_custom_path(self, tmp_path):
        """load(path) must read the supplied YAML file."""
        cfg = tmp_path / "custom-grid.yaml"
        cfg.write_text(
            "model:\n"
            "  checkpoint_path: /custom/model.pth\n"
            "overview:\n"
            "  image_settings:\n"
            "    beam_type: ELECTRON\n"
            "    hfw: 200.0e-6\n"
            "    filename: custom-overview\n"
        )
        params = GridScreeningParameters.load(path=cfg)
        assert params.model.checkpoint_path == "/custom/model.pth"
        assert params.overview.image_settings.beam_type == BeamType.ELECTRON
        assert params.overview.image_settings.hfw == pytest.approx(200e-6)
        assert params.overview.image_settings.filename == "custom-overview"

    def test_load_partial_yaml_uses_defaults(self, tmp_path):
        """Fields absent from the YAML fall back to dataclass defaults."""
        cfg = tmp_path / "partial.yaml"
        cfg.write_text("model:\n  checkpoint_path: only-this.pth\n")
        params = GridScreeningParameters.load(path=cfg)
        assert params.model.checkpoint_path == "only-this.pth"
        assert params.overview.nrows == 3
        assert params.post_processing.min_cell_diameter_um == pytest.approx(5.0)
        assert params.stage_positions == []

    def test_load_returns_independent_instances(self, tmp_path):
        """Two consecutive load() calls must return independent objects."""
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("model:\n  checkpoint_path: a.pth\n")
        a = GridScreeningParameters.load(cfg)
        b = GridScreeningParameters.load(cfg)
        a.model.checkpoint_path = "modified"
        assert b.model.checkpoint_path != "modified"
