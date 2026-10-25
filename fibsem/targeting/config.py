from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fibsem.structures import FibsemStagePosition, OverviewAcquisitionSettings


@dataclass
class GridPreparationParameters:
    """Optional GIS deposition step run before grid screening.

    When ``enabled``, the workflow moves to ``deposition_position`` (typically
    the current stage position, grabbed in the UI), purges the GIS line, runs a
    timed deposition, then moves to ``center_position`` (the centre of the grid).
    All durations are in seconds. Positions may be ``None`` until captured.
    """

    enabled: bool = False
    deposition_position: Optional[FibsemStagePosition] = None  # grid 1 GIS deposition
    deposition_position_2: Optional[FibsemStagePosition] = None  # grid 2 GIS deposition
    sputter_position: Optional[FibsemStagePosition] = None  # grid 1 micro-sputter
    sputter_position_2: Optional[FibsemStagePosition] = None  # grid 2 micro-sputter
    center_position: Optional[FibsemStagePosition] = None  # grid 1 centre
    center_position_2: Optional[FibsemStagePosition] = (
        None  # grid 2 centre (2nd loaded grid)
    )
    purge_duration: float = 5.0  # s — valve open with needle retracted (purge line)
    deposition_duration: float = 25.0  # s — valve open with needle inserted
    heat_needle: bool = True  # heat needle to temperature before opening valve
    autofocus: bool = True  # SEM autofocus + link Z before insertion (needle height)
    port: Optional[str] = None  # GIS port (None = default)
    gas: Optional[str] = None  # gas name (multichem only)
    insert_position: Optional[str] = None  # insert position (multichem only)

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "deposition_position": (
                self.deposition_position.to_dict()
                if self.deposition_position is not None
                else None
            ),
            "deposition_position_2": (
                self.deposition_position_2.to_dict()
                if self.deposition_position_2 is not None
                else None
            ),
            "sputter_position": (
                self.sputter_position.to_dict()
                if self.sputter_position is not None
                else None
            ),
            "sputter_position_2": (
                self.sputter_position_2.to_dict()
                if self.sputter_position_2 is not None
                else None
            ),
            "center_position": (
                self.center_position.to_dict()
                if self.center_position is not None
                else None
            ),
            "center_position_2": (
                self.center_position_2.to_dict()
                if self.center_position_2 is not None
                else None
            ),
            "purge_duration": self.purge_duration,
            "deposition_duration": self.deposition_duration,
            "heat_needle": self.heat_needle,
            "autofocus": self.autofocus,
            "port": self.port,
            "gas": self.gas,
            "insert_position": self.insert_position,
        }

    @staticmethod
    def from_dict(d: dict) -> "GridPreparationParameters":
        dep = d.get("deposition_position")
        dep2 = d.get("deposition_position_2")
        spt = d.get("sputter_position")
        spt2 = d.get("sputter_position_2")
        ctr = d.get("center_position")
        ctr2 = d.get("center_position_2")
        return GridPreparationParameters(
            enabled=d.get("enabled", False),
            deposition_position=(FibsemStagePosition.from_dict(dep) if dep else None),
            deposition_position_2=(
                FibsemStagePosition.from_dict(dep2) if dep2 else None
            ),
            sputter_position=(FibsemStagePosition.from_dict(spt) if spt else None),
            sputter_position_2=(FibsemStagePosition.from_dict(spt2) if spt2 else None),
            center_position=(FibsemStagePosition.from_dict(ctr) if ctr else None),
            center_position_2=(FibsemStagePosition.from_dict(ctr2) if ctr2 else None),
            purge_duration=d.get("purge_duration", 5.0),
            deposition_duration=d.get("deposition_duration", 25.0),
            heat_needle=d.get("heat_needle", True),
            autofocus=d.get("autofocus", True),
            port=d.get("port", None),
            gas=d.get("gas", None),
            insert_position=d.get("insert_position", None),
        )


@dataclass
class ModelParameters:
    checkpoint_path: str = ""

    def to_dict(self) -> dict:
        return {"checkpoint_path": self.checkpoint_path}

    @staticmethod
    def from_dict(d: dict) -> "ModelParameters":
        return ModelParameters(checkpoint_path=d.get("checkpoint_path", ""))


@dataclass
class SampleGridParameters:
    """Physical sample and grid geometry. All distances in metres."""

    lamella_width: float = 11e-6  # m
    lamella_height: float = 8e-6  # m
    hole_width: float = 90e-6  # m
    pitch: float = 125e-6  # m
    bar_width: float = 35e-6  # m
    milling_angle: float = 11.0  # degrees
    grid_type: str = "Square"

    def to_dict(self) -> dict:
        return {
            "lamella_width": self.lamella_width,
            "lamella_height": self.lamella_height,
            "hole_width": self.hole_width,
            "pitch": self.pitch,
            "bar_width": self.bar_width,
            "milling_angle": self.milling_angle,
            "grid_type": self.grid_type,
        }

    @staticmethod
    def from_dict(d: dict) -> "SampleGridParameters":
        return SampleGridParameters(
            lamella_width=d.get("lamella_width", 11e-6),
            lamella_height=d.get("lamella_height", 8e-6),
            hole_width=d.get("hole_width", 90e-6),
            pitch=d.get("pitch", 125e-6),
            bar_width=d.get("bar_width", 35e-6),
            milling_angle=d.get("milling_angle", 11.0),
            grid_type=d.get("grid_type", "Square"),
        )


@dataclass
class TargetScoringWeights:
    """Per-component weights for the composite candidate scoring function.

    Each weight is a non-negative float; 0.0 disables that component.
    Contamination/crack weights default to 2.0 (hard disqualifiers deserve more pull).
    """

    cell_size: float = 1.0  # reward size close to ideal diameter
    centrality: float = 1.0  # reward proximity to grid-hole centre
    contamination_free: float = 2.0  # penalise contamination in candidate region
    crack_free: float = 2.0  # penalise cracks in candidate region
    contains_target: float = 1.0  # reward cell-class coverage in candidate region

    max_targets_per_grid_square: int = 1  # top N per hole; 0 = keep all
    restrict_to_single_target: bool = (
        True  # override: one target from most central hole
    )

    def to_dict(self) -> dict:
        return {
            "cell_size": self.cell_size,
            "centrality": self.centrality,
            "contamination_free": self.contamination_free,
            "crack_free": self.crack_free,
            "contains_target": self.contains_target,
            "max_targets_per_grid_square": self.max_targets_per_grid_square,
            "restrict_to_single_target": self.restrict_to_single_target,
        }

    @staticmethod
    def from_dict(d: dict) -> "TargetScoringWeights":
        # backward compat: old configs stored a bool select_best_per_grid_square
        if "max_targets_per_grid_square" not in d:
            legacy = d.get("select_best_per_grid_square", True)
            max_n = 1 if legacy else 0
        else:
            max_n = d["max_targets_per_grid_square"]
        return TargetScoringWeights(
            cell_size=d.get("cell_size", 1.0),
            centrality=d.get("centrality", 1.0),
            contamination_free=d.get("contamination_free", 2.0),
            crack_free=d.get("crack_free", 2.0),
            contains_target=d.get("contains_target", 1.0),
            max_targets_per_grid_square=max_n,
            restrict_to_single_target=d.get("restrict_to_single_target", True),
        )


@dataclass
class PostProcessingParameters:
    min_cell_diameter_um: float = 5.0  # µm — minimum cell diameter to detect
    max_cell_diameter_um: float = 30.0  # µm — maximum cell diameter to detect
    reject_threshold: float = (
        0.2  # fraction — rejects targets overlapping milling patterns
    )
    edge_mask_padding: int = 50  # pixels — buffer from image edges
    milling_stages: list = field(
        default_factory=list
    )  # list[FibsemMillingStage] for overlap filtering
    scoring_weights: TargetScoringWeights = field(default_factory=TargetScoringWeights)
    grid: SampleGridParameters = field(default_factory=SampleGridParameters)

    def to_dict(self) -> dict:
        return {
            "min_cell_diameter_um": self.min_cell_diameter_um,
            "max_cell_diameter_um": self.max_cell_diameter_um,
            "reject_threshold": self.reject_threshold,
            "edge_mask_padding": self.edge_mask_padding,
            "scoring_weights": self.scoring_weights.to_dict(),
            "grid": self.grid.to_dict(),
        }

    @staticmethod
    def from_dict(d: dict) -> "PostProcessingParameters":
        # backward compat: old configs stored grid geometry under top-level "scoring" key
        grid_dict = d.get("grid", d.get("scoring", {}))
        return PostProcessingParameters(
            min_cell_diameter_um=d.get("min_cell_diameter_um", 5.0),
            max_cell_diameter_um=d.get("max_cell_diameter_um", 30.0),
            reject_threshold=d.get("reject_threshold", 0.2),
            edge_mask_padding=d.get("edge_mask_padding", 50),
            milling_stages=[],
            scoring_weights=TargetScoringWeights.from_dict(
                d.get("scoring_weights", {})
            ),
            grid=SampleGridParameters.from_dict(grid_dict),
        )


@dataclass
class GridScreeningParameters:
    """Top-level configuration bundle for the grid screening workflow."""

    overview: OverviewAcquisitionSettings = field(
        default_factory=OverviewAcquisitionSettings
    )
    model: ModelParameters = field(default_factory=ModelParameters)
    post_processing: PostProcessingParameters = field(
        default_factory=PostProcessingParameters
    )
    grid_preparation: GridPreparationParameters = field(
        default_factory=GridPreparationParameters
    )
    protocol_path: str = ""  # default fibsemOS task protocol ("" = built-in default)
    stage_positions: list[FibsemStagePosition] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "overview": self.overview.to_dict(),
            "model": self.model.to_dict(),
            "post_processing": self.post_processing.to_dict(),
            "grid_preparation": self.grid_preparation.to_dict(),
            "protocol_path": self.protocol_path,
            "stage_positions": [p.to_dict() for p in self.stage_positions],
        }

    @staticmethod
    def from_dict(d: dict) -> "GridScreeningParameters":
        # backward compat: old configs had a top-level "scoring" key; pass it
        # into PostProcessingParameters.from_dict via the combined dict
        pp_dict = d.get("post_processing", {})
        if "scoring" in d and "grid" not in pp_dict:
            pp_dict = {**pp_dict, "scoring": d["scoring"]}
        return GridScreeningParameters(
            overview=OverviewAcquisitionSettings.from_dict(d.get("overview", {})),
            model=ModelParameters.from_dict(d.get("model", {})),
            post_processing=PostProcessingParameters.from_dict(pp_dict),
            grid_preparation=GridPreparationParameters.from_dict(
                d.get("grid_preparation", {})
            ),
            protocol_path=d.get("protocol_path", ""),
            stage_positions=[
                FibsemStagePosition.from_dict(p) for p in d.get("stage_positions", [])
            ],
        )

    @staticmethod
    def load(path: "str | Path") -> "GridScreeningParameters":
        """Load parameters from a YAML file; absent keys take the dataclass defaults."""
        from fibsem.utils import load_yaml

        return GridScreeningParameters.from_dict(load_yaml(Path(path)) or {})
