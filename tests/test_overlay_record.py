"""A placement over a grid's overview survives the experiment file (FIB-1030).

`OverlayRecord` on `GridRecord.overlays`: round trip through the dict form, a grid
saved before the field existed still loads, one lattice of grid bars per grid, and an
experiment written and read back carries the record. No Qt, so this runs on every CI
job.
"""

import os

from fibsem.applications.autolamella.structures import (
    Experiment,
    GridRecord,
    OverlayRecord,
)


def _bars(**overrides):
    record = OverlayRecord(
        kind="gridbar", dx=12.5e-6, dy=-8e-6, rotation=-7.25, pitch=1e-4, bar_width=2e-5
    )
    for name, value in overrides.items():
        setattr(record, name, value)
    return record


class TestTheRecordRoundTrips:
    def test_every_field_survives_the_dict_form(self):
        record = _bars()
        record.fit = {"pairs": [[1, 2, 3, 4]], "rms": 0.5}
        back = OverlayRecord.from_dict(record.to_dict())
        assert back == record

    def test_an_image_record_keeps_its_source_and_reference(self):
        record = OverlayRecord(
            kind="image",
            source="fm/overview.ome.tiff",
            reference="SEM Overview/overview-1.tif",
            view="SEM @ SEM",
            scale=1.02,
        )
        back = OverlayRecord.from_dict(record.to_dict())
        assert (back.source, back.reference, back.view, back.scale) == (
            record.source,
            record.reference,
            record.view,
            record.scale,
        )

    def test_a_grid_saved_before_the_field_existed_still_loads(self):
        legacy = GridRecord(name="grid-oak").to_dict()
        del legacy["overlays"]
        grid = GridRecord.from_dict(legacy)
        assert grid.overlays == []
        assert grid.overlay_of("gridbar") is None


class TestOneLatticePerGrid:
    def test_setting_the_bars_again_replaces_them(self):
        grid = GridRecord(name="grid-oak")
        grid.set_overlay(_bars())
        grid.set_overlay(_bars(dx=1e-6))
        assert len(grid.overlays) == 1
        assert grid.overlay_of("gridbar").dx == 1e-6

    def test_a_record_with_an_id_replaces_by_id(self):
        grid = GridRecord(name="grid-oak")
        first = _bars()
        grid.set_overlay(first)
        again = OverlayRecord.from_dict(first.to_dict())
        again.rotation = 90.0
        grid.set_overlay(again)
        assert [o.rotation for o in grid.overlays] == [90.0]

    def test_images_are_kept_apart_from_the_bars_and_from_each_other(self):
        grid = GridRecord(name="grid-oak")
        grid.set_overlay(_bars())
        grid.set_overlay(OverlayRecord(kind="image", source="a.ome.tiff"))
        grid.set_overlay(OverlayRecord(kind="image", source="b.ome.tiff"))
        assert len(grid.overlays) == 3
        assert grid.overlay_of("gridbar").pitch == 1e-4


class TestTheExperimentCarriesIt:
    def test_saved_and_loaded_back(self, tmp_path):
        experiment = Experiment(path=tmp_path, name="overlay-record-test")
        os.makedirs(str(experiment.path), exist_ok=True)
        grid = experiment.add_grid(GridRecord(name="grid-oak"))
        grid.set_overlay(_bars())
        experiment.save()

        loaded = Experiment.load(os.path.join(str(experiment.path), "experiment.yaml"))
        back = loaded.get_grid_by_name("grid-oak").overlay_of("gridbar")
        assert back is not None
        assert (back.dx, back.dy, back.rotation) == (12.5e-6, -8e-6, -7.25)
        assert (back.pitch, back.bar_width) == (1e-4, 2e-5)
