"""The export renderer: what the metadata bar says, and what the picture looks like.

Qt-free, like the module, so this runs on every CI job rather than only ui-tests.
"""

import os
from datetime import datetime

import numpy as np
import pytest

from fibsem.fm.structures import (
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
)
from fibsem.imaging.export import (
    CROSSHAIR_COLOR,
    ExportOptions,
    adjust_contrast,
    auto_contrast_limits,
    default_export_name,
    default_options,
    export_shape,
    format_si,
    from_fibsem_image,
    from_fluorescence_image,
    image_fields,
    load_export_image,
    render_export,
    save_export,
)
from fibsem.structures import (
    BeamSettings,
    BeamType,
    FibsemDetectorSettings,
    FibsemExperimentRef,
    FibsemImage,
    MicroscopeState,
)


def _sem_image(voltage=2e3, current=50e-12, experiment=None) -> FibsemImage:
    image = FibsemImage.generate_blank_image(resolution=(1536, 1024), hfw=150e-6)
    image.data[:] = 64
    image.metadata.microscope_state = MicroscopeState(
        electron_beam=BeamSettings(
            BeamType.ELECTRON,
            voltage=voltage,
            beam_current=current,
            working_distance=4e-3,
        ),
        electron_detector=FibsemDetectorSettings(type="ETD", mode="SE"),
    )
    if experiment is not None:
        image.metadata.experiment = experiment
    return image


def _fm_image() -> FluorescenceImage:
    data = np.zeros((2, 3, 256, 256), dtype=np.uint16)
    data[0, :, 100:120, 100:120] = 40000
    data[1, :, 10:20, 10:20] = 40000
    channels = [
        FluorescenceChannelMetadata(
            name=name,
            excitation_wavelength=wavelength,
            power=0.5,
            exposure_time=0.1,
            gain=1.0,
            offset=0.0,
            color=color,
            objective_magnification=100,
            objective_numerical_aperture=0.85,
        )
        for name, wavelength, color in [("DAPI", 365, "cyan"), ("GFP", 470, "green")]
    ]
    return FluorescenceImage(
        data=data,
        metadata=FluorescenceImageMetadata(
            acquisition_date="2026-10-02T14:48:00",
            pixel_size_x=65e-9,
            pixel_size_y=65e-9,
            channels=channels,
            z_positions=[0.0, 0.5e-6, 1.0e-6],
        ),
    )


def _values(fields):
    return {f.key: f.value for f in fields}


@pytest.mark.parametrize(
    "value, unit, expected",
    [
        (150e-6, "m", "150 µm"),
        (97.65625e-9, "m", "97.7 nm"),
        (2e3, "V", "2 kV"),
        (50e-12, "A", "50 pA"),
        (1e-6, "s", "1 µs"),
        (0.9999996e-3, "m", "1 mm"),  # rounds up past the prefix boundary
        (0, "m", "0 m"),
    ],
)
def test_format_si(value, unit, expected):
    assert format_si(value, unit) == expected


def test_beam_image_fields():
    export = from_fibsem_image(_sem_image())
    assert export.kind == "SEM"
    assert _values(export.fields) == {
        "detector": "ETD · SE",  # mode given as "SE" here; see the abbreviation test
        "hfw": "150 µm",
        "pixel_size": "97.7 nm",
        "voltage": "2 kV",
        "current": "50 pA",
        "working_distance": "4 mm",
        "dwell_time": "1 µs",
    }


@pytest.mark.parametrize(
    "make, export",
    [(_sem_image, from_fibsem_image), (_fm_image, from_fluorescence_image)],
)
def test_image_fields_are_the_export_bar(make, export):
    """The canvas reads the same fields the export draws, from one place."""
    image = make()
    info = image_fields(image)
    exported = export(image)
    assert info.kind == exported.kind
    assert info.fields == exported.fields
    assert info.provenance == exported.provenance


def test_image_fields_never_read_the_pixels(monkeypatch):
    """Cheap enough for every image the canvas is handed: no conversion, no projection."""
    import fibsem.imaging.export as export_module

    def refuse(*args, **kwargs):
        raise AssertionError("image_fields touched the pixels")

    for name in ("_to_uint8", "_normalize", "projection_layers", "composite_fm_layers"):
        monkeypatch.setattr(export_module, name, refuse)
    assert _values(image_fields(_sem_image()).fields)["hfw"] == "150 µm"
    assert image_fields(_fm_image()).kind == "FM"


def test_image_fields_without_metadata():
    image = _sem_image()
    image.metadata = None
    info = image_fields(image)
    assert (info.kind, info.fields, info.provenance) == ("Image", [], [])


def test_hfw_comes_from_the_pixel_size_not_the_request():
    """image_settings.hfw is what was asked for; the pixel size is what was recorded."""
    image = _sem_image()
    image.metadata.image_settings.hfw = 999e-6
    assert _values(from_fibsem_image(image).fields)["hfw"] == "150 µm"


def test_a_value_the_file_does_not_record_is_left_out():
    export = from_fibsem_image(_sem_image(voltage=None))
    assert "voltage" not in export.field_keys()
    assert "voltage" not in default_options(export).fields


def test_provenance_names_the_lamella_and_task():
    ref = FibsemExperimentRef(name="exp", item_name="lamella-03", task_name="Polishing")
    export = from_fibsem_image(_sem_image(experiment=ref))
    values = _values(export.provenance)
    assert values["item"] == "lamella-03"
    assert values["task"] == "Polishing"
    assert values["experiment"] == "exp"


def test_no_lamella_or_task_outside_a_workflow():
    export = from_fibsem_image(_sem_image())
    assert "item" not in export.provenance_keys()
    assert "task" not in export.provenance_keys()


def test_a_grid_is_an_item_like_a_lamella():
    ref = FibsemExperimentRef(name="exp", item_name="grid-01", task_name="Screen")
    item = from_fibsem_image(_sem_image(experiment=ref)).provenance[1]
    assert (item.key, item.name, item.value) == (
        "item",
        "Item",
        "grid-01",
    )


@pytest.mark.parametrize(
    "timestamp",
    [
        datetime(2026, 8, 19, 15, 44).timestamp(),
        "08/19/2026 15:44:41",  # as written in a real v3 image
        "2026-08-19T15:44:41",
    ],
)
def test_the_date_reads_from_any_recorded_timestamp(timestamp):
    image = _sem_image()
    image.metadata.microscope_state.timestamp = timestamp
    assert _values(from_fibsem_image(image).provenance)["date"] == "2026-08-19 15:44"


def test_an_unreadable_timestamp_is_left_out():
    image = _sem_image()
    image.metadata.microscope_state.timestamp = "not a date"
    assert "date" not in from_fibsem_image(image).provenance_keys()


def test_an_image_without_metadata_still_exports():
    image = FibsemImage(data=np.zeros((100, 150), dtype=np.uint8), metadata=None)
    export = from_fibsem_image(image)
    assert export.kind == "Image"
    assert export.pixel_size is None
    out = render_export(export, default_options(export))
    assert out.shape[1] == 150  # no scalebar to draw, but nothing fails


def test_the_bar_goes_below_the_image():
    export = from_fibsem_image(_sem_image())
    out = render_export(export, default_options(export))
    assert out.shape[1] == 1536
    assert out.shape[0] > 1024
    # The image rows are untouched by the bar: they are still the image.
    assert out.shape[2] == 3 and out.dtype == np.uint8


def test_no_fields_and_no_provenance_means_no_bar():
    export = from_fibsem_image(_sem_image())
    out = render_export(export, ExportOptions(scalebar=False))
    assert out.shape == (1024, 1536, 3)
    assert np.all(out == 64)


def test_provenance_adds_a_row():
    ref = FibsemExperimentRef(name="exp", item_name="lamella-03", task_name="Polishing")
    export = from_fibsem_image(_sem_image(experiment=ref))
    without = render_export(export, default_options(export))
    options = default_options(export)
    options.provenance = ["item", "task"]
    with_row = render_export(export, options)
    assert with_row.shape[0] > without.shape[0]


def test_scale_two_repeats_every_pixel():
    export = from_fibsem_image(_sem_image())
    out = render_export(export, ExportOptions(scalebar=False, scale=2))
    assert out.shape == (2048, 3072, 3)


@pytest.mark.parametrize("scale", [1, 2])
@pytest.mark.parametrize("provenance", [[], ["item", "task", "date"]])
def test_export_shape_is_the_rendered_shape(scale, provenance):
    """The dialog reports this size without rendering; it must not drift from it."""
    ref = FibsemExperimentRef(name="exp", item_name="lamella-03", task_name="Polishing")
    export = from_fibsem_image(_sem_image(experiment=ref))
    options = default_options(export)
    options.scale = scale
    options.provenance = provenance
    assert export_shape(export, options) == render_export(export, options).shape[:2]


def test_the_crosshair_is_drawn_at_the_centre():
    export = from_fibsem_image(_sem_image())
    out = render_export(export, ExportOptions(scalebar=False, crosshair=True))
    centre = out[512, 768].astype(int)
    # Blended at 0.8 over grey 64: dominated by the crosshair's yellow.
    assert centre[0] > 200 and centre[1] > 190 and centre[2] < 120
    assert tuple(out[0, 0]) == (64, 64, 64)
    assert CROSSHAIR_COLOR[0] == 255


def test_the_scalebar_is_drawn_in_the_chosen_corner():
    export = from_fibsem_image(_sem_image())
    right = render_export(export, ExportOptions(scalebar_location="lower right"))
    left = render_export(export, ExportOptions(scalebar_location="lower left"))
    assert not np.all(right[900:, 1400:] == 64)
    assert np.all(right[900:, :100] == 64)
    assert not np.all(left[900:, :100] == 64)


def test_fluorescence_fields_and_channels():
    export = from_fluorescence_image(_fm_image())
    assert export.kind == "FM"
    assert _values(export.fields) == {
        "objective": "100× · 0.85 NA",
        "hfw": "16.6 µm",
        "pixel_size": "65 nm",
        "z": "MIP · 3 × 500 nm",
    }
    assert [c.name for c in export.channels] == ["DAPI", "GFP"]
    assert export.channels[0].color == (0, 255, 255)


def test_the_legend_is_drawn_top_right():
    export = from_fluorescence_image(_fm_image())
    with_legend = render_export(export, ExportOptions(scalebar=False, legend=True))
    without = render_export(export, ExportOptions(scalebar=False, legend=False))
    assert not np.array_equal(with_legend[:40, -80:], without[:40, -80:])
    assert np.array_equal(with_legend[:40, :80], without[:40, :80])


def test_load_a_saved_beam_image(tmp_path):
    ref = FibsemExperimentRef(name="exp", item_name="lamella-03", task_name="Polishing")
    path = _sem_image(experiment=ref).save(str(tmp_path / "ref_image.tif"))
    export = load_export_image(path)
    assert export.path == path
    assert _values(export.fields)["hfw"] == "150 µm"
    assert _values(export.provenance)["item"] == "lamella-03"


@pytest.mark.parametrize("extension", [".png", ".tif"])
def test_save_round_trip(tmp_path, extension):
    from PIL import Image

    export = from_fibsem_image(_sem_image())
    out = render_export(export, default_options(export))
    path = save_export(out, str(tmp_path / f"export{extension}"))
    assert np.array_equal(np.array(Image.open(path)), out)


def test_default_export_name_sits_beside_the_source():
    folder = os.path.join("data", "lamella-03")  # the platform's separator
    export = from_fluorescence_image(_fm_image())
    export.path = os.path.join(folder, "fm_stack.ome.tiff")
    assert default_export_name(export) == os.path.join(folder, "fm_stack_export.png")
    expected_tif = os.path.join(folder, "fm_stack_export.tif")
    assert default_export_name(export, ".tif") == expected_tif


def _gradient_image() -> FibsemImage:
    """A left-to-right ramp, 0 to 255: contrast changes are easy to read off it."""
    image = _sem_image()
    image.data[:] = np.linspace(0, 255, 1536).astype(np.uint8)[None, :]
    return image


def test_detector_modes_are_abbreviated():
    image = _sem_image()
    image.metadata.microscope_state.electron_detector.mode = "SecondaryElectrons"
    assert _values(from_fibsem_image(image).fields)["detector"] == "ETD · SE"
    image.metadata.microscope_state.electron_detector.mode = "SomeVendorMode"
    assert (
        _values(from_fibsem_image(image).fields)["detector"] == "ETD · SomeVendorMode"
    )


def test_default_contrast_leaves_the_image_as_acquired():
    export = from_fibsem_image(_gradient_image())
    out = render_export(export, ExportOptions(scalebar=False))
    assert np.array_equal(out, export.rgb)


def test_contrast_limits_clip_and_stretch():
    export = from_fibsem_image(_gradient_image())
    options = ExportOptions(scalebar=False, contrast_min=0.25, contrast_max=0.75)
    out = render_export(export, options)[0, :, 0]
    assert out[: 1536 // 5].max() == 0  # below the min: black
    assert out[-1536 // 5 :].min() == 255  # above the max: white
    assert out[768] in (127, 128)  # the middle stays in the middle


def test_gamma_above_one_darkens():
    export = from_fibsem_image(_gradient_image())
    plain = render_export(export, ExportOptions(scalebar=False, gamma=1.0))
    darker = render_export(export, ExportOptions(scalebar=False, gamma=2.0))
    assert darker.mean() < plain.mean()


def test_adjust_contrast_matches_its_definition():
    norm = np.linspace(0.0, 1.0, 11, dtype=np.float32)
    out = adjust_contrast(norm, 0.2, 0.8, 2.0)
    expected = ((np.clip(norm, 0.2, 0.8) - 0.2) / 0.6) ** 2.0
    assert np.allclose(out, expected)


def test_auto_contrast_uses_the_1st_and_99th_percentiles():
    export = from_fibsem_image(_gradient_image())
    lo, hi = auto_contrast_limits(export)
    assert lo == pytest.approx(0.01, abs=0.01)
    assert hi == pytest.approx(0.99, abs=0.01)


def test_fluorescence_is_not_contrast_adjusted_as_a_whole():
    """Each channel is contrasted on its own; one min/max over the blend would not do."""
    export = from_fluorescence_image(_fm_image())
    assert not export.adjustable
    plain = render_export(export, ExportOptions(scalebar=False, legend=False))
    options = ExportOptions(scalebar=False, legend=False, contrast_min=0.5, gamma=2.0)
    assert np.array_equal(render_export(export, options), plain)


def test_provenance_wraps_rather_than_running_off_the_edge():
    """A long provenance row on a narrow image wraps; the shape stays predictable."""
    ref = FibsemExperimentRef(
        name="AutoLamella-2026-09-13-20-00-DEV-TEST",
        item_name="my great grid",
        task_name="SEM Overview",
    )
    image = FibsemImage.generate_blank_image(resolution=(400, 300), hfw=150e-6)
    image.metadata.experiment = ref
    export = from_fibsem_image(image)
    options = ExportOptions(provenance=["experiment", "item", "task"])
    out = render_export(export, options)
    assert export_shape(export, options) == out.shape[:2]
    one_line = render_export(export, ExportOptions(provenance=["item"]))
    assert out.shape[0] > one_line.shape[0]


def test_z_slices_are_counted_from_the_data():
    """A real file lists a z-position per channel per slice; the count is the stack's."""
    image = _fm_image()
    image.metadata.z_positions = [0.0, 0.5e-6, 1.0e-6] * 2  # two channels' worth
    assert _values(from_fluorescence_image(image).fields)["z"] == "MIP · 3 × 500 nm"


def test_a_hidden_channel_leaves_the_blend_and_the_legend():
    export = from_fluorescence_image(_fm_image())
    both = render_export(export, ExportOptions(scalebar=False, legend=False))
    no_dapi = render_export(
        export, ExportOptions(scalebar=False, legend=False, hidden_channels=[0])
    )
    # DAPI (cyan) is the square at 100:120; GFP (green) at 10:20 is untouched.
    assert both[110, 110].any() and not no_dapi[110, 110].any()
    assert np.array_equal(both[15, 15], no_dapi[15, 15])

    with_legend = render_export(export, ExportOptions(scalebar=False))
    hidden_legend = render_export(
        export, ExportOptions(scalebar=False, hidden_channels=[0])
    )
    assert not np.array_equal(with_legend[:40, -120:], hidden_legend[:40, -120:])


def test_hiding_a_channel_does_not_stick():
    export = from_fluorescence_image(_fm_image())
    before = render_export(export, ExportOptions(scalebar=False))
    render_export(export, ExportOptions(scalebar=False, hidden_channels=[0, 1]))
    assert np.array_equal(render_export(export, ExportOptions(scalebar=False)), before)


def test_every_channel_hidden_is_a_black_image():
    export = from_fluorescence_image(_fm_image())
    out = render_export(export, ExportOptions(scalebar=False, hidden_channels=[0, 1]))
    assert out.shape == (256, 256, 3) and not out.any()
