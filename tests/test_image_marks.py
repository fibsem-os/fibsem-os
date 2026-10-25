"""Marks drawn on an image (fibsem.imaging.marks) and their labels placed
where they read (fibsem.imaging.labels)."""

import base64
import io

import numpy as np
from PIL import Image

from fibsem.imaging.export import ExportImage, ExportOptions
from fibsem.imaging.labels import LabelPlacer, label_size
from fibsem.imaging.marks import canvas, draw_marks, marked_picture, scaled_marks
from fibsem.structures.image import FibsemImage
from fibsem.theme.tokens import ACCENT_COLOR, AGENT_MARK_COLOR
from tests.fixtures.fm_stacks import planes_stack


def _rgb(colour):
    return np.asarray(Image.new("RGB", (1, 1), colour)).reshape(3).astype(int)


def _near(pixels, colour, tolerance=40):
    """Where *pixels* are *colour*."""
    return np.abs(pixels.astype(int) - _rgb(colour)).max(axis=-1) < tolerance


def _picture(marks, colour=AGENT_MARK_COLOR, size=(400, 300)):
    grey = np.full((size[1], size[0]), 90, dtype=np.uint8)
    image = ExportImage(rgb=grey, pixel_size=None, kind="Image")
    return marked_picture(image, marks, colour, ExportOptions(scalebar=False), 1024)


def _overlap(a, b):
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    return width > 0 and height > 0


def test_a_label_is_sized_to_the_picture_and_never_tiny():
    assert label_size(1000) == 24
    assert label_size(200) == 12


def test_labels_keep_clear_of_each_other_the_edges_and_what_is_drawn_over():
    figure, axes, scale = canvas(np.zeros((300, 400), dtype=np.uint8), 1024)
    corner = (300.0, 250.0, 400.0, 300.0)  # a scale bar, drawn later
    placer = LabelPlacer(axes, scale, [corner])
    placed = [
        placer.place((390.0, 280.0, 396.0, 286.0), "by the scale bar", "white"),
        placer.place((200.0, 150.0, 206.0, 156.0), "first", "white"),
        placer.place((204.0, 150.0, 210.0, 156.0), "second", "white"),
        placer.place((0.0, 0.0, 6.0, 6.0), "in the corner", "white"),
    ]
    for box in placed:
        assert 0 <= box[0] and box[2] <= 400 and 0 <= box[1] and box[3] <= 300
        assert not _overlap(box, corner)
    for i, a in enumerate(placed):
        for b in placed[i + 1 :]:
            assert not _overlap(a, b)


def test_every_kind_of_mark_is_drawn_in_the_colour_of_whoever_marked_it():
    marks = {
        "points": [{"x": 60, "y": 60, "shape": "target"}],
        "areas": [{"x": 150, "y": 40, "width": 80, "height": 50}],
        "circles": [{"x": 320, "y": 80, "radius": 30}],
        "lines": [{"x1": 40, "y1": 220, "x2": 360, "y2": 220, "arrow": True}],
    }
    agent = _picture(marks)
    assert _near(agent[48:73, 48:73], AGENT_MARK_COLOR).any()  # the target
    assert _near(agent[38:43, 150:230], AGENT_MARK_COLOR).any()  # the box's top
    assert _near(agent[75:86, 288:293], AGENT_MARK_COLOR).any()  # the circle's left
    assert _near(agent[216:225, 100:300], AGENT_MARK_COLOR).any(axis=0).all()
    assert not _near(agent, ACCENT_COLOR, 20).any()

    operator = _picture(marks, ACCENT_COLOR)
    assert _near(operator[216:225, 100:300], ACCENT_COLOR).any(axis=0).all()
    assert not _near(operator, AGENT_MARK_COLOR, 20).any()


def test_a_label_reads_on_a_see_through_box_not_an_outline():
    marks = {"points": [{"x": 100, "y": 150, "label": "target"}]}
    labelled = _picture(marks).astype(int)
    plain = _picture({"points": [{"x": 100, "y": 150}]}).astype(int)
    changed = np.abs(labelled - plain).max(axis=-1) > 0
    rows, cols = np.nonzero(changed)
    # Right of the mark, darker than the image behind it but not black.
    assert cols.min() > 100
    box = labelled[rows.min() + 4 : rows.max() - 4, cols.min() + 4 : cols.max() - 4]
    assert 15 < np.median(box) < 90


def test_marks_move_with_the_image_they_are_drawn_on():
    marks = {
        "points": [{"x": 10, "y": 20}],
        "lines": [{"x1": 0, "y1": 0, "x2": 10, "y2": 10}],
    }
    moved = scaled_marks(marks, 2.0, 0.5)
    assert moved["points"][0] == {"x": 20, "y": 10}
    assert moved["lines"][0] == {"x1": 0, "y1": 0, "x2": 20, "y2": 5}
    assert marks["points"][0] == {"x": 10, "y": 20}


def test_the_evidence_picture_draws_the_operator_s_marks_blue(tmp_path):
    from fibsem.ai.agent.drawing import evidence_png

    frame = FibsemImage.generate_blank_image(resolution=(200, 100), hfw=100e-6)
    path = str(tmp_path / "frame.tif")
    frame.save(path)
    marks = {"areas": [{"x": 20, "y": 20, "width": 160, "height": 60}]}
    for by, colour in (("operator", ACCENT_COLOR), ("agent", AGENT_MARK_COLOR)):
        png = evidence_png({"image": path, "marks": marks, "by": by})
        pixels = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
        assert pixels.shape[1] == 200
        assert _near(pixels[18:23, 40:160], colour).any()


def test_draw_marks_with_none_draws_nothing():
    figure, axes, scale = canvas(np.zeros((50, 50), dtype=np.uint8), 1024)
    before = len(axes.lines) + len(axes.patches) + len(axes.texts)
    draw_marks(axes, None, scale, LabelPlacer(axes, scale), AGENT_MARK_COLOR)
    assert len(axes.lines) + len(axes.patches) + len(axes.texts) == before


def _stack(path):
    """A two-channel stack: red top left, green bottom right."""
    from fibsem.fm.structures import (
        FluorescenceChannelMetadata,
        FluorescenceImage,
        FluorescenceImageMetadata,
    )

    data = np.zeros((2, 3, 256, 256), dtype=np.uint16)
    data[0, :, :128, :128] = 4000
    data[1, :, 128:, 128:] = 4000
    channels = [
        FluorescenceChannelMetadata(
            name=name,
            excitation_wavelength=488.0,
            power=0.5,
            exposure_time=0.1,
            gain=1.0,
            offset=0.0,
            color=colour,
        )
        for name, colour in (("red", "red"), ("green", "green"))
    ]
    metadata = FluorescenceImageMetadata(
        acquisition_datetime="2026-01-01T00:00:00",
        pixel_size_x=1e-7,
        pixel_size_y=1e-7,
        resolution=(256, 256),
        channels=channels,
        z_positions=[-1e-6, 0.0, 1e-6],
    )
    FluorescenceImage(data=data, metadata=metadata).save(path)


def test_marks_draw_on_a_stack_s_max_projection_in_the_channels_shown(tmp_path):
    from fibsem.imaging.marks import drawn_picture

    path = str(tmp_path / "stack.ome.tiff")
    _stack(path)
    marks = {"points": [{"x": 64, "y": 64, "shape": "target"}]}
    both = drawn_picture(path, marks, ACCENT_COLOR)
    assert both.shape[1] == 256 and both.shape[0] > 256  # the metadata bar
    assert _near(both[50:80, 50:80], ACCENT_COLOR).any()
    green = both[200, 200]
    assert green[1] > 100 and green[0] < 60

    red_only = drawn_picture(path, None, channels=["red"])
    assert red_only[200, 200].max() < 30


def test_a_picture_shows_the_plane_its_marks_are_on():
    from fibsem.imaging.marks import plane_shown

    target = {"x": 1, "y": 2, "z": 2, "label": "POI"}
    surface = {"x": 3, "y": 4, "z": 0, "label": "surface"}
    loose = {"x": 5, "y": 6}
    # One plane: that plane's marks and those on no plane.
    assert plane_shown({"points": [target, loose]}) == (
        2,
        {"points": [target, loose]},
    )
    assert plane_shown({"points": [target, surface]}, 0) == (
        0,
        {"points": [surface]},
    )
    # Marks on several planes: the projection, each label saying its plane.
    plane, shown = plane_shown({"points": [target, surface, loose]})
    assert plane is None
    assert shown["points"][0]["label"] == "POI · plane 3"
    assert shown["points"][1]["label"] == "surface · plane 1"
    assert "label" not in shown["points"][2]
    assert plane_shown({"points": [{"x": 1, "y": 2, "z": 1.6}]})[0] == 2
    # An image with no planes draws them all, labelled.
    assert plane_shown({"points": [target]}, stack=False)[0] is None
    assert plane_shown(None) == (None, None)


def test_marks_draw_on_the_plane_they_are_on(tmp_path):
    from fibsem.imaging.marks import drawn_pair, drawn_picture

    path = str(tmp_path / "stack.ome.tiff")
    planes_stack(path)
    on_two = {"points": [{"x": 80, "y": 20, "z": 2}]}
    picture = drawn_picture(path, on_two, ACCENT_COLOR)
    # Plane 2's block, not plane 0's, and the mark on it.
    assert picture[48, 80].max() > 100 and picture[48, 16].max() < 30
    assert _near(picture[10:30, 70:90], ACCENT_COLOR).any()
    # Asked for plane 0: plane 0's block, and the mark on plane 2 left out.
    other = drawn_picture(path, on_two, ACCENT_COLOR, plane=0)
    assert other[48, 16].max() > 100 and other[48, 80].max() < 30
    assert not _near(other[10:30, 70:90], ACCENT_COLOR).any()

    pair = drawn_pair(
        path,
        path,
        {"points": [{"x": 16, "y": 20, "z": 0}, {"x": 80, "y": 20, "z": 2}]},
        before_plane=0,
        after_plane=2,
    )
    left, right = pair[:, :96], pair[:, -96:]
    assert left[48, 16].max() > 100 and right[48, 80].max() > 100
    assert _near(left[10:30, 6:26], AGENT_MARK_COLOR).any()
    assert not _near(left[10:30, 70:90], AGENT_MARK_COLOR).any()
    assert _near(right[10:30, 70:90], AGENT_MARK_COLOR).any()


def test_a_plane_is_for_a_fluorescence_stack(tmp_path):
    import pytest

    from fibsem.imaging.marks import drawn_picture

    path = str(tmp_path / "stack.ome.tiff")
    planes_stack(path)
    with pytest.raises(ValueError):
        drawn_picture(path, None, plane=3)
    beam = str(tmp_path / "beam.tif")
    FibsemImage.generate_blank_image(resolution=(64, 64)).save(beam)
    with pytest.raises(ValueError, match="fluorescence stack"):
        drawn_picture(beam, None, plane=0)


def test_evidence_keeps_the_plane_a_mark_is_on(tmp_path):
    import pytest

    from fibsem.workflows import evidence

    planes_stack(str(tmp_path / "stack.ome.tiff"))
    FibsemImage.generate_blank_image(resolution=(64, 64)).save(
        str(tmp_path / "beam.tif")
    )
    root = lambda: str(tmp_path)  # noqa: E731

    kept = evidence.save(
        root,
        image="stack.ome.tiff",
        marks={"points": [{"x": 1, "y": 2, "z": 2}, {"x": 3, "y": 4}]},
        by=evidence.OPERATOR,
    )
    assert [p.get("z") for p in kept["marks"]["points"]] == [2, None]
    # A fitted z is kept as it is and drawn on the nearest plane.
    fitted = evidence.save(
        root, image="stack.ome.tiff", marks={"points": [{"x": 1, "y": 2, "z": 1.6}]}
    )
    assert fitted["marks"]["points"][0]["z"] == 1.6

    for image, marks, says in (
        ("stack.ome.tiff", {"points": [{"x": 1, "y": 2, "z": 3}]}, "0 to 2"),
        (
            "stack.ome.tiff",
            {"areas": [{"x": 1, "y": 2, "width": 3, "height": 4, "z": -1}]},
            "marks is",
        ),
        ("stack.ome.tiff", {"points": [{"x": 1, "y": 2, "z": True}]}, "marks is"),
        ("stack.ome.tiff", {"points": [{"x": 1, "y": 2, "z": 2.6}]}, "0 to 2"),
        (
            "beam.tif",
            {"lines": [{"x1": 1, "y1": 2, "x2": 3, "y2": 4, "z": 0}]},
            "fluorescence stack",
        ),
    ):
        with pytest.raises(evidence.EvidenceRefused, match=says):
            evidence.save(root, image=image, marks=marks)
    png = io.BytesIO()
    Image.new("RGB", (8, 8)).save(png, format="PNG")
    with pytest.raises(evidence.EvidenceRefused, match="fluorescence stack"):
        evidence.save(
            root,
            upload=base64.b64encode(png.getvalue()).decode(),
            marks={"points": [{"x": 1, "y": 2, "z": 0}]},
        )
