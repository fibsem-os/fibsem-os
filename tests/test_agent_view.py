"""What an image tool answers the agent (`fibsem.ai.agent.mcp.agent_view`):
every fact it acts on, each once, in less text. Real answers, from a Demo
image saved in an experiment folder; the sizes print with ``-s``."""

import copy
import json
import os

import pytest

from fibsem.ai.agent.mcp.agent_view import (
    VIEWED,
    dumps,
    relative,
    rounded,
    view,
)
from fibsem.analysis.tools import ImageTools
from fibsem.codec import encode
from fibsem.server import files
from fibsem.server.images import image_answer
from fibsem.structures.settings import BeamType, ImageSettings

# An image answer as beam_acquire_image answered it on a Windows instrument PC.
EXPERIMENT = r"C:\Users\User\Desktop\2026\AutoLamella-2026-11-05-11-09"
SAVED = EXPERIMENT + r"\grids\puck 26-8-2\full_mag_out\puck-26-8-2_sem_max_fov_eb.tif"
ACQUIRED = {
    "summary": {
        "kind": "image",
        "beam_type": "ELECTRON",
        "acquired_at": "2026-11-05T11:20:31.123456-07:00",
        "shape": [2048, 3072],
        "dtype": "uint8",
        "hfw": 0.004,
        "pixel_size": 1.0661175728009987e-06,
        "stage_position": {
            "x": 3.648800000046002e-09,
            "y": -1.5403999998828543e-10,
            "z": -1.0000231999999953e-05,
            "r_deg": 0.0,
            "t_deg": -0.0063,
            "coordinate_system": "SPECIMEN",
        },
        "scan_rotation_deg": 180.0,
        "reduced_area": None,
        "beam": {
            "voltage": 2000.0,
            "current": 2.5e-11,
            "working_distance": 0.009624611498499741,
            "dwell_time": 3e-07,
        },
        "detector": {"type": "ETD", "mode": "SecondaryElectrons"},
        "item": None,
        "task": None,
        "path": SAVED,
    },
    "path": SAVED,
    "preview": {
        "width": 768,
        "height": 512,
        "full_width": 3072,
        "full_height": 2048,
        "beam_type": "ELECTRON",
        "hfw": 0.004,
        "pixelsize": {"x": 1.0661175728009987e-06, "y": 1.0661175728009985e-06},
        "preview_pixelsize": 4.264470291203995e-06,
        "image_index": 0,
    },
}

SIZES = {}


def _measured(name, before, after):
    SIZES[name] = (len(json.dumps(before)), len(dumps(after)))
    print(f"\n{name}: {SIZES[name][0]} -> {SIZES[name][1]} characters")


def test_floats_keep_five_figures_and_never_less_than_a_nanometre():
    assert rounded(1.0661175728009987e-06) == 1.0661e-06
    assert rounded(4.264470291203995e-06) == 4.2645e-06
    # A stage coordinate keeps nanometres, whatever its size.
    assert rounded(0.0123456789123) == 0.012345679
    assert rounded(-1.0000231999999953e-05) == -1.0e-05
    assert rounded(1234.5678) == 1234.6
    # A POSIX time keeps its seconds.
    assert rounded(1763581169.257) == 1763581169.0
    assert rounded(0.0) == 0.0


def test_a_path_inside_the_experiment_is_relative_to_it():
    assert relative(SAVED, EXPERIMENT) == (
        "grids/puck 26-8-2/full_mag_out/puck-26-8-2_sem_max_fov_eb.tif"
    )
    # Windows paths compare without case; one elsewhere stays whole.
    assert relative(SAVED.lower(), EXPERIMENT).startswith("grids/")
    assert relative(r"D:\elsewhere\a.tif", EXPERIMENT) == r"D:\elsewhere\a.tif"
    assert relative("/data/exp/a/b.tif", "/data/exp") == "a/b.tif"
    assert relative("/data/exp-2/b.tif", "/data/exp") == "/data/exp-2/b.tif"
    assert relative("a/b.tif", "/data/exp") == "a/b.tif"


def test_an_acquired_image_says_each_fact_once():
    seen = view(copy.deepcopy(ACQUIRED), "beam_acquire_image", EXPERIMENT)
    _measured("beam_acquire_image", ACQUIRED, seen)
    assert seen["path"] == (
        "grids/puck 26-8-2/full_mag_out/puck-26-8-2_sem_max_fov_eb.tif"
    )
    summary = seen["summary"]
    assert "path" not in summary and "kind" not in summary
    assert not {"reduced_area", "item", "task"} & set(summary)
    assert summary["pixel_size"] == 1.0661e-06
    assert summary["stage_position"]["z"] == -1.0e-05
    assert summary["stage_position"]["coordinate_system"] == "SPECIMEN"
    assert summary["acquired_at"] == ACQUIRED["summary"]["acquired_at"]
    # The preview keeps its own size and scale, and its place among the images.
    assert seen["preview"] == {
        "width": 768,
        "height": 512,
        "preview_pixelsize": 4.2645e-06,
        "image_index": 0,
    }
    assert SIZES["beam_acquire_image"][1] < SIZES["beam_acquire_image"][0] * 0.6


def test_a_tool_that_is_not_an_image_tool_keeps_its_answer():
    proposals = {"values": {"depth": 1.234567891e-06}, "path": SAVED}
    assert view(copy.deepcopy(proposals), "proposal_list", EXPERIMENT) == proposals
    assert "proposal_list" not in VIEWED


@pytest.fixture
def experiment(tmp_path, microscope):
    """An experiment folder holding a Demo SEM image, as beam_acquire_image
    saves one: (folder, the image's relative path, the image)."""
    root = os.path.realpath(tmp_path)
    folder = os.path.join(root, "grids", "puck 26-8-2", "full_mag_out")
    os.makedirs(folder)
    image = microscope.acquire_image(
        ImageSettings(resolution=(1536, 1024), hfw=4e-3, beam_type=BeamType.ELECTRON)
    )
    path = os.path.join(folder, "overview_eb.tif")
    image.save(path)
    image.filepath = path
    return root, os.path.relpath(path, root).replace(os.sep, "/"), image


def _without_previews(answer):
    from fibsem.ai.agent.mcp.sidecar import _pop_previews

    _pop_previews(answer)
    return answer


def test_eight_cropped_cells_say_what_they_share_once(experiment):
    root, path, _ = experiment
    shown = files.read(root, path, cells=True)["preview"]["cells"]["items"]
    names = [cell["name"] for cell in shown][:8]
    before = _without_previews(files.read(root, path, crops=names))
    seen = view(copy.deepcopy(before), "analysis_crop_image", root)
    _measured("analysis_crop_image (8 cells)", before, seen)

    assert seen["path"] == path
    assert not {"name", "info", "modified"} & set(seen)
    assert [crop["name"] for crop in seen["crops"]] == names
    for crop, was in zip(seen["crops"], before["crops"]):
        assert crop["preview"]["region"] == was["preview"]["region"]
        assert crop["preview"]["image_index"] == was["preview"]["image_index"]
        assert crop["stage"]["x"] == pytest.approx(was["stage"]["x"], abs=1e-9)
        # The preview's size and scale: its own, else what every crop shares.
        scale = {**seen["crop_preview"], **crop["preview"]}
        assert scale["width"] == was["preview"]["width"]
        assert scale["height"] == was["preview"]["height"]
        assert scale["preview_pixelsize"] == pytest.approx(
            was["preview"]["preview_pixelsize"], rel=1e-4
        )
        assert not {"beam_type", "pixelsize", "full_width"} & set(scale)
    assert SIZES["analysis_crop_image (8 cells)"][1] < (
        SIZES["analysis_crop_image (8 cells)"][0] / 2
    )


def test_a_drawing_answers_what_it_computed_not_the_marks_it_was_given(experiment):
    root, path, _ = experiment
    marks = {
        "points": [
            {"x": 100 + 50 * i, "y": 200, "label": f"site-{i}"} for i in range(6)
        ]
        + [{"x": 700, "y": 300}],
        "outlines": [
            {
                "points": [[10, 10], [400, 12], [410, 300], [12, 310], [5, 150]],
                "label": "square A",
            }
        ],
        "lines": [{"x1": 10, "y1": 900, "x2": 1110, "y2": 900, "measure": True}],
    }
    drawn = ImageTools(lambda: root).draw_image(path, marks)
    before = _without_previews(encode(drawn))
    seen = view(copy.deepcopy(before), "analysis_draw_image", root)
    _measured("analysis_draw_image", before, seen)

    assert seen["path"] == os.path.dirname(path) + "/drawn/overview_eb.png"
    assert "image" not in seen
    points = seen["marks"]["points"]
    assert points[0] == {
        "label": "site-0",
        "stage": {
            k: pytest.approx(v, abs=1e-9)
            for k, v in before["marks"]["points"][0]["stage"].items()
        },
    }
    # An unlabelled mark keeps where it is, to tell it by.
    assert set(points[-1]) == {"x", "y", "stage"}
    assert seen["marks"]["outlines"][0].keys() == {"label", "stage"}
    line = seen["marks"]["lines"][0]
    assert line["label"] == before["marks"]["lines"][0]["label"]
    assert {"stage1", "stage2"} <= set(line)
    assert seen["plot"] == {"image_index": 0}
    assert seen["layers"]["SEM"]["p99"] == before["layers"]["SEM"]["p99"]
    assert SIZES["analysis_draw_image"][1] < SIZES["analysis_draw_image"][0] * 0.6


def test_a_saved_image_answer_names_its_file_inside_the_experiment(experiment):
    root, path, image = experiment
    before = _without_previews(image_answer(image))
    seen = view(copy.deepcopy(before), "beam_acquire_image", root)
    assert seen["path"] == path
    # Not under the open experiment (another one, read by name): kept whole.
    elsewhere = view(copy.deepcopy(before), "beam_acquire_image", root + "-other")
    assert elsewhere["path"] == image.filepath
