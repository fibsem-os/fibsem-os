"""An image from other software, imported onto the Overview tab (FIB-1030).

Load… takes any image now. One that says where it was taken is placed from its
metadata, as before; any other goes through the import dialog, starts upright at the
centre of the view, and is kept as our own OME-TIFF in the folder its record will
look in. A restore never asks anything.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import tifffile

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QDialog

import fibsem.config as fibsem_config
from fibsem import utils
from fibsem.fm.reader import assumed_geometry, assumed_pose
from fibsem.fm.structures import (
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
)
from fibsem.projection import FMStageProjection
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemStagePosition,
    ImageSettings,
)
from fibsem.ui.widgets.canvas.aligned_images import decompose, image_map
from fibsem.ui.widgets.fm_import_dialog import ImportImageDialog
from fibsem.ui.widgets.overview_widget import FibsemOverviewWidget

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(scope="module")
def microscope():
    path = os.path.join(
        os.path.dirname(fibsem_config.__file__),
        "config",
        "sim-arctis-configuration.yaml",
    )
    scope, _ = utils.setup_session(manufacturer="Demo", config_path=path)
    return scope


@pytest.fixture(autouse=True)
def _destroy_widgets(destroy_widgets_after_test):
    """See tests/ui/test_overview_widget.py: the widget leaves top-levels behind."""


@pytest.fixture
def toasts(monkeypatch):
    said = []
    monkeypatch.setattr(
        "fibsem.ui.widgets.overview_widget.notification_service.show_toast",
        lambda text, *a, **k: said.append(text),
    )
    return said


def _beam_image(scope, orientation, beam_type):
    pose = scope.get_orientation(orientation)
    hfw = 128 * 2e-6
    image = FibsemImage.generate_blank_image(resolution=(128, 128), hfw=hfw)
    state = scope.get_microscope_state(beam_type=beam_type)
    state.stage_position = FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=pose.r, t=pose.t)
    image.metadata.image_settings = ImageSettings(hfw=hfw, beam_type=beam_type)
    image.metadata.microscope_state = state
    image.metadata.system_info = scope.system.info
    image.metadata.hardware_geometry = scope.hardware_geometry()
    return image


SQUARE = ("SEM", BeamType.ELECTRON)
FORESHORTENED = ("MILLING", BeamType.ION)


@pytest.fixture
def widget(microscope, tmp_path):
    w = FibsemOverviewWidget(microscope)
    w.resize(900, 700)
    w.set_save_directory(str(tmp_path / "experiment"))
    w.set_image(_beam_image(microscope, *SQUARE))
    yield w
    w.close()


def _foreign_tiff(folder, name="foreign.tif"):
    """Two channels by three z-slices, as another microscope's ImageJ export."""
    data = np.zeros((3, 2, 32, 40), dtype=np.uint16)  # Z, C, Y, X
    data[:, 0, 4:12, 4:12] = 3000
    data[:, 1, 20:28, 30:38] = 3000
    path = os.path.join(folder, name)
    tifffile.imwrite(
        path,
        data,
        imagej=True,
        resolution=(1 / 0.5, 1 / 0.5),
        metadata={"axes": "ZCYX", "unit": "micron"},
    )
    return path


def _placeable_fm(scope, folder):
    """One of our own acquisitions: it says where it was taken."""
    pose = scope.get_orientation("FM")
    image = FluorescenceImage(
        data=np.ones((1, 1, 16, 16), dtype=np.uint16),
        metadata=FluorescenceImageMetadata(
            acquisition_date="2026-09-23T10:00:00",
            pixel_size_x=1e-6,
            pixel_size_y=1e-6,
            stage_position=FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=pose.r, t=pose.t),
            channels=[
                FluorescenceChannelMetadata(
                    name="GFP",
                    excitation_wavelength=488.0,
                    power=0.5,
                    exposure_time=0.1,
                    gain=1.0,
                    offset=0.0,
                )
            ],
        ),
    )
    image.metadata.geometry = scope.fm_image_geometry()
    path = os.path.join(folder, "ours.ome.tiff")
    image.save(path)
    return path


def _view_centre(widget):
    (x0, x1), (y0, y1) = widget.canvas._ax.get_xlim(), widget.canvas._ax.get_ylim()
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


@pytest.fixture
def accept(monkeypatch):
    """The dialog, answered with Import after whatever `before` sets on it."""
    from types import SimpleNamespace

    answers = SimpleNamespace(seen=[], before=[])

    def exec_(dialog):
        answers.seen.append(dialog)
        for hook in answers.before:
            hook(dialog)
        return QDialog.Accepted

    monkeypatch.setattr(ImportImageDialog, "exec_", exec_)
    return answers


class TestTheStartingPlacement:
    @pytest.mark.parametrize("view", [SQUARE, FORESHORTENED])
    def test_the_centre_lands_on_the_target_upright(self, microscope, view):
        w = FibsemOverviewWidget(microscope)
        w.resize(900, 700)
        w.set_image(_beam_image(microscope, *view))
        frame = w._frame()
        w.aligned_images.refresh(frame)
        # Preconditions, said: without a scale every canvas point is the origin.
        assert frame is not None and w.canvas.reference_pixel_size
        projection = FMStageProjection(
            geometry=assumed_geometry(microscope), pixel_size=1e-6, shape=(40, 60)
        )
        target = (123.0, -45.0)

        base = w.aligned_images.base_at(projection, assumed_pose(microscope), target)

        a, offset = image_map(projection, base, frame.projection, frame.origin)
        assert w.canvas.metres_to_canvas(*offset) == pytest.approx(target, abs=1e-6)
        assert abs(decompose(a)[1]) == pytest.approx(0.0, abs=1e-6)
        w.close()

    def test_upright_on_an_instrument_whose_sem_view_is_turned(self, microscope):
        """The Arctis runs its SEM at a 180 degree scan rotation. There the assumed
        pose alone shows the image upside down; the opposite rotation is taken."""
        import math

        before = microscope.get("scan_rotation", BeamType.ELECTRON)
        microscope.set("scan_rotation", math.pi, BeamType.ELECTRON)
        try:
            w = FibsemOverviewWidget(microscope)
            w.resize(900, 700)
            w.set_image(_beam_image(microscope, *SQUARE))
            frame = w._frame()
            w.aligned_images.refresh(frame)
            projection = FMStageProjection(
                geometry=assumed_geometry(microscope), pixel_size=1e-6, shape=(40, 60)
            )
            pose = assumed_pose(microscope)
            base = w.aligned_images.base_at(projection, pose, (0.0, 0.0))
            a, _ = image_map(projection, base, frame.projection, frame.origin)
            assert abs(decompose(a)[1]) == pytest.approx(0.0, abs=1e-6)
            assert base.r == pytest.approx(pose.r + math.pi)  # the opposite one
            w.close()
        finally:
            microscope.set("scan_rotation", before, BeamType.ELECTRON)


class TestLoadAsksOnlyWhenItMustAndKeepsACopy:
    def test_our_own_image_is_placed_without_asking(
        self, widget, microscope, tmp_path, accept
    ):
        path = _placeable_fm(microscope, str(tmp_path))
        key = widget.open_aligned_image(path)
        assert key is not None and accept.seen == []
        assert widget.aligned_images.get(key).path == path

    def test_a_foreign_image_is_asked_about_and_starts_at_the_view_centre(
        self, widget, tmp_path, accept
    ):
        original = _foreign_tiff(str(tmp_path))
        before = open(original, "rb").read()

        key = widget.open_aligned_image(original)

        assert len(accept.seen) == 1
        record = widget.aligned_images.get(key)
        assert record.overlay.centre == pytest.approx(_view_centre(widget), abs=1e-6)
        assert abs(record.overlay.rotation) == pytest.approx(0.0, abs=1e-6)
        assert record.placement == (0.0, 0.0, 0.0, 1.0)
        # The copy is ours, in the folder a record looks in; the original is untouched.
        folder = widget.aligned_image_folder()
        assert os.path.dirname(record.path) == folder
        assert record.path.endswith("foreign.ome.tiff")
        assert open(original, "rb").read() == before
        copy = FluorescenceImage.load(record.path)
        assert copy.data.shape == (2, 3, 32, 40)  # channels first, as confirmed
        assert copy.metadata.pixel_size_x == pytest.approx(0.5e-6)
        assert widget._can_place(copy)

    def test_the_copy_lays_back_where_it_was_imported(self, widget, tmp_path, accept):
        key = widget.open_aligned_image(_foreign_tiff(str(tmp_path)))
        record = widget.aligned_images.get(key)
        centre = record.overlay.centre
        widget.remove_aligned_image(key)

        again = widget.load_aligned_image(record.path)  # as a restore does

        assert len(accept.seen) == 1  # no second question
        placed = widget.aligned_images.get(again).overlay.centre
        assert placed == pytest.approx(centre, abs=1e-6)

    def test_what_the_user_set_in_the_dialog_is_what_is_imported(
        self, widget, tmp_path, accept
    ):
        def answer(dialog):
            dialog.btn_swap.click()  # read the other way: 3 channels by 2 z
            dialog.spin_pixel_size.setValue(0.25)
            dialog.check_flip.setChecked(True)

        accept.before.append(answer)
        key = widget.open_aligned_image(_foreign_tiff(str(tmp_path)))

        copy = FluorescenceImage.load(widget.aligned_images.get(key).path)
        assert copy.data.shape == (3, 2, 32, 40)
        assert copy.metadata.pixel_size_x == pytest.approx(0.25e-6)
        assert "mirrored" in copy.metadata.description

    def test_a_png_is_always_asked_about(self, widget, tmp_path, accept):
        from PIL import Image

        path = str(tmp_path / "screenshot.png")
        Image.fromarray(np.zeros((20, 30, 3), dtype=np.uint8)).save(path)
        key = widget.open_aligned_image(path)
        assert len(accept.seen) == 1
        assert widget.aligned_images.get(key).path.endswith("screenshot.ome.tiff")

    def test_importing_the_same_file_twice_never_overwrites(
        self, widget, tmp_path, accept
    ):
        original = _foreign_tiff(str(tmp_path))
        first = widget.aligned_images.get(widget.open_aligned_image(original)).path
        second = widget.aligned_images.get(widget.open_aligned_image(original)).path
        assert first != second and os.path.isfile(first) and os.path.isfile(second)
        assert second.endswith("foreign-2.ome.tiff")

    def test_cancel_adds_nothing_and_writes_nothing(
        self, widget, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(ImportImageDialog, "exec_", lambda d: QDialog.Rejected)
        assert widget.open_aligned_image(_foreign_tiff(str(tmp_path))) is None
        assert widget.aligned_images.keys() == []
        assert not os.path.exists(widget.aligned_image_folder())

    def test_with_no_view_to_place_it_in_it_says_so_and_does_not_ask(
        self, widget, tmp_path, accept, toasts, monkeypatch
    ):
        monkeypatch.setattr(widget, "_frame", lambda *a, **k: None)
        assert widget.open_aligned_image(_foreign_tiff(str(tmp_path))) is None
        assert accept.seen == []
        assert any("overview first" in text for text in toasts)

    def test_a_restore_never_asks(self, widget, tmp_path, accept, toasts):
        assert widget.load_aligned_image(_foreign_tiff(str(tmp_path))) is None
        assert accept.seen == []

    def test_the_host_decides_the_folder(self, widget, tmp_path, accept):
        grid_folder = str(tmp_path / "grids" / "grid-1" / "Aligned Images")
        widget.aligned_image_folder = lambda: grid_folder
        key = widget.open_aligned_image(_foreign_tiff(str(tmp_path)))
        assert os.path.dirname(widget.aligned_images.get(key).path) == grid_folder


class TestTheAssumedCamera:
    def test_is_the_systems_fm_camera_when_there_is_one(self, microscope):
        assert assumed_geometry(microscope) == microscope.fm_image_geometry()
        assert assumed_pose(microscope) == microscope.get_orientation("FM")


class TestTheHostPointsImportsAtTheGrid:
    def test_the_folder_is_the_grid_under_the_stage(self, tmp_path):
        """So the record that keeps the image needs no second copy; with no grid
        under the stage, no folder, and the widget's own default applies."""
        from types import SimpleNamespace

        from fibsem.applications.autolamella.structures import Experiment, GridRecord
        from fibsem.applications.autolamella.ui.autolamella_overview_tab import (
            AutoLamellaOverviewTab,
        )

        experiment = Experiment(path=tmp_path, name="import-folder-test")
        grid = GridRecord(name="grid-oak")
        host = SimpleNamespace(
            current_grid=grid,
            experiment=experiment,
            ALIGNED_IMAGES_DIR=AutoLamellaOverviewTab.ALIGNED_IMAGES_DIR,
        )
        folder = AutoLamellaOverviewTab._aligned_image_folder(host)
        assert folder == os.path.join(str(experiment.grid_path(grid)), "Aligned Images")
        host.current_grid = None
        assert AutoLamellaOverviewTab._aligned_image_folder(host) is None


class TestTheMapIsMadeOncePerView:
    """A drag redraws on every mouse move and a stage move rebuilds an equal frame;
    each map made runs three stage transforms. Made once per view, not per redraw."""

    def test_a_drag_and_an_equal_frame_reuse_it_a_new_view_remakes_it(
        self, widget, microscope, tmp_path, monkeypatch
    ):
        import fibsem.ui.widgets.canvas.aligned_images as aligned

        key = widget.load_aligned_image(_placeable_fm(microscope, str(tmp_path)))
        made = []
        real = aligned.image_map
        monkeypatch.setattr(
            aligned, "image_map", lambda *a, **k: made.append(1) or real(*a, **k)
        )
        record = widget.aligned_images.get(key)
        cx, cy = record.overlay.centre
        for step in range(10):
            record.overlay.moved.emit(cx + step, cy)
        widget._refresh_context_overlays()  # an equal frame, rebuilt
        assert made == []

        widget.set_image(_beam_image(microscope, *FORESHORTENED))  # another view
        assert made == [1]
        assert widget.aligned_images.get(key).squash < 1.0  # and it is that view's

    def test_a_removed_image_forgets_its_map(self, widget, microscope, tmp_path):
        key = widget.load_aligned_image(_placeable_fm(microscope, str(tmp_path)))
        widget.remove_aligned_image(key)
        assert key not in widget.aligned_images._maps


class TestTheCopyIsCompressed:
    def test_as_the_files_people_bring_are(self, widget, tmp_path, accept):
        """zlib, which any install reads, with the predictor that makes 16-bit
        stacks compress."""
        key = widget.open_aligned_image(_foreign_tiff(str(tmp_path)))
        with tifffile.TiffFile(widget.aligned_images.get(key).path) as tif:
            assert tif.pages.first.compression.name == "ADOBE_DEFLATE"
            assert tifffile.PREDICTOR(tif.pages.first.predictor).name == "HORIZONTAL"
