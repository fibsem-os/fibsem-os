"""The latest image of each beam, kept on the session (FIB-1451)."""

from fibsem.imaging.latest import LatestImages
from fibsem.structures import BeamType, FibsemImage


def _image(beam_type):
    image = FibsemImage.generate_blank_image(resolution=[32, 32], hfw=1e-6)
    image.metadata.image_settings.beam_type = beam_type
    return image


def test_each_beam_keeps_its_own_latest():
    images = LatestImages()
    sem, fib = _image(BeamType.ELECTRON), _image(BeamType.ION)

    images.set(sem)
    images.set(fib)

    assert images.get(BeamType.ELECTRON) is sem
    assert images.get(BeamType.ION) is fib


def test_a_named_beam_wins_over_the_metadata():
    images = LatestImages()
    frame = _image(BeamType.ELECTRON)

    images.set(frame, BeamType.ION)

    assert images.get(BeamType.ION) is frame
    assert images.get(BeamType.ELECTRON) is None


def test_the_image_widget_reads_and_writes_the_session_s_images(qapp):
    from fibsem import utils
    from fibsem.structures import ImageSettings
    from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget

    microscope, _ = utils.setup_session(manufacturer="Demo")
    images = LatestImages()
    widget = FibsemImageSettingsWidget(
        microscope=microscope,
        image_settings=ImageSettings(resolution=[64, 64], hfw=1e-5),
        parent=None,
        images=images,
    )
    try:
        assert widget.ib_image is images.get(BeamType.ION)
        assert images.get(BeamType.ION).metadata.beam_type is BeamType.ION
        frame = _image(BeamType.ION)
        widget.ib_image = frame
        assert images.get(BeamType.ION) is frame
    finally:
        widget.deleteLater()


def test_an_image_widget_without_a_session_follows_the_microscope(qapp):
    from fibsem import utils
    from fibsem.structures import ImageSettings
    from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget

    microscope, _ = utils.setup_session(manufacturer="Demo")
    widget = FibsemImageSettingsWidget(
        microscope=microscope,
        image_settings=ImageSettings(resolution=[64, 64], hfw=1e-5),
        parent=None,
    )
    try:
        frame = _image(BeamType.ELECTRON)
        microscope.imaging.show(frame)
        assert widget.eb_image is frame
    finally:
        widget.deleteLater()


def test_images_shown_and_live_frames_are_kept_until_unfollowed():
    from fibsem import utils

    microscope, _ = utils.setup_session(manufacturer="Demo")
    images = LatestImages()
    images.follow(microscope)

    shown, live = _image(BeamType.ION), _image(BeamType.ELECTRON)
    microscope.imaging.show(shown)
    microscope.beams[BeamType.ELECTRON].live_frame.emit(live)
    assert images.get(BeamType.ION) is shown
    assert images.get(BeamType.ELECTRON) is live

    images.follow(None)
    microscope.imaging.show(_image(BeamType.ION))
    assert images.get(BeamType.ION) is shown


def test_the_session_keeps_the_microscope_s_images_without_a_window():
    from fibsem import utils
    from fibsem.workflows.session import AppSession

    microscope, _ = utils.setup_session(manufacturer="Demo")
    session = AppSession()
    session.microscope = microscope

    frame = _image(BeamType.ION)
    microscope.imaging.show(frame)
    assert session.images.get(BeamType.ION) is frame

    session.microscope = None
    microscope.imaging.show(_image(BeamType.ION))
    assert session.images.get(BeamType.ION) is frame
