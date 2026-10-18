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
        widget._on_acquire(frame)
        assert images.get(BeamType.ION) is frame
    finally:
        widget.deleteLater()
