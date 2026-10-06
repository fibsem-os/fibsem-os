"""How the FM camera is mounted is the camera device's to say (FIB-1164's FM side).

A fixed flip between the sensor and the stage is a fact about the mount. The camera
device reports it, so an FM on its own computer brings it, and the FM API applies it
to every frame before the user's own transform, as the old FM classes did.
"""

import numpy as np
import pytest

from fibsem.devices.drivers.demo import bind_demo_fm
from fibsem.devices.fm import mount_transform_from_name, mount_transform_name
from fibsem.fm.microscope import FluorescenceMicroscope
from fibsem.fm.structures import CameraImageTransform

FRAME = np.arange(6).reshape(2, 3)


def _fm(mount=None):
    config = {"mount_transform": mount_transform_name(mount)} if mount else None
    devices = bind_demo_fm(config=config)
    return FluorescenceMicroscope(devices)


def test_a_camera_is_mounted_straight_unless_its_configuration_says():
    fm = _fm()

    assert fm.devices["camera"].mount_transform.get_value() == "none"
    assert fm.mount_transform is CameraImageTransform.NONE


def test_the_mount_comes_before_the_users_transform():
    fm = _fm(CameraImageTransform.FLIP_X)
    fm.set_image_transform(CameraImageTransform.FLIP_Y)

    assert fm.devices["camera"].mount_transform.get_value() == "flip-x"
    assert fm.mount_transform is CameraImageTransform.FLIP_X
    np.testing.assert_array_equal(fm._apply_image_transform(FRAME), FRAME[::-1, ::-1])


def test_an_image_comes_out_in_the_stages_axes():
    fm = _fm(CameraImageTransform.FLIP_XY)
    raw = fm.devices["camera"].acquire()
    fm.devices["camera"]._acquire = lambda: raw

    image = fm.acquire_image()

    np.testing.assert_array_equal(image.data, raw[::-1, ::-1])


def test_the_mount_cannot_be_changed_from_a_client():
    fm = _fm()

    assert not fm.devices["camera"].mount_transform.settable


def test_a_camera_from_before_the_parameter_is_mounted_straight():
    fm = _fm(CameraImageTransform.FLIP_X)
    params = dict(fm.devices["camera"].parameters)
    params.pop("mount_transform")
    fm.devices["camera"]._bound = params

    assert fm.mount_transform is CameraImageTransform.NONE


@pytest.mark.parametrize("transform", list(CameraImageTransform))
def test_the_names_go_both_ways(transform):
    assert mount_transform_from_name(mount_transform_name(transform)) is transform


def test_an_unknown_name_says_what_it_could_be():
    with pytest.raises(ValueError, match="none, flip-x, flip-y, flip-xy"):
        mount_transform_from_name("rotate-90")


def test_the_camera_takes_the_mount_from_the_fm_entrys_keys_and_ignores_the_rest():
    devices = bind_demo_fm(
        config={"mount_transform": "flip-y", "port": 8001, "driver": "remote"},
    )

    assert devices["camera"].mount_transform.get_value() == "flip-y"


def test_the_demo_fm_takes_its_configuration_too():
    from fibsem import utils

    microscope, _ = utils.setup_session(manufacturer="Demo")
    devices = bind_demo_fm(microscope, config={"mount_transform": "flip-x"})

    assert devices["camera"].mount_transform.get_value() == "flip-x"
