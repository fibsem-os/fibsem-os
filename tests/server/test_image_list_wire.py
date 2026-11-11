"""An image list (reference images, reference image sets) crosses the wire as
images when a program asks for the data, its shape and Nones kept."""

import numpy as np

from fibsem.server.images import image_list_bytes, image_list_from_bytes
from fibsem.structures.image import FibsemImage


def test_an_image_list_round_trips_with_its_shape():
    a = FibsemImage.generate_blank_image(resolution=(32, 24), random=True)
    b = FibsemImage.generate_blank_image(resolution=(16, 8), random=True)
    images = image_list_from_bytes(image_list_bytes([[a, None], [None, b]]))
    assert images[0][1] is None and images[1][0] is None
    np.testing.assert_array_equal(images[0][0].data, a.data)
    np.testing.assert_array_equal(images[1][1].data, b.data)
