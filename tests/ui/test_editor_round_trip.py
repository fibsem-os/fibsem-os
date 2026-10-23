"""Saving from an editor writes back the values it was given.

The µm spin boxes used to convert back with ``value * 1e-6``, which turned a
field of view of ``0.0001`` into ``0.00009999999999999999``: a save changed a
value nobody touched. Dividing by ``1e6`` is exact for the values a spin box
shows.
"""

import pytest

pytest.importorskip("PyQt5")

from fibsem import utils  # noqa: E402
from fibsem.milling.tasks import FibsemMillingTaskConfig  # noqa: E402
from fibsem.structures import ReferenceImageParameters  # noqa: E402
from fibsem.ui.widgets.milling_task_config_widget import (  # noqa: E402
    MillingTaskConfigWidget,
)
from fibsem.ui.widgets.reference_image_parameters_widget import (  # noqa: E402
    ReferenceImageParametersWidget,
)

FIELDS_OF_VIEW = [1e-4, 80e-6, 150e-6, 25e-6, 0.3e-3]


@pytest.mark.parametrize("fov", FIELDS_OF_VIEW)
def test_the_milling_field_of_view_survives_a_save(qapp, fov):
    microscope, _ = utils.setup_session(manufacturer="Demo", setup_logging=False)
    widget = MillingTaskConfigWidget(microscope=microscope)
    widget.update_from_settings(FibsemMillingTaskConfig(field_of_view=fov))

    assert widget.get_settings().field_of_view == fov


@pytest.mark.parametrize("fov", FIELDS_OF_VIEW)
def test_the_reference_fields_of_view_survive_a_save(qapp, fov):
    widget = ReferenceImageParametersWidget()
    widget.update_from_settings(
        ReferenceImageParameters(field_of_view1=fov, field_of_view2=fov)
    )

    settings = widget.get_settings()
    assert (settings.field_of_view1, settings.field_of_view2) == (fov, fov)
