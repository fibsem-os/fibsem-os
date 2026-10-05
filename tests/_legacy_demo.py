"""A session on ``LegacyDemoMicroscope``, the Demo before devices.

No configuration selects it, so tests that compare against it, or that bind the
demo drivers over its simulated parts, build it here: exactly as
``utils.setup_session`` builds the Demo, with the legacy class in its place.
"""

from unittest import mock


def setup_legacy_session(**kwargs):
    from fibsem import utils
    from fibsem.microscopes.simulator import LegacyDemoMicroscope

    kwargs.setdefault("manufacturer", "Demo")
    with mock.patch(
        "fibsem.microscopes.device_demo.DemoMicroscope", LegacyDemoMicroscope
    ):
        return utils.setup_session(**kwargs)
