"""The stage device says whether it carries its own one-grid shuttle.

`_create_sample_stage` used to ask `stage_is_compustage`; it asks the stage device now,
so each driver answers for its own stage type.
"""

from fibsem.devices.stage import Stage
from fibsem.drivers.autoscript.devices import AutoscriptCompustage, AutoscriptStage


def test_only_the_thermo_compustage_carries_a_builtin_shuttle():
    assert AutoscriptCompustage.has_builtin_shuttle(
        object.__new__(AutoscriptCompustage)
    )
    assert not AutoscriptStage.has_builtin_shuttle(object.__new__(AutoscriptStage))


def test_a_stage_has_no_builtin_shuttle_by_default():
    assert Stage.has_builtin_shuttle(object.__new__(Stage)) is False
