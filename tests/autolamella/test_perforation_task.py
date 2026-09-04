"""The perforation milling task: registration, defaults and config round trip.

Ported from the circular-suspension plugin, where it was registered through the
``fibsem.tasks`` entry point. In tree it is a built-in, so what needs pinning is the
registration itself (a task the registry does not know is silently dropped when a
protocol names it — ``load_task_config`` logs a warning and moves on) and the config
round trip, since ``orientation`` reaches disk through the base's ``parameters``
subdict rather than a field of its own.
"""

from fibsem.applications.autolamella.protocol.constants import PERFORATION_KEY
from fibsem.applications.autolamella.workflows._default_milling_config import (
    DEFAULT_MILLING_CONFIG,
)
from fibsem.applications.autolamella.workflows.tasks import get_tasks
from fibsem.applications.autolamella.workflows.tasks.perforation import (
    MillPerforationTask,
    MillPerforationTaskConfig,
)


def test_the_task_is_registered():
    assert get_tasks()[MillPerforationTaskConfig.task_type] is MillPerforationTask


def test_the_default_config_carries_the_perforation_milling():
    config = MillPerforationTaskConfig()

    assert list(config.milling) == [PERFORATION_KEY]
    (stage,) = config.milling[PERFORATION_KEY].stages
    assert stage.pattern.name == "ArrayPattern"
    assert (stage.pattern.n_columns, stage.pattern.n_rows) == (2, 5)
    assert stage.pattern.use_circle is True


def test_the_default_milling_is_a_copy_not_the_shared_default():
    """Two tasks editing one shared config would edit each other's."""
    a, b = MillPerforationTaskConfig(), MillPerforationTaskConfig()

    a.milling[PERFORATION_KEY].stages[0].pattern.n_rows = 99

    assert b.milling[PERFORATION_KEY].stages[0].pattern.n_rows == 5
    assert DEFAULT_MILLING_CONFIG[PERFORATION_KEY].stages[0].pattern.n_rows == 5


def test_config_round_trip_keeps_the_orientation():
    config = MillPerforationTaskConfig(orientation="FIB")

    restored = MillPerforationTaskConfig.from_dict(config.to_dict())

    assert restored.orientation == "FIB"
    assert list(restored.milling) == [PERFORATION_KEY]
    assert restored.milling[PERFORATION_KEY].stages[0].pattern.n_rows == 5


def test_an_explicit_milling_config_is_left_alone():
    """__post_init__ only fills an empty milling dict."""
    config = MillPerforationTaskConfig(milling={})
    config_with = MillPerforationTaskConfig(
        milling={"custom": DEFAULT_MILLING_CONFIG[PERFORATION_KEY]}
    )

    assert list(config.milling) == [PERFORATION_KEY]
    assert list(config_with.milling) == ["custom"]
