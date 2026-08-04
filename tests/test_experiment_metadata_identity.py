"""Image metadata carries a stable experiment key, not just a display name.

Metadata embedded in an image is a snapshot: authoritative only when the experiment
record is missing, with the record winning on conflict (FIB-445 D2). That rule needs a
key that does not change -- given only a name, a renamed experiment is indistinguishable
from a different one. Up to v0.5.2 the `id` field held the name. See FIB-446.
"""

from fibsem.config import METADATA_VERSION, UNVERSIONED_METADATA
from fibsem.structures import FibsemExperiment


def test_the_id_and_the_name_are_separate_fields():
    experiment = FibsemExperiment(id="7c1e...uuid", name="my-experiment")

    d = experiment.to_dict()
    assert d["id"] == "7c1e...uuid"
    assert d["name"] == "my-experiment"


def test_round_trip_keeps_both():
    original = FibsemExperiment(id="7c1e...uuid", name="my-experiment")

    restored = FibsemExperiment.from_dict(original.to_dict())

    assert restored.id == original.id
    assert restored.name == original.name


def test_a_pre_rename_file_is_recognisable_by_its_missing_name():
    """The era signal is field presence, not a version lookup: no `name` means `id`
    holds one. Not backfilled, so a reader is never handed a name claiming to be an ID."""
    old_file = {"id": "my-experiment", "method": "null"}

    restored = FibsemExperiment.from_dict(old_file)

    assert restored.name is None
    assert restored.id == "my-experiment"


def test_an_unversioned_file_reads_as_older_than_v1_not_as_current():
    """settings.get("version", METADATA_VERSION) claimed an unversioned file was
    current. Absent means written before versioning existed."""
    assert UNVERSIONED_METADATA != METADATA_VERSION
    assert UNVERSIONED_METADATA == "v0"
