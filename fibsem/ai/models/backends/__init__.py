"""The backends that run models, by name: the only code that knows a framework.

A backend loads one model and turns an image into a mask; the manager
(`fibsem.ai.models.manager`) does the rest from the model's card.
"""

from typing import Dict, Type

from fibsem.ai.models.backends.fake import FakeSegmentationModel

BACKENDS: Dict[str, Type] = {
    FakeSegmentationModel.name: FakeSegmentationModel,
}
