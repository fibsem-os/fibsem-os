"""The backends that run models, by name: the only code that knows a framework.

A backend loads one model and turns an image into a mask; the manager
(`fibsem.ai.models.manager`) does the rest from the model's card.
"""

from typing import Dict, Type

from fibsem.ai.models.backends.fake import FakeSegmentationModel
from fibsem.ai.models.backends.nnunet import NnunetModel
from fibsem.ai.models.backends.smp import SmpModel

# Importing a backend imports no framework: each one imports its own when a
# model loads, and asks whether its packages are there without importing them.
BACKENDS: Dict[str, Type] = {
    backend.name: backend for backend in (FakeSegmentationModel, NnunetModel, SmpModel)
}
