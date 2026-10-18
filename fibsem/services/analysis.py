"""Analysis services: capabilities that read saved data and use no device.

Correlation and inference work on saved images and move nothing, so they need
no microscope. The App Server serves these under ``/services/{name}``
whether or not the microscope is connected or its routes are on, so a PC with
no microscope (the standalone correlation UI, a script, a GPU box) serves them
too. The app's widgets and scripts call the libraries directly
(`fibsem.correlation`, `fibsem.ai.models`); a service is the remote API over the same library, for
the agent, a browser and remote hosts, with no logic of its own.
"""

from typing import Dict

from fibsem.services.core import Service


def analysis_services() -> Dict[str, Service]:
    """A new copy of each analysis service, by name, with no microscope."""
    from fibsem.services.correlation import Correlation
    from fibsem.services.inference import Inference

    services = [Correlation(), Inference()]
    for service in services:
        service.connect()
    return {service.name: service for service in services}
