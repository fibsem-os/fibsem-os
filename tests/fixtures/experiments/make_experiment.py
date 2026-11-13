"""Write a small AutoLamella experiment with the installed fibsem release, on
the Demo microscope: three lamellae, the first three tasks of the shipped
protocol run on two of them. Run with that release's Python:

    python make_experiment.py <output folder>

Written for the 0.5 releases; a later one may need its imports changed.
"""

import os
import sys

os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import fibsem
import fibsem.applications.autolamella as autolamella
from fibsem import utils
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.applications.autolamella.workflows.tasks.manager import run_tasks

RUN_ON = ["first", "second"]


def main(out: str) -> None:
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    protocol = AutoLamellaTaskProtocol.load(
        os.path.join(
            os.path.dirname(autolamella.__file__), "protocol", "task-protocol.yaml"
        )
    )
    experiment = Experiment(path=out, name="release-experiment")
    os.makedirs(experiment.path, exist_ok=True)
    experiment.task_protocol = protocol
    for name in (*RUN_ON, "untouched"):
        experiment.add_new_lamella(
            microscope_state=microscope.get_microscope_state(),
            task_config=protocol.task_config,
            name=name,
        )
    experiment.save()
    protocol.save(os.path.join(experiment.path, "protocol.yaml"))
    for task in list(protocol.task_config)[:3]:
        run_tasks(microscope, experiment, [task], required_lamella=RUN_ON)
    experiment.save()
    print(f"fibsem {fibsem.__version__}: wrote {experiment.path}")


if __name__ == "__main__":
    main(sys.argv[1])
