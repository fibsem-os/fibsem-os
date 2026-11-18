"""What the simulated cases share: the grid cases made from one overview
(``grid``), and live runs, where a case runs on the App Server with the
simulator as its microscope, so the agent works the instrument and is
scored on what it did there, from the sample the simulator drew.

A live case (``sim_run:`` in its case.yaml) runs through the same runner,
report and review page as any other: the runner connects the case's
microscope, sets the run up, and starts the agent, fibsem-agent or a
scripted one (``fibsem.ai.evals.sim.scripted``) that costs nothing.
"""
