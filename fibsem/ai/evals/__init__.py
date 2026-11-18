"""Offline agent evals: frozen decision points from past runs, scored in code.

A case is a folder holding ``case.yaml`` (task, brief, inputs, answer key),
the inputs under ``experiment/`` and what the run did under ``reference/``.
The agent answers in ``answer.json``; the scorers here compare it with the
key. No model is involved in scoring.

The package: ``core`` (the machinery every task shares), ``sim`` (what the
simulated cases share) and ``tasks``, one folder per task with everything
for it: ``task*.py`` (its scoring, figure and key check), ``make*.py`` (its
cases from the simulator) and ``scripted.py`` (its baseline).

A live case (``sim_run:``, ``fibsem.ai.evals.sim``) is run on the
simulator instead: the agent works the microscope, and is scored on what the
run left in the experiment against what the simulator drew. ``run
--scripted`` runs it with a scripted agent, which spends nothing.
"""
