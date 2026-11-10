"""Offline agent evals: frozen decision points from past runs, scored in code.

A case is a folder holding ``case.yaml`` (task, brief, inputs, answer key),
the inputs under ``experiment/`` and what the run did under ``reference/``.
The agent answers in ``answer.json``; the scorers here compare it with the
key. No model is involved in scoring.
"""
