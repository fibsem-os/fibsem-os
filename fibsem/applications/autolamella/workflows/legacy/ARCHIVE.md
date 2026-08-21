# Archived: the legacy AutoLamella workflows

This branch exists so these files stay findable. They were deleted from `main`
on 2026-08-21; this branch is `main` as of `7970224c`, immediately before
that deletion.

Git would have kept them in history regardless — the point of the branch is that
you do not have to know when they were removed in order to find them.

## What is here

| file | lines |
| ---- | ----- |
| `autoliftout.py` | 1,576 |
| `serial.py` | 1,075 |
| `experimental.py` | 141 |

These implement the liftout and serial-liftout workflows. They were moved into
`legacy/` on 2026-04-16 ("move deprecated workflow files to legacy") and nothing
has been done to them deliberately since — every later commit touching them was
collateral from a repo-wide sweep.

## They do not run

Do not assume this is working code that merely lost its callers. As of the
archive point, none of the three modules can even be imported:

    autoliftout   ImportError: cannot import name 'AutoLamellaStage'
    serial        ImportError: cannot import name 'AutoLamellaProtocol'
    experimental  ImportError: cannot import name '_draw_milling_stages_on_image'

Each references a name that no longer exists in the codebase. Reviving them means
porting to the current task-based workflow API, not fixing three imports.

## Why they were removed rather than repaired

Nothing imported them, they could not run, and they carried 42 lint findings —
6.7% of the repository's parked total — in code that could not execute. Repairing
imports in unrunnable code would have made it look maintained.

The *capability* may still matter: there are trained models
(`autoliftout-serial-01-34.pt`) and paper datasets referencing these workflows.
This branch is the starting point if liftout is ever brought back.

## Restoring

    git checkout archive/legacy-workflows -- fibsem/applications/autolamella/workflows/legacy/
