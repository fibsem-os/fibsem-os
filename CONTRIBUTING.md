# Contributing

Conventions that are not obvious from reading the code, and that CI will otherwise catch
late or not at all.

## Pull requests

**Aim for about five files in a pull request.** A target to design toward, not a limit
to obey: the constraint is reviewability, which is set by what a person can hold in
their head rather than by a file count. A machine-generated sweep with an AST-level
safety check and a green suite is still too large to review at twenty files — verifying
it well does not buy back the size of the thing someone has to read.

Go over five when the extra files are **cohesive** — one change rather than a sweep —
and **unavoidable**, in the sense that they break without it. Say so in the pull request
body, and say what they contain: "six test files, four of them a one-line fixture
change" is the sentence that makes eight files readable. Do not contort the design to
hit the number; a forwarding shim that exists only to keep the count down is worse than
the extra file.

For a mechanical change across many files, slice by directory or module so each pull
request is one coherent area, and open them independently off `main` rather than
stacking. More pull requests is the accepted cost.

This bites hardest under format-on-touch (below), where editing a file forces a full
reformat of it, so file count drives diff size far more than the change itself does.

**Put the reasoning in the pull request body.** Design and planning documents stay local
and untracked — write one if it helps, but do not add it to the tree. Anything a reviewer
needs (the why, the key decisions, follow-ups deliberately excluded) gets inlined.

**Swapping something in production is two pull requests.** This project drives working
instruments, and a broken intermediate state is an operator mid-session, not a failing
test. The pull request that points production at a new implementation leaves the old one
on disk and unused; a follow-up deletes it once the new one has been exercised. Otherwise
a revert has to resurrect the deleted file, which turns a one-click rollback into a merge.

Before staging such a swap, look for consumers that call **both** surfaces — those break
the moment you swap either one alone.

## Commit messages

### Release notes go in the commit

A change a user would notice carries a `Release-Note:` trailer:

```
[ui] Toasts always show, instead of never (FIB-781)

<the usual explanation of what changed and why>

Release-Note: Toasts now appear on the window that raised them, rather than always on the main window.
```

The changelog for a release is then one command:

```bash
git log v0.5.1..HEAD --format='%(trailers:key=Release-Note,valueonly)' | grep .
```

The commit is the right home for it because it cannot drift from the code it describes —
it ships in the same commit.

**The trailer must be the last paragraph of the message.** Git does not parse a trailer
with prose after it, and it does not warn: the line is simply invisible to every tool that
reads it. Squash merges take the pull request body verbatim, so in practice the
`Release-Note:` line must be the last thing in the pull request body.

Write it for a user: what they can now do, or what now behaves differently. One sentence.
A pure refactor with no user-facing effect does not need one.

**Removing a feature flag always needs one.** It produces no user-visible diff to review
and a very user-visible change in behaviour, which is exactly the combination that goes
unrecorded.

### Housekeeping

- No AI or assistant attribution in commit messages or pull request bodies.
- This repository is public. Keep user names, site names and instrument identifiers out of
  commit messages, pull request bodies and code comments.

## Python version

`requires-python = ">=3.8"`, and CI builds 3.8 through 3.13. A green local run on a modern
interpreter proves nothing about the two oldest jobs.

**Use `Optional[X]` and `Union[X, Y]` from `typing` in function signatures, not `X | Y`.**
PEP 604 unions in a signature are evaluated at runtime, so on 3.8 and 3.9 the module fails
at *collection*, not at the call. Same for `list[str]` and `dict[str, int]` as builtin
generics. A file with `from __future__ import annotations` defers evaluation and may use
the newer syntax.

It is not only syntax: standard-library APIs have version floors too — `ast.unparse` is
3.9+, `str.removeprefix` and `str.removesuffix` are 3.9+. Both have turned CI red here on
the 3.8 job alone.

## Formatting and lint

The `lint` job runs **two** things, and `ruff check` passing locally is not enough:

```bash
ruff check .
ruff format --check $(git diff --name-only --diff-filter=d $(git merge-base origin/main HEAD) HEAD -- '*.py')
```

The second is **format-on-touch**: the tree converts to `ruff format` file by file rather
than in a flag day, so a file must be formatted once anything in it is edited. `main` stays
mixed for a while, which is the accepted cost. ruff is pinned — keep the pin in the
workflow and the `dev` extra in step.

When the pre-existing formatting debt in a file is large, put it in its own `[format]`
commit at the bottom of the branch rather than mixing it with the change, so the review is
not four hundred lines of whitespace around thirty lines of substance.

## Tests

**Run the files your change affects, not the whole suite.** The full suite takes several
minutes; run it before pushing rather than after every edit.

**Always set `QT_QPA_PLATFORM=offscreen`** for anything touching `tests/ui/`:

```bash
QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_something.py -q
```

**CI is thinner than a development environment.** It installs `.[test]`, not `.[ui]`, so
the napari and PyQt5 tests `importorskip` and are skipped there. A UI test passing locally
is not evidence CI ran it.

**Prefer real objects to stand-ins.** A fake encodes the author's assumption about the
collaborator, so the test can only confirm that assumption and never contradict it. Before
writing one, check whether the real object is constructible — usually it is, and it absorbs
schema changes that a `SimpleNamespace` breaks on.

**Run the application for anything about wiring.** Widget tests assert on the state they
name, so a setup method truncated part-way leaves every named attribute intact and drops
only the unnamed remainder — which no assertion covers. Changes that looked correct under
hundreds of green tests have been visibly broken on first launch.

## User interface

New widgets use the shared napari-dark palette. **Import the role-named tokens from
`fibsem/ui/tokens.py`** — `SURFACE_COLOR`, `PANEL_COLOR`, `BORDER_COLOR`, `TEXT_COLOR`,
`ACCENT_COLOR` and the rest — rather than pasting hex values into a widget, and reach for
the prebuilt stylesheets in `fibsem/ui/stylesheets.py` for buttons and progress bars.

Render an offscreen screenshot (`widget.grab().save(...)`) to check layout before calling
a widget done.

## Network access

**Nothing reaches the network unless a user asked it to.** Any feature that does is
opt-in — the preference defaults to `False` — and the enabling check **fails closed**: if
the preference cannot be read, do not call out. Give every request a timeout and run it off
the interface thread.

This applies to fetching public data as much as to uploading anything. fibsemOS runs on
instrument PCs, often on institutional, regulated or air-gapped networks, and an
unannounced outbound connection at startup is a compliance question for the operator
sitting at the machine.

## Releases

See [RELEASE.md](RELEASE.md).
