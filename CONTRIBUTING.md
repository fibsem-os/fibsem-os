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

`requires-python = ">=3.10"`, and CI builds 3.10 through 3.13. A green local run on a newer
interpreter proves nothing about the 3.10 job.

`X | Y` unions and builtin generics such as `list[str]` are safe in signatures. What still
bites is the standard library: `tomllib`, `typing.Self`, `enum.StrEnum`, `ExceptionGroup`
and `datetime.UTC` are 3.11+, and `typing.override` is 3.12+. Each passes locally and fails
only on the 3.10 job.

0.5.3 is the last release that supports Python 3.8 and 3.9. A fix for those installs is
cut from `release/v0.5.x`.

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

**CI is thinner than a development environment.** The build matrix installs `.[test]`, not
`.[ui]`, so the PyQt5 tests `importorskip` and are skipped there; only the `ui-tests` job
runs them. No job installs `[labelling]`, so nothing that needs napari runs on CI at all. A
UI test passing locally is not evidence CI ran it.

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

## Times

A recorded time is read on other machines, in other zones, years later. Every time
fibsemOS stores goes through `fibsem/util/timestamps.py` and follows these rules
(FIB-1190):

1. **In memory, an aware `datetime`.** Take one with `timestamps.now()`, never
   `datetime.now()`. A naive datetime does not say which instant it is, and comparing
   one with an aware one raises `TypeError`.
2. **On disk, ISO 8601 with the UTC offset:** `2026-09-13T21:19:40.974286-06:00`. Write
   it with `.isoformat()` on an aware datetime (or `now_iso()`); read it with
   `to_datetime`, or `to_aware` for a field that only holds instants. Parsing keeps the
   offset the value was written with, so the site's clock time survives a load and a
   save.
3. **Convert to the viewer's zone only to display,** with `format_time`. Comparing,
   subtracting and sorting aware datetimes goes by the instant, whatever their offsets.
   Displays show no offset. The exception is what keeps a run on the instrument's
   clock (FIB-1196). The replay reads with `wall_time_of`, which drops an offset rather
   than converting it. A report converts into the instrument's zone as the experiment
   recorded it, `experiment.session.zone`, and says so once in its header.
4. **Name the moment `*_at`** (`started_at`, `captured_at`), one field per moment, and
   derive durations from two of them rather than storing one.
5. **Default with `field(default_factory=now)`,** never `= now()` in a class body,
   which runs once when the module is imported (FIB-487).
6. **POSIX only at the edges.** A vendor's time or a file's mtime becomes an aware
   datetime as soon as it is read (`from_posix`). Do not add a POSIX field. The ones
   that exist keep their keys and are written as ISO 8601 with offset from now on
   (FIB-1197): task `start_timestamp`/`end_timestamp` first, the `created_at` fields
   next. `microscope_state.timestamp` stays a float.
7. **Old files keep reading as they did.** `to_datetime` reads every older form: a POSIX
   float, AutoScript's `%m/%d/%Y %H:%M:%S` string, a naive ISO string, and it reads
   them where the key is now ISO, so an older file loads and shows the same times. A
   POSIX float is an instant, so it is written back as ISO with this machine's offset
   when the file is next saved. A naive value stays naive, because its zone is
   unknown: never give it one by guessing. Backward compatibility is the goal; an
   older fibsem need not open a newer file.

Filenames and folder names (`overview-21-18-22`, `DATETIME_FILE`) are names, on the
local clock, and nothing parses them as times.

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
