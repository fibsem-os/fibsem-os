# AGENTS.md

**Read [CONTRIBUTING.md](CONTRIBUTING.md) first.** Everything in it applies to you: the
five-file target for pull requests, the `Release-Note:` commit trailer and where it must sit,
the Python 3.8 floor in signatures, format-on-touch, `QT_QPA_PLATFORM=offscreen`, the
palette tokens, and the rule that nothing reaches the network unless a user asked it to.

This file covers only what is specific to working here as an agent.

## Do not run the full test suite by default

Run the files your change affects. The full suite takes several minutes, and running it
after every edit spends most of a session waiting. Run it once before pushing, or when
asked.

## Green tests are weaker evidence here than usual

Two reasons, both of which have produced confidently wrong reports:

- **napari cannot be constructed under `QT_QPA_PLATFORM=offscreen`** — its GL canvas
  aborts the process with no traceback. Any test of a napari path is therefore against a
  stub, and proves nothing about the real widget.
- **CI installs `.[test]`, not `.[ui]`.** The Qt tests `importorskip` there. A UI test you
  watched pass locally may not have run in CI at all.

For anything about wiring, say plainly that you have not run the application, or run it.

## Say what you did not verify

Prefer "I checked X by reading the code, not by running it" over silence. A claim that
turns out to rest on inspection when the reader assumed execution costs more than the
hedge does.

## Attribution

Do not add AI or assistant attribution to commit messages or pull request bodies.
