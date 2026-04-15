---
name: review-changes
description: Review current git changes for quality, correctness, and issues
argument-hint: [optional: path/to/file.py]
disable-model-invocation: true
---

Review the current git changes and provide structured feedback.

## Step 1 — Get the diff

If `$ARGUMENTS` is provided, run:
```
git diff HEAD -- $ARGUMENTS
git diff -- $ARGUMENTS
```

If no arguments, run:
```
git diff HEAD
git diff
```

Combine both outputs (HEAD covers staged+committed-but-unpushed; the second covers unstaged). If both are empty, report "No changes found."

## Step 2 — Analyze

Read any files referenced in the diff that need more context to understand the change properly.

Evaluate the diff across these dimensions:

**Correctness**
- Logic errors, off-by-one, wrong conditions
- Broken edge cases (None, empty list, uninitialized state)
- Signal/slot wiring that could fire at the wrong time or double-fire
- Mutation of data that shouldn't be mutated

**Code quality**
- Duplicate logic that could be shared
- Dead code (imports, variables, branches that are now unreachable)
- Unnecessary complexity

**Project conventions** (fibsem-specific)
- Qt widgets: `_setup_ui → _connect_signals` init order
- `blockSignals` used in `set_*` / `update_from_*` methods
- `TitledPanel` + `add_header_widget` used (not raw `QGroupBox`)
- `ValueSpinBox` / `ValueComboBox` used (not raw `QDoubleSpinBox`)
- Stylesheets from `fibsem.ui.stylesheets` (not hardcoded)
- List widget drag-drop uses `dropEvent` override pattern

**Potential issues**
- Race conditions or ordering dependencies
- Missing guard for `None` / empty state
- UI state that can get out of sync

## Step 3 — Report

Structure the output as:

### Overview
One short paragraph describing what the change does.

### Issues
Real bugs or correctness problems. Number each one. For each:
- Where it is (file:line)
- What's wrong
- Suggested fix (code snippet if helpful)

If none: "No issues found."

### Suggestions
Non-blocking improvements (style, simplification, conventions). Keep these brief.

If none: "No suggestions."

### Verdict
One line: **Looks good** / **Minor issues** / **Needs fixes** — with a one-sentence summary.