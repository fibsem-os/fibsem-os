---
name: review-widget
description: Audit a fibsem Qt widget file against established patterns and suggest fixes
argument-hint: [path/to/widget.py]
disable-model-invocation: true
---

Audit the widget at `$ARGUMENTS` against the fibsem widget conventions.

## Step 1 — Read the file
Read the full widget file at `$ARGUMENTS`.

## Step 2 — Check against each pattern

**Init order**
- [ ] `__init__` calls `_setup_ui()` → `_connect_signals()` → `_update_visibility()` in that order

**TitledPanel usage**
- [ ] Uses `TitledPanel` (not `QGroupBox`) for panels that have header widgets
- [ ] Header widgets added via `add_header_widget()` (inserts before collapse btn)
- [ ] Collapse button is always last in header

**Advanced fields**
- [ ] Advanced-only widgets collected in `self._adv_widgets` list
- [ ] `_adv_widgets` built at the END of `_setup_ui` (after all widgets exist)
- [ ] `_update_visibility()` iterates `_adv_widgets` — no manual show/hide elsewhere

**Signal blocking**
- [ ] `blockSignals(True/False)` used in all `set_*` / `update_from_*` methods
- [ ] After `blockSignals(False)`, header icon buttons explicitly synced if they depend on checkbox state

**Controls**
- [ ] `ValueSpinBox` used instead of raw `QDoubleSpinBox` (gets `WheelBlocker` + `setKeyboardTracking(False)`)
- [ ] `ValueComboBox` used instead of `_create_combobox_control` for new controls
- [ ] No `WheelBlocker` added manually if using `ValueSpinBox`/`ValueComboBox`

**List widgets (if applicable)**
- [ ] Drag-drop uses `dropEvent` override + `item.data(Qt.ItemDataRole.UserRole)` (not `model().rowsMoved`)
- [ ] Checked state tracked in `_checked: Dict[int, bool]`
- [ ] `_on_reordered` rebuilds all `setItemWidget` calls from UserRole data

**Stylesheets**
- [ ] Imports from `fibsem.ui.stylesheets` (not hardcoded color strings in buttons)
- [ ] `LIST_WIDGET_STYLESHEET` applied if widget contains a `QListWidget`

## Step 3 — Report

For each failed check, explain:
1. What the issue is and where in the file
2. What it should look like instead (show a corrected snippet)

For checks that pass, just list them as ✓ without elaboration.

At the end, ask the user if they'd like you to apply any or all of the fixes.
