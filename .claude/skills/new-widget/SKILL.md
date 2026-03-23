---
name: new-widget
description: Scaffold a new fibsem Qt widget following established TitledPanel/signal patterns
argument-hint: [WidgetName] [purpose]
disable-model-invocation: true
---

You are scaffolding a new Qt widget named $0 for purpose: $1.

## Conventions to follow

**Init order**: always `_setup_ui()` → `_connect_signals()` → `_update_visibility()`

**Panels**: use `TitledPanel` (from `fibsem.ui.widgets.custom_widgets`) instead of `QGroupBox` for any panel that needs header widgets (tune/enable buttons, collapse). Dark header `#1e2124`, collapse button always last via `add_header_widget()`.

**Advanced fields**: collect advanced-only widgets in `self._adv_widgets: list`. `_update_visibility()` iterates this list, showing/hiding based on an advanced toggle. Build `_adv_widgets` at the end of `_setup_ui` after all widgets exist.

**Signals**: use `blockSignals(True/False)` when populating controls from data in `set_*` / `update_from_*` methods. After `blockSignals(False)`, explicitly sync any header icon buttons that weren't reached by signals (e.g. `_on_alignment_checkbox_changed()`).

**Spinboxes**: use `ValueSpinBox` (suffix, min, max, step, decimals all optional). Sets `setKeyboardTracking(False)` and `WheelBlocker` automatically.

**Combos**: use `ValueComboBox(items, value, unit, format_fn)`. Sets `WheelBlocker` automatically.

**Icons**: use `QIconifyIcon` from superqt. Common: `mdi:tune` (settings), `mdi:power`, `mdi:eye`/`mdi:eye-off`, `mdi:refresh`, `mdi:chevron-up`/`mdi:chevron-down`.

**Stylesheets**: import from `fibsem.ui.stylesheets` — `PRIMARY_BUTTON_STYLESHEET`, `SECONDARY_BUTTON_STYLESHEET`, `LIST_WIDGET_STYLESHEET`.

**File location**: `fibsem/ui/widgets/` for general widgets, `fibsem/ui/` for top-level composite widgets.

## Step 1 — Explore
Read 1–2 similar existing widgets to understand the exact import style and patterns used. Good references:
- `fibsem/ui/widgets/beam_settings_widget.py` — `_adv_widgets`, `TitledPanel`, `ValueSpinBox`
- `fibsem/ui/widgets/milling_stage_list_widget.py` — list widget with drag-drop
- `fibsem/ui/widgets/custom_widgets.py` — `TitledPanel`, `ValueSpinBox`, `ValueComboBox` definitions

## Step 2 — Scaffold
Create `fibsem/ui/widgets/$0.py` (or the appropriate path) with:
- Class docstring explaining purpose
- `__init__(self, parent=None)` calling `super().__init__(parent)` then the three init methods
- `_setup_ui()` — all widget creation and layout
- `_connect_signals()` — all signal→slot connections
- `_update_visibility()` — show/hide advanced widgets
- `get_settings()` / `set_settings()` (or equivalent) methods if the widget wraps a data structure

## Step 3 — Verify
Print the class structure and ask the user if any adjustments are needed before finishing.
