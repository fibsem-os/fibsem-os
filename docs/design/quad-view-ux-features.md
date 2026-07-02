# Quad-View Main Tab — UX Features

Follow-on to the main-tab quad-view cutover (`quad-view-microscope-display.md`,
`canvas-overlay-state-model.md`). Now that the AutoLamella main tab and the standalone
`FibsemUI` are viewer-less matplotlib canvases driven by `MicroscopeViewController`, this
adds four operator-facing conveniences to the 2×2 display:

1. **Selected view** — the canvas the user most recently clicked, highlighted with a primary
   border; the target for view-scoped actions (toolbar, hotkeys).
2. **Full screen** — expand one view to fill the quad, hide the others.
3. **Hotkeys** — F5 (toggle full screen), F6 (acquire selected view), Esc (exit full screen).
4. **Working-distance control** — Shift+scroll on the SEM/FIB canvas nudges beam working
   distance, mirroring the FM objective's Shift+scroll.

These are independent of the overlay/reducer model — they operate on the *view* layer
(`QuadViewWidget`) and the *seam* (`MicroscopeViewController`), not on `SceneModel`.

## Design seam

The clean split, consistent with the existing architecture:

- **`QuadViewWidget`** (`quad_view.py`) owns the *mechanics* — panel frames + borders,
  show/hide for full screen, the mouse-press event filter. It is the view.
- **`MicroscopeViewController`** exposes a small *API* the rest of the app calls
  (`selected_view`, `view_selected` signal, `toggle_fullscreen()`/`set_fullscreen()`) and
  forwards to the widget. It is the seam.
- **`LamellaEditorView`** shares the controller but has a *stacked* (one-page) layout, not a
  2×2 grid. Selected-view and full-screen are `QuadViewWidget`-only; the controller methods
  **no-op gracefully** when the view is not a `QuadViewWidget` (guard with `hasattr` / an
  optional interface method), so the editor is unaffected.

## Scope

- **In:** `QuadViewWidget`, `MicroscopeViewController`, `FibsemImageCanvas` (a small toolbar
  visibility + selection hook), and `AutoLamellaMainUI` (hotkey install + F6 routing).
- **Out (this pass):**
  - Interaction gating by selection — all four canvases stay fully interactive
    (click / double-click-to-move / scroll work anywhere); selection is *visual + toolbar +
    hotkey target only*.
  - View-selection hotkeys (1–4 to pick a cell, arrows to move selection) — deferred.
  - A dedicated WD control block / WD info-bar readout — WD reuses the existing
    `FibsemBeamSettingsWidget` spinbox; only the interactive Shift+scroll is new.
  - Full-screen / selection for the standalone `FibsemUI` — its main window can adopt the
    same controller API later; not wired here.

---

## 1. Selected view

**Decision:** visual highlight + "only the selected view's toolbar is shown" + it is the
target for F5/F6. All canvases remain interactive.

### State & API

- `QuadViewWidget`: `selected: Optional[Union[BeamType, str]]` (`BeamType.ELECTRON`/`ION` or
  `"fm"`), `set_selected(key)`, and `view_selected = pyqtSignal(object)`.
- `MicroscopeViewController`: `selected_view` property + a forwarded `view_selected` signal.
- **Default selection:** select SEM (`BeamType.ELECTRON`) on first image / connect, so exactly
  one view is always selected (and exactly one toolbar is always visible).
- The placeholder cell is inert — not selectable. The FM cell **is** selectable and becomes the
  F6 target.

### Detection (the subtle part)

The existing `canvas_clicked` signal is too narrow for "most recently clicked": it fires only
on a left-click, on empty area, with no active overlay, drag < 3px
(`image_canvas.py` `_on_release`). Selection must register on *any* press — including
right-click, and including while an overlay owns input.

Use a Qt-level **`eventFilter`** installed by `QuadViewWidget` on each canvas (and the FM inner
canvas, `fm_widget.canvas`) catching `QEvent.MouseButtonPress` → map `sender()` → key →
`set_selected`. This sits *below* the matplotlib click gating, so it catches everything without
touching `FibsemImageCanvas`'s event handlers.

### Visual

Restructure `_titled()` (`quad_view.py:56`) to return a `QFrame` we keep references to
(`self._panels: Dict[key, QFrame]`). Toggle a border stylesheet on selection change — keep a
2px border present at all times (`transparent` when unselected, the primary accent from
`stylesheets` when selected) so there is **no layout shift**.

### "Only show selected view toolbar"

Add `FibsemImageCanvas.set_toolbar_visible(bool)` — show/hide the `_overlay_buttons` list (and
the FM layers button, which is added via `add_toolbar_button`). On selection change, hide every
canvas's toolbar except the selected one's.

**Edge case — non-selected canvas mid overlay-edit.** Because canvases stay interactive, a
canvas can own an active overlay (e.g. FIB alignment editing) while a *different* canvas is
selected. Hiding the non-selected toolbar would hide that canvas's contextual mode toggle
(`btn_mode`), stranding the edit. Resolution: `set_toolbar_visible(False)` keeps `btn_mode`
visible **when it is active** (i.e. hide only the generic buttons; the contextual mode toggle
follows overlay-mode state, not selection). This avoids surprising the user and never moves
selection implicitly.

---

## 2. Full screen

**Decision:** visibility-toggle of the existing splitter panels (no reparenting, no
`QStackedWidget`) — fully reversible and preserves each canvas's zoom / overlays / contrast.

### Layout

The quad is nested splitters:

```
root (H) ── left (V):  SEM  |  FM
         └─ right (V):  FIB  |  placeholder
```

Full-screening a cell = **hide its vertical-splitter sibling + hide the other vertical
splitter**:

| Full-screen target | Hide |
|---|---|
| SEM | FM panel + right splitter |
| FM  | SEM panel + right splitter |
| FIB | placeholder panel + left splitter |
| placeholder | (not applicable — inert) |

The remaining vertical splitter then shows only the target cell, and `root` shows only that
splitter.

### API

- `QuadViewWidget`: store references to the two vertical splitters + the panels + a saved-sizes
  dict; `set_fullscreen(key | None)` (None → restore all + saved sizes) and
  `toggle_fullscreen()` (targets the current selection; no-op if nothing selected).
- **Invariant:** `set_fullscreen(key)` also **selects** `key` — so the one visible cell always
  shows its toolbar (selection drives toolbar visibility). Without this, full-screening a
  non-selected cell would leave it with no toolbar.
- `MicroscopeViewController` forwards both; no-ops on `LamellaEditorView`.
- Optional (nice-to-have): a per-canvas `mdi:fullscreen` toolbar button that full-screens that
  cell, complementing F5 for discoverability.

---

## 3. Hotkeys

**Decision:** F5 / F6 / Esc only, scoped to the microscope tab.

There is no existing shortcut infrastructure in this UI, so this is greenfield. Add
`_install_shortcuts()` on `AutoLamellaMainUI` — it is the coordinator that can reach both the
controller (F5/Esc) and the acquisition widgets (F6).

- Use `QShortcut(QKeySequence(...), container, context=Qt.WidgetWithChildrenShortcut)` on the
  microscope-tab container, so the keys fire only when focus is within that tab (not the
  minimap / editor / workflow tabs).
- Keep a single **key → callback dict** as the source of truth (easy to extend later with
  selection keys).

| Key | Action |
|---|---|
| F5  | `view_controller.toggle_fullscreen()` |
| Esc | exit full screen (`set_fullscreen(None)`) |
| F6  | acquire selected view |

**F6 routing:** `beam = view_controller.selected_view`; ELECTRON →
`autolamella_ui.image_widget.acquire_sem_image()`, ION → `acquire_fib_image()`
(`FibsemImageSettingsWidget.py:444`); FM → `fm_control_widget.acquire_image()`. Guard when the
widget is `None` (disconnected) or already acquiring (`is_acquiring`).

### View menu

The same actions are surfaced in the main window's existing **View** menu
(`AutoLamellaSingleWindowUI._create_menu_bar`) for discoverability:

- **Full Screen** (checkable, shows the `F5` hint), **Exit Full Screen** (`Esc`, enabled only
  when full-screened), a **Full Screen View ▸** submenu (SEM / FIB / Fluorescence → full-screen
  a specific cell), and **Acquire Selected View** (`F6`).
- The `"\t<key>"` suffix in the action text is a **display hint only** — the real bindings are
  the tab-scoped `QShortcut`s, so there's no double-binding and no ambiguous-shortcut warning.
- `view_menu.aboutToShow` → `_sync_view_menu` refreshes the checkable / enabled state from
  `view_controller.fullscreen` each time the menu opens (no fullscreen-changed signal needed).

---

## 4. FIBSEM working-distance control

**Decision:** Shift+scroll on the SEM/FIB canvas nudges beam working distance and syncs the
existing beam-settings spinbox. No new control block.

### What already exists

- `microscope.get_working_distance(beam)` / `set_working_distance(wd, beam)` for all backends
  (`microscope.py:987`).
- `FibsemBeamSettingsWidget` already has a per-beam `working_distance_spinbox` wired to
  `set_working_distance` (`beam_settings_widget.py:147`).

This is **beam / column working distance (focus)** — the direct analogue of objective focus —
**not** stage Z.

### What's new

Mirror `ObjectiveControlWidget._on_canvas_scroll` + its debounced move
(`objective_control_widget.py:452`):

- Wire `sem_canvas.canvas_scrolled` / `fib_canvas.canvas_scrolled` → a per-beam WD handler.
  Shift+scroll → step WD by a configured step; plain scroll stays zoom (the canvas already
  suppresses zoom when modifiers are present, `image_canvas.py` `_on_scroll`).
- **Debounce** the hardware move (`qdebounced`, ~150 ms) so rapid notches coalesce into one
  `set_working_distance`, exactly like the objective wheel.
- Update the beam-settings `working_distance_spinbox` immediately for visual feedback (block
  signals), then move on settle.

### Safety

WD is beam focus (a lens parameter), **not** a physical objective — so unlike the FM objective
it carries no collision risk and needs neither a large-change confirmation nor an acquisition
lockout. The only guard is clamping to the spinbox WD range.

### Implemented

- Handler lives on `FibsemBeamSettingsWidget` (per beam): `_on_canvas_scroll` +
  `_execute_wd_wheel_move_impl` (`qdebounced`, 150 ms) + `_set_working_distance_spinbox`.
  One scroll notch = `WD_WHEEL_STEP_MM` = **1 µm** (0.001 mm); the WD spinbox was bumped to
  3 decimals so the step is representable. The spinbox updates immediately (blocked signals)
  and the debounced move coalesces the burst.
- **On-canvas feedback:** a transient top-centre flash (`FibsemImageCanvas.flash_message`,
  ~1.2 s auto-clear, independent of the hint / info bar) shows `WD x.xxx mm` on the canvas
  being scrolled and fades after scrolling stops. The handler targets the emitting canvas via
  `sender()`, and the flash is re-created in `set_array` so it stays visible across live
  acquisition frames (which recomposite the canvas).
- Wired in `FibsemImageSettingsWidget.setup_connections` — `controller.sem_canvas` /
  `fib_canvas` `canvas_scrolled` → the SEM / FIB beam widget's `_on_canvas_scroll`; connections
  are stored and torn down in `_teardown_connections` (the canvases outlive the widget), which
  also cancels any pending debounce.
- **No acquisition lockout and no large-change confirmation** — WD is beam focus, so it must
  stay adjustable while scanning (that's how you focus) and a big move isn't a collision risk.
  Both intentionally differ from the FM objective. The only guard is clamping to the spinbox range.
- Headless smoke: `test_working_distance_scroll` (8).

---

## Staging

Small, focused PRs (project pattern), each with a headless smoke under
`fibsem/ui/widgets/tests/`:

1. **Selected-view foundation** — `_titled`→`QFrame`, event-filter selection, borders,
   `set_toolbar_visible`, controller `selected_view` + `view_selected`, default-select SEM.
   *Smoke:* selection transitions + toolbar visibility (incl. the mid-edit `btn_mode`
   exception).
2. **Full screen** — splitter show/hide + saved-sizes restore; controller forwarding + editor
   no-op. *Smoke:* per-target hide/show invariants + round-trip restore.
3. **Hotkeys** (depends on 1+2) — `_install_shortcuts`, F5/Esc/F6 routing + guards.
   *Smoke:* F6 dispatches to the right acquire call per selected beam; guards when
   disconnected / acquiring.
4. **WD Shift+scroll** (independent) — canvas-scroll handler + debounce + spinbox sync +
   safety. *Smoke:* step math, debounce coalescing, large-change guard, acquisition lockout.

## Open items / future

- View-selection hotkeys (1–4, arrows) and Esc-to-deselect.
- Bringing selected-view + full screen to the standalone `FibsemUI`.
- A dedicated WD control block + WD readout in each canvas info bar (next to STAGE / MILLING
  ANGLE), if the Shift+scroll-only surface proves too thin.
- Per-canvas `mdi:fullscreen` toolbar button.
