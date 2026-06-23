# Grid Workflow System & Stage Components

**Status:** Draft / design
**Scope:** Stage hardware abstractions (`SampleGrid`, `GridSlot`, `SampleHolder`, `SampleGridLoader`, `Stage`) and the grid-level task workflow system, plus a proposed path to bring grid workflows to parity with the existing AutoLamella task system.

---

## 1. Overview & Motivation

AutoLamella's existing automation is built around the **lamella** as the unit of work: each lamella carries its own configuration, state and history, and a `TaskManager` drives a matrix of *(lamella × task)* work items through a workflow (rough milling, polishing, etc.). This system is mature, persisted to disk, and UI-driven.

A second axis of automation sits *above* the lamella: the **grid**. Before any lamella exists, we need to acquire grid overviews, clean grids, screen them for targets, and — on systems with an autoloader — physically exchange grids in and out of the microscope. These operations act on a whole grid, not on an individual lamella, so they don't fit the lamella task model directly.

The codebase already contains the building blocks for this:

- A set of **stage hardware abstractions** (`SampleGrid`, `GridSlot`, `SampleHolder`, `SampleGridLoader`, `Stage`) that model where grids physically live and how they are loaded. These are solid and unit-tested.
- A **grid task prototype** (`GridTask` / `GridTaskConfig`) that mirrors the shape of the lamella task system but is currently a thin, early-stage implementation with explicit TODOs.

This document describes both layers as they exist today and proposes how to grow the grid workflow layer to the same maturity as the lamella task system, so that grid-level operations become first-class, configurable, persisted, and orchestrated workflows.

---

## 2. Concepts & Terminology

- **SampleGrid** — a physical TEM grid (or specimen) that carries the regions of interest. The atomic unit of *sample* the workflow operates over. Defined purely by identity + geometry (`name`, `description`, `radius`).
- **GridSlot** — a fixed mechanical position on the holder, identified by `name`/`index`, with a known stage `position`. A slot is a *location*; it may be empty or hold exactly one `SampleGrid`.
- **SampleHolder** — the static structure mounted on the stage that physically holds grids. It is a **static transform on top of the stage**: via its slots' positions, plus the system pre-tilt and reference rotation, it defines where each grid sits in stage coordinates. It does not move grids itself — it is the fixed frame of reference.
- **SampleGridLoader** — the **robotic actuator** that exchanges grids between storage and the holder's slots. Where the holder is static geometry, the loader is the active component that changes which `SampleGrid` occupies which `GridSlot`. Present only on hardware that supports automated exchange (CompuStage); on other systems grids are loaded manually and the loader is `None`.
- **Stage** — the high-level interface that owns the holder (and optional loader), performs all motion, and answers "which slot/grid am I at?" via tolerance matching.

> **Holder vs. Loader, in one line:** the **holder** is *where grids can be*; the **loader** is *what moves grids there*.

---

## 3. Stage Components

All stage hardware abstractions live in [`fibsem/microscopes/_stage.py`](../../fibsem/microscopes/_stage.py) and are exercised by [`tests/test_sample_holder.py`](../../tests/test_sample_holder.py).

### Composition hierarchy

```
FibsemMicroscope
└── Stage                         (motion + slot awareness)
    ├── SampleHolder              (static structure on the stage)
    │   └── GridSlot × capacity   (fixed positions; each holds 0..1 grid)
    │       └── SampleGrid        (the physical grid, when loaded)
    └── SampleGridLoader?         (robotic actuator; CompuStage only)
            mutates slot.loaded_grid
```

The holder defines *geometry* (where slots are); the loader defines *exchange* (which grid is in which slot). The `Stage` ties them together and is the only object that actually moves.

### SampleGrid — [`_stage.py:31`](../../fibsem/microscopes/_stage.py#L31)

A physical TEM grid or specimen. It is intentionally minimal — identity plus geometry:

```python
@dataclass
class SampleGrid:
    name: str
    description: str = ""
    radius: float = field(default=GRID_RADIUS)  # GRID_RADIUS = 1e-3 m (1 mm)
```

`radius` is currently descriptive geometry only — it is *not* read during slot matching. `Stage.current_slot` matches against the fixed module constant `GRID_RADIUS` (the value `SampleGrid.radius` happens to default to), not the per-grid field. `to_dict`/`from_dict` provide YAML serialization.

### GridSlot — [`_stage.py:58`](../../fibsem/microscopes/_stage.py#L58)

A *fixed* mechanical slot on a holder. A slot is a location that may hold one grid or none:

```python
@dataclass
class GridSlot:
    name: str
    index: int
    position: Optional[FibsemStagePosition] = None  # None for magazine slots
    loaded_grid: Optional[SampleGrid] = None
```

`position` is optional: a holder *working* slot has a calibrated stage position, but a loader *magazine* slot is just storage and carries none.

### SampleHolder — [`_stage.py:108`](../../fibsem/microscopes/_stage.py#L108)

The static structure bolted to the stage that physically holds grids:

```python
@dataclass
class SampleHolder:
    name: str = "Sample Holder"
    description: str = ""
    capacity: int = 2                   # 1..12 slots
    slots: dict[str, GridSlot] = field(default_factory=dict)
```

Key behaviours:

- **Pre-tilt / reference rotation** are *not* stored on the holder; they are read from the system stage configuration via the bound parent microscope:
  - `pre_tilt` → `microscope.system.stage.shuttle_pre_tilt`
  - `reference_rotation` → `microscope.system.stage.rotation_reference`
  This keeps the holder a pure *static transform* layered on top of the system stage geometry.
- **Lookup helpers**: `find_slot_for_grid(grid)` / `find_slot_by_grid_name(name)` resolve which slot a grid occupies; **`occupied_slots`** ([`_stage.py:155`](../../fibsem/microscopes/_stage.py#L155)) returns the working slots that currently hold a grid (i.e. what's *in the beam* — the canonical "loaded" set, used by the reachability checks below).
- **`_ensure_slots()`** materialises exactly `capacity` slots (`Slot-01`, `Slot-02`, …), adding missing ones and trimming extras.
- **Persistence**: `to_dict`/`from_dict`, plus `save(path)` / `load(path)` to YAML. Holder configuration files: [`fibsem/config/sample-holder.yaml`](../../fibsem/config/sample-holder.yaml) (user) and `default-sample-holder.yaml` (fallback).

### SampleGridLoader — [`_stage.py:226`](../../fibsem/microscopes/_stage.py#L226)

The robotic actuator that exchanges grids in and out of slots. It models an autoloader **magazine** (its own `capacity` + `slots` of storage) and moves grids between the magazine and the holder's working slot:

```python
class SampleGridLoader:
    def load_grid(self, slot_name: str, grid: SampleGrid) -> None: ...
    def unload_grid(self, slot_name: str) -> None: ...
    def run_inventory(self) -> List[GridSlot]: ...        # scan the magazine
    def assign_grid(self, slot_name, grid) / find_grid(name) -> ...
    @property
    def loaded_magazine_slots(self) -> List[GridSlot]: ...
```

The loader's job is to mutate `slot.loaded_grid` — to change *which* grid occupies a slot. It is only instantiated for systems that model an autoloader (CompuStage / real ThermoFisher autoloaders); on other backends the loader is `None` and grids are placed into slots manually (via the UI or config). Working-slot occupancy ("what's loaded") is read from `SampleHolder.occupied_slots`, not the loader.

**Real hardware**: `AutoscriptSampleLoader` ([`fibsem/microscopes/autoscript.py`](../../fibsem/microscopes/autoscript.py)) maps a ThermoFisher autoloader (`specimen.autoloader`) onto this interface — see the companion design doc [`autoscript-sample-loader.md`](autoscript-sample-loader.md).

### Stage — [`_stage.py:312`](../../fibsem/microscopes/_stage.py#L312)

The high-level stage interface. It owns a `holder` and optional `loader`, performs all motion, exchanges grids, and tracks slot/grid context:

- **Motion**: `move_absolute`, `move_relative`, `stable_move`, `vertical_move`, `move_to_milling_angle`, `move_to_orientation`, `home`, and the slot-aware `move_to_slot(slot_name)` / `move_to_grid(grid_name)`.
- **Grid exchange**: `ensure_loaded(grid_name)` ([`_stage.py:433`](../../fibsem/microscopes/_stage.py#L433)) brings a grid into the working slot — a no-op on a static shuttle (it's already there), a loader exchange on an autoloader — raising `GridExchangeError` ([`_stage.py:26`](../../fibsem/microscopes/_stage.py#L26)) on failure; `unload()` retracts the working slot(s). These are the stage primitives the grid workflow and the Sample UI drive (the grid manager's `ensure_loaded` is a thin delegate).
- **Slot awareness**: `current_slot` / `current_grid` resolve against the *live* stage position within `GRID_RADIUS` (`is_close2`); the generalised `slot_at_position(pos)` / `grid_at_position(pos)` ([`_stage.py:350`](../../fibsem/microscopes/_stage.py#L350)) resolve a slot/grid for *any* position — used to stamp a manually-placed lamella with the grid it physically sits on.

### Backend construction — [`_stage.py:476`](../../fibsem/microscopes/_stage.py#L476)

`_create_sample_stage(microscope)` decides the holder/loader configuration per backend:

- **CompuStage** → a single-slot holder (`capacity=1`) plus an active `SampleGridLoader` (models an autoloader).
- **Other backends** → load the holder from `SAMPLE_HOLDER_CONFIGURATION_PATH` (falling back to the default), stamp each slot's `r`/`t` with the SEM orientation, and set `loader = None` (manual loading).

### UI

- [`SampleHolderWidget`](../../fibsem/ui/widgets/sample_holder_widget.py) — edit the holder, capture slot positions from the live stage, load/clear grids, auto-save to YAML; and `LoaderMagazineWidget` for the autoloader magazine.
- [`SampleWidget`](../../fibsem/ui/widgets/sample_widget.py) composes the two (magazine above holder) and owns the threaded load/unload, driving `Stage.ensure_loaded` / `Stage.unload`. It is pure hardware staging — **no `Experiment` coupling** — and is hosted in the Sample tab; hosts react to a changed working slot via its `state_changed` signal.

---

## 4. Grid Workflows

> **Status:** the prototype `grid_tasks.py` described in earlier revisions of this doc has been replaced by a mature, lifecycle-driven system that mirrors the lamella task system. It lives in the package [`fibsem/applications/autolamella/workflows/tasks/grid/`](../../fibsem/applications/autolamella/workflows/tasks/grid/) (`base.py`, `imaging.py`, `cryo.py`, `milling.py`, `targeting.py`, `registry.py`). Phases 1–4 of the [proposed design](#6-proposed-design-path-to-parity) are implemented.

### GridTaskConfig — [`grid/base.py:36`](../../fibsem/applications/autolamella/workflows/tasks/grid/base.py#L36)

A lean per-task config: `task_type`/`display_name` ClassVars + `task_name`, with **flat `to_dict`/`from_dict`** (each task-specific field at the top level) and a `parameters` property for generic UI form generation. Configs are stored in a `GridTaskProtocol` on `Experiment.grid_protocol` and read back at run time.

### GridTask — [`grid/base.py:93`](../../fibsem/applications/autolamella/workflows/tasks/grid/base.py#L93)

A `GridTask` is bound to a `GridRecord` (the workflow unit), resolves its hardware slot live via the holder, and runs a full **`pre_task → _run → post_task`** lifecycle (with `on_failure`) that writes progress into the record's `task_state` / `task_history` — matching `AutoLamellaTask`. Shared helpers include `record_result(...)`, `acquire_grid_reference_image(...)`, `wait_with_progress(...)`, the supervise primitives `validate` / `ask_user` / `update_status_ui`, and `create_lamella(stage_position, name)` ([`grid/base.py:207`](../../fibsem/applications/autolamella/workflows/tasks/grid/base.py#L207)) — the integration point for grid tasks that emit lamellae (stamps the grid's id; see §7).

### Implemented tasks

| Task | What it does |
|---|---|
| `AcquireOverviewImageGridTask` | Tiled overview of the grid (`tiled_image_acquisition_and_stitch`); saves the stitched image + a thumbnail. The image carries the acquisition `MicroscopeState` (used by the overview-as-canvas placement, §7). |
| `AcquireImageTask` | Single hi-res image at a configurable beam voltage/current. |
| `CryoCleaningGridTask` | FIB cryo-cleaning for a fixed duration (`_stop_event`-aware); records an ION reference image. |
| `CryoDepositionGridTask` | GIS deposition via the `run_gis_deposition` microscope facade (countdown progress); records an SEM reference image. |
| `CryoSputterGridTask` | Sputter coating via `run_sputter_coater` (indeterminate progress); records an SEM reference image. |

Still stub `_run` (log-only): `AutoLamellaTargetingGridTask` (blocked on `fibsem/targeting`), `AcquireFluorescenceOverviewImageTask`, `ParallelTrenchMillingGridTask`.

### Registry & orchestration

- `GRID_TASK_REGISTRY` ([`grid/registry.py:50`](../../fibsem/applications/autolamella/workflows/tasks/grid/registry.py#L50)) — string→class map; `run_grid_task` reads the **saved** config (deep-copied per run) from the protocol.
- **`GridTaskManager`** ([`grid_manager.py`](../../fibsem/applications/autolamella/workflows/tasks/grid_manager.py)) drives a grid-outer `(grid × task)` queue (reusing the shared `TaskQueue`), loading each grid once via `Stage.ensure_loaded` before its task group, with skip/stop, hooks, supervise, and UI status signals — the grid analogue of the lamella `TaskManager`.

See §6 for the phase-by-phase history and §7 for the data model.

---

## 5. Relationship to the Lamella Task System

The grid task system is intentionally a sibling of the AutoLamella task system. The lamella system is the mature template to mirror:

- **`AutoLamellaTask`** ([`base.py:95`](../../fibsem/applications/autolamella/workflows/tasks/base.py#L95)) — lifecycle `pre_task() → _run() → post_task()`, hooks, stop-event; `post_task` appends to `lamella.task_history` and writes back per-lamella config.
- **`AutoLamellaTaskConfig`** ([`structures.py:173`](../../fibsem/applications/autolamella/structures.py#L173)) — `task_type`/`display_name`, shared `milling` / `reference_imaging`, `to_dict`/`from_dict`; discovered via `BUILTIN_TASKS` + plugins (`get_tasks()`).
- **`TaskManager`** (`workflows/tasks/manager.py`) + **`TaskQueue`** (`workflows/tasks/queue.py`) — build and execute a *(lamella × task)* work matrix, applying skip/dependency logic, emitting UI signals, and saving the `Experiment` after each item.
- **`Lamella`** ([`structures.py:686`](../../fibsem/applications/autolamella/structures.py#L686)) / **`Experiment`** ([`structures.py:928`](../../fibsem/applications/autolamella/structures.py#L928)) / **`AutoLamellaTaskProtocol`** ([`structures.py:414`](../../fibsem/applications/autolamella/structures.py#L414)) — per-unit state + history, the collection, and the protocol (per-task config + ordered workflow with dependencies, supervision, scheduling).

### Mapping table

The grid system was built to mirror the lamella one; the two are now at parity (the "proposed" column of earlier revisions is implemented):

| Concern | Lamella system | Grid system |
|---|---|---|
| Unit of work | `Lamella` | `GridRecord` (workflow entity under `Experiment`; distinct from the hardware `SampleGrid`) |
| Task base | `AutoLamellaTask` | `GridTask` ([`grid/base.py:93`](../../fibsem/applications/autolamella/workflows/tasks/grid/base.py#L93)) — same `pre_task`/`post_task` lifecycle |
| Task config | `AutoLamellaTaskConfig` | `GridTaskConfig` ([`grid/base.py:36`](../../fibsem/applications/autolamella/workflows/tasks/grid/base.py#L36)) — flat `to_dict`/`from_dict` |
| Registry | `BUILTIN_TASKS` + `get_tasks()` | `GRID_TASK_REGISTRY` |
| State / history | `task_state`, `task_history` | per-grid `task_state` / `task_history` on `GridRecord` |
| Protocol | `AutoLamellaTaskProtocol` | `GridTaskProtocol` on `Experiment.grid_protocol` |
| Orchestration | `TaskManager` + `TaskQueue` | `GridTaskManager` + shared `TaskQueue` (grid-outer) |
| Persistence | `experiment.yaml` per-lamella config | grid configs + results on `Experiment` |
| UI | task config editor, workflow widget | Grids tab (cards + Protocol + Results) + grid workflow widget |

---

## 6. Proposed Design (path to parity)

The goal is to grow the grid workflow layer to match the lamella task system, reusing its patterns (and, where sensible, its code) rather than building a divergent system. Proposed in phases so each step is independently useful.

### Phase 1 — Grid record, lifecycle & state ✅ *implemented*
Introduce a minimal `GridRecord` — the first-class workflow entity under `Experiment`, distinct from the hardware `SampleGrid` (see [Decisions](#8-decisions)) — carrying identity plus per-grid `task_state` + `task_history`. Give `GridTask` a `pre_task` / `post_task` lifecycle matching `AutoLamellaTask` that writes progress into the record. This makes grid-task progress observable (UI status), resumable, and auditable — the single biggest gap today, and the foundation the later phases build on. See [Data Model](#7-data-model) for the record shape and its relationship to lamella.

### Phase 2 — Config persistence ✅ *implemented*
Add `to_dict` / `from_dict` to `GridTaskConfig` (flat serialization; nested dataclasses self-serialize) and store grid task configs in a **grid protocol** (`GridTaskProtocol`) on `Experiment`, mirroring `AutoLamellaTaskProtocol`. `run_grid_task` now reads saved config from `experiment.grid_protocol` (keyed by task_name), falling back to defaults only when none is saved — removing the standing TODO. Typed reconstruction via `load_grid_task_config` (task_type → registry → `config_cls.from_dict`). Serialized within `experiment.yaml` (back-compat: `.get("grid_protocol", {})`).

### Phase 3 — Orchestration ✅ *implemented*
Route grid tasks through orchestration that reuses the shared `TaskQueue` so grid workflows inherit skip conditions, stop handling and (eventually) status signals. The work matrix is *(grid × task)* in **grid-outer** order.
- **3a — `GridTaskManager`** ([`grid_manager.py`](../../fibsem/applications/autolamella/workflows/tasks/grid_manager.py)) drives a grid-outer queue (new additive `unit_outer` flag on `TaskQueue.build_from_matrix` — no rename of the mature lamella path; the `WorkItem.lamella_name → unit_name` rename is folded into the later base extraction). Skip on not-required / grid-failure; stop event threaded into tasks; `ensure_loaded(record)` brings a grid into the working slot (no-op on a static shuttle, loader exchange otherwise), raising `GridExchangeError` to halt on exchange failure.
- **3b — Magazine model.** `SampleGridLoader` now owns its own `capacity` + `slots` (the magazine, human-loaded) distinct from the holder working slot, with `run_inventory()` / `assign_grid()` / `find_grid()`. `ensure_loaded` sources the real `SampleGrid` (with geometry) from the magazine; `sync_grids_from_holder` also enumerates the magazine inventory on autoloader systems.

### Phase 4 — Results & UI ✅ *implemented*
`GridRecord` carries screening `results`, and the grid-workflow UI is built. Hardware staging (holder + magazine) lives in a **Sample tab** (`SampleWidget`, §3); the experiment-side grid work lives in a **"Grids" tab** (`GridTabWidget`) with **cards + Protocol + Results** sub-tabs, and grid runs in a **Grids sub-tab of the Workflow tab** (`GridWorkflowWidget`). Per-task config editors, run-order + supervise, hooks, the shared progress timeline, and the per-grid Results view (overview hero + task history + a per-grid lamella table) are all in place. See [Phase 4 UI design](#phase-4-ui-design) for the original breakdown.

### Phase 5 — Screening integration *(in progress / blocked)*
Wire the `fibsem/targeting` grid-screening pipeline (overview acquisition → segmentation → target scoring/selection) in as grid tasks that emit `Lamella` targets — closing the loop from grid-level screening to lamella-level milling. The **manual** half is shipped: the overview-as-canvas placement (§7, §9 Q2) lets a user place lamella targets on a grid's overview by hand. The **automated** half (`AutoLamellaTargetingGridTask`) is still a stub, blocked on `fibsem/targeting` landing on this branch; `GridTask.create_lamella` is the ready integration point.

---

### Phase 4 UI design

Hosting: one **"Grids" tab** with sub-sections **Magazine + Holder | Protocol | Run | Results**. `SampleHolderWidget` exists but is not currently hosted anywhere — wiring it into the app is part of this work. Results are **minimal** for Phase 4 (status + overview image + task-history timeline); rich screening results (segmentation, target overlays) arrive with Phase 5.

| # | Widget | Build | Reuses / mirrors |
|---|---|---|---|
| 1 | `GridProtocolEditor` | new (thin container) | `AutoLamellaProtocolTaskConfigEditor`; generic `AutoLamellaTaskParametersConfigWidget` for `.parameters`; `overview_acquisition_settings_widget` for the nested `settings` field. **No milling viewer.** Persists to `experiment.grid_protocol`. |
| 2 | `GridListWidget` | new | `LamellaListWidget` — lists `GridRecord`s with `task_state` status |
| 3 | `GridWorkflowWidget` | new (container) | `LamellaWorkflowWidget` + reuse `WorkflowConfigWidget` (task-agnostic); Run/Stop → `GridTaskManager.run` on a thread |
| 4 | `LoaderMagazineWidget` ✅ built | new | `SampleHolderWidget` row/edit patterns (see below) |
| 5 | `GridResultsWidget` | new (minimal) | `autolamella_overview_image_widget` + `workflow_timeline_widget` |
| 6 | `GridRecord.results: dict` | backend addition | feeds #5 (overview image path, etc.) |

**Magazine + Holder sub-section** (detailed first). Hosts two widgets side by side, mirroring `SampleHolderWidget`'s structure (form + `QListWidget` of row widgets + edit panel + auto-save):

- **`SampleHolderWidget`** (existing, finally hosted) — the holder *working* slots. Used directly on a static shuttle (grids live in slots); on an autoloader it shows the single working slot.
- **`LoaderMagazineWidget`** (new) — the loader *magazine* (storage inventory). Bound to `microscope._stage.loader`; hidden/disabled when `loader is None` (static shuttle).
  - **Top:** loader name + capacity (read-only), an **"Unload"** button (retract the working-slot grid) and a **"Run Inventory"** button → `loader.run_inventory()` (hardware presence scan).
  - **Rows:** one `_MagazineSlotRowWidget` per magazine slot. Columns: **slot number** | inline **Name** | inline **Description** | **status dot** (clickable) | **Load → beam**. No stage position (magazine slots are storage), no bottom edit panel, no checkbox, and no clear button.
  - **Clickable status dot.** The dot is both indicator and control. Clicking it toggles availability: empty → available (and **auto-names** the grid `Grid-NN` from the slot number, kept unique) → click again to clear. Typing a name also marks the slot available. Names are editable; the auto-name just removes the need to type one before loading.
  - **Status dot** colours: **gray** = empty, **white** = available, **green** = loaded in the working slot (beam). "In beam" is an object-identity check against the holder working slots (robust across renames — same grid object). Clicking a green dot is a no-op (use Load/Unload). Load is enabled in the white state once the grid has a name (auto-naming means that's immediate).
  - **Duplicate names rejected.** Names are the grid identity (load + `GridRecord` lookup), so a name already used by another magazine slot is refused inline (field reverts, red border, "Name already in use").
  - **Load / Unload.** Each named row's **Load → beam** emits `load_requested(grid_name)`; the widget-level **Unload** emits `unload_requested()`. The Grids-tab controller actions both (running `GridTaskManager.ensure_loaded` on a thread for load, retracting the working slot for unload, with `GridExchangeError` handling) and refreshes the holder — so an operator can stage/retract a grid outside a workflow run.
  - **Signals:** `magazine_changed()`, `presence_toggled(slot_name, bool)`, `load_requested(grid_name)`, `unload_requested()`.
  - **Persistence:** none in Phase 4 — magazine state is in-memory only; where it persists (holder config vs. experiment) is decided later.
- **View coordination.** Once a grid is loaded, the magazine slot and the holder working slot reference the *same* `SampleGrid` object (same physical grid). The Grids-tab controller therefore refreshes the holder view on `magazine_changed` (and the magazine on holder edits) so both stay in sync. Editing a loaded grid is **not** disabled — it is a view-refresh concern, not a model conflict.

**Run sub-section** (detailed). Mirrors the lamella run flow ([`LamellaWorkflowWidget`](../../fibsem/ui/widgets/lamella_workflow_widget.py) + `_start_run_workflow_thread` → daemon thread → `manager.run(parent_ui=self)` → `workflow_update_signal` → Stop = `manager.stop()`).

- **`GridListWidget`** (new, mirrors `LamellaListWidget`) — checkbox multi-select list of `GridRecord`s with a status badge from `task_state.status`. `set_experiment(exp)`, `get_selected_grids() → List[GridRecord]`, signal `grid_selection_changed(list)`.
- **`GridWorkflowWidget`** (new container, mirrors `LamellaWorkflowWidget`) — `GridListWidget` (top) + reused `WorkflowConfigWidget` (bottom). The config widget is task-agnostic; populate it with task entries derived from `experiment.grid_protocol.task_config` keys. For Phase 4 the supervise / schedule / dependency columns are **hidden** (via the existing `enable_*_button` toggles) — grid Run is just **task selection + grid-outer order**. Exposes `get_selected_grids()`, `get_selected_tasks()` (in display order).
- **Run / Stop** live in the Run sub-section header (not the global status-bar button), enabled when ≥1 grid **and** ≥1 task selected. Run → confirm dialog → `_start_grid_workflow_thread(task_names, grid_names)`: a daemon thread builds `GridTaskManager(microscope, experiment, parent_ui=self)` and calls `manager.run(task_names, grid_names)`; a finished signal re-enables Run. Stop → `manager.stop()` (already threaded through `_stop_event` and checked each queue item in Phase 3a).
- **Backend addition (status):** `GridTaskManager` currently runs headless. Phase 4 adds **minimal status emission** — on each item it calls `parent_ui.grid_workflow_update_signal.emit({...})` (mirroring `TaskManager._emit_status`), and the UI refreshes the `GridListWidget` badges + a status label. Rich progress/timeline reuse is deferred (consistent with minimal Results).
- **Note:** a *persisted* grid workflow order/supervision (`GridTaskProtocol.workflow_config`, analogous to `AutoLamellaWorkflowConfig`) is a future addition; Phase 4 Run uses an ephemeral selection/order.

## 7. Data Model

### Grid ↔ Lamella relationship ✅ *implemented*

Every lamella is physically located on a grid, so this containment is part of the **data model**, not something inferred from stage coordinates at runtime — a lamella belongs to its grid whether or not that grid is currently mounted.

The relationship is captured by a single back-reference, with the grid→lamella direction *derived* rather than stored:

```
Lamella.grid_id: Optional[str]   →  GridRecord._id     (stable, experiment-scoped)
GridRecord.name                  →  SampleGrid.name    (hardware link, by name)
```

- **Back-reference up, not a stored child list.** `Lamella.grid_id` ([`structures.py:703`](../../fibsem/applications/autolamella/structures.py#L703)) is the single source of truth; the grid→lamella direction is a derived view via `Experiment.get_lamellae_for_grid(grid)` (`[l for l in positions if l.grid_id == grid._id]`). A separately stored child list would drift out of sync with `positions`. The reverse convenience is `Experiment.get_grid_for_lamella(lamella)` (resolves `grid_id` via `get_grid_by_id`).
- **Reference the workflow `GridRecord`, not the hardware `SampleGrid`.** The lamella belongs to a *grid in this experiment* — the entity with state, history and a lifecycle, living in the same `experiment.yaml`. Linking by stable `_id` (mirroring `Lamella._id`) survives display-name renames; the hardware-grid link (`GridRecord.name → SampleGrid.name`) is a separate, lower hop.
- **Set at creation time.** Stamp `grid_id` when the lamella is born via the `add_new_lamella(..., grid_id=...)` parameter ([`structures.py:1521`](../../fibsem/applications/autolamella/structures.py#L1521)): from the detected grid in the screening path (Phase 5), or from `stage.current_grid` ([`_stage.py:272`](../../fibsem/microscopes/_stage.py#L272)) → matching `GridRecord` when added manually. Defaults to `None` (unlinked) — legacy/single-grid lamellae load with no grid, preserving back-compat.
- **Orphan on grid removal.** `Experiment.remove_grid(name)` clears `grid_id` on any linked lamellae (orphaning, not deleting them) so removal stays non-destructive; the lamellae remain in `positions`, unlinked.

**On-disk layout is unchanged.** Lamella keep their flat `experiment.path / petname` directories — the relationship lives in data, not in paths. This keeps lamella paths invariant under grid reassignment (only `grid_id` changes, no directories move, no stamped `acquisition.imaging.path` values break) and avoids migrating existing experiments. Grid-task outputs continue to nest under `experiment.path / <grid name> / …` because they are grid-scoped artifacts; a lamella is its own first-class entity and need not share that layout.

### Worked example: grid record creation

Creation happens in two stages — grids land on the **holder** (hardware), then get **mirrored** into the experiment as `GridRecord`s. The holder owns physical presence; everything transient (slot, position) is resolved live by name.

**Step 1 — grids onto the holder (hardware, already exists).** Both paths end at `slot.loaded_grid`:

```python
# CompuStage / autoloader — robotic exchange
microscope._stage.loader.load_grid("Slot-01", SampleGrid(name="grid-aspen"))
microscope._stage.loader.load_grid("Slot-02", SampleGrid(name="grid-birch"))

# Static multi-slot shuttle — set via SampleHolderWidget or sample-holder.yaml,
# loaded into the Stage at startup by _create_sample_stage()
```

Nothing experiment-side has happened yet — this is purely hardware state.

**Step 2 — the proposed `GridRecord`.** Note what it does *not* store: the slot, the stage position, or a copy of the `SampleGrid` geometry.

```python
@evented
@dataclass
class GridRecord:
    name: str                                    # links to SampleGrid.name (hardware)
    _id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_state: AutoLamellaTaskState = field(default_factory=AutoLamellaTaskState)
    task_history: List[AutoLamellaTaskState] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)   # overview image, segmentation, targets
```

**Step 3 — mirror the holder into the experiment.** An idempotent sync mirroring `add_lamella`'s dedupe-then-append shape ([`structures.py:1227`](../../fibsem/applications/autolamella/structures.py#L1227)):

```python
# on Experiment
def add_grid(self, grid: GridRecord) -> None:
    if any(g.name == grid.name for g in self.grids):
        raise ValueError(f"Grid {grid.name} already exists in the experiment.")
    self.grids.append(grid)

def sync_grids_from_holder(self, microscope: FibsemMicroscope) -> None:
    """Create GridRecords for any loaded grid not yet tracked. Idempotent."""
    tracked = {g.name for g in self.grids}
    for slot in microscope._stage.holder.slots.values():
        if slot.loaded_grid is not None and slot.loaded_grid.name not in tracked:
            self.add_grid(GridRecord(name=slot.loaded_grid.name))
    self.save()
```

Call site is just `exp.sync_grids_from_holder(microscope)`. It is safe to call after every load/unload, on experiment load, or before a grid workflow run — matching by `name` means already-tracked grids are skipped. On an autoloader where one slot cycles many grids, each newly loaded grid name adds a record on the next sync; unloading a grid leaves its record (and history) intact.

**Step 4 — resolve the live slot (not stored).** Exactly what `GridTask.slot` already does ([`grid/base.py`](../../fibsem/applications/autolamella/workflows/tasks/grid/base.py)):

```python
slot = microscope._stage.holder.find_slot_by_grid_name(grid_record.name)
```

**Step 5 — persistence.** `GridRecord` joins `positions` in `Experiment.to_dict`/`from_dict` ([`structures.py:988`](../../fibsem/applications/autolamella/structures.py#L988)). Reading with `.get("grids", [])` keeps old `experiment.yaml` files (no `grids` key) loading as empty — no migration needed:

```python
# to_dict
"grids": [g.to_dict() for g in self.grids],
# from_dict
for grid_dict in ddict.get("grids", []):
    experiment.grids.append(GridRecord.from_dict(grid_dict))
```

**Lamella attachment.** `add_new_lamella(..., grid_id=None)` ([`structures.py:1589`](../../fibsem/applications/autolamella/structures.py#L1589)) takes the originating grid's `_id` and stamps it onto the new lamella. A caller creating a lamella from a target on the current grid resolves the grid the stage is on first:

```python
current = microscope._stage.current_grid                     # SampleGrid or None
record = experiment.get_grid_by_name(current.name) if current else None
experiment.add_new_lamella(state, task_config, grid_id=record._id if record else None)
```

### GUI + workflow wiring (implemented)

The link is wired end-to-end:
- **Manual creation** stamps `grid_id` from the new lamella's *own* position — `Stage.grid_at_position(pos)` ([`_stage.py`](../../fibsem/microscopes/_stage.py)) resolves position → holder slot → mounted grid → `GridRecord`; all click-to-place sites funnel through `AutoLamellaUI.add_new_lamella`, so they inherit it.
- **Workflow creation** uses `GridTask.create_lamella(stage_position, name)` ([`grid/base.py`](../../fibsem/applications/autolamella/workflows/tasks/grid/base.py)), which stamps the task's own `grid._id` — the integration point for the Phase 5 targeting task.
- **Display** — grid cards show a lamella count; the Results tab shows a per-grid lamella table (left of the overview, click → focus in the Lamella tab) via `get_lamellae_for_grid`; and the lamella list rows carry a **grid chip** (which grid, dimmed when not loaded) with the stage-dependent controls (move-to / update) disabled when that grid isn't in the beam.
- **Placement** — clicking a grid's overview opens the overview-as-canvas dialog to place/move/delete lamella positions by hand (see §9 Q2).

### Reachability: the loaded grid is the active scope (implemented)

A lamella's `stage_position` is only physically valid when *its* grid is loaded — on an autoloader every grid occupies the **same** working-slot coordinates, so a lamella shown/milled against the wrong grid points at the wrong surface. `Experiment.get_loaded_grids(microscope)` (grids in `holder.occupied_slots`), `is_lamella_reachable`, and `unreachable_lamellae` model this. Two consumers:
- **Overview** (`FibsemMinimapWidget`) filters markers to lamellae on the loaded grid(s) (`grid_id is None` always shown), since the single-grid overview can't meaningfully place off-grid positions.
- **Lamella run** (`AutoLamellaMainUI._on_run_workflow_clicked`) **blocks** when the selection includes lamellae whose grid isn't loaded — a pre-flight guard; the lamella `TaskManager` itself is unchanged. Multi-grid execution (grid-outer load-and-mill) is deferred; the intended shape is **B** — the grid workflow drives loading and runs the lamella manager scoped to each grid's lamellae (see [Decisions](#8-decisions)).

---

## 8. Decisions

- **Grid workflow record placement (resolved).** Grids become a first-class workflow entity, `GridRecord`, held under `Experiment` (e.g. `grids: EventedList[GridRecord]`) and kept distinct from the hardware `SampleGrid` (geometry, owned by the holder config). Workflow state — identity, `task_state`, `task_history`, screening results — lives on `GridRecord`; it links to hardware by name (`GridRecord.name → SampleGrid.name`) and resolves its current slot live via the holder rather than freezing slot occupancy into the experiment. The grid↔lamella containment is modelled by a `Lamella.grid_id → GridRecord._id` back-reference (**implemented**: field + `get_lamellae_for_grid` / `get_grid_for_lamella` / `get_grid_by_id` helpers + orphan-on-`remove_grid`); see [Grid ↔ Lamella relationship](#grid--lamella-relationship--implemented). To avoid disrupting the mature lamella pipeline, `Experiment.positions` stays the canonical flat list of lamella initially, with per-grid lamella derived by filtering on `grid_id`.
- **Holder & loader semantics (resolved).** One orchestration path covers both the static multi-slot shuttle and the CompuStage autoloader:
  - **Two-level model: magazine → working slot.** The loader owns a **magazine** — its own set of storage slots at a fixed capacity (existing loaders hold 12), filled by a human operator — *distinct* from the holder's working slot(s) in the beam. This extends `SampleGridLoader` with its own `capacity` + `slots` (today it only mutates `holder.slots`). The magazine is the inventory of grids available to load; the working slot is where exactly one of them sits at a time.
  - **Inventory & naming.** Presence is detected, not assumed. `SampleGridLoader` gains a `run_inventory()` method that scans which magazine slots hold a grid. The user assigns a name/description to a grid only for a *loaded* slot — a magazine slot on an autoloader, or a holder slot on a static shuttle; empty slots cannot be named. `GridRecord`s are the workflow inventory, created from these named grids — from the loader's magazine on an autoloader, or via `sync_grids_from_holder` on a static shuttle (no loader).
  - **Implicit exchange.** When a task's requested grid differs from the grid currently in the holder's working slot, the orchestrator performs the exchange implicitly via an `ensure_loaded(record)` step — robotically moving the grid from its magazine slot into the working slot (and the previous grid back) — rather than modelling it as a first-class task. On a static shuttle (no loader) every grid is already in a holder slot, so `ensure_loaded` is a no-op followed by a stage move.
  - **Iteration order.** Grid-outer by default (load a grid, run all its tasks, then exchange) to amortise exchanges; task-outer ordering is permitted only when there is no loader (static shuttle). Implemented as an ordering option on `TaskQueue.build_from_matrix`.
  - **Failure handling.** If an exchange fails, halt the workflow and raise — no skip, no retry.
  - `ensure_loaded` lives in the grid manager's per-grid loop and becomes an overridable no-op for lamella when the shared base is extracted.
- **Orchestration: share the queue, parallel manager, extract a base later (resolved).** The orchestration core is generic; only five seams bind it to lamella (unit collection `positions`, accessor `get_lamella_by_name`, the unit state interface `is_failure`/`has_completed_task`/`task_state`/`task_config`, the task registry, and status-dict key naming). Therefore: (1) reuse `TaskQueue` directly — it is already unit-agnostic; the only change is renaming `WorkItem.lamella_name` → `unit_name` with a back-compat alias for the UI status dict. (2) Implement a thin `GridTaskManager` mirroring `TaskManager` short-term rather than refactoring the working lamella path up-front. (3) Once both managers are concrete, extract a `BaseTaskManager` that injects the five seams via a small "workflow unit" interface (`.name`, `.is_failure`, `.has_completed_task`, `.task_state`, `.task_config`), which both `Lamella` and `GridRecord` satisfy. This shares code (so the grid side does not permanently lag) while avoiding a premature, risky refactor of the mature pipeline.
- **Config schema reuse: lean grid configs, flat serialization (resolved).** `GridTaskConfig` stays lean — it does **not** inherit `milling` / `reference_imaging` (lamella-specific; no current grid task patterns-mills or does alignment-reference imaging). A grid task that genuinely mills (e.g. the `ParallelTrenchMilling` stub) adds its own `milling` field opt-in.
  - **Keep the `parameters` *property*** (the list of task-specific field names). It is what the task-config UI iterates to auto-generate the editor form per task type ([`autolamella_task_config_widget.py:327`](../../fibsem/ui/widgets/autolamella_task_config_widget.py#L327)), and the Phase 4 grid editor will want the same.
  - **Serialize grid configs flat, not via a `parameters` subdict.** Each field self-serializes; nested dataclasses (e.g. `settings: OverviewAcquisitionSettings`) call their own `to_dict`/`from_dict`. This is type-correct (the subdict's `getattr` dump assumes primitives and already needs hand-patched exclusions on the lamella side, e.g. `spot_burn` excluding `coordinates`) and transparent.
  - **Consequence:** grid and lamella on-disk formats diverge by design. The eventual shared `TaskConfigBase` (extracted in the later refactor) shares **identity + the `parameters` property**, but each side keeps its own `to_dict`/`from_dict`. Killing the lamella subdict is out of scope (it is already on disk and would need a `protocol.yaml` migration).

## 9. Open Questions

1. **Per-grid protocols vs one shared protocol (deferred — global for now).** Today `Experiment.grid_protocol` is a *single shared* `GridTaskProtocol`: every grid runs the same task configs. The lamella side is different — each `Lamella` carries its **own** `task_config` (per-unit, editable), seeded from the shared `Experiment.task_protocol` template. **Decision (2026-06):** keep the grid protocol **global** for now — the initial screening workflow is expected to be uniform across grids, and a global protocol is simpler. So the Grids-tab right panel's **Protocol** sub-tab edits the shared `grid_protocol` (it does *not* follow card selection), while **Results** is per-grid.
   - **Future option (the hybrid, mirroring lamella):** give `GridRecord` its own `task_config` seeded from `grid_protocol` (the template) when a grid is added; `run_grid_task` would read `grid.task_config` instead of the shared protocol. This is worth doing if grids need per-grid tuning (ice thickness, sample quality, milling current). It also makes the UI more coherent — both Protocol and Results in the Grids tab would be per-grid (selection-following) — and *eases* the deferred `BaseTaskManager`/`TaskConfigBase` unification (grid would then mirror lamella's per-unit-config shape). The uniform case stays easy via the template-seed default. Revisit when a real per-grid need appears; until then the global protocol stands. (Per-grid *workflow* — different grids running different task *sets* — is already covered at run time by the Workflow tab's grid×task selection, so it doesn't need per-grid workflow definitions.)
2. **Overview-as-canvas: place lamella positions on the grid overview ✅ *implemented*.** Clicking a grid's overview in the Results tab opens a non-modal [`LamellaSelectionDialog`](../../fibsem/ui/widgets/lamella_selection_dialog.py): a matplotlib `OverviewCanvas` (minimap-styled markers + milling-FOV boxes, zoom/pan, fit, crosshair, scalebar, contrast/gamma via the reusable [`ContrastGammaControl`](../../fibsem/ui/widgets/contrast_gamma_control.py)) beside the grid's lamella list. Right-click adds / moves the selected position; the per-row trash deletes; edits are a working set committed on Accept. Each new lamella uses the **overview image's recorded `MicroscopeState`** (so it's placed in the frame the overview was acquired in) and is stamped with the grid's id via `add_new_lamella(microscope_state=…, grid_id=…)`. This is the manual half of the screening→targeting loop; the per-grid lamella table also appears in the Results tab. The remaining open piece here is whether a *move* should rewrite the whole pose or only the position (see the follow-up notes).
3. **Targeting coupling (deferred).** Where does the `fibsem/targeting` screening pipeline sit relative to grid tasks — is screening *one* grid task that produces targets, or a multi-task sub-workflow (acquire → segment → score → select)? Deferred until `fibsem/targeting` lands on this branch (currently on `feat-automated-ml-grid-targeting`). Pointers for when we pick it up: the pipeline entry point is `run_detection_pipeline(model, image, ...) -> SegmentationTargetResult`, with `generate_screened_positions(model, run, ...)` looping it over a `GridScreeningRun`'s images. `SegmentationTargetResult.targets` yields enabled `LamellaTarget`s (each already carrying `stage_position` + `poi`), which is the natural hand-off into `Experiment.add_new_lamella` with `grid_id` stamped (the Phase 5 loop).
