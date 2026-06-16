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

## 3. Stage Components (current state)

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

### SampleGrid — [`_stage.py:24`](../../fibsem/microscopes/_stage.py#L24)

A physical TEM grid or specimen. It is intentionally minimal — identity plus geometry:

```python
@dataclass
class SampleGrid:
    name: str
    description: str = ""
    radius: float = field(default=GRID_RADIUS)  # GRID_RADIUS = 1e-3 m (1 mm)
```

`radius` is currently descriptive geometry only — it is *not* read during slot matching. `Stage.current_slot` matches against the fixed module constant `GRID_RADIUS` (the value `SampleGrid.radius` happens to default to), not the per-grid field. `to_dict`/`from_dict` provide YAML serialization.

### GridSlot — [`_stage.py:51`](../../fibsem/microscopes/_stage.py#L51)

A *fixed* mechanical slot on a holder. A slot is a location with a known stage position; it may hold one grid or none:

```python
@dataclass
class GridSlot:
    name: str
    index: int
    position: FibsemStagePosition       # where the stage moves to reach this slot
    loaded_grid: Optional[SampleGrid] = None
```

### SampleHolder — [`_stage.py:88`](../../fibsem/microscopes/_stage.py#L88)

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
- **Lookup helpers**: `find_slot_for_grid(grid)` and `find_slot_by_grid_name(name)` resolve which slot a grid occupies.
- **`_ensure_slots()`** materialises exactly `capacity` slots (`Slot-01`, `Slot-02`, …), adding missing ones and trimming extras.
- **Persistence**: `to_dict`/`from_dict`, plus `save(path)` / `load(path)` to YAML. Holder configuration files: [`fibsem/config/sample-holder.yaml`](../../fibsem/config/sample-holder.yaml) (user) and `default-sample-holder.yaml` (fallback).

### SampleGridLoader — [`_stage.py:189`](../../fibsem/microscopes/_stage.py#L189)

The robotic actuator that exchanges grids in and out of slots:

```python
class SampleGridLoader:
    def load_grid(self, slot_name: str, grid: SampleGrid) -> None: ...
    def unload_grid(self, slot_name: str) -> None: ...
    @property
    def loaded_slots(self) -> List[GridSlot]: ...
```

The loader's job is to mutate `slot.loaded_grid` — i.e. to change *which* grid occupies a slot. It is only instantiated for **CompuStage** systems (which model an autoloader); on other backends the loader is `None` and grids are placed into slots manually (via the UI or config).

### Stage — [`_stage.py:220`](../../fibsem/microscopes/_stage.py#L220)

The high-level stage interface. It owns a `holder` and optional `loader`, performs all motion, and tracks slot/grid context:

- **Motion**: `move_absolute`, `move_relative`, `stable_move`, `vertical_move`, `move_to_milling_angle`, `move_to_orientation`, `home`, and the slot-aware `move_to_slot(slot_name)` / `move_to_grid(grid_name)`.
- **Slot awareness**: `current_slot` returns the slot whose `position` matches the live stage position within `GRID_RADIUS` tolerance on the x/y axes (`is_close2`); `current_grid` returns the grid loaded in that slot, if any.

### Backend construction — [`_stage.py:329`](../../fibsem/microscopes/_stage.py#L329)

`_create_sample_stage(microscope)` decides the holder/loader configuration per backend:

- **CompuStage** → a single-slot holder (`capacity=1`) plus an active `SampleGridLoader` (models an autoloader).
- **Other backends** → load the holder from `SAMPLE_HOLDER_CONFIGURATION_PATH` (falling back to the default), stamp each slot's `r`/`t` with the SEM orientation, and set `loader = None` (manual loading).

### UI

[`fibsem/ui/widgets/sample_holder_widget.py`](../../fibsem/ui/widgets/sample_holder_widget.py) provides `SampleHolderWidget` (and slot-row / grid-edit sub-widgets) for editing the holder, capturing slot positions from the live stage, loading/clearing grids, and auto-saving to YAML.

---

## 4. Grid Workflows (current state)

The grid task prototype lives in [`fibsem/applications/autolamella/workflows/tasks/grid_tasks.py`](../../fibsem/applications/autolamella/workflows/tasks/grid_tasks.py). It deliberately mirrors the *shape* of the lamella task system, but is an early implementation.

### GridTaskConfig — [`grid_tasks.py:23`](../../fibsem/applications/autolamella/workflows/tasks/grid_tasks.py#L23)

```python
@dataclass
class GridTaskConfig(ABC):
    task_type: ClassVar[str]
    display_name: ClassVar[str]
    task_name: str = ""
```

Compared with the lamella `AutoLamellaTaskConfig`, this base has no shared imaging/milling fields and **no `to_dict`/`from_dict`** — configs cannot yet be serialized as part of a protocol.

### GridTask — [`grid_tasks.py:31`](../../fibsem/applications/autolamella/workflows/tasks/grid_tasks.py#L31)

```python
class GridTask(ABC):
    def __init__(self, microscope, config, grid, experiment, parent_ui=None, task_manager=None): ...

    @property
    def slot(self) -> Optional[GridSlot]:
        return self.microscope._stage.holder.find_slot_for_grid(self.grid)

    def run(self):
        self._run()                         # no pre/post lifecycle

    def _move_to_grid_slot_position(self, orientation: str): ...
```

A `GridTask` is bound to a `SampleGrid` (not a `Lamella`), resolves its slot via the holder, and can move to that slot in a target orientation. Crucially, `run()` calls `_run()` directly — there is **no `pre_task`/`post_task` lifecycle, no `task_state`, and no `task_history`** (contrast with `AutoLamellaTask`, below).

### Implemented tasks

| Task | Config | What it does |
|---|---|---|
| `AcquireOverviewImageGridTask` | `AcquireOverviewImageGridTaskConfig` | Moves to the grid slot and acquires a tiled (default 3×3) overview via `tiled_image_acquisition_and_stitch`. |
| `AcquireImageTask` | `AcquireImageGridTaskConfig` | Single high-resolution image at a configurable voltage, saving and restoring microscope state. |
| `CryoCleaningGridTask` | `CryoCleaningGridTaskConfig` | FIB cryo-cleaning for a fixed duration; honours the `_stop_event`. |

Stub configs with no task implementation yet: `CryoDepositionGridTaskConfig`, `CryoSputterGridTaskConfig`, `ParallelTrenchMillingGridTaskConfig`.

### Registry & runners

- `GRID_TASK_REGISTRY` ([`grid_tasks.py:266`](../../fibsem/applications/autolamella/workflows/tasks/grid_tasks.py#L266)) — a string→class map (currently 3 entries).
- `run_grid_task` / `run_grid_tasks` ([`grid_tasks.py:273`](../../fibsem/applications/autolamella/workflows/tasks/grid_tasks.py#L273)) — minimal nested loops over grids and tasks.

The runner instantiates each config with **defaults** (`task_cls.config_cls()`) rather than reading a saved configuration, and carries the explicit TODO *"add task config to experiment, integrate into runner."* The commented-out lines show the intended direction (pull config from `experiment.task_protocol`).

### Current limitations (the design surface)

1. **No task lifecycle / state.** `GridTask` lacks `pre_task`/`post_task`; there is no per-grid `task_state` or `task_history`, so grid-task progress is neither observable nor resumable.
2. **No config persistence.** `GridTaskConfig` has no `to_dict`/`from_dict`, and grid task configs are not stored on `Experiment` or in YAML. The runner ignores any saved config.
3. **No orchestration.** Grid runs are ad-hoc loops with no skip/dependency/scheduling logic and no UI status signals — unlike the lamella `TaskManager`/`TaskQueue`.
4. **Grid is not a first-class workflow entity.** `SampleGrid` is a *hardware* descriptor; there is no workflow-level grid record under `Experiment` carrying state, history, and results the way `Lamella` does.
5. **No grid-workflow UI** analogous to the lamella task-config and workflow-order editors.
6. **(Adjacent) Screening pipeline.** `fibsem/targeting/` (grid screening / automated lamella targeting, currently on branch `feat-automated-ml-grid-targeting`) is the likely first real consumer of grid workflows — overview acquisition feeding segmentation and target selection.

---

## 5. Relationship to the Lamella Task System

The grid task system is intentionally a sibling of the AutoLamella task system. The lamella system is the mature template to mirror:

- **`AutoLamellaTask`** ([`base.py:95`](../../fibsem/applications/autolamella/workflows/tasks/base.py#L95)) — lifecycle `pre_task() → _run() → post_task()`, hooks, stop-event; `post_task` appends to `lamella.task_history` and writes back per-lamella config.
- **`AutoLamellaTaskConfig`** ([`structures.py:173`](../../fibsem/applications/autolamella/structures.py#L173)) — `task_type`/`display_name`, shared `milling` / `reference_imaging`, `to_dict`/`from_dict`; discovered via `BUILTIN_TASKS` + plugins (`get_tasks()`).
- **`TaskManager`** (`workflows/tasks/manager.py`) + **`TaskQueue`** (`workflows/tasks/queue.py`) — build and execute a *(lamella × task)* work matrix, applying skip/dependency logic, emitting UI signals, and saving the `Experiment` after each item.
- **`Lamella`** ([`structures.py:686`](../../fibsem/applications/autolamella/structures.py#L686)) / **`Experiment`** ([`structures.py:928`](../../fibsem/applications/autolamella/structures.py#L928)) / **`AutoLamellaTaskProtocol`** ([`structures.py:414`](../../fibsem/applications/autolamella/structures.py#L414)) — per-unit state + history, the collection, and the protocol (per-task config + ordered workflow with dependencies, supervision, scheduling).

### Mapping table

| Concern | Lamella system (mature) | Grid system (today) | Grid system (proposed) |
|---|---|---|---|
| Unit of work | `Lamella` | `SampleGrid` (hardware only) | `GridRecord` (first-class workflow entity under `Experiment`) |
| Task base | `AutoLamellaTask` (`base.py:95`) | `GridTask` (`grid_tasks.py:31`) | + `pre_task`/`post_task` lifecycle |
| Task config | `AutoLamellaTaskConfig` (`structures.py:173`) | `GridTaskConfig` (`grid_tasks.py:23`) | + `to_dict`/`from_dict`, shared imaging/milling bases |
| Registry | `BUILTIN_TASKS` + `get_tasks()` | `GRID_TASK_REGISTRY` | unified discovery + plugins |
| State / history | `task_state`, `task_history` | — | per-grid state + history |
| Protocol | `AutoLamellaTaskProtocol` | — | grid protocol (configs + ordered workflow) |
| Orchestration | `TaskManager` + `TaskQueue` | `run_grid_task(s)` ad-hoc loops | shared/extended `TaskManager` |
| Persistence | `experiment.yaml` per-lamella config | none | grid configs + results on `Experiment` |
| UI | task config editor, workflow widget | — | grid task config + workflow editor |

---

## 6. Proposed Design (path to parity)

The goal is to grow the grid workflow layer to match the lamella task system, reusing its patterns (and, where sensible, its code) rather than building a divergent system. Proposed in phases so each step is independently useful.

### Phase 1 — Grid record, lifecycle & state ✅ *implemented*
Introduce a minimal `GridRecord` — the first-class workflow entity under `Experiment`, distinct from the hardware `SampleGrid` (see [Decisions](#8-decisions)) — carrying identity plus per-grid `task_state` + `task_history`. Give `GridTask` a `pre_task` / `post_task` lifecycle matching `AutoLamellaTask` that writes progress into the record. This makes grid-task progress observable (UI status), resumable, and auditable — the single biggest gap today, and the foundation the later phases build on. See [Data Model](#7-data-model) for the record shape and its relationship to lamella.

### Phase 2 — Config persistence
Add `to_dict` / `from_dict` to `GridTaskConfig` and store grid task configs in a **grid protocol** on `Experiment`, mirroring `AutoLamellaTaskProtocol`. Update `run_grid_task` to read saved config from the protocol instead of instantiating defaults (`config_cls()`), removing the standing TODO.

### Phase 3 — Orchestration
Route grid tasks through the existing `TaskManager` / `TaskQueue` (or a shared base class extracted from them) so grid workflows inherit skip conditions, dependency ordering, scheduling, stop handling, and UI status signals. The work matrix becomes *(grid × task)* over the holder's loaded slots.

### Phase 4 — Results & UI
Flesh out `GridRecord` (introduced in Phase 1) with screening `results`, and add the grid-workflow UI: grid task-config and workflow-order editor widgets analogous to [`autolamella_task_config_editor.py`](../../fibsem/ui/widgets/autolamella_task_config_editor.py) and [`workflow_config_widget.py`](../../fibsem/ui/widgets/workflow_config_widget.py).

### Phase 5 — Screening integration
Wire the `fibsem/targeting` grid-screening pipeline (overview acquisition → segmentation → target scoring/selection) in as grid tasks that emit `Lamella` targets — closing the loop from grid-level screening to lamella-level milling. This is the headline workflow the whole effort enables: load a grid → screen it → produce lamella targets → mill them.

---

## 7. Data Model

### Grid ↔ Lamella relationship

Every lamella is physically located on a grid, so this containment is part of the **data model**, not something inferred from stage coordinates at runtime — a lamella belongs to its grid whether or not that grid is currently mounted.

The relationship is captured by a single back-reference, with the grid→lamella direction *derived* rather than stored:

```
Lamella.grid_id: Optional[str]   →  GridRecord._id     (stable, experiment-scoped)
GridRecord.name                  →  SampleGrid.name    (hardware link, by name)
```

- **Back-reference up, not a stored child list.** `Lamella.grid_id` is the single source of truth; `GridRecord.lamellae` is a derived view (`[l for l in experiment.positions if l.grid_id == self._id]`). A separately stored child list would drift out of sync with `positions`.
- **Reference the workflow `GridRecord`, not the hardware `SampleGrid`.** The lamella belongs to a *grid in this experiment* — the entity with state, history and a lifecycle, living in the same `experiment.yaml`. Linking by stable `_id` (mirroring `Lamella._id`) survives display-name renames; the hardware-grid link (`GridRecord.name → SampleGrid.name`) is a separate, lower hop.
- **Set at creation time.** Stamp `grid_id` when the lamella is born: from the detected grid in the screening path (Phase 5), or from `stage.current_grid` ([`_stage.py:272`](../../fibsem/microscopes/_stage.py#L272)) → matching `GridRecord` when added manually.

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

**Step 4 — resolve the live slot (not stored).** Exactly what `GridTask.slot` already does ([`grid_tasks.py:53`](../../fibsem/applications/autolamella/workflows/tasks/grid_tasks.py#L53)):

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

**Lamella attachment.** When a lamella is created from a target on the current grid, `add_new_lamella` ([`structures.py:1271`](../../fibsem/applications/autolamella/structures.py#L1271)) stamps `grid_id` from the grid the stage is on:

```python
current = microscope._stage.current_grid                     # SampleGrid or None
record = next((g for g in self.grids if g.name == current.name), None)
lamella.grid_id = record._id if record else None
```

---

## 8. Decisions

- **Grid workflow record placement (resolved).** Grids become a first-class workflow entity, `GridRecord`, held under `Experiment` (e.g. `grids: EventedList[GridRecord]`) and kept distinct from the hardware `SampleGrid` (geometry, owned by the holder config). Workflow state — identity, `task_state`, `task_history`, screening results — lives on `GridRecord`; it links to hardware by name (`GridRecord.name → SampleGrid.name`) and resolves its current slot live via the holder rather than freezing slot occupancy into the experiment. The grid↔lamella containment is modelled by a `Lamella.grid_id → GridRecord._id` back-reference; see [Grid ↔ Lamella relationship](#grid--lamella-relationship). To avoid disrupting the mature lamella pipeline, `Experiment.positions` stays the canonical flat list of lamella initially, with per-grid lamella derived by filtering on `grid_id`.
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

1. **Targeting coupling (deferred).** Where does the `fibsem/targeting` screening pipeline sit relative to grid tasks — is screening *one* grid task that produces targets, or a multi-task sub-workflow (acquire → segment → score → select)? Deferred until `fibsem/targeting` lands on this branch (currently on `feat-automated-ml-grid-targeting`). Pointers for when we pick it up: the pipeline entry point is `run_detection_pipeline(model, image, ...) -> SegmentationTargetResult`, with `generate_screened_positions(model, run, ...)` looping it over a `GridScreeningRun`'s images. `SegmentationTargetResult.targets` yields enabled `LamellaTarget`s (each already carrying `stage_position` + `poi`), which is the natural hand-off into `Experiment.add_new_lamella` with `grid_id` stamped (the Phase 5 loop).
