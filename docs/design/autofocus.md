# Autofocus: Consolidation & Strategy Extension

> **Status:** Part 1 (consolidation) implemented in [PR #107](https://github.com/fibsem-os/fibsem-os/pull/107) · Part 2 (strategy extension) proposed, not yet implemented.

This document covers two related things:

1. **Consolidation** — how the duplicated FIB-SEM and FM autofocus types/methods were merged into one canonical set, and the reusable recipe for doing the same to other auto-functions.
2. **Strategy extension** — a simple, forward-looking design for making each focus pass pluggable (e.g. `sweep` vs `hill_climb`), which also finishes the consolidation by unifying the two remaining sweep engines.

Consolidation comes first because the strategy extension builds directly on it.

---

## Part 1 — Consolidating FIB-SEM / FM autofocus

### What existed before

Autofocus was implemented twice, with parallel-but-incompatible types:

| Concern | FIB-SEM (`autofunctions.autofocus`) | FM (`fm.structures` / `fm.calibration`) |
|---|---|---|
| Result | `AutoFocusResult` — iteration objects, `save`/`load`, plot | `AutoFocusResult` — parallel lists (`z_positions`, `scores`, `images`), recursive `iterations`, no `save`/`load` |
| Settings | `AutoFocusSettings` — `passes: list[FocusSweepPass]` | `AutoFocusSettings` — fixed `coarse_*` / `fine_*` |
| Method enum | `str` | `FocusMethod(Enum)` |
| Run fn | `run_auto_focus` (multi-pass WD sweep) | `run_autofocus` + `run_coarse_fine_autofocus` (objective-z sweep) |
| Strategy layer | — | `fm.strategy` package (ABC + generics + pydantic) wrapping the run fns with no real polymorphism |

Two `AutoFocusResult` classes, two `AutoFocusSettings` classes, a duplicated `FocusMethod`, and a 460-line strategy package that added indirection without behaviour.

### The consolidation recipe

The pattern we applied — reusable for any duplicated FIB-SEM/FM auto-function (ACB, charge neutralisation, etc.):

1. **Pick the canonical home.** Lower-level, hardware-agnostic code lives in `fibsem.autofunctions.*`. `fibsem.fm.*` may depend on `autofunctions`, never the reverse — so canonical types go in `autofunctions`.
2. **Unify the data shape first.** The result/settings types must describe the *same data*, independent of hardware. Here: iterations store a raw `np.ndarray` (not `FibsemImage`), since that's all the focus metric needs; domain-only fields (`channel_name` for FM; `probe_resolution`/`probe_dwell_time`/`use_autocontrast` for FIB-SEM) coexist on one dataclass and are simply ignored by the domain that doesn't use them.
3. **Re-export for source compatibility.** `fm.structures` re-exports `AutoFocusResult`, `AutoFocusSettings`, `FocusMethod`, `FocusSweepPass` from `autofunctions.autofocus`, so existing `from fibsem.fm.structures import ...` call sites keep working with no edits.
4. **Provide a factory + legacy loader, not a shim.** `AutoFocusSettings.from_coarse_fine(...)` builds the common two-pass config; `from_dict` accepts both the new `passes` schema and the legacy `coarse_*`/`fine_*` and `n_steps` formats. This is the migration surface — old protocols on disk still load.
5. **Delete the abstraction that wasn't paying rent.** The `fm.strategy` package was removed; `run_autofocus` / `run_coarse_fine_autofocus` are called directly.

### Canonical types (today)

```python
class FocusMethod(str, Enum):           # str-Enum: serialises as its value, compares as str
    LAPLACIAN = "laplacian"; SOBEL = "sobel"; VARIANCE = "variance"; TENENGRAD = "tenengrad"

@dataclass
class FocusSweepPass:
    search_range: float = 5e-3          # total span (positions cover ±range/2)
    step_size: float = 0.5e-3
    enabled: bool = True
    @property
    def n_steps(self) -> int: ...       # derived: round(search_range / step_size)

@dataclass
class AutoFocusSettings:
    method: FocusMethod = FocusMethod.TENENGRAD
    passes: list = None                 # list[FocusSweepPass]; coarse → fine
    probe_resolution: tuple = (768, 512)   # FIB-SEM only
    probe_dwell_time: float = 0.5e-6       # FIB-SEM only
    reduced_area: FibsemRectangle = None
    use_autocontrast: bool = True          # FIB-SEM only
    channel_name: Optional[str] = None     # FM only

@dataclass
class AutoFocusIteration:
    working_distance: float; focus_score: float; pass_index: int; image: np.ndarray

@dataclass
class AutoFocusResult:
    image: np.ndarray; working_distance: float; initial_working_distance: float
    focus_score: float; iterations: list[AutoFocusIteration]
    settings: AutoFocusSettings = None; method: Optional[str] = None
```

### What still isn't consolidated

The **two sweep executors remain duplicated**:

- FIB-SEM `_run_sweep` moves *working distance* (`microscope.set_working_distance`), acquires via `microscope.acquire_image`, scores `probe.filtered_data`.
- FM `run_autofocus` / `run_coarse_fine_autofocus` move the *objective* (`objective.move_absolute`), acquire via `fm.acquire_image`, crop to ROI, score via `calculate_focus_quality`.

The control flow (sweep positions, argmax, move-to-best, pass tagging, cancellation) is the same in both; only the move + acquire + score primitives differ. Part 2 closes this gap.

One piece of that duplication is already factored out: converting a `FocusSweepPass`
into a relative sweep is centralised in **`ZParameters.from_focus_pass(pass)`**
(used by both `run_coarse_fine_autofocus` and the tileset autofocus in
`fm/acquisition.py`). It lives on `ZParameters` because the conversion targets an
fm-level type — `autofunctions` can't import it. Note that once the Part 2 driver
lands, the per-pass path stops going through `ZParameters` at all (the driver
visits positions directly via `move`); `from_focus_pass` is the interim dedup and
remains useful for callers that still want a `ZParameters` (e.g. building a z-stack
from a pass).

---

## Part 2 — Per-pass focus strategy

### Goal

Let each pass choose *how* it searches, not just its range/step. Default stays a full grid sweep; opt into alternatives like an iterative hill-climb:

```python
FocusSweepPass(search_range=100e-6, step_size=10e-6, strategy=FocusStrategy.HILL_CLIMB)
```

This must **not** reintroduce the deleted strategy-class layer. A strategy is only the question *"which position do I visit next?"* — independent of how a position is measured.

### Design

Add a per-pass enum (serialises and maps onto existing params):

```python
class FocusStrategy(str, Enum):
    SWEEP = "sweep"
    HILL_CLIMB = "hill_climb"

@dataclass
class FocusSweepPass:
    search_range: float = 5e-3
    step_size: float = 0.5e-3
    enabled: bool = True
    strategy: FocusStrategy = FocusStrategy.SWEEP   # NEW
```

Pull the hardware mechanics into **two callbacks** each executor supplies, and make the strategy a tiny function driving them:

```python
def _run_pass(p, center, move, measure):
    """move(pos): position the WD / objective.
       measure(pos) -> AutoFocusIteration at pos (records into the shared list, returns it).
       Returns the best iteration; leaves the hardware parked at it."""
    def evaluate(pos):
        move(pos)
        return measure(pos)

    climb = _hill_climb if p.strategy is FocusStrategy.HILL_CLIMB else _sweep
    best = climb(p, center, evaluate)
    move(best.working_distance)          # driver guarantees the final park position
    return best

def _sweep(p, center, evaluate):
    half = p.search_range / 2
    its = [evaluate(z) for z in np.linspace(center - half, center + half, p.n_steps + 1)]
    return max(its, key=lambda it: it.focus_score)

def _hill_climb(p, center, evaluate):
    half, step = p.search_range / 2, p.step_size
    best = evaluate(center)
    direction = max((+1, -1), key=lambda d: evaluate(center + d * step).focus_score)
    pos = center + direction * step
    while abs(pos - center) <= half:     # bounded by search_range
        it = evaluate(pos)
        if it.focus_score <= best.focus_score:
            break                        # greedy stop on first non-improvement
        best, pos = it, pos + direction * step
    return best
```

`move` is the primitive that differs between domains — and it's the *only* hardware difference, so factoring it out is what lets one driver serve both:

```python
# FIB-SEM
move = lambda wd: microscope.set_working_distance(wd, beam_type)
def measure(wd):
    probe = microscope.acquire_image(image_settings=probe_settings)
    score = float(np.mean(focus_fn(probe.filtered_data.astype(np.float32))))
    it = AutoFocusIteration(working_distance=wd, focus_score=score,
                            pass_index=pass_index, image=probe.data)
    iterations.append(it)
    return it

# FM — same shape, two lines differ:
move = lambda z: microscope.objective.move_absolute(z)
def measure(z):
    img = microscope.acquire_image()
    data = img.crop(roi) if roi is not None else img.data
    score = calculate_focus_quality(data, method=method)
    it = AutoFocusIteration(working_distance=z, focus_score=score,
                            pass_index=pass_index, image=data)
    iterations.append(it)
    return it
```

### Cancellation

Both run functions today poll `stop_event` before each position and restore the
initial position on cancel. The driver keeps that behaviour without threading a
flag through every strategy: **`measure` owns the check and raises**, the driver
lets it propagate, and the run function catches it once.

```python
class AutofocusCancelled(Exception):
    pass

def measure(pos):                       # inside each executor's closure
    if stop_event is not None and stop_event.is_set():
        raise AutofocusCancelled
    ...                                 # acquire + score + record

# run_auto_focus / run_coarse_fine_autofocus:
initial = read_current_position()
try:
    best = _run_pass(p, center, move, measure)
except AutofocusCancelled:
    move(initial)                       # restore, then return None / partial result
    return None
```

This keeps strategies oblivious to cancellation (they just call `evaluate`),
puts the one check at the single I/O point, and centralises restore-on-cancel in
the run function — which also fixes the current duplication where each path
re-implements the poll + restore.

### Why this is the sweet spot

- **Finishes the consolidation.** Both `_run_sweep` (FIB-SEM) and the per-pass body of `run_coarse_fine_autofocus` (FM) collapse to "build `move`/`measure`, call `_run_pass`". The extension *removes* the duplication from Part 1 rather than adding code.
- **No class hierarchy.** Strategies are pure functions over an `evaluate` callback; adding one is a function plus an enum value.
- **Driver owns the final move-to-best**, so callers can't forget it (both current paths do this manually — easy to get wrong).
- **No wasted acquisition** — `measure` runs once per visited position; the end-of-pass park is a pure `move`.
- **`measure(pos)` takes the commanded position** so the recorded `working_distance` is the requested value, not a re-read of (possibly drifted) hardware state.

### Caveats (bake into defaults)

- **Hill-climb assumes a unimodal curve.** Noise or a local dip triggers the greedy stop early. Natural config: **coarse = `SWEEP` (global), fine = `HILL_CLIMB` (fast local refinement)** — exactly the two-pass shape we already have.
- **Eval count is bounded** by `search_range / step_size` either way, so hill-climb never costs more than a sweep and usually far less.
- Always evaluating `center` first guarantees the result is no worse than "stay put."

### Testing

Strategies never touch hardware, so they unit-test against a synthetic score curve with fake `move`/`measure` closures (e.g. a Gaussian peaked at a known WD) — no microscope, no Qt. Assert hill-climb converges within `step_size` of the peak and visits ≤ `n_steps + 1` positions.

### UI (optional, later)

A small `QComboBox` as an extra control in each `_PassRowWidget` (the row infra already exists). Until then, default `SWEEP` keeps current behaviour and the field is editable only via the protocol/dict.

### Serialisation

`strategy` is a `(str, Enum)` field on `FocusSweepPass`, so `dataclasses.asdict` serialises it directly; `_sweep_pass_from_dict` reads it with a `SWEEP` default for backward compatibility.

---

## Alternatives considered

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Strategy granularity | **Per-pass** (`FocusSweepPass.strategy`) | Per-run (`AutoFocusSettings.strategy`) | Per-pass lets coarse stay a global sweep while fine refines via hill-climb — the common case. A per-run flag couldn't express that mix. |
| Strategy mechanism | **Plain functions over an `evaluate` callback** | Reinstate a strategy class hierarchy (the deleted `fm.strategy` ABC + generics + pydantic) | That layer added 460 lines and indirection with no real polymorphism. A function + enum value is the whole cost of a new strategy. |
| Iteration image type | **`np.ndarray`** | `FibsemImage` | The focus metric only needs the pixel array; carrying full metadata coupled the canonical type to FIB-SEM imaging and bloated `save`. |
| FM source compatibility | **Re-export canonical types from `fm.structures`** | Edit every `from fibsem.fm.structures import ...` call site | Re-export is a one-line change per type and keeps the migration reviewable; call sites stay untouched. |
| Cancellation plumbing | **`measure` raises; run fn catches** | Thread a `stop_event` / return-sentinel through every strategy | Keeps strategies pure and puts the single check at the single I/O point. |

## Known gaps / open questions

- **Tileset acquisition ignores the coarse pass.** `fm/acquisition.py` builds one `ZParameters` from the *last enabled pass* and reuses it for **all** modes (`ONCE`, `EACH_ROW`, `EACH_TILE`), so a configured coarse pass never runs during tiling — contradicting the code's own "coarse only initially" comment. Intended follow-up: `ONCE` → `run_coarse_fine_autofocus` (full passes); per-row/per-tile → fine pass only. Deferred deliberately (behaviour change).
- **`AutofocusWidget` channel sync.** `set_autofocus_settings` / `update_channels` emit `settings_changed` inconsistently and don't reconcile a `channel_name` that is `None` or no longer in the channel list (combo and model can disagree). Low impact; needs a channel to be removed/renamed mid-session.
- **Two focus-quality entry points.** FIB-SEM scores via `get_focus_measure_function(...)` on `filtered_data`; FM via `calculate_focus_quality(...)`. The driver unifies the *sweep*, but the scoring call still differs per domain — fine for now, but a candidate for a single `score(image, method)` once the driver lands.
- **Future strategies the framework enables** (each is one function + enum value): golden-section search, parabolic-fit refinement around the running best, adaptive step (halve `step_size` near the peak).

## Non-goals

- New focus *metrics* (the `FocusMethod` set is unchanged).
- Changing acquisition cadence or the autofocus *scheduling* modes (`AutoFocusMode`).
- A general optimiser framework — strategies are deliberately small, 1-D, bounded searches over working distance.

---

## Scope summary

| Change | Files | Effort |
|---|---|---|
| `FocusStrategy` enum + `FocusSweepPass.strategy` + serialisation | `autofunctions/autofocus.py` | Small |
| `_run_pass` + `_sweep`/`_hill_climb` driver | `autofunctions/autofocus.py` | Small |
| FIB-SEM `run_auto_focus` → use driver | `autofunctions/autofocus.py` | Small |
| FM `run_coarse_fine_autofocus` → use driver | `fm/calibration/__init__.py` | Small |
| Strategy unit tests | `tests/...` | Small |
| Per-pass strategy combo (optional) | `ui/fm/widgets/autofocus_widget.py` | Medium |
