# Odemis FM Driver — Findings, Update Plan & Testing

**Status:** analysis complete, implementation not started
**Date:** 2026-07-21
**Scope:** `fibsem/fm/odemis.py` (unchanged since PR #90), compared against `fibsem/fm/autoscript.py`, `fibsem/fm/microscope.py` (base ABCs), and the FM UI/workflow consumers.
**Verified against:** [delmic/odemis](https://github.com/delmic/odemis) master @ `b83df59` (2026-07-20). Source references below are paths within that repo (`src/odemis/...`). Target scopes run the latest odemis (≥ 3.8, per D6), so master is the correct reference.

## Context

`OdemisFluorescenceMicroscope` was written alongside the original Arctis FM support and predates the workflow layer (coincident milling tasks, movement guards), the objective-control UI, and the polish applied to the ThermoFisher driver. The architecture is sound — component ABCs, live reads from hardware VAs, `MD_FAV_POS_ACTIVE/DEACTIVE` for insert/retract (the same convention odemis's own METEOR posture manager uses, `acq/move.py:1319`) — but four findings break core functionality outright and several more are safety or parity gaps.

## Findings

### P0 — broken functionality (confirmed)

**F1. Wavelength setters always select the lowest available wavelength (unit mismatch).**
`available_excitation_wavelengths` converts choices to nm (`odemis.py:474`), but the setter converts the incoming nm value to metres before comparing against those nm values (`odemis.py:526-528`): `abs(365 − 5.5e-7)` < `abs(635 − 5.5e-7)`, so the smallest wavelength always wins. Identical bug in the emission setter (`odemis.py:589-593`). Every `set_channel()` therefore configures the wrong filter band.
*Verified:* stream `excitation`/`emission` VAs are `unit="m"` (`acq/stream/_live.py:1106,1112`); spectra are 5-tuples in metres, centre at index 2 (`driver/simulated.py:45-55`).
*Fix:* build a `{nm: choice}` mapping in one pass and select by nm distance — no double conversion, and it removes the fragile two-separate-iterations-of-a-set index mapping (`exc_choices = set(...)`, `_live.py:1099`).

**F2. `objective.state` is a method returning non-standard values.**
Base + TFS define `state` as a property returning `Literal["Inserted", "Retracted", "Busy", "Error", "Other"]`; Odemis defines a *method* returning `"INSERTED"/"RETRACTED"/"UNKNOWN"` (`odemis.py:217`). Every consumer compares `fm.objective.state == "Inserted"`: workflow tasks (`mill_coincident.py:156`, `acquire_fluorescence.py:190`, `select_fluorescence_position.py:58`, `manager.py:152`), the movement guard (`fibsem/microscope.py:2101`), `FMAcquisitionWidget.py:790`, `FibsemMovementWidget.py:316`, `objective_control_widget.py:243`. On Odemis this compares a bound method to a string → always `False`: the system permanently believes the objective is retracted; coincident-milling tasks refuse to run.
*Fix:* convert to `@property`, return the base Literal values, keep the FAV-position comparison with a ±100 µm tolerance (use `"Other"` for in-between, matching the base Literal).

**F3. Camera geometry double-counts binning.**
`_pixel_size`/`_resolution` are cached once at init (`odemis.py:272-273`) and the base properties then scale by current binning. The odemis backend already does that: `MD_PIXEL_SIZE = sensor_px × binning / magnification`, re-published live on every binning or magnification change (`odemisd/mdupdater.py:204-235` — "we compute PIXEL_SIZE every time the LENS_MAG *or* binning change"), and camera drivers rescale the `resolution` VA when binning changes (`driver/simcam.py:_setBinning` — "adapt resolution so that the AOI stays the same"). Correct only if binning happened to be 1 at init; a leftover binning from a previous session corrupts pixel size, FOV, and the stage-movement math in `acquisition.py:443`.
*Fix:* override `pixel_size` to read `getMetadata()[MD_PIXEL_SIZE]` live and `resolution` to read `resolution.value` live, with **no** binning arithmetic. (Each read is a Pyro round-trip; if profiling ever cares, cache with invalidation via `binning.subscribe` — start with the simple correct version.)

**F4. Power units: fraction vs watts.**
`ChannelSettings.power` is a 0–1 fraction (UI shows %, `channel_settings_widget.py:104`); TFS maps it to brightness 0–1. Odemis `stream.power` is in **watts** (`FluoStream.power` inherits `unit=emitter.power.unit`; Light emitters declare `unit="W"`, `driver/simulated.py:50`). `power=0.01` means 0.01 W — wrong illumination, and because `power.clip_on_range = True` (`_live.py:1120`) out-of-range values **clip silently** rather than raise, so there's no error to notice.
*Fix:* normalise fraction ↔ watts through `stream.power.range` in the driver (setter: `frac × range[1]`; getter returns fraction so metadata stays cross-driver comparable). See decision D1.

### P1 — robustness & safety

**F5. `move_absolute` ignores the user safety limit.**
Base and TFS clip to `limit_position` (`microscope.py:170`, `autoscript.py:169-176`); Odemis calls `moveAbs` directly (`odemis.py:180-193`). The objective-limit spinbox (`objective_control_widget.py:383`) — the anti-crash-into-sample protection — has no effect on Odemis hardware. Also no pre-clip to the axis range: odemis raises `ValueError` on out-of-range moves (`model/_components.py:939-956`), surfacing as unhandled exceptions from UI values.

**F6. `limits` not overridden.**
Inherits sim constants (−12 mm, +10 mm); the objective widget uses them for spinbox ranges (`objective_control_widget.py:120`). Should be `self._focuser.axes["z"].range`.

**F7. Init fragility → whole FM silently disabled.**
`self._focus_position = self.active_position["z"]` (`odemis.py:50`) raises `TypeError` when `MD_FAV_POS_ACTIVE` is absent; `camera_md[MD_PIXEL_SIZE]` / `[MD_BASELINE]` (`odemis.py:273-274`) raise `KeyError` — **MD_BASELINE is only published by three camera drivers** (tucsen, andorcam3, simcam). Any failure is swallowed by `odemis_microscope.py:266` into `fm = None` with a log line. Use `.get()` fallbacks; only hard-fail on components that are genuinely required.

**F7b. `add_odemis_path()` reads `/etc/odemis.conf` unguarded** (`odemis_microscope.py:38-58`) at import time — `FileNotFoundError` on any machine without it (dev boxes, Windows support machines), which also blocks unit-testing the driver module. Wrap in try/except and skip silently when absent. *Prerequisite for the unit-test plan.*

**F8. `acquire_image` returns `None` on failure** (`odemis.py:299-301`) despite `-> np.ndarray`; the base then builds `FluorescenceImage(data=None)` and crashes far from the cause. Re-raise instead (TFS lets exceptions propagate).

**F9. Emission setter crashes on `"Fluorescence"`.**
`ChannelSettings.emission_wavelength: Optional[Union[str, float]]` — TFS-side channel files legitimately contain `"Fluorescence"`. The Odemis emission setter has no type guard, so a str hits `value *= 1e-9` → `TypeError`.
*Fix (D3 resolved):* map str values to the emission band matching the current excitation via `odemis.util.fluo.get_one_band_em(bands, ex_band)` (`util/fluo.py:53`) — the same helper family FluoStream uses internally for its excitation↔emission guessing. Follow-up (out of scope here): expose explicit per-band emission selection to the user instead of closest-match magic.

### P2 — parity gaps vs the TFS driver

**F10. No live/fast acquisition path.** The base worker runs a full `acqmng.acquire()` per frame: per-frame future creation, stream activate (filter setup + light **on**), one frame, deactivate (light **off**). Seconds per frame plus LED toggling. Design below (§ Live acquisition).

**F11. Camera `gain` is a silent no-op.** No override → writes land on the inherited sim `_gain`; `camera_widget.py:73` displays it and `get_metadata` records it, but hardware never changes. Wire to the camera's `gain` VA when present (`model.hasVA`), else warn-once no-op (see decision D4).

**F12. Binning validation hardcoded** to `[1, 2, 4, 8]` instead of the camera's `binning.range` (simcam supports up to 16). Add `available_binnings` (powers of two within range) for TFS parity.

**F13. No `exposure_time_limits`.** Available via `exposureTime.range` (FloatContinuous, `simcam.py:128`).

**F14. `power_limits` is a method on Odemis (`odemis.py:414`) but a property on TFS (`autoscript.py:450`)** — the base ABC defines neither, which is why they diverged. Promote into the ABC (see decision D5).

### P3 — typing & cleanup

- **F15.** `parent: "Client"` forward refs on `OdemisObjectiveLens`/`OdemisCamera` (`odemis.py:32,257`) — no `Client` class exists; should be `"OdemisFluorescenceMicroscope"`.
- **F16.** `focuser: model.Actuator = None`, `light_source: model.Emitter = None` → `Optional[...]`; `OdemisFilterSet.__init__(parent=None)` untyped; `deactive_position` missing return annotation.
- **F17.** `available_*_wavelengths` annotated `Tuple[float, ...]` but return lists; emission list contains `None` → `List[Optional[float]]` in practice.
- **F18.** `OdemisCamera.offset` overrides the property without a setter, killing the inherited setter (latent — nothing assigns it today).
- **F19.** Dead code: `closest_* is None` checks after `min()` can never fire; `_excitation_wavelength = None` shadows a base attribute typed `float`.
- **F20.** `acquire_image` actually returns `model.DataArray` (ndarray subclass carrying `.metadata`) — fine at runtime; worth using that metadata (see Phase 3).

## Verified odemis behaviour (reference)

| Fact | Source (delmic/odemis) |
|---|---|
| `excitation`/`emission` VAs in metres; excitation choices = `set(emitter.spectra.value)` of 5-tuples (centre = index 2); emission choices from filter `band` axis; `BAND_PASS_THROUGH = "pass-through"` | `acq/stream/_live.py:1096-1113`, `driver/simulated.py:45-55`, `model/_metadata.py:135` |
| `stream.power` = per-channel `FloatContinuous`, `unit="W"`, `clip_on_range=True` (silent clip, no raise); `.range` re-slices per selected excitation channel | `acq/stream/_live.py:1116-1121, 1140-1145`; `driver/simulated.py:50` |
| VAs only push to hardware while the stream is **active**; `_onActive(True)` runs `_setup_emission()` then `_setup_excitation()`; deactivate calls `_stop_light()` — so set-VAs-then-acquire is valid, settings land at acquisition time | `_live.py:1126-1155` |
| `MD_PIXEL_SIZE = sensor_px × binning / mag`, live-updated on binning/mag changes; `MD_BINNING`, `MD_LENS_MAG` alongside | `odemisd/mdupdater.py:204-235` |
| Camera `resolution` VA rescales when binning changes (AOI preserved) | `driver/simcam.py:_setBinning`; same in andorcam3 etc. |
| `MD_BASELINE` published only by tucsen / andorcam3 / simcam | grep `driver/` |
| `acqmng.acquire(streams)` → future → `(list[DataArray], Exception | None)` | `acq/acqmng.py:56-72` |
| `moveAbs` raises `ValueError` outside `axes[*].range` | `model/_components.py:939-956` |
| `MD_FAV_POS_ACTIVE/DEACTIVE` = engage/park convention, read unguarded by odemis's own METEOR posture manager (guaranteed by a correct microscope file) | `model/_metadata.py:220-221`, `acq/move.py:1319` |
| `is_active` drives acquisition (dataflow subscribe); `should_update` is a GUI-only flag ("no direct effect") | `_base.py:180-185`, `_live.py:110-130` |
| `LiveStream.single_frame_acquisition` VA: activate → one frame via `getSingleFrame()` → auto-deactivate | `_live.py:57,122,224` |
| `DataFlow.get(asap=False)` returns a frame guaranteed acquired after the call | `model/_dataflow.py:340` |
| Stream image projection throttled to ~10 Hz in a background thread (histogram + RGB compute per frame while active) | `_base.py:1064-1100` |

## Live acquisition design (F10)

The base worker already exposes the seam: `_acquisition_worker` checks `hasattr(self.camera, "_start_fast_acquisition")` (`fibsem/fm/microscope.py:921`) — the same hook the TFS driver uses. Implement it odemis-natively: activate the stream once, subscribe a raw-frame listener to the camera dataflow (dataflows are multi-listener), and let frames push at the exposure-limited rate.

```python
def _start_fast_acquisition(self):
    stream = self._stream

    def on_frame(dataflow, data):
        self.parent._construct_image(data)   # emits acquisition_signal, rate-limited by base

    try:
        stream.is_active.value = True        # light on, filters set once, continuous mode
        self._camera.data.subscribe(on_frame)
        self.parent._stop_acquisition_event.wait()
    finally:
        self._camera.data.unsubscribe(on_frame)
        stream.is_active.value = False       # light off, camera stops
```

Notes:
- Frame pacing handled by the existing `_rate_limit` in `_construct_image`.
- Exposure/binning VAs are writable during live; drivers apply next frame. Binning changes frame shape mid-stream — display must tolerate.
- Frames arrive on an odemis driver thread → psygnal, same threading shape as the TFS fast path.
- Light stays on for the whole session (fast, no LED churn; photobleaching is user-controlled via start/stop, same tradeoff as the odemis GUI).
- **Snapshot during live:** with the stream active, `camera.data.get(asap=False)` returns a fresh frame without stopping the live stream — removes the stop/acquire/restart dance for workflows that grab stills while streaming.
- Single-shot workflow acquisitions stay on `acqmng.acquire` (correct ordering, metadata observer, cancellable future). `single_frame_acquisition` exists but is clunkier (no future) — not worth adopting.
- Accepted inefficiency: the active stream computes histogram + RGB projections (~10 Hz throttle) that fibsem never reads. If profiling cares, subclass `FluoStream` with a no-op `_updateImage`.

## Implementation plan

Ordered so each phase is a small, independently reviewable PR; P0s first. F7b lands first because the unit tests depend on it.

**Phase 1 — correctness (P0): F7b, F1, F2, F3, F4**
1. Guard `add_odemis_path()` (F7b) — unblocks unit testing.
2. Wavelength setters: one-pass `{nm: choice}` mapping for excitation and emission; keep closest-match semantics; drop dead `is None` checks (F19 partial).
3. `state` → property, base Literal values, ±100 µm tolerance, `"Other"` between positions.
4. `pixel_size`/`resolution` live-read overrides (no binning arithmetic).
5. Power fraction↔watts normalisation via `stream.power.range` (per D1).
- *Acceptance:* unit suite below passes; a saved multi-channel `ChannelSettings` list round-trips through `set_channel` selecting the correct bands on a stubbed stream.

**Phase 2 — robustness (P1): F5, F6, F7, F8, F9**
1. `limits` from `focuser.axes["z"].range`; `move_absolute` clips to user `limit_position` **and** axis range (warn on clip, matching base behaviour).
2. Init `.get()` fallbacks: `MD_BASELINE → 0`; missing `MD_FAV_POS_ACTIVE` → fall back to current position with a clear warning (don't kill the driver).
3. `acquire_image` raises on failure.
4. Emission setter: map str values via `fluo.get_one_band_em` (D3).

**Phase 3 — live acquisition (P2): F10 (+ F20 metadata)**
1. `_start_fast_acquisition` as designed above.
2. Optional: `_construct_image` consumes `DataArray.metadata` (authoritative per-frame `MD_PIXEL_SIZE`, `MD_ACQ_DATE`, `MD_EXP_TIME`) when present.
3. Optional: snapshot-during-live helper using `data.get(asap=False)`.
- *Acceptance:* live view on the sim backend streams at exposure-limited rate (not seconds/frame); start/stop leaves light off and stream inactive, including on exceptions.

**Phase 4 — parity & typing (P2/P3): F11–F19 (+ carried items)**
1. Gain wiring via `model.hasVA(camera, "gain")` (per D4): getter returns None without a hardware VA (metadata stays honest — `FluorescenceChannelMetadata.gain` becomes `Optional[float]`, camera widget disables the spinbox); setter is a warn-once no-op.
2. `available_binnings` + `exposure_time_limits` from the camera VAs (choices when enumerated, powers of two within range otherwise); binning validation against them.
3. Promote `power_limits` / `exposure_time_limits` / `available_binnings` into the base ABCs with sim defaults; the base binning setter validates against `available_binnings` (replacing the old `raise Warning`); TFS already conforms (no autoscript changes needed — the promotion formalizes its existing surface).
4. Annotation sweep (F15–F18), remove dead attributes (F19); `offset` stays a read-only property by intent (documented).
5. Per-instance acquisition state: `_stop_acquisition_event` / `_acquisition_thread` move from shared class attributes into `__init__`. (The same pattern exists on `FibsemMicroscope` in `fibsem/microscope.py:154` — out of FM scope, tracked separately.)
6. F20 (carried from Phase 3): `_construct_image(data, frame_metadata=None)` — per-frame values (`pixel_size`, `acquisition_date`, `exposure_time`) override the state snapshot; the odemis microscope overrides `_construct_image` to auto-extract them from `DataArray.metadata`, so both single-shot and live paths stay correctly stamped even when settings change mid-stream.

## Testing

### Unit tests (no hardware, run everywhere) — `tests/fm/test_odemis_driver.py`

`fibsem.fm.odemis` imports odemis at module level, so tests inject stub modules into `sys.modules` **before** first import (fixture in `tests/fm/conftest.py`, alongside the existing sim-arctis session fixture): `odemis`, `odemis.model`, `odemis.acq`, `odemis.acq.acqmng`, `odemis.acq.stream`, `odemis.util.dataio`. Stubs needed: `FakeVA` (value/choices/range/subscribe), `model.getComponent(role=...)` returning stub focuser/ccd/light/filter, a stub `FluoStream` carrying `excitation`/`emission`/`power` VAs, `acqmng.acquire` returning a fake future. Requires F7b so importing `fibsem.microscopes.odemis_microscope` doesn't open `/etc/odemis.conf`.

Cases (regression tests for each P0/P1):
- **F1:** choices = four 5-tuples in metres → `excitation_wavelength = 635` selects the 635 nm tuple (not 365); same for emission bands; `emission_wavelength = None` → `BAND_PASS_THROUGH`; round-trip getter returns nm.
- **F2:** position vs FAV metadata → `"Inserted"` / `"Retracted"` / `"Other"`, values ∈ base Literal; `state == "Inserted"` comparison works (property, not method).
- **F3:** stub `MD_PIXEL_SIZE` and `resolution.value`; change binning on the stub (updating both, as the backend does) → `pixel_size`/`resolution`/`field_of_view` correct, no double count; init at binning 2 also correct.
- **F4:** `power.range = (0, 0.4)` → `set_power(0.5)` sets 0.2 W; getter returns 0.5; metadata records the fraction.
- **F5/F6:** `limits` == stub axis range; `move_absolute` beyond `limit_position` clips + warns; beyond axis range never reaches `moveAbs`.
- **F7:** missing `MD_BASELINE` / `MD_FAV_POS_ACTIVE` → driver constructs, sensible fallbacks, warning logged.
- **F8:** stub acquire failure → raises (no `None` return).
- **F9:** `emission_wavelength = "Fluorescence"` → selects the band matching the current excitation (stub `get_one_band_em`), no `TypeError`.
- **F10:** `_start_fast_acquisition` subscribes + activates; pushed frames emit `acquisition_signal`; setting the stop event unsubscribes + deactivates; teardown also runs when the callback raises.
- **F11–F13:** gain maps to VA when present / warns when absent; `available_binnings` ⊆ range; `exposure_time_limits` == VA range.

### Integration tests (odemis sim backend, Linux only, opt-in marker)

odemis ships METEOR sim configs (`install/linux/usr/share/odemis/sim/meteor-sim.odm.yaml`, `meteor-tfs3-sim.odm.yaml`, ...). With a backend running (`odemis-start .../meteor-sim.odm.yaml`), mark tests `@pytest.mark.odemis` (skipped unless backend reachable). These exercise the *real* VA semantics and the backend-side MetadataUpdater — the things stubs can't prove:
- set binning → `pixel_size` and `resolution` track correctly (validates F3 against the real updater);
- excitation/emission setters land the requested band on the stream;
- `acquire_image` returns a uint16 frame with plausible metadata;
- insert/retract transitions `state` through the Literal values;
- live acquisition: ≥ N frames in T seconds at a given exposure (catches any per-frame re-setup regression); light off after stop.

### On-hardware checklist (METEOR)

1. `odemis --version` → check out matching tag in the clone, re-verify F1/F3/F4 semantics (all long-stable core API, expected no drift).
2. Camera model → confirms `MD_BASELINE` presence (Zyla/andorcam3 publishes it; others need the F7 fallback).
3. Light `power.range` per channel → sanity-check the fraction→W mapping against what users expect from the odemis GUI (which displays watts — see D1).
4. `MD_FAV_POS_ACTIVE/DEACTIVE` present in the microscope file; insert/retract from the fibsem objective widget; state chip correct at both ends and mid-travel.
5. Objective limit spinbox actually clamps a commanded overshoot (F5).
6. Restart-persistence: leave binning 4, restart fibsem → pixel size/FOV still correct (F3).
7. Channel correctness: acquire each configured channel on a fluorescent test sample; confirm the expected filter wheel/LED behaviour per channel (F1).
8. Live view: frame rate ≈ 1/exposure; exposure change applies within a frame or two; stop turns the light off; filter-wheel setup happens once at start, not per frame.
9. FOV cross-check against a feature of known size (grid bar) — validates the full F3 pixel-size chain.
10. Workflow smoke: coincident-milling task precondition (`state == "Inserted"`) passes with the objective inserted (F2).

Per repo convention (manual GUI harnesses live outside `tests/`), add a small on-scope smoke script covering 4–8 (insert → per-channel acquire → 10 s live → snapshot-during-live → retract, printing timings) rather than pytest.

## Decisions

Resolved with Patrick, 2026-07-21 (D5 still open):

- **D1 — power semantics. RESOLVED: fraction 0–1.** `ChannelSettings.power` stays a 0–1 fraction everywhere (UI %); the odemis driver maps fraction↔watts via `power.range`. Note the fraction re-scales if `power.range` changes per excitation channel — acceptable, since range is re-sliced per channel (`_live.py:1140-1145`).
- **D2 — metadata honesty. RESOLVED: record the fraction.** `FluorescenceChannelMetadata.power`/`gain` record the driver-normalised fraction, not raw hardware units, for cross-driver comparability. Raw watts remain available via the odemis `DataArray.metadata` if Phase 3 adopts it.
- **D3 — `"Fluorescence"` emission on odemis. RESOLVED: map it.** Map str values to the emission band matching the current excitation via `fluo.get_one_band_em(bands, ex_band)` (`util/fluo.py:53`). **Follow-up:** users should eventually get explicit control over emission-band selection (per-band choice in the channel UI rather than closest-match inference) — tracked as future work, not part of this update.
- **D4 — gain without a hardware VA. RESOLVED: warn-once no-op.** Report the hardware value only when a `gain` VA exists (else omit/None in metadata) — never echo back a value that didn't reach hardware.
- **D5 — ABC promotion timing. RESOLVED: Phase 4.** "Promotion" = adding the members the drivers invented ad-hoc (`power_limits`, `exposure_time_limits`, `available_binnings`) to the base ABCs in `fibsem/fm/microscope.py` with simulator defaults, and re-declaring `state` there so its property-ness and Literal values are enforced by the contract rather than by convention — the absence of these from the ABC is why TFS (property) and Odemis (method) drifted apart (F14). Doing it in Phase 4 keeps Phases 1–3 pure-odemis diffs. Note: the ABC also becomes the wire contract for remote control (§ below), which raises its importance.
- **D6 — odemis version. RESOLVED: latest (≥ 3.8).** Scopes track the latest odemis, so master @ `b83df59` is the verification reference; everything cited (VA units, MetadataUpdater, FAV_POS, `clip_on_range`, `single_frame_acquisition`) is long-stable core API.

## Remote control from the support PC (Phase 5, future)

**Goal:** the METEOR (odemis) runs on its own Linux PC; the FIB/SEM `FibsemMicroscope` (Autoscript) runs on the Windows support PC. Support controlling the METEOR from the support PC as well as locally.

### Constraints (verified)

- **The odemis backend is unreachable over the network.** Its Pyro4 daemons bind to Unix domain sockets under `/var/run/odemisd` (`model/_core.py:311` — `Pyro4.Daemon.__init__(self, unixsocket=...)`). No TCP option; and odemis isn't installable on Windows anyway. → `OdemisFluorescenceMicroscope` **must** run on the Linux box; remote control means a bridge service there plus a client on Windows.
- **The site already runs this kind of bridge — in the opposite direction.** Delmic's xtadapter runs on the Windows PC and odemis connects to it via `Pyro5.api.Proxy(f"PYRO:Microscope@{address}:{port}")` (default port 4243) with msgpack_numpy for arrays (`driver/autoscript_client.py:109,132`). We'd add the mirror-image bridge for the FM hardware. Same trust model (unauthenticated service on the private instrument LAN).
- **odemis has no HTTP/REST layer** to piggyback on (no fastapi/flask anywhere in the tree).

### Architecture: boundary at the fibsem ABC

Put the network boundary at the `FluorescenceMicroscope` ABC, not at the odemis layer:

- **Server (Linux):** a thin service wrapping the (fixed) `OdemisFluorescenceMicroscope`, exposing the ABC surface. Runs as a systemd unit next to odemisd (`After=odemisd`), binds to the instrument-LAN interface only.
- **Client (Windows):** `RemoteFluorescenceMicroscope(FluorescenceMicroscope)` — a proxy implementing the same ABC by forwarding calls. Everything above the ABC (FM widgets, `acquisition.py`, autofocus, workflow tasks) works unchanged.
- **The stage stays local.** On the support PC the stage/beams are driven by local Autoscript; only the FM components (objective, camera, light, filter) cross the network. That's exactly the ABC boundary — no split-brain stage control. (`RemoteFluorescenceMicroscope.parent` is the local Autoscript `FibsemMicroscope`, so `get_metadata()` stage positions and orientation checks are local and authoritative.)
- This is why D5 (ABC promotion) matters beyond tidiness: **the ABC is the wire contract.**

### Transport: FastAPI + one websocket (recommended)

| Option | Pros | Cons |
|---|---|---|
| **FastAPI REST + WS** (recommended) | Self-documenting (OpenAPI), curl-debuggable, clean streaming/eventing via one WS channel, Windows client needs only `httpx` + `websockets`, team-familiar | More code than Pyro |
| Pyro5 + msgpack_numpy (match xtadapter) | Least code; transparent object proxying; precedent on site | Push events/live frames need client-side Pyro daemon (callbacks) — clunky; version lock-step both ends |
| gRPC | Typed contract, first-class streaming | protoc toolchain ceremony; overkill for an internal lab tool unless non-Python clients are coming |

Design points:

1. **Control plane vs data plane.** REST for get/set/insert/retract/`set_channel`/single acquire; one WS for live frames, acquisition-progress events, and state-change pushes. Image frames as **binary** (raw uint16 + small JSON header: dtype, shape, channel metadata) — never JSON/base64 arrays. 4512² × uint16 ≈ 40 MB/frame; with typical binning 2–4 and the existing `_rate_limit` throttle, a few fps ≈ tens of MB/s — fine on the instrument GigE LAN (comparable to the Autoscript traffic odemis already puts on the same wire).
2. **Signals over the wire.** Server subscribes `acquisition_signal`/`acquisition_progress_signal` → WS messages; the client re-emits into local psygnal signals so widgets connect exactly as they do today.
3. **Live mode mapping.** `start_acquisition`/`stop_acquisition` map to explicit server endpoints driving the server-side `_start_fast_acquisition` (Phase 3). Client-side loops (z-stack, autofocus in `acquisition.py`/`calibration`) run unchanged through the proxy — per-step round-trip (~2–5 ms LAN) is negligible against 50–500 ms exposures.
4. **Chattiness.** Add a bulk `/state` endpoint (objective position + state, camera settings, filter, power) for UI polling instead of N property GETs.
5. **Concurrency/ownership.** The odemis backend is multi-client by design (odemis GUI + CLI + our server already coexist; VAs are last-writer-wins). Start with last-writer-wins plus a visible "connected clients" indicator; add an optional soft lease only if concurrent frontends bite in practice.
6. **Failure & safety.** Server-side watchdog: if the live-view consumer disconnects, stop acquisition (light **off**) after a short grace period — illumination must never stay on because a client vanished. Reconnect logic client-side; idempotent state reads; jobs (if any) carry IDs. No retract-on-disconnect (objective position isn't a hazard by itself; stage is local).
7. **Serialization.** `ChannelSettings`/`ZParameters`/metadata already have `to_dict`/`from_dict` — reuse as the JSON schema. `FluorescenceImage` = binary + metadata dict.
8. **Config & versioning.** `microscope-configuration.yaml`: `fm: {type: odemis-remote, host: ..., port: ...}` alongside the unchanged local `type: odemis`. A handshake endpoint reports fibsem + odemis versions to catch skew.
9. **Local use goes through the same server (recommended).** The Linux PC's own fibsem-os connects to `host: localhost` — the FM server becomes the single owner of the FM session (stream activation, live-acquisition state, light state), mirroring odemis's own daemon model (odemisd owns hardware; GUI/CLI are clients). Benefits: one state machine instead of two uncoordinated stacks (an in-process driver and a remote server activating independent `FluoStream`s would fight over light/filters with no visibility); the remote path gets exercised daily on the Linux box rather than rotting until a support-PC session; and live frames **fan out** — one acquisition, N websocket subscribers, so the Linux PC and support PC can watch the same live stream simultaneously. Loopback overhead is negligible (µs round-trips; frame serialization ≈ a memcpy at a few fps). The in-process `type: odemis` path remains for development/debugging, and the server itself consumes the driver in-process.
10. **Watchdog with multiple subscribers:** reference-count live-view subscribers; stop acquisition (light off) when the count reaches zero or the initiating client disconnects without handover — not merely when *a* client drops.
11. **Non-goal noted:** remote desktop to the Linux box is the zero-code alternative but doesn't give single-UI operation with the FIB on the support PC — which is the point.

### Extensibility to other devices

The server is **driver-agnostic by construction**: it wraps the `FluorescenceMicroscope` ABC, not the odemis driver. Any concrete implementation — a future non-odemis FM, a pymmcore/Micro-Manager-backed system, the simulator — can sit behind the identical server on whatever PC hosts its control software; the client proxy and config (`fm: {type: fm-remote, host, port}`) don't change, and the handshake reports what's behind the server. Extending to a new FM = write the driver (needed anyway) + define its safe-state action for the watchdog. A simulated FM behind the server doubles as a network-testable fake device.

For a different *kind* of device (non-FM), the same pattern applies but needs its own ABC written to the "remotable contract" rules the FM ABC (post-Phase 4) demonstrates: serializable arguments/returns (`to_dict` dataclasses, arrays, primitives — no live object handles over the wire); coarse-grained operations over property-twiddling; signals declared at the ABC level so the proxy can re-emit; explicit start/stop lifecycle for long-running work; and cross-device references resolved locally (the `parent`/stage pattern), never remoted. The transport/framing/watchdog/handshake infrastructure should be kept separable from the FM-specific surface so it can be reused — but don't build a generic device-server framework up front; extract the shared core when a second device type actually arrives.

### Testing (remote tier)

- Loopback: server + `RemoteFluorescenceMicroscope` on one machine over the stubbed driver — proxy fidelity per ABC member (the Phase 4 ABC contract doubles as the conformance checklist; run the same unit suite against the remote proxy).
- Sim-backend over a real network hop: live view fps, frame integrity (dtype/shape/metadata round-trip).
- Disconnect drills: kill the client mid-live → light off within the grace period; kill the server mid-workflow → client raises cleanly, UI recovers on reconnect.
