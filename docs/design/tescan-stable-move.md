# Tescan Sample-Plane Stage Movement (`stable_move`)

## Problem

On FIBSEM systems the sample sits on a pre-tilted shuttle above the stage, so the
sample plane is not parallel to the stage plane. A `stable_move` is a move along
the **sample plane** (so the feature stays centered and in focus/eucentric), which
requires converting an image-space displacement into stage-axis moves.

The conversion depends on where the stage translation axes sit relative to the
tilt axis, and this differs between vendors:

- **ThermoFisher**: x/y/z sit **above** the tilt axis. The y/z axes rotate with
  stage tilt — z is always perpendicular to the stage plane, y always lies in it.
- **Tescan**: z sits **below** the tilt axis. The translation axes are fixed in
  the **chamber frame** — z always moves vertically regardless of stage tilt.

The Thermo math lives in `ThermoMicroscope._y_corrected_stage_movement`
(`fibsem/microscope.py`). The Tescan implementation was a placeholder
(`x=-dx, y=-dy, z=0`), which is only correct when the sample is flat to the SEM.
This doc derives the correct Tescan equations.

## Conventions and notation

All angles in radians unless noted.

| Symbol | Meaning | Source |
|---|---|---|
| `θ` | current stage tilt | `get_stage_position().t` |
| `α` | corrected pre-tilt = `PRETILT_SIGN × shuttle_pre_tilt` | `system.stage.shuttle_pre_tilt`; sign flips when rotated 180° to face the FIB |
| `η` | beam column tilt from vertical | 0° (SEM), 55° (Tescan FIB) |
| `φ` | **sample inclination** relative to chamber horizontal | `φ = θ − α` |
| `dy` | image-space displacement along image y | input |
| `d` | true distance along the sample plane | derived |

Chamber frame: z up (toward the SEM column), y horizontal, tilt is a rotation
about the x-axis.

**Convention check** (against known ThermoFisher values, which use the same
`corrected_pretilt_angle` definition): with a 35° pre-tilt shuttle, `φ = 0`
(flat to SEM) at `θ = 35°`; rotated 180° to the FIB (`α = −35°`), flat-to-ion
(`φ = 52°`) at `θ = 17°`. Both match the Aquilos reference values, so
`φ = θ − α` is consistent with the existing codebase conventions.

## Derivation

The conversion is two independent steps. Step 1 is identical for both vendors;
only step 2 differs.

### Step 1 — perspective correction (image plane → sample plane)

The image displacement `dy` is the projection of the true sample-plane distance
`d` onto the plane perpendicular to the viewing beam. The angle between the
sample plane and the beam-perpendicular plane is `(φ − η)`:

```
d = dy / cos(φ − η)          # η = 0 for SEM, 55° for FIB
```

This is the same as Thermo's
`y_sample_move = expected_y / cos(stage_tilt + perspective_tilt_adjustment)`.

### Step 2 — decompose `d` into stage axes

A unit vector along the sample-plane y-direction, expressed in the chamber
frame, is `(0, cos φ, sin φ)`.

**Thermo** (y/z rotate with the stage): express the same vector in the tilted
stage frame — the stage tilt θ cancels and only the shuttle offset remains:

```
y_stage = d · cos(α)
z_stage = −d · sin(α)
```

**Tescan** (y/z fixed in the chamber frame): no cancellation — decompose along
the full chamber-frame inclination. Tescan's +z increases **downward** (away
from the SEM column), opposite to Thermo RAW, hence the negated z:

```
y_stage =  d · cos(φ) =  d · cos(θ − α)
z_stage = −d · sin(φ) = −d · sin(θ − α)
```

The two formulations agree at `θ = 0`, where the frames coincide.

### Combined Tescan equations

```
φ  = θ − α
d  = dy / cos(φ − η)

y_stage =  dy · cos(φ) / cos(φ − η)
z_stage = −dy · sin(φ) / cos(φ − η)
x_stage = dx                            # tilt is about x; x passes through
```

### Sanity check — SEM case

For the vertical SEM beam (`η = 0`) the cosines cancel:

```
y_stage = dy            (exactly, at any tilt)
z_stage = −dy · tan(φ)
```

This is intuitively right for a chamber-fixed y-axis: sliding the sample
horizontally under a vertical beam maps 1:1 to image coordinates regardless of
tilt. The entire correction is the z move that keeps the tilted sample surface
at eucentric height / in focus. It also shows why the old placeholder
(`y=-dy, z=0`) worked when the sample was flat to the SEM (`φ = 0`).

### Vertical move (coincidence correction)

A coincidence correction is a purely vertical chamber move. On Tescan this is a
pure z move (z is chamber-vertical — no y compensation needed, unlike Thermo).
The FIB-image displacement `dy` projects onto the vertical with factor
`sin(η)`:

```
dz = dy / sin(η) = dy / cos(90° − η)     # η = 55° → dz ≈ dy / 0.819
```

This mirrors Thermo's `dy / cos(90° − column_tilt)` perspective factor. The
previous Tescan implementation used `dz = dy` (factor omitted).

### Inverse (stage move → image displacement)

Used by `project_stable_move` consumers (minimap, tiled imaging) to invert
positions. From the forward equations:

```
d  = y_stage / cos(φ)     (or −z_stage / sin(φ), whichever component is larger)
dy = d · cos(φ − η)
```

The `−z_stage / sin(φ)` branch only runs once `|φ| > 45°` — reachable in the
ion-facing orientation, where the pre-tilt sign flips and `φ = θ + α`. It
carries the z sign independently of the `y_stage` branch, so both must be
updated together whenever the z convention changes; `test_inverse_uses_sin_branch_past_45_degrees`
pins it.

The inverse functions take **raw stage deltas** (`pos − base_position`), the
same contract as the Thermo versions. Since `stable_move` applies the empirical
stage-axis inversion (`y_stage = −y_chamber`), the Tescan inverse undoes it
internally (`dy = −dy`) before applying the equations above. Callers that also
use `delta.x` must apply the matching x inversion themselves (see
`calculate_reprojected_stage_position2`).

## Code mapping

All in `fibsem/microscopes/tescan.py` (`TescanMicroscope`):

| Function | Change |
|---|---|
| `_y_corrected_stage_movement` | Rewritten with the chamber-frame decomposition above. Returns the physical chamber-frame move, before any stage-axis inversion. |
| `stable_move` | Uses `_y_corrected_stage_movement`; keeps the pre-existing empirical `x=-dx, y=-…` stage-axis inversion, applied after the trig so z stays independent. |
| `project_stable_move` | Implemented as the pure-math equivalent of `stable_move` (mirrors Thermo). |
| `vertical_move` | Adds the `1/sin(η)` FIB perspective factor. |
| `_inverse_y_corrected_stage_movement` | Implemented as the inverse above (raw stage deltas in, image dy out). |

And in `fibsem/imaging/tiled.py`:

| Function | Change |
|---|---|
| `_inverse_y_corrected_stage_movement` | Dispatches to the Tescan version when `image.metadata.system.info.manufacturer == "Tescan"`. |
| `_inverse_y_corrected_stage_movement_tescan` | New: standalone Tescan inverse computed from image metadata (mirror of the microscope method). |
| `calculate_reprojected_stage_position2` | Applies the Tescan x-axis inversion to `delta.x`. |

Note: `get_scan_rotation`/`set_scan_rotation` use **radians** (codebase-wide
convention, matching Thermo and the image-metadata parser), converting at the
Tescan API boundary, which is in degrees. All `isclose(scan_rotation, np.pi)`
checks are therefore consistent with the value stored in image metadata by
`get_beam_settings`. (Historically `get_scan_rotation` returned degrees while
`get_beam_settings` persisted that degrees value into metadata consumed as
radians by `calculate_reprojected_stage_position2` — at 180° scan rotation the
click projection flipped but the reprojection did not, placing added minimap
positions at the opposite corner.)

## Hardware verification

The trigonometry is derived, but three sign conventions could not be confirmed
from the codebase alone. One is now settled; the other two remain flagged in
code with `TODO(hardware-verify)` comments:

1. ~~**Tescan +z direction**~~ — **RESOLVED on hardware, 2026-07-23.** Tescan +z
   increases **downward** (away from the SEM column), opposite to Thermo RAW, so
   `z_move` is negated in `_y_corrected_stage_movement` and `vertical_move`.
   Flipping z alone corrected the move, which also confirms the tilt *magnitude*
   model — note that flipping z and flipping the tilt angle are **not**
   equivalent: negating z changes only the vertical component, whereas negating φ
   also rescales `y` through `d = dy/cos(φ − η)` (the `η` offset breaks the
   evenness of `cos`). They coincide only when `η = 0`.
2. **Tilt sense** — assumes positive stage tilt tips the sample toward the FIB
   (same sense as Thermo). If opposite, use `−stage_tilt` in `φ`.
3. **The empirical `x=-dx, y=-dy` inversion** — preserved from the previous
   implementation as a stage-axis orientation, applied to x/y *after* the trig
   (z is a physically independent sign). It may instead have been a scan
   rotation artifact.

**Suggested test**: at a known pre-tilt with the stage tilted (φ ≠ 0), perform a
`stable_move` on a feature in the SEM. The feature must stay centered **and in
focus**. A wrong z sign shows up as a focus/eucentric error of `2·dy·tan(φ)`;
a wrong y inversion moves the feature the wrong way. Repeat in the FIB view to
check the perspective factor, and run `vertical_move` from the FIB view to
verify the coincidence factor and z sign.
