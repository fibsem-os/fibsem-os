# fibsemOS MCP Server

The fibsemOS MCP server exposes FIB/SEM microscope control as tools callable by Claude (via Claude Code or any MCP-compatible client). Claude can acquire images, move the stage, draw milling patterns, and run complete milling tasks — all by issuing natural-language instructions that the model translates into tool calls.

---

## Installation

```bash
# From the fibsem repo root:
pip install -e ".[mcp]"
```

The `mcp` extra installs the `mcp` package (FastMCP). All other fibsem dependencies are included in the base install.

---

## Configuring Claude Code

Add the server to your project's `.claude/settings.json` (or `~/.claude/settings.json` for global use):

```json
{
  "mcpServers": {
    "fibsem": {
      "command": "python",
      "args": ["-m", "fibsem.mcp.server"]
    }
  }
}
```

Restart Claude Code after editing settings. You should see `fibsem` listed under connected MCP servers. All tools appear with the prefix `mcp__fibsem__` in the tool list.

---

## Connection Modes

### Direct (in-process)

Claude calls the microscope drivers directly in the same process as the MCP server. Use this when Claude Code is running **on the microscope PC**.

```
connect_microscope(manufacturer="Demo")                        # offline testing
connect_microscope(manufacturer="ThermoFisher", ip_address="192.168.1.1")
connect_microscope(manufacturer="TESCAN", ip_address="192.168.1.2")
```

### Remote (via FibsemServer)

The microscope PC runs a lightweight HTTP server (`fibsem.server`). Claude connects to it over the network. Use this when Claude Code is on a **separate machine** from the microscope.

**On the microscope PC:**

```bash
python -m fibsem.server.server --manufacturer ThermoFisher --ip-address 192.168.1.1 --port 8001
```

The Swagger UI is available at `http://<microscope-pc>:8001/docs`.

**In Claude Code:**

```
connect_microscope_remote(host="192.168.1.100", port=8001)
```

---

## Tool Reference

### Connection

| Tool | Description |
|------|-------------|
| `connect_microscope` | Connect in-process (Demo / ThermoFisher / TESCAN) |
| `connect_microscope_remote` | Connect via FibsemServer REST API |

### State

| Tool | Description |
|------|-------------|
| `get_stage_position` | Current X/Y/Z/rotation/tilt (mm and degrees) |
| `get_microscope_state` | Full state: stage + both beams + detectors |
| `get_beam_settings` | Voltage, current, WD, HFW, scan rotation, shift, stigmation |
| `get_stage_orientation` | SEM / FIB / MILLING / FM / NONE |
| `get_milling_angle` | Current milling angle computed from stage tilt and column tilt |
| `get_milling_state` | IDLE / RUNNING / PAUSED / STOPPING / ERROR |

### Imaging

| Tool | Description |
|------|-------------|
| `acquire_image` | Acquire one image; returns text metadata + JPEG |
| `acquire_both_beams` | Acquire SEM + FIB pair; returns text + JPEG for each |
| `auto_focus` | Auto-adjust working distance |
| `autocontrast` | Auto brightness/contrast |

### Beam Control

| Tool | Description |
|------|-------------|
| `set_field_of_view` | Set HFW (magnification) in µm |
| `beam_on_off` | Turn beam on or off |
| `blank_unblank` | Blank / unblank beam to protect sample |
| `reset_beam_shifts` | Zero both beam shifts |

### Stage Movement

| Tool | Description |
|------|-------------|
| `move_stage` | Absolute move (mm / degrees); omit any axis to leave unchanged |
| `move_stage_relative` | Relative offset (mm / degrees) |
| `move_to_milling_angle` | Tilt stage to a target milling angle (degrees) |
| `move_flat_to_beam` | Orient sample flat/perpendicular to the chosen beam |
| `link_stage` | Synchronise stage Z with working distance (eucentric linking) |

### Milling — Primitives

Call these sequentially: **setup → draw → run → finish**.

| Tool | Description |
|------|-------------|
| `setup_milling` | Set ion beam current, voltage, application file, patterning mode |
| `draw_rectangle` | Draw a rectangle pattern (with optional cleaning cross-section) |
| `draw_line` | Draw a line pattern |
| `draw_circle` | Draw a circle or arc pattern |
| `clear_patterns` | Clear all patterns without restoring imaging conditions |
| `run_milling` | Execute current patterns (blocking or async) |
| `finish_milling` | Clear patterns and restore ion beam to imaging settings |
| `stop_milling` | Halt a running milling operation |
| `estimate_milling_time` | Estimated time for current patterns (seconds) |

### Milling — High-Level

| Tool | Description |
|------|-------------|
| `run_milling_task` | Run a complete multi-stage milling task from a JSON config (setup → draw → run → finish in one call) |

### Interactive

| Tool | Description |
|------|-------------|
| `pick_point` | Show last image in a window, user clicks a point; returns offset from centre in µm. Requires a graphical display (`$DISPLAY`). |

---

## Example Sessions

### 1 — Survey and navigate

```
"Connect to the Demo microscope, acquire an overview at 400 µm, then acquire a
higher-magnification image at 50 µm."
```

Claude will:
1. `connect_microscope(manufacturer="Demo")`
2. `acquire_both_beams(hfw_um=400, autocontrast=True)` → inspect SEM + FIB
3. `acquire_image(beam_type="ion", hfw_um=50)`

### 2 — Mill a rectangle

```
"Set up milling at 0.1 nA, draw a 5×2 µm rectangle centred 3 µm to the right,
estimate the time, then run it."
```

Claude will:
1. `setup_milling(milling_current_na=0.1)`
2. `draw_rectangle(width_um=5, height_um=2, centre_x_um=3)`
3. `estimate_milling_time()`
4. `run_milling(milling_current_na=0.1)`
5. `finish_milling()`

### 3 — Full milling task from JSON

```
"Run a polish pass: 60 pA, Si-ccs, two cleaning cross-section trenches
(9 µm wide, 0.7 µm height, 0.3 µm spacing, 1 µm deep) at 60 µm field of view."
```

Claude will build the `FibsemMillingTaskConfig` JSON and call:

```
run_milling_task(config_json='{
  "name": "Polish",
  "field_of_view": 60e-6,
  "stages": [{
    "milling": {"milling_current": 60e-12, "application_file": "Si-ccs"},
    "pattern": {
      "name": "Trench",
      "width": 9e-6, "depth": 1e-6,
      "upper_trench_height": 0.7e-6,
      "lower_trench_height": 0.7e-6,
      "spacing": 0.3e-6
    }
  }]
}')
```

---

## Lamella Agent Workflow

This section describes how Claude can autonomously prepare a cryo-lamella using only MCP tools. The workflow is **semi-supervised**: Claude handles all microscope operations and provides visual assessments after each imaging step, but pauses at key checkpoints for human confirmation before irreversible steps.

### How Claude assesses image quality

`acquire_image` and `acquire_both_beams` return JPEG images directly to Claude via the MCP `ImageContent` type. Because Claude is a multimodal model it can visually inspect each image and reason about:

- **Contamination** — dark spots, ice contamination, charging artefacts
- **Focus / drift** — blurring, image shift between acquisitions
- **Curtaining** — vertical streaking in the FIB view of the lamella
- **Lamella thickness** — estimated from the bright band visible in the FIB image
- **Pattern placement** — whether drawn patterns are correctly centred on the target

This real-time visual feedback is the key advantage of the MCP approach over scripted automation.

### Prerequisites

- Microscope connected, sample loaded and cryo-stage at temperature
- Ion beam tuned and focused at the desired current for overview imaging
- Eucentric height set, or user ready to set it

### Step-by-step sequence

```
1. CONNECT
   connect_microscope(manufacturer="ThermoFisher", ip_address="<ip>")
   ─────────────────────────────────────────────────────────────────
   Or for remote connection from a separate PC:
   connect_microscope_remote(host="<microscope-pc>", port=8001)

2. SURVEY
   acquire_both_beams(hfw_um=400, autocontrast=True)
   → Claude inspects SEM + FIB pair and identifies candidate target regions.

3. NAVIGATE TO TARGET
   acquire_image(beam_type="electron", hfw_um=150)
   pick_point(title="Click the lamella target site")
     → Returns offset from image centre in µm.
   move_stage_relative(dx_mm=<offset_x/1000>, dy_mm=<offset_y/1000>)

4. FOCUS AND LINK EUCENTRIC HEIGHT
   auto_focus(beam_type="electron")
   link_stage()
   → Synchronises Z with working distance. Required before tilting.

5. MOVE TO MILLING ANGLE
   move_to_milling_angle(milling_angle_deg=15.0)
   acquire_both_beams(hfw_um=100)
   → Claude verifies the FIB view shows the correct lamella orientation
     and the sample surface is in the field of view.

6. MILL FIDUCIAL  (marks a reference point for repeat visits)
   run_milling_task(config_json='{
     "name": "Fiducial",
     "field_of_view": 50e-6,
     "stages": [{
       "milling": {"milling_current": 100e-12, "application_file": "Si"},
       "pattern": {
         "name": "Rectangle",
         "width": 1e-6, "height": 10e-6, "depth": 1e-6,
         "centre_x": -15e-6, "rotation": 0.7854
       }
     }]
   }')
   acquire_image(beam_type="ion", hfw_um=50)
   → Claude verifies fiducial is sharp, correctly placed, and not overlapping
     the target lamella site.

7. ROUGH MILL — Trench 1  (bulk material removal, ~0.74 nA)
   run_milling_task(config_json='{
     "name": "Rough Mill 01",
     "field_of_view": 100e-6,
     "stages": [{
       "milling": {"milling_current": 740e-12, "application_file": "Si-ccs"},
       "pattern": {
         "name": "Trench",
         "width": 10e-6, "depth": 2e-6,
         "upper_trench_height": 3.5e-6,
         "lower_trench_height": 3.5e-6,
         "spacing": 4.6e-6
       }
     }]
   }')
   acquire_both_beams(hfw_um=80)
   → Claude assesses trench depth, symmetry, and sample drift.
     If drift is large, calls auto_focus + link_stage before continuing.

8. ROUGH MILL — Trench 2  (reduced spacing, ~0.2 nA)
   run_milling_task(config_json='{
     "name": "Rough Mill 02",
     "field_of_view": 80e-6,
     "stages": [{
       "milling": {"milling_current": 200e-12, "application_file": "Si-ccs"},
       "pattern": {
         "name": "Trench",
         "width": 9.5e-6, "depth": 2e-6,
         "upper_trench_height": 2e-6,
         "lower_trench_height": 2e-6,
         "spacing": 1.6e-6
       }
     }]
   }')
   acquire_both_beams(hfw_um=60)
   → Claude estimates lamella thickness from the FIB image and reports to user.

   *** HUMAN CHECKPOINT ***
   Claude presents the images and asks the user to confirm lamella thickness
   is acceptable before the polishing step. The user may adjust pattern
   parameters (spacing, depth) before approving.

9. POLISH  (final surface finish, ~60 pA)
   run_milling_task(config_json='{
     "name": "Polish",
     "field_of_view": 60e-6,
     "stages": [{
       "milling": {"milling_current": 60e-12, "application_file": "Si-ccs"},
       "pattern": {
         "name": "Trench",
         "width": 9e-6, "depth": 1e-6,
         "upper_trench_height": 0.7e-6,
         "lower_trench_height": 0.7e-6,
         "spacing": 0.3e-6
       }
     }]
   }')
   acquire_both_beams(hfw_um=30, save=True, path="/data", filename="lamella_final")
   → Claude evaluates the finished lamella: thickness uniformity, surface
     quality, contamination, and any curtaining artefacts. Reports findings.

10. RESTORE IMAGING CONDITIONS
    move_flat_to_beam(beam_type="electron")
    blank_unblank(action="blank", beam_type="ion")
    → Stage returned to SEM position; ion beam blanked to protect sample.
```

### Typical milling parameters (cryo-lamella, Si/protein sample)

| Step | Current | Application | Width | Spacing |
|------|---------|-------------|-------|---------|
| Fiducial | 100 pA | Si | 1 × 10 µm | — |
| Rough 1 | 0.74 nA | Si-ccs | 10 µm wide | 4.6 µm |
| Rough 2 | 0.2 nA | Si-ccs | 9.5 µm wide | 1.6 µm |
| Polish | 60 pA | Si-ccs | 9 µm wide | 0.3 µm |

Adjust width to match your target lamella length. Increase depth if multiple passes are needed.

### Agent reasoning loop

After every imaging call, Claude follows this decision tree:

1. **Is the sample still in the field of view?** If not, call `move_stage_relative` to re-centre.
2. **Is the image in focus?** If not, call `auto_focus` + `link_stage`.
3. **Is there significant drift?** If >5 µm estimated shift, repeat the previous step before continuing.
4. **Are patterns correctly positioned?** If not, call `clear_patterns`, adjust offsets, and redraw.
5. **Proceed** with the next step.

---

## Known Limitations

- **No beam-shift alignment** — the full autolamella workflow uses cross-correlation beam-shift alignment between steps. This is not currently exposed as an MCP tool, so pattern placement relies on stage positioning accuracy and visual inspection by Claude. For high-accuracy work, use the full AutoLamella application.

- **`pick_point` requires a display** — `pick_point` opens a matplotlib window and requires `$DISPLAY` to be set. It will return a clear error in headless environments (e.g., SSH without X11 forwarding).

- **No manipulator or GIS control** — manipulator insertion/retraction and gas injection (GIS/sputter/cryo deposition) are not exposed via MCP.

- **Single microscope session** — the server holds one global microscope connection. Multiple simultaneous Claude sessions sharing the same server process will conflict.

- **`asynch=True` milling** — if `run_milling(asynch=True)` is used, do not call `finish_milling` until `get_milling_state()` returns `IDLE`.
