---
name: autolamella-coincidence-milling
description: Polishing with FM stacks between passes, on an instrument whose FM and FIB look at the same point (such as the Arctis) — Iterative Polishing following a target and a control pass by pass. Read fibsem-agent, fibsem-microscope, autolamella, autolamella-polishing and autolamella-correlative first, then this when the FM is coincident with the FIB.
---

# Milling with a coincident FM

Read fibsem-agent, fibsem-microscope, autolamella, autolamella-polishing and
autolamella-correlative first. This skill is for an instrument whose FM and
FIB look at the same point, so the FM can image the lamella between passes
without moving the stage. Without a coincident FM it doesn't apply: polish
by autolamella-polishing's rules, with an SEM image each pass.

**Experimental, and only when the operator turns it on.** Coincidence milling
needs an instrument whose software restrictions on running the FM and the
FIB together an operator has disabled: if they haven't, don't use it. Never
start it yourself, at any level. Propose it, and use it only after the
operator says yes for this run, and only at Drive.

This is about the FM and the FIB. Bringing the SEM and FIB onto the same
feature is fibsem-microscope's *Coincidence: do it yourself*.

## Iterative Polishing with the FM

Autolamella-polishing's *Iterative Polishing* is the loop. With a coincident FM, each
pass also takes FM stacks and follows a target:

```json
"strategy": {"name": "Iterative Polishing",
             "config": {"face": "top", "acquire_sem_image": true,
                        "objective_position": "<this pass's, in m>",
                        "baseline": "<the last pass's Pass data>",
                        "targets": [[212.4, 198.1]], "control": [140.0, 260.0],
                        "target_channel": {"name": "red",
                                           "excitation_wavelength": 550,
                                           "emission_filter": {"name": "<filter_set's>"},
                                           "power": 0.01, "exposure_time": 0.1}}}
```

`target_channel` is the channel the target shows in: its excitation in nm (550,
not 5.5e-07) and an emission filter from the FM's `filter_set` choices. Left
without one, the stack is reflection, and the target doesn't show in it.

`service_wait_for_run` milling then also answers each target's refit,
signal, ratio to the control and change from the baseline, and the stacks'
paths.

- **Before the first pass.** Take a FIB image and measure the band, take an
  FM stack, pick the target and a control punctum away from the mill, and
  note the objective's position.
- **Each pass.** Set `objective_position` each pass: the objective drifts
  between passes (about 90 nm down a pass on one Arctis), so measure it on
  yours. Pass the last pass's `Pass data` as `baseline` and its refitted
  points as `targets`.
- **Where each face goes** (rectangles, from one Arctis grid): the face
  farther from the target starts 100 nm inside the target's half-maximum
  edge and moves 100 nm a pass; the nearer face starts 150 nm outside it and
  moves 50 nm a pass.
- **Stop and ask the operator** (guidance, not hard limits), as well as at
  autolamella's stop points, when: the target's signal against the control
  falls two passes running, or a lot in one pass; the target's fit fails;
  the FWHM nears its floor with the objective in (about 450 nm on one grid);
  the reflection image shows damage.
- **Thickness with the objective in** reads differently from the operator's
  measurement with it out (560 against 483 nm, 474 against 467 nm on one
  run). Report your last reading as objective-in.

## Workers

When a worker polishes the lamella (autolamella-lead, *Workers*), its brief
also names this skill and gives it the target and control from the last
correlation, the `target_channel`, the objective's position and this
skill's stop-and-ask list. The worker cannot ask the operator: when one of
those stops is hit, it stops polishing and returns what it saw, and the
lead asks the operator. The worker still asks the Verifier before every pass
that would mill.

