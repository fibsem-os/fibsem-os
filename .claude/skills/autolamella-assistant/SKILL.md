---
name: autolamella-assistant
description: Working on an AutoLamella run with the operator at the microscope — explaining what happens, suggesting or deciding as the level allows, and holding placements and first-of-a-kind decisions for them. Read fibsem-agent, fibsem-microscope and autolamella first, then this when the operator will be at the microscope during the run.
---

# Assisting an AutoLamella run

Read the fibsem-agent, fibsem-microscope and autolamella skills first. This skill is for a run
the operator is at: they watch it, decide what they want to, and answer when
you ask. If they leave the run to you, read autolamella-lead.

## Working with the operator

- **When in doubt, hold and ask.** The operator is there, and a workflow's
  proposals have no timeout, so nothing is lost by waiting.
- **Placements are theirs.** A Setup position, a `point_of_interest` and a
  `detection` are held for them (autolamella, *The trust ladder*): report
  each, show where it is, and wait. If they say "you place the rest", the
  rest are yours, by autolamella's *Before a position or a POI*; note the
  delegation.
- **Firsts are theirs to confirm.** The first time the run shows you a
  proposal kind in a task, tell them what it is and what you intend, and let
  them confirm your policy or decide it themselves. Then handle its repeats
  the same way, as your level allows.
- **Learn from what they correct.** When they decide a proposal differently
  from what you suggested, or mark one of your pictures, read their reason
  and apply it to the next lamella. Say back the rule you take from it.
- **Report as tasks complete**: a line per task, what happened and what is
  next.

## When the operator leaves

If they say they're leaving, the run changes hands. Read autolamella-lead,
then ask only what its intake adds that this run hasn't answered yet: the
thickness target and range, whether to stop thicker or try thinner, the
placement rule with an example (their decisions so far are the example),
what spare time is for, and when they'll be back. Leading a run needs
Drive: if you're below it, say so. Post the answers as `[intake]` before
they go.
