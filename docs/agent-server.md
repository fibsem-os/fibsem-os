# The Agent, Explained

What the AutoLamella agent server is, what an AI agent can see and do through
it, and how to connect one — locally or from another machine.

(Setup commands for the `fibsem-mcp` sidecar live in
[fibsem/mcp/README.md](../fibsem/mcp/README.md); this guide is the *what* and
the *why*.)

## What it is

When you enable the agent server, AutoLamella hosts a small, token-protected
API on your machine while a microscope is connected. An AI agent (Claude,
Codex, or anything that speaks MCP or plain HTTP) can connect to it and watch
your session: the workflow, the queue, the images on screen, the questions the
workflow is asking. With permission you grant per session, it can also act —
answer those questions, start and stop workflows, change supervision.

Nothing about your workflow changes when the feature is off, and nothing about
it changes when the feature is on but no agent is connected. The server is an
observer with a doorbell, not a second driver.

## Turning it on

1. **Preferences → Agent**: tick *Enable Agent Server*. This is the durable
   switch; it also holds the hand-over time (below).
2. Connect a microscope. The server starts with the session and stops with it.
3. **Tools → Agent Server** shows the live session: whether it is running, the
   session token, and what the agent may do.

## What the agent can see (always, while running)

Reading is on for any connected agent — it is the point of the feature:

- **Status and queue** — what is running, which item, what is next.
- **The protocol** — task list, supervision settings, schedules.
- **What your screen shows** — the SEM/FIB images on display, as small
  previews, with when they were acquired. Not a fresh acquisition: the agent
  sees what you see.
- **The standing question** — when a supervised task asks something, the agent
  sees the same question, including its images and the live position of a
  marker you are dragging.
- **Events** — task starts and finishes, milling progress, questions raised
  and answered (and by whom), as a live feed.
- **Run history** — the run summary and each item's produced files.
- **Each item's details** — status, failure flag, point of interest,
  alignment area, milling angle, and where its poses put the stage.

Every image travels as a downscaled preview that carries its own scale: the
preview's size, the source image's size, the field width, and the physical
size of a source pixel. Positions are never left to guesswork — point
questions state their convention in the payload (metres, origin at the image
centre, +y up), detected features are in source-image pixels, and alignment
areas are fractions of the frame. An agent drawing on or measuring against a
preview should always use the scale in the payload it arrived with — the live
view's field width can differ from the acquisition's.

## What the agent may do (with your permission)

In **Tools → Agent Server**, the *Act* switch lets the agent:

- **answer supervision questions** — through exactly the same path as your
  Yes/No buttons. First answer wins: you can always answer first, and your
  click beats the agent's every time. Every answer is attributed — the line
  under the buttons reads `agent · answered Run Milling · 14:22:31`.
  For the two geometry questions (alignment area, point of interest) an
  answer can carry an adjustment: the agent's proposed rectangle or marker
  is placed on your screen through the same widgets you would drag, then
  accepted — you see exactly what it chose, out-of-bounds proposals are
  refused before anything moves, and the record marks the answer as
  adjusted. Milling parameters cannot travel this way; only the geometry
  the question itself is about;
- **start and stop workflows** — the same as your Run and Stop buttons
  (stopping, like the stop-milling command, actually works for *any* connected
  agent regardless of permission: the emergency brake is never gated);
- **change supervision** — flip a task between automated, supervised, and
  agent-supervised, mid-run;
- **re-queue a task** — "run 03's fiducial again" during a run.

A separate **Configure** switch lets the agent edit an item's details
(point of interest, alignment area, description, defect verdict), schedule
tasks ("start polishing at 6am"), and edit task parameters — an
item's milling depths, currents, imaging settings — as targeted patches
against exactly the state it last read (a concurrent edit of yours makes the
agent's write stale, and it is refused, never merged). A task already
running has copied its settings and is refused outright; pending tasks pick
changes up when they start. Every applied change is recorded on the
timeline, old value and new. It is deliberately its own switch: answering
questions and rewriting protocols are different levels of trust.

Permissions last the session only. They are granted in the dialog, die when
the app closes, and never persist — every session starts read-only.

## What the agent cannot do

Command hardware. There is no permission that lets an agent move the stage,
acquire, or mill directly — the switch exists in the dialog so the ladder is
visible, and it is disabled. Milling happens only the way it always has:
through a workflow's own steps, with its checks.

## Agent-supervised tasks

In the workflow tab, each task's supervision control cycles **Automated →
Supervised → Agent** (the third option appears only when the feature is
enabled). A task set to *Agent* raises its questions to the connected agent:

- the window border turns **purple** while it runs, and the supervision chip
  shows **✦ Agent** — you can walk away;
- if the agent answers, you never notice; the timeline records who answered;
- if the agent goes quiet for longer than *Hand questions to me after*
  (Preferences → Agent, default 5 minutes), the question becomes yours: the
  ordinary orange border, attention button, and sound, plus a message saying
  why. This hand-over runs inside AutoLamella, so it works even if the agent's
  own process has died;
- and the app knows whether anyone is actually out there: a watching agent is
  in touch with the server every half-minute, so if it hasn't been heard from
  — it never connected, or its session died — a question is handed to you
  right away instead of waiting out the timer. The Agent Server dialog shows
  when the agent was last heard from.

You can always answer any question yourself, whoever it is addressed to.

## The dashboard

**Open Dashboard** in Tools → Agent Server opens a read-only monitor page in
your browser — the same session the agent sees, for human eyes: experiment
and workflow state, a card per item with its latest recorded image, a review
strip of each completed task's final image, and progress through the tasks,
plus the live workflow queue and event feed while a run is up. It updates
itself from the event stream; there is nothing to refresh.

It is a monitor, not a second cockpit: questions are answered in AutoLamella.
The page is served by the agent server itself, so it exists wherever the
server does — on this machine by default. The button hands the page your
session token in the URL fragment, which stays in the browser (fragments are
never sent over the network or written to server logs).

## Connecting an agent

**On the same machine** — nothing to copy. The running server writes a
discovery file (`~/.fibsem/agent-server.json`) that the `fibsem-mcp` sidecar
finds by itself:

```bash
claude mcp add fibsem -- fibsem-mcp
```

**From another machine** — copy the session token from Tools → Agent Server
and point the sidecar at the microscope PC:

```bash
claude mcp add fibsem -- fibsem-mcp --url http://<microscope-pc>:8001 --token <token>
```

(Or set `FIBSEM_SERVER_URL` and `FIBSEM_SERVER_TOKEN`.) The server binds to
localhost by default; reaching it from another machine means running it on a
reachable address — do that only on a network you trust, because the
connection is plain HTTP: anyone who can read the traffic can read the token.

Any MCP-speaking agent works, and so does anything that can send HTTP with a
bearer token — the sidecar is convenience, not a requirement.

## The security model, briefly

- **One token per session**, generated fresh at server start, shown only in
  the dialog. It is the whole key: no token, no access.
- **Access fails closed.** Reading needs a valid token; acting needs the
  *Act* permission; hardware has no permission at all.
- **Permissions last one session** — granted by whoever is at the
  microscope, dead when the app closes. There is nothing in any config file
  that arms anything.
- **Localhost by default.** Out of the box, only processes on the same
  machine — running as someone who can read your files anyway — can even
  attempt to connect.
- **One server per machine.** A second AutoLamella (or a bench server) refuses
  to start its own while one is alive.
- **Everything is attributed.** Answers carry who gave them; the event stream
  and the log keep the record.

## When something is off

- *The dialog says "Not running"* — the feature is enabled but no microscope
  is connected yet, or the enable box was ticked after connecting (it starts
  on the next connect, or immediately after saving Preferences).
- *The sidecar exits with "no server found"* — it waited for a server that
  never appeared; start AutoLamella (with the feature on) first, or pass
  `--url`/`--token` explicitly.
- *An agent's answer is refused with `stale_prompt`* — the question changed
  between the agent reading it and answering (often: you answered first).
  That refusal is correct behaviour; the agent re-reads and continues.
