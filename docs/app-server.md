# The App Server and the Agent, Explained

What the AutoLamella App Server is, what an AI agent can see and do through
it, and how to connect one — locally or from another machine.

(Setup commands for the `fibsem-mcp` sidecar live in
[fibsem/ai/agent/mcp/README.md](../fibsem/ai/agent/mcp/README.md); this guide is the *what* and
the *why*.)

## What it is

When you enable the App Server, AutoLamella hosts a small, token-protected
API on your machine while a microscope is connected. An AI agent (Claude,
Codex, or anything that speaks MCP or plain HTTP) can connect to it and watch
your session: the workflow, the queue, the images on screen, the proposals the
workflow is stopped on. Raise its level to Drive and it can also act — decide
those proposals, start and stop workflows, change supervision, edit task
parameters.

Nothing about your workflow changes when the feature is off, and nothing about
it changes when the feature is on but no agent is connected. The server is an
observer with a doorbell, not a second driver.

## Turning it on

1. **Tools → App Server**: tick *Run the App Server*. It is saved, so the
   next launch starts the same way.
2. Connect a microscope. The server starts with the session and stops with it.
3. **Tools → App Server** also shows the server: whether it is running, which
   services it serves, when the agent was last heard from, and the
   dashboard. *Serve the microscope* there also serves the routes that
   reach the instrument, everything under `/microscope`: who it is, its
   state, its devices and its services (milling, stage movement, spot burn,
   sample handling, imaging, alignment), reads included. It is saved too, and changing
   it restarts a running server; a connected agent carries on. It starts
   off, because a read of the instrument is hardware access, not
   monitoring. Only the agent's level resets at every launch.
4. The **Agent chip** in the status bar shows and sets the agent's level
   (below). Its menu's *Connect an agent…* shows how to connect one.

## The agent's level

One knob, on the Agent chip in the status bar, set in two clicks:

| Level | The agent can |
|---|---|
| **Off** | nothing: every request is refused |
| **Watch** | see everything below and add notes; never act |
| **Assist** | also stop what it started itself, suggest decisions on proposals, and ask you to approve anything else it wants done |
| **Drive** | also decide proposals, start and stop (a run, or a mill it didn't start), change supervision, re-queue, and edit configuration; with the microscope served, use it too |

Every launch starts at **Watch**. The level lasts the session: nothing in any
file raises it. Changing it is recorded on the event stream, with you as the
actor.

While the agent is at Assist or Drive the chip's menu leads with **Stop agent**. It drops
the agent to Watch at once, with no confirmation, and the agent is told it was
stopped. If the agent is using the microscope, Stop agent also stops what it
is doing: a mill, a spot burn, an auto focus or live imaging. A stage move
already under way finishes. A running workflow keeps running; its own Stop is
beside the chip.

Using the microscope directly (moving the stage, acquiring, milling) needs
Drive and *Serve the microscope*. At Assist the agent asks for it.

## Approving the agent's requests (Assist)

At Assist the agent asks before it acts. When it does, the Agent chip turns
amber and says how many actions it is asking for ("Agent asks · 3"). Click
the chip: the request is at the top of its menu. Each action is shown with
what it acts on and what it changes (a setting shows old → new). These
lines are written by the app from the actions themselves, not by the agent.
Under them is the agent's reason. The picture beside it is the image the agent
judged from, its evidence, with any points or areas it marked: a beam image has
the actions' values drawn on it too, while an image the agent drew itself is
shown as it is. Without evidence the picture is the image the values were
measured on, and "No evidence" shows under the reason. Evidence is kept in the
experiment under `evidence/`, and Recent has an "evidence" link on each row
that cites some.

- **Approve** runs the actions in order, each checked as if the agent had
  made the call at Drive. The block shows each one's progress. If one is
  refused (a run has the microscope, a setting changed since the agent read
  it), the rest are not run, and the status bar's failure line says how far
  it got and why.
- **Reject…** runs nothing. You can tell the agent why; it gets your words.

One request waits at a time. Changing the level or pressing Stop agent
cancels a waiting request, and stops an approved one before its next action.

Each request is saved in the experiment's record (`experiment.yaml`) as an
`actions` proposal, with who approved or rejected it and why. A request still
waiting when the app closes reads as withdrawn the next time the experiment
is opened.

## What the agent can see (Watch and above)

Reading is the point of the feature:

- **Status and queue** — what is running, which item, what is next.
- **The protocol** — task list, supervision settings, schedules.
- **What your screen shows** — the SEM/FIB images on display, as small
  previews, with when they were acquired. Not a fresh acquisition: the agent
  sees what you see.
- **Pending proposals** — when a supervised task stops on a proposal (a
  position, a point of interest, a detection, a mill's pass), the agent sees
  the same proposal, with the image its values sit on.
- **Events** — task starts and finishes, milling progress, proposals asked
  and decided (and by whom), as a live feed.
- **Run history** — the run summary and each item's produced files.
- **Each item's details** — status, failure flag, point of interest,
  alignment area, milling angle, and where its poses put the stage.
- **Grids** — each grid the experiment records, its quality verdict and task
  history, its SEM/FIB/FM overviews as previews, and where your items fall on
  an overview (computed in the app, in that image's pixels, with each item's
  status). Records only: which slot a grid is in is a live hardware fact the
  agent does not get here.

Every image travels as a downscaled preview that carries its own scale: the
preview's size, the source image's size, the field width, and the physical
size of a source pixel. Positions are never left to guesswork — a point
proposal states its convention in the payload (metres, origin at the image
centre, +y up), detected features are in source-image pixels, and alignment
areas are fractions of the frame. An agent drawing on or measuring against a
preview should always use the scale in the payload it arrived with — the live
view's field width can differ from the acquisition's.

## What the agent may do (Drive)

At Drive the agent may:

- **decide proposals** — through the same decision as your Continue. The
  first decision wins: you can always decide first. Every decision is
  attributed to whoever made it, in the History tab and on the record. A
  decision on a position, an alignment area or a detection can carry the
  agent's own values; out-of-bounds values are refused before anything is
  written;
- **start and stop workflows** — the same as your Run and Stop buttons, and
  stop milling. Stopping a run, or a mill the workflow or you started, needs
  Drive or a request you approve at Assist; at Assist the agent stops only its
  own command. Every stop carries the agent's reason, which the record keeps.
  At Watch it cannot stop anything: a watching agent that sees trouble tells
  you, and the Stop button is yours;
- **change supervision** — flip a task between automated and supervised,
  mid-run;
- **re-queue a task** — "run 03's fiducial again" during a run.

It may also edit an item's details
(point of interest, alignment area, description, defect verdict), schedule
tasks ("start polishing at 6am"), and edit task parameters — an
item's milling depths, currents, imaging settings — as targeted patches
against exactly the state it last read (a concurrent edit of yours makes the
agent's write stale, and it is refused, never merged). A task already
running has copied its settings and is refused outright; pending tasks pick
changes up when they start. Every applied change is recorded on the
timeline, old value and new.

At Drive the agent can also start a **grid screening run** — the Grids
view's Run or Screen all grids, remotely — over the grid tasks in your
protocol. It is refused while any run is going, and a good agent shows you
the plan (every grid, its load, its tasks) before it starts one.

## Using the microscope (Drive, microscope served)

With *Serve the microscope* on, an agent at Drive may also use the
microscope directly: move the stage, acquire images, change beam and detector
settings, and mill. Each command is in control of the microscope while it
runs, so it never acts beside anything else:

- while a workflow run or a microscope script is in control, the command is
  refused with `409 microscope_busy`, naming who is in control;
- while the run waits on a proposal, the run lends control: whoever decides
  it may act, one command at a time, and the run carries on when it ends;
- stopping never takes control, so a stop is never refused by whoever is in
  control. The agent stops its own command at Assist; anything else (a mill
  the workflow or you started) needs Drive. A stop when nothing is milling
  answers 409 `NothingToStop`.

Your own buttons don't take control yet, so nothing stops an agent's command
running beside something you started by hand. Set the agent to Watch while
you work at the microscope yourself.

The app follows: a stage move shows in its stage position like one of yours
would. With the microscope not served, every `/microscope` route answers 404
`microscope_off`, saying so, at any level from Watch up (at Off, every
request answers 403 `agent_off`).

## Supervised tasks and the agent

A supervised task's proposal goes to you and to any connected agent alike.
The first decision wins, and you can always decide first. The window shows the
ordinary orange border, attention button and sound whether or not an agent is
connected, and the record keeps who decided. At Watch the agent sees the
proposal and can tell you what it would decide. At Assist it can suggest a
decision: the Agent chip says "Agent suggests", and its card shows the
agent's reason and the suggested value drawn on the proposal's image. Approve
decides the proposal as suggested, as your decision, and the record names the
agent as the one who suggested it; Decline drops the suggestion, with an
optional reason that goes back to the agent, and the proposal stays up for
you. At Drive it may decide it.

## The dashboard

**Open Dashboard** in Tools → App Server opens a read-only monitor page in
your browser — the same session the agent sees, for human eyes: experiment
and workflow state, a card per item with its latest recorded image, a review
strip of each completed task's final image, and progress through the tasks,
plus the live workflow queue and event feed while a run is up. It updates
itself from the event stream; there is nothing to refresh.

It is a monitor, not a second cockpit: proposals are decided in AutoLamella.
The page is served by the App Server itself, so it exists wherever the
server does — on this machine by default. The button hands the page the
dashboard's own token in the URL fragment, which stays in the browser
(fragments are never sent over the network or written to server logs). That
token is read-only: it reads the session's monitoring routes and nothing
else, so the link can't be used to act on the session or reach the
microscope, and it is never written to the discovery file the agent uses.
*Copy link* beside the button copies the same link, to open the page in
another browser. The agent's level doesn't affect the dashboard.

## Connecting an agent

**On the same machine**, use *Copy for your agent* in *Connect an agent…*
(the Agent chip's menu) and paste the text into Claude Code. The agent
registers the sidecar by its absolute path, checks the connection with
`fibsem-mcp --check`, and reports its level. For Claude Desktop the text says
which entry to merge into `mcpServers` in its config file. The token isn't
in the text: the running server writes a discovery file
(`~/.fibsem/app-server.json`; `agent-server.json` before v0.6.0, still read)
that the `fibsem-mcp` sidecar finds by itself. By hand, that's:

```bash
claude mcp add -s user fibsem -- fibsem-mcp
```

`-s user` registers it for all your projects; without it, `claude mcp add`
covers only the current directory's project.

**From another machine** — copy the agent token from *Connect an agent…* and
point the sidecar at the microscope PC:

```bash
claude mcp add -s user fibsem -- fibsem-mcp --url http://<microscope-pc>:8001 --token <token>
```

(Or set `FIBSEM_SERVER_URL` and `FIBSEM_SERVER_TOKEN`.) The server binds to
localhost by default; reaching it from another machine means running it on a
reachable address — do that only on a network you trust, because the
connection is plain HTTP: anyone who can read the traffic can read the token.

Any MCP-speaking agent works, and so does anything that can send HTTP with a
bearer token — the sidecar is convenience, not a requirement.

## The security model, briefly

- **Two tokens per session**, generated fresh at server start: the agent's,
  shown only in *Connect an agent…*, and the dashboard's, which reads the
  monitoring routes and nothing else. No token, no access.
- **Access fails closed.** Reading needs a valid token and a level above Off;
  acting needs Drive; using the microscope needs Drive and the microscope
  served.
- **The level lasts one session** — set by whoever is at the microscope,
  back to Watch when the app restarts. A microscope reconnect or a services
  change restarts the server but not the session: the level and the agent's
  token stand, so a connected agent carries on. There is nothing in any
  config file that raises it.
- **Localhost by default.** Out of the box, only processes on the same
  machine — running as someone who can read your files anyway — can even
  attempt to connect.
- **One server per machine.** A second AutoLamella refuses
  to start its own while one is alive.
- **Everything is attributed.** Decisions carry who made them; the event stream
  and the log keep the record.

## When something is off

- *The dialog says "Not running"* — *Run the App Server* is off, or it is on
  but no microscope is connected yet; it starts when one connects.
- *The sidecar exits with "no server found"* — it waited for a server that
  never appeared; start AutoLamella (with the feature on) first, or pass
  `--url`/`--token` explicitly.
- *The agent says it can't act* — it is at Watch. Raise it from the Agent chip
  if you want it to: to Assist, where it asks you before each action, or Drive; refusals name the level (`level`, or `ask_first` at Assist),
  a stop (`agent_stopped`) or Off (`agent_off`).
