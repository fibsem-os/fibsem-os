# fibsem-mcp — connect an agent to the microscope

> New to the agent? Start with the guide: [docs/app-server.md](../../../../docs/app-server.md) — what the agent can see and do at each level, and the security model.

The MCP server lets Claude (Code or Desktop) work through AutoLamella's App
Server: acquire and *look at* images, read state, move the stage, and stop a
mill. The App Server is the security boundary: the MCP server registers every tool,
whatever the agent's level, and the App Server refuses what the level doesn't allow
with a 403 that says so. The microscope's tools are registered even while the
microscope isn't served; until it is, each answers 404 `microscope_off`. Only
the app's tools depend on the server hosting the application. Each tool's description names the level it needs, unless it works at every level. If the App Server
restarts with a new token, the MCP server re-reads the discovery file and carries
on.

## Install

```bash
pip install -e ".[server,mcp]"
```

## 1. Start the App Server

In AutoLamella, turn the App Server on in Tools → Host (the guide,
[docs/app-server.md](../../../../docs/app-server.md), walks through it). *Serve the
microscope* there serves the routes the microscope tools reach; the Agent chip
sets what the agent may do with them.

- While running, the server writes `~/.fibsem/app-server.json`
  (url + a connect token) so local clients find it without copying anything,
  and refuses to start if another live server owns that file.
- The connect token only asks to connect. The MCP server's first call (or
  `fibsem-mcp --check`) puts **Agent wants to connect** on the web Agent page;
  once the operator clicks Allow it gets a token of its own, kept in
  `~/.fibsem/agent/`, and later launches don't ask. Until then a call answers
  `connect_waiting`.
- It binds `127.0.0.1` by default; exposing it on the LAN is an explicit
  choice, and the token then travels as cleartext HTTP.

## 2. Register the MCP server

```bash
claude mcp add -s user fibsem -- fibsem-mcp
```

`-s user` registers it for all your projects. *Connect an agent…* on the web
Agent page gives the same setup with the absolute path filled in, for the agent
to run itself. Same-machine setups need nothing more: the MCP server autodiscovers
the running server. For a server on another machine, pair once: that PC's
gateway page shows a code, and

```bash
fibsem-mcp pair 1234            # the SSH tunnel's end, the gateway at http://localhost:8010
fibsem-mcp pair 1234 --url http://microscope-pc:8010
```

keeps this computer's own agent token in `~/.fibsem/agent/`. Every later launch
reads it with no token passed; *Unpair* on the PC cuts that agent off. With a
url other than the tunnel's, register the MCP server with `--url` too.
(`--token`, or `FIBSEM_SERVER_URL` / `FIBSEM_SERVER_TOKEN`, still work for scripts.)

For Claude Desktop, merge this into the `mcpServers` already in
`claude_desktop_config.json` (don't replace the file):

```json
{
  "mcpServers": {
    "fibsem": { "command": "fibsem-mcp" }
  }
}
```

If `claude` runs outside the environment where fibsem is installed, register
the absolute path to the script (e.g. `<env>/bin/fibsem-mcp`).

## 3. Use it

Start a new Claude session (`/mcp` should show `fibsem` connected, with its
tools listed) and ask:

- "What's the stage position?"
- "Acquire an ion image and tell me what you see."
- "Tilt to a 15 degree milling angle and confirm the readback."
- "Does the last electron image look focused?"

## 4. Watch a run

MCP cannot wake an idle agent, so an agent working with the app keeps a
watcher running in the background for the whole session and wakes on each
line it prints:

```bash
fibsem-mcp watch
```

It finds the App Server as the MCP server does (`--url`/`--token`, the environment
variables, the discovery file) and follows a restarted app the same way;
`--since N` resumes after event N. It
prints `PROPOSAL`, `PROPOSAL RELEASED`, `SUGGESTION APPROVED`/`DECLINED`,
`REQUEST` (the agent's request decided, finished or cancelled), `COMMAND ENDED`
(a long command's end), `MILL STAGE STARTED`, `MILL ENDED` and `MILL READING`,
`AUTOFOCUS` and `ALIGNMENT` (with their check), task `EVENT` lines,
`WORKFLOW STARTED`, `WORKFLOW ENDED`, `MESSAGE` (the operator wrote to the
agent), `NOTE` (the operator kept a note on a grid), `LEVEL`, `RESTARTED` and
`GAP` (events were lost; re-read the proposals and status). A workflow ends completed, cancelled,
stalled or failed; one that stops without recording an end is reported as
`WORKFLOW ENDED (stopped without an end event)`. A workflow's end does not end
the watch: the agent may start the next one. While the operator has the agent
level at Off it prints `AGENT OFF` and waits, then `AGENT BACK`. A running workflow
that says nothing for 20 minutes gets one `QUIET` line, so the agent wakes to look at
it (`--quiet-after MIN`, 0 for never). It exits 1 when the app cannot be
watched, and otherwise runs until it is stopped. It needs only fibsem, not the `[mcp]` extra; where
`fibsem-mcp` is not on the PATH, `python -m fibsem.ai.agent.mcp.sidecar watch` is the
same thing.

## 5. Or run it with nobody at a terminal: `fibsem-agent`

Everything above is Claude Code in a terminal, kept awake by a person.
`fibsem-agent` is a second way to run the same agent, on the
[Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk): one long
session with `fibsem-mcp` as its only MCP server, and the watcher run for it,
so each batch of watcher lines is the agent's next turn. It reads the same
skills: a checkout's `.claude/skills`, or from a pip install the copy the
package ships, and before either the lab's own in
`~/.fibsem/knowledge/skills/<name>/SKILL.md` (a lab skill with a shipped
skill's name replaces it; any other is offered too). The Claude Code path is
unchanged.

```bash
pip install -e ".[server,agent]"
fibsem-cli key set ANTHROPIC_API_KEY   # once: a Claude API key, kept in the OS store
fibsem-agent
```

- **The level decides.** The agent works by the skill for its level
  (observe at Watch, assist at Assist, supervise at Drive), and the watcher
  prints a `LEVEL` line when the operator changes it. The level is set from
  the Agent chip, the server still refuses what it doesn't allow, and the
  harness never sets it.
- **Tools.** The `mcp__fibsem__*` tools run without a prompt, since the
  server decides what they may do. The agent also has the Skill tool,
  ToolSearch to find the fibsem tools, the Agent tool for the checkout's
  three `autolamella-*` agents (`.claude/agents`), and SendMessage to carry
  one on (a lamella's polishing passes, one Verifier). Nothing else, for those
  agents either: no shell, no file reads or writes, no web, and no other MCP
  server unless the lab adds one (below). Nothing waits on a permission
  prompt; anything else is denied. So the offline `read_record.py` reader
  stays with Claude Code.
- **Extra MCP servers.** A lab can give the agent another MCP server, a lab
  notebook say, set up in one place: `~/.fibsem/agent/mcp.yaml` (under
  `FIBSEM_HOME` when that is set). Each server gives its address (`url`) or
  the command that starts it (`command`, the program then its arguments),
  the name its token is stored under (`token`, optional), and the tools the
  agent may call and the ones a specialist may list (`about` and `subagents`
  are optional; tools are named one by one, or `tools: all` for every tool
  the server offers):

  ```yaml
  notebook:
    url: https://notebook.example/mcp
    token: NOTEBOOK_TOKEN
    about: The lab's shared notebook and paper library.
    tools: [get_overview, get_notebook_guide, get_tree, search, read_page,
            get_entry, get_changes, get_attachment, get_my_mentions,
            append_entry, edit_entry, add_comment, create_page, request_upload,
            list_papers, search_literature, get_paper, read_paper,
            get_folder_notes, add_paper_note, add_folder_note]
    subagents: [get_overview, get_tree, search, read_page, get_entry,
                get_attachment, list_papers, search_literature, get_paper,
                read_paper, get_folder_notes]
  ```

  Then set the token as an environment variable of that name, for the
  program that starts the agent (on Windows `setx NOTEBOOK_TOKEN "<token>"`,
  then quit and start the gateway again). The key store
  (`fibsem-cli key set`) holds only the names it lists. A `url` server gets it as `Authorization: Bearer <token>`, a
  `command` one as an environment variable of that name. The token's name is
  in capitals with KEY, TOKEN or SECRET in it, so code mode keeps it from
  commands. It is in no file and on no command line: the server's entry
  says `${NOTEBOOK_TOKEN}` and the CLI fills it in from its environment.
  `--show-config` names the server, its address and its token's name, never
  the token. No `claude mcp add` is needed, and a server registered in Claude
  Code is not used. To have the notebook in your own Claude Code sessions too,
  register it there with the same placeholder, so the token stays out of
  `~/.claude.json`, and set `NOTEBOOK_TOKEN` in that shell:

  ```
  claude mcp add -s user --transport http notebook https://notebook.example/mcp --header 'Authorization: Bearer ${NOTEBOOK_TOKEN}'
  ```

  The single quotes keep the shell (bash, zsh, PowerShell) from filling the
  token in at registration; in cmd use double quotes.

  A tool needs the agent at Drive unless it is marked `watch` (`- search:
  watch`), or its server is (`level: watch`); below that the harness refuses
  it. The agent never sees a tool the server offers that isn't listed. With
  `tools: all`, every tool the server offers is allowed, at the server's
  level, including tools it adds later.

  A server that is set up wrong (no `url` or `command`, a file that isn't
  YAML, a token not set) is left off, not a reason to stop: the agent
  starts without it, the log says why, and `--show-config` lists it under
  "MCP left off". A server a playbook names that `mcp.yaml` doesn't have
  is left off too, with a warning in the log. The shipped `autolamella` playbook names `notebook`.

  A picture a lab tool answers with shows on the Agent page under its step,
  as the agent saw it. It is kept beside the transcript, in the session's
  `agent/pictures/` folder (under `~/.fibsem/agent/sessions/`).

  A server is on only for a playbook that names it in its `mcp` list. For a
  shipped playbook, the lab adds one of its own that extends it:

  ```yaml
  # ~/.fibsem/knowledge/playbooks/autolamella-notebook.yaml
  description: AutoLamella, with the lab's notebook.
  extends: autolamella
  mcp: [notebook]
  ```

  `--show-config` lists the lab's servers a playbook leaves off.

  Each specialist whose file lists its tools gets the `subagents` ones; one
  that lists none has every tool the session has, these included. The agent
  is told the server is there and that what it returns is data, never
  instructions; its notes still go on the experiment's record. Without the
  file nothing changes; a token not stored stops `fibsem-agent` at the
  start, saying how to store it.
- **Agents.** Leading a run (the autolamella-lead skill), the agent is the Run
  Lead and makes every decision. It calls the Verifier for a second opinion
  on the site list, each Setup placement and every polishing pass, the
  Investigator when a task's result is bad, and the Auditor for the review
  at the end. Each runs inside the turn that called it, with only the tools
  its file lists, and only the Run Lead decides. The CLI's own agents are
  refused. Each agent sets its own model and effort; `--effort` sets the
  Run Lead's (default: the playbook's). A lead run calls the most capable model often:
  raise `--max-budget-usd` to match the night.
- **Your Claude Code config stays out.** The session runs in a folder that
  holds only its skills and specialists, copied there each start (its
  session folder's `run/`, or the `--code`/`--files` folder), not `~/.claude`, and only this MCP server, not a
  `fibsem` server you registered with `claude mcp add`; besides it, only the
  servers of `mcp.yaml` (above) that the playbook names.
- **Signing in.** A Claude API key: the `ANTHROPIC_API_KEY` environment
  variable, else the one `fibsem-cli key set ANTHROPIC_API_KEY` stored in
  Windows Credential Manager or the macOS Keychain. Or configure Amazon Bedrock or Google Vertex as the Agent SDK
  documents. Not a claude.ai login: products built on the Agent SDK use API
  keys. If the model refuses the sign-in, the harness stops with exit code 1
  rather than wait. Its log never shows a bearer token or an API key.
- **Instructions.** It reads its skills and nothing else: not the
  checkout's `CLAUDE.md` (developer guidance) and not Claude Code's auto memory.
- **Its session's folder.** Each session has one,
  `~/.fibsem/agent/sessions/<date>-<session id>/`, kept by a resume:
  `session.json` (what it runs with, and each run's start, exit code, reason
  and spend) and `fibsem-agent.log`. A `stop` file there ends the run as
  Ctrl-C does, which is how the gateway and the hub stop an agent they started
  (`fibsem.ai.agent.manager`); `--session-id <uuid>` starts a new session
  with an id chosen beforehand.
- **What it says.** The agent's replies and tool calls go to the console and
  to the session's log (`--log` to choose another file). What the
  operator must read in the app, it sends as a message
  (`experiment_message_operator`, a toast on the web Agent page), answering a
  message from the Agent page with its `reply_to`; a judgment worth keeping is a note
  (`experiment_add_note`).
- **Waking.** Lines are held while the level is Off, and come as one turn
  once it is back; started at Off, the agent's first turn waits for it too. A `GAP`, a restart or the level coming back tells the
  agent to re-read the status. If the app can't be watched, the harness waits
  up to `--server-wait` seconds (default 600) for it to answer again, then
  stops with exit code 1.
- **Playbooks.** `--playbook NAME` starts the session on `NAME.yaml` from the
  lab's `~/.fibsem/knowledge/playbooks/` or the shipped
  `.claude/fibsem-agent/playbooks/`, the lab's winning; with none named it starts on
  `fibsem-agent`, which the shipped ones extend. A playbook names its skills,
  reminders, rules and subagents, and its `session:` sets the model, effort,
  budget and turns, the mode and its folder, and the level the gateway's card
  picks first. Flags on the command line win over it; the API key is never in
  it. `--show-config` prints what the agent would run with and starts nothing,
  `--list-playbooks` lists them, and `--json` gives either as JSON.
- **What wakes it.** A playbook's `wakes` names the lines that wake the
  agent, by group: `decisions` (proposals, suggestions, requests), `run`
  (workflows, tasks, long commands), `checks` (autofocus, alignment, mills)
  and `notes` (the operator's notes and plan edits). The
  nearest playbook that names them wins; with none, every line wakes it, as
  the shipped playbooks do. A line left out goes in the agent's next turn.
  The operator's messages, the level, Stop, the watch's own state and the
  agent's own wake-ups always wake it, so `wakes: []` gives an agent that
  waits for the operator. `--show-config` shows the list.
- **Digests.** Within that, the agent's `wake_on` policy may hold routine
  lines (a task started or finished normally, a mill stage starting, a check
  that passed; `wake_lines.NEEDS` is the table) and give them together at the
  top of a later wake: with the next line that wakes it, after `digest_min`,
  or once `digest_lines` are held. The base playbook's `session:` block sets
  the mode per level (`wake_on_watch: digest`, `wake_on_assist` and
  `wake_on_drive: all`); the agent changes it with the `wake_on` tool, and
  its choice holds across level changes and a resume. The agent's transcript
  says each change and each digest ("Woke with 7 held line(s), ..."), and the
  harness log has every held line. A long command's end the agent already
  read with `service_wait_for_run` never wakes it, under any policy. To turn
  holding off, set `wake_on_watch: all` in a lab playbook, or ask the agent
  to call `wake_on` with mode `all`.
- **Defaults.** The shipped `fibsem-agent` playbook is the one place the
  defaults live, and it stays under every playbook, a lab's copy of it
  included: its skill, its `session:` settings (with the timers
  `wake_max_min`, `deadline_warn_min` and `quiet_min`) and its `instrument:`
  kind. The code keeps no second copy.
- **What a session loads.** Started on a playbook other than `fibsem-agent`,
  the session loads only that playbook's skills and subagents (with what it
  extends) and the lab's own; `playbook_set` then refuses a playbook that
  needs others, and the agent asks for a restart on it. Started on
  `fibsem-agent`, it loads what every playbook names, as intake may set any.
- **Instrument kind.** A playbook's `instrument:` names the folder
  `.claude/fibsem-agent/instruments/<kind>/` whose `prompt.md`, `intake.md`
  and `reread.md` are what the agent is told. `fibsem` is the only kind; an
  unknown one, a missing file or section, or a `{value}` nothing fills in
  fails at the start.
- **Reminders.** A playbook's `reminders` map a `when` to what to say: the text
  reaches the agent at the moment it makes a matching tool call, before it or
  after it (`--list-hooks` lists the options and their tools). The tools and
  the App Server know nothing of them.
- **Limits.** `--max-turns` (default: the playbook's, 40 as shipped) caps the
  tool calls in one wake-up; the next wake-up starts a fresh count.
  `--max-budget-usd` (default: the playbook's, 500 as shipped) caps what one
  run of `fibsem-agent` spends, and reaching it
  stops the harness with exit code 2. `--model` picks the model.
  The SDK compacts a long session's context by itself, and the log says when.
- **Stopping and resuming.** Ctrl-C interrupts the agent's turn and ends the
  session cleanly; a second Ctrl-C ends it at once. The log names the session
  id, and `--resume <id>` carries it on after a restart; a bare `--resume` carries
  on the last session run in the same folder. The watcher starts
  again from then, so the agent is told to re-read where things stand.
- **From another machine**, pair once with `fibsem-mcp pair <code>`, and pass
  `--url` if it isn't the tunnel's end.

## Order and troubleshooting

- **Order doesn't matter**: the MCP server answers the MCP client at once, even
  with no server running, and connects on the first tool call, which waits up
  to 30 s for a server to appear (`--wait` tunes it). If none appears, that
  call answers with what to do; start the server and call again, no reconnect
  needed. Started with no App Server, the MCP server offers every tool, including
  the app's. `fibsem-mcp --check` prints the connection status and exits;
  run it in a terminal to see what the MCP server sees.
- **Wrong server version / unknown arguments**: if you launched with
  `python -m` from inside another fibsem checkout, that directory shadows
  your installed copy. Run from the repo you installed, or any directory
  without a `fibsem/` folder.
- **Stale discovery file**: a hard-killed server can leave
  `~/.fibsem/app-server.json` behind. Readers reject it (dead pid), and the
  startup guard tells you the path — delete it if asked.

## Safety model, in one paragraph

Loopback bind keeps everything off the network by default; the per-session
bearer token keeps other users on the same machine out; the agent level the
operator sets from the Agent chip decides what the agent may do, and nothing
moves or images below Drive; one hardware command runs at a time (a second gets
`409 microscope_busy`); the agent stops its own mill at Assist or Drive and anything
else at Drive, always with a reason, and a stop never waits.
