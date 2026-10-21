# fibsem-mcp — connect an agent to the microscope

> New to the agent? Start with the guide: [docs/app-server.md](../../../../docs/app-server.md) — what the agent can see and do at each level, and the security model.

The MCP sidecar lets Claude (Code or Desktop) work through AutoLamella's App
Server: acquire and *look at* images, read state, move the stage, and stop a
mill. The server is the security boundary: the sidecar registers every tool,
whatever the agent's level, and the server refuses what the level doesn't allow
with a 403 that says so. The microscope's tools are registered even while the
microscope isn't served; until it is, each answers 404 `microscope_off`. Only
the app's tools depend on the server hosting the application. Each tool's description names the level it needs. If the server
restarts with a new token, the sidecar re-reads the discovery file and carries
on.

## Install

```bash
pip install -e ".[server,mcp]"
```

## 1. Start the App Server

In AutoLamella, turn the App Server on in Tools → App Server (the guide,
[docs/app-server.md](../../../../docs/app-server.md), walks through it). *Serve the
microscope* there serves the routes the microscope tools reach; the Agent chip
sets what the agent may do with them.

- While running, the server writes `~/.fibsem/app-server.json`
  (url + token) so local clients connect without copying anything, and
  refuses to start if another live server owns that file.
- It binds `127.0.0.1` by default; exposing it on the LAN is an explicit
  choice, and the token then travels as cleartext HTTP.

## 2. Register the sidecar

```bash
claude mcp add -s user fibsem -- fibsem-mcp
```

`-s user` registers it for all your projects. In AutoLamella, *Copy for your
agent* in the Agent chip's *Connect an agent…* gives the same setup with the
absolute path filled in, for the agent to run itself. Same-machine setups need
nothing more: the sidecar autodiscovers the running
server. For a server on another machine, pair once: *Pair an agent* in
*Connect an agent…* (or the chat page's Computers panel) shows a code, and

```bash
fibsem-mcp pair 1234            # the SSH tunnel's end, http://localhost:8001
fibsem-mcp pair 1234 --url http://microscope-pc:8001
```

keeps this computer's own agent token in `~/.fibsem/agent/`. Every later launch
reads it with no token passed; *Unpair* on the PC cuts that agent off. With a
url other than the tunnel's, register the sidecar with `--url` too.
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

It finds the server as the sidecar does (`--url`/`--token`, the environment
variables, the discovery file) and follows a restarted app the same way;
`--since N` resumes after event N. It
prints `PROMPT`, `ANSWERED`, `PROMPT WITHDRAWN`, `PROPOSAL`,
`PROPOSAL RELEASED`, `REQUEST` (the agent's request decided, finished or
cancelled), `COMMAND ENDED` (a long command's end), task `EVENT` lines,
`WORKFLOW STARTED`, `WORKFLOW ENDED` and `GAP` (events were lost; re-read the
prompt and status). A workflow ends completed, cancelled, stalled or failed;
one that stops without recording an end is reported as
`WORKFLOW ENDED (stopped without an end event)`. A workflow's end does not end
the watch: the agent may start the next one. While the operator has the agent
level at Off it prints `AGENT OFF` and waits, then `AGENT BACK`. It exits 1 when the app cannot be
watched, and otherwise runs until it is stopped. It needs only fibsem, not the `[mcp]` extra; where
`fibsem-mcp` is not on the PATH, `python -m fibsem.ai.agent.mcp.sidecar watch` is the
same thing.

## 5. Or run it with nobody at a terminal: `fibsem-agent`

Everything above is Claude Code in a terminal, kept awake by a person.
`fibsem-agent` is a second way to run the same agent, on the
[Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk): one long
session with this sidecar as its only MCP server, and the watcher run for it,
so each batch of watcher lines is the agent's next turn. It reads the same
skills, from this checkout's `.claude/skills`. The Claude Code path is
unchanged.

```bash
pip install -e ".[server,agent]"
export ANTHROPIC_API_KEY=...       # a Claude API key
fibsem-agent
```

- **The level decides.** The agent works by the skill for its level
  (observe at Watch, assist at Assist, supervise at Drive), and the watcher
  prints a `LEVEL` line when the operator changes it. The level is set from
  the Agent chip, the server still refuses what it doesn't allow, and the
  runner never sets it.
- **Tools.** The `mcp__fibsem__*` tools run without a prompt, since the
  server decides what they may do. The agent also has the Skill tool,
  ToolSearch to find the fibsem tools, and the Agent tool for the checkout's
  three `autolamella-*` agents (`.claude/agents`). Nothing else, for those
  agents either: no shell, no file reads or writes, no web. Nothing waits on
  a permission prompt; anything else is denied. So the offline
  `read_record.py` reader stays with Claude Code.
- **Agents.** In the supervise skill's lead mode the agent is the Run
  Lead and makes every decision. It calls the Verifier for a second opinion
  on the site list, each Setup placement and every polishing pass, the
  Investigator when a task's result is bad, and the Auditor for the review
  at the end. Each runs inside the turn that called it, with only the tools
  its file lists, and only the Run Lead decides. The CLI's own agents are
  refused. Each agent sets its own model and effort; `--effort` sets the
  Run Lead's (default high). A lead run calls the most capable model often:
  raise `--max-budget-usd` to match the night.
- **Your Claude Code config stays out.** The session loads this checkout's
  project settings and skills, not `~/.claude`, and only this sidecar, not a
  `fibsem` server you registered with `claude mcp add`.
- **Signing in.** Set `ANTHROPIC_API_KEY` to a Claude API key in its
  environment, or configure Amazon Bedrock or Google Vertex as the Agent SDK
  documents. Not a claude.ai login: products built on the Agent SDK use API
  keys. If the model refuses the sign-in, the runner stops with exit code 1
  rather than wait. Its log never shows a bearer token or an API key.
- **Instructions.** It reads this checkout's skills and nothing else: not the
  checkout's `CLAUDE.md` (developer guidance) and not Claude Code's auto memory.
- **What it says.** The agent's replies and tool calls go to the console and
  to `~/.fibsem/agent/fibsem-agent-<time>.log` (`--log` to choose). What the
  operator must see in the app, it puts in a note (`experiment_add_note`),
  answering a message from the Agent card with its `reply_to`.
- **Waking.** Lines are held while the level is Off, and come as one turn
  once it is back; started at Off, the agent's first turn waits for it too. A `GAP`, a restart or the level coming back tells the
  agent to re-read the status. If the app can't be watched, the runner waits
  up to `--server-wait` seconds (default 600) for it to answer again, then
  stops with exit code 1.
- **Limits.** `--max-turns` (default 40) caps the tool calls in one
  wake-up; the next wake-up starts a fresh count. `--max-budget-usd`
  (default 5) caps what one run of `fibsem-agent` spends, and reaching it
  stops the runner with exit code 2. `--model` picks the model.
  The SDK compacts a long session's context by itself, and the log says when.
- **Stopping and resuming.** Ctrl-C interrupts the agent's turn and ends the
  session cleanly; a second Ctrl-C ends it at once. The log names the session
  id, and `--resume <id>` carries it on after a restart. The watcher starts
  again from then, so the agent is told to re-read where things stand.
- **From another machine**, pair once with `fibsem-mcp pair <code>`, and pass
  `--url` if it isn't the tunnel's end.

## Order and troubleshooting

- **Order doesn't matter**: the sidecar answers the MCP client at once, even
  with no server running, and connects on the first tool call, which waits up
  to 30 s for a server to appear (`--wait` tunes it). If none appears, that
  call answers with what to do; start the server and call again, no reconnect
  needed. Started with no server, the sidecar offers every tool, including
  the app's. `fibsem-mcp --check` prints the connection status and exits;
  run it in a terminal to see what the sidecar sees.
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
