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
server. For a server on another machine (or a non-default setup):

```bash
claude mcp add -s user fibsem -- fibsem-mcp --url http://microscope-pc:8001 --token <token>
```

(`FIBSEM_SERVER_URL` / `FIBSEM_SERVER_TOKEN` environment variables also work.)

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
prints `PROMPT`, `ANSWERED`, `PROMPT WITHDRAWN`, `QUESTION`,
`QUESTION RELEASED`, `REQUEST` (the agent's request decided, finished or
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
