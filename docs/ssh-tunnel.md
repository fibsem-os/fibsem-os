# Running the agent on another computer

The agent can run on a laptop or a server instead of the microscope PC. It reaches the
microscope PC's gateway through an SSH tunnel, and the gateway passes its calls to the App
Server. The gateway stays bound to `127.0.0.1` on the microscope PC, the tunnel carries its
traffic encrypted, and nothing is opened to the lab network except SSH. One port carries
everything: the agent's calls, the web app and pairing.

```
your computer                               microscope PC
fibsem-agent ─┐
browser      ─┴► localhost:8010 ══ ssh ══►  127.0.0.1:8010  gateway ──► App Server
```

Both computers need the same fibsemOS version. Your computer needs
`pip install "fibsem[agent]"` and an API key; the microscope PC needs no internet.

## On the microscope PC (once, as an administrator)

Turn on Windows' built-in SSH server. In PowerShell, run as administrator:

```powershell
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Start-Service sshd
Set-Service -Name sshd -StartupType Automatic
```

This also opens port 22 in Windows Firewall. If the lab network has its own firewall,
ask for port 22 from your computer to the microscope PC.

Make a standard (not administrator) Windows account for the tunnel, such as `fibsem-tunnel`.
It needs nothing but the login. Don't use an administrator account: their keys go in one
shared file, `C:\ProgramData\ssh\administrators_authorized_keys`, that lets the key into
every administrator account.

Add your computer's key (see below) to that account's
`C:\Users\fibsem-tunnel\.ssh\authorized_keys`, with this in front of it on the same line:

```
restrict,port-forwarding,permitopen="127.0.0.1:8010",command="echo tunnel only" ssh-ed25519 AAAA... agent-laptop
```

`restrict` turns off everything, `port-forwarding` and `permitopen` allow only the
gateway's port, and `command` replaces any command the key asks to run, so the key opens
the tunnel and nothing else.

Use one key per computer. To stop a computer connecting, delete its line, and
unpair it on the gateway's page, **Pairing**.

## On your computer (once)

Make a key, and give the `.pub` file to whoever set up the microscope PC:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/microscope
```

Add the microscope PC to `~/.ssh/config`:

```
Host microscope
    HostName <microscope-pc>
    User fibsem-tunnel
    IdentityFile ~/.ssh/microscope
    LocalForward 8010 127.0.0.1:8010
    ServerAliveInterval 15
    ServerAliveCountMax 3
    ExitOnForwardFailure yes
```

`LocalForward` listens on your computer's `127.0.0.1` only. Don't add `-g` or
`GatewayPorts`: that would open the tunnel to your computer's network. On a computer
other people log in to, they can reach `localhost:8010` too; they still need a token, which
is kept readable by you alone.

## Each session

Start the tunnel and leave it running:

```bash
ssh -N microscope
```

A dropped tunnel is not restarted by itself. To restart it after a network blip, run it in a
loop, or with `autossh -M 0 -N microscope` (`brew install autossh`, `apt install autossh`):

```bash
while true; do ssh -N microscope; sleep 5; done
```

The first time, pair the agent at the gateway. On the gateway's page, open **Pairing →
Pair a computer…** and pick **Agent**. On your computer:

```bash
fibsem-mcp pair <code>
```

The token is kept in `~/.fibsem/agent/` on your computer, so later sessions don't pair
again. Then run the agent against the tunnel's end:

```bash
fibsem-agent --url http://localhost:8010 --playbook <playbook> --goal "..."
```

or, for Claude Code:

```bash
claude mcp add fibsem -- fibsem-mcp --url http://localhost:8010
```

## Claude Code with nothing installed

The gateway also serves the fibsem MCP itself, at `/mcp/fibsem`. Claude Code (or any MCP
client that speaks streamable HTTP) can use it through the tunnel with no fibsem on your
computer. The microscope PC needs the `mcp` extra (`pip install "fibsem[mcp]"`).

Pair once with the code from **Pairing → Pair a computer… → Agent**, and keep the token
it answers (it is the agent's; keep it to yourself):

```bash
curl -s -X POST http://localhost:8010/pair -H 'Content-Type: application/json' \
    -d '{"code": "<code>", "name": "my-laptop"}'
```

Then add the MCP with that token:

```bash
claude mcp add --transport http fibsem http://localhost:8010/mcp/fibsem \
    --header "Authorization: Bearer <token>"
```

The tools are the same ones `fibsem-mcp` serves, with two differences: a file you fetch
comes back in the answer rather than being saved on your computer, and evidence is sent as
its bytes (`upload`, base64) or named as an experiment image. Options the stdio sidecar takes
as flags go on the address: `?instrument=<kind>`, `?record=no`, `?always_load=<tool>,...`.

Nothing wakes the agent between turns unless it watches. The MCP tells it how at the
start: it runs, in the background,

```bash
curl -sN -H "Authorization: Bearer <token>" http://localhost:8010/mcp/fibsem/watch
```

which prints the same lines as `fibsem-mcp watch`, one per event worth waking for. Started
again, it carries on after the last event it printed. A client that can't run commands in
the background calls the `app_wait` tool instead, which answers the same lines; nothing
wakes such a client between calls, so it suits work with a person watching, not
unattended runs. Agents run by `fibsem-agent` still start their own `fibsem-mcp`.

The web app is at `http://localhost:8010/ui/agent` and the gateway's page at
`http://localhost:8010/ui/gateway`, paired once with a code (**Pair a computer** in the
gateway's window).

## When the tunnel drops

The run on the microscope carries on: the app does not need the agent. `fibsem-agent`
waits up to 10 minutes for the gateway to answer again (`--server-wait`, in seconds),
then stops. Restart the tunnel within that time and the agent watches again, and is told it
missed events in between.

The tunnel's requests reach the gateway from the PC itself. With **Always allow agents
logged in to this PC** ticked, the agent is still paired by its code: only a program on the
PC can read the connect token in the gateway's `gateway.json`.
