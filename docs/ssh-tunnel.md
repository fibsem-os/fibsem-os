# Running the agent on another computer

The agent can run on a laptop or a server instead of the microscope PC. It reaches the
microscope PC's App Server through an SSH tunnel. The App Server and the gateway stay
bound to `127.0.0.1` on the microscope PC, the tunnel carries their traffic encrypted, and
nothing is opened to the lab network except SSH.

```
agent computer                              microscope PC
fibsem-agent ──► localhost:8001 ══ ssh ══►  127.0.0.1:8001  App Server
browser      ──► localhost:8010 ══ ssh ══►  127.0.0.1:8010  gateway
```

Both computers need the same fibsemOS version. The agent computer needs
`pip install "fibsem[agent]"` and an API key; the microscope PC needs no internet.

## On the microscope PC (once, as an administrator)

Turn on Windows' built-in SSH server. In PowerShell, run as administrator:

```powershell
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Start-Service sshd
Set-Service -Name sshd -StartupType Automatic
```

This also opens port 22 in Windows Firewall. If the lab network has its own firewall,
ask for port 22 from the agent computer to the microscope PC.

Make a standard (not administrator) Windows account for the tunnel, such as `fibsem-tunnel`.
It needs nothing but the login. Don't use an administrator account: their keys go in one
shared file, `C:\ProgramData\ssh\administrators_authorized_keys`, that lets the key into
every administrator account.

Add the agent computer's key (see below) to that account's
`C:\Users\fibsem-tunnel\.ssh\authorized_keys`, with this in front of it on the same line:

```
restrict,port-forwarding,permitopen="127.0.0.1:8001",permitopen="127.0.0.1:8010",command="echo tunnel only" ssh-ed25519 AAAA... agent-laptop
```

`restrict` turns off everything, `port-forwarding` and `permitopen` allow only the two
ports, and `command` replaces any command the key asks to run, so the key opens the tunnel
and nothing else.

Use one key per agent computer. To stop a computer connecting, delete its line, and
unpair it in the web app's **Settings → Paired computers**.

## On the agent computer (once)

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
    LocalForward 8001 127.0.0.1:8001
    LocalForward 8010 127.0.0.1:8010
    ServerAliveInterval 15
    ServerAliveCountMax 3
    ExitOnForwardFailure yes
```

`LocalForward` listens on the agent computer's `127.0.0.1` only. Don't add `-g` or
`GatewayPorts`: that would open the tunnel to the agent computer's network. On a computer
other people log in to, they can reach `localhost:8001` too; they still need a token, which
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

The first time, pair the agent with the App Server. On the microscope PC, open the web
app's **Settings → Paired computers → Pair a computer…** and pick **Agent**. On the agent
computer:

```bash
fibsem-mcp pair <code>
```

The token is kept in `~/.fibsem/agent/` on the agent computer, so later sessions don't pair
again. Then run the agent against the tunnel's end:

```bash
fibsem-agent --url http://localhost:8001 --playbook <playbook> --goal "..."
```

or, for Claude Code:

```bash
claude mcp add fibsem -- fibsem-mcp --url http://localhost:8001
```

The web app is at `http://localhost:8001/ui/agent` and the gateway's page at
`http://localhost:8010/ui/gateway`, each paired once with a code (**Pair a computer** in
the gateway's window).

## When the tunnel drops

The run on the microscope carries on: the app does not need the agent. `fibsem-agent`
waits up to 10 minutes for the App Server to answer again (`--server-wait`, in seconds),
then stops. Restart the tunnel within that time and the agent watches again, and is told it
missed events in between.

The tunnel's requests reach the App Server from the PC itself. With **Always allow agents
logged in to the microscope PC** ticked, the agent is still paired by its code: only a
program on the PC can read the discovery file's connect token.
