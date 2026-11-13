import { type ReactNode, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { api } from "@/lib/api";
import {
  agentMeta,
  at,
  Dot,
  Failed,
  FIELD,
  Hint,
  type HubState,
  type Instrument,
  leaf,
  LevelControl,
  RED,
  StartForm,
  uptime,
  useHubAction,
} from "@/pages/hub/shared";

function Line({
  label,
  children,
  actions,
}: {
  label: string;
  children: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <div className="grid min-h-[52px] grid-cols-[110px_minmax(0,1fr)_auto] items-center gap-3 border-t border-border px-4 py-3 first:border-t-0">
      <span className="text-muted-foreground">{label}</span>
      <span className="flex min-w-0 items-center gap-2">{children}</span>
      <span className="flex items-center gap-2">{actions}</span>
    </div>
  );
}

const MUTED = "text-muted-foreground";

/** What can't start an agent for it yet, in a few words. */
function notReady(i: Instrument): string | undefined {
  if (!i.reachable) return "Its PC isn't reachable.";
  if (!i.paired || !i.gateway) return "Pair this hub with it first.";
  if (i.gateway.mode !== "running") return "Its app isn't running.";
  return undefined;
}

export function AgentCard({ instrument: i }: { instrument: Instrument }) {
  const end = useHubAction(() =>
    api<HubState>(`${at(i.name)}/agent/end`, { method: "POST" }),
  );
  const agent = i.session ?? i.agent;
  return (
    <Card className="flex flex-col gap-2.5 p-4 text-[13px]">
      <div className="flex items-center gap-2">
        <Dot colour={agent ? "ok" : "none"} />
        <span className="font-semibold text-muted-foreground">Agent</span>
      </div>
      {agent ? (
        <>
          <div className="text-[15px]">
            {agent.goal || agent.playbook || "Running"}
          </div>
          <div className={MUTED}>{agentMeta(i, !i.session)}</div>
          {i.session ? (
            <div className="mt-1.5 flex items-center gap-3">
              <span className="text-[12.5px] text-muted-foreground">Level</span>
              <LevelControl instrument={i} />
              <button
                type="button"
                className={`${RED} ml-auto`}
                disabled={end.isPending}
                onClick={() => end.mutate()}
              >
                End
              </button>
            </div>
          ) : (
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              Started at its PC
              <Hint>Its level and its End are on the microscope PC.</Hint>
            </div>
          )}
          <Failed error={end.error} />
        </>
      ) : (
        <StartForm key={i.name} instrument={i} disabled={notReady(i)} />
      )}
    </Card>
  );
}

/** One instrument: its microscope and app, and the agent to start or end. */
export function InstrumentView({ instrument: i }: { instrument: Instrument }) {
  const stop = useHubAction(() =>
    api<HubState>(`${at(i.name)}/app/stop`, { method: "POST" }),
  );
  const apps = i.gateway?.apps ?? [];
  const [app, setApp] = useState("");
  const chosen = app || apps[0]?.key || "";
  const start = useHubAction(() =>
    api<HubState>(`${at(i.name)}/app/start`, {
      method: "POST",
      body: JSON.stringify({ app: chosen }),
    }),
  );
  const running = i.gateway?.mode === "running";
  if (!i.reachable || !i.gateway)
    return (
      <Card className="flex flex-col gap-1 p-4 text-[13px]">
        <span className="flex items-center gap-2">
          <Dot colour={i.reachable ? "warn" : "off"} />
          {i.reachable === false
            ? "Not reachable"
            : i.reachable === null
              ? "Checking"
              : "Not paired yet"}
        </span>
        {i.why && <span className={MUTED}>{i.why}</span>}
      </Card>
    );
  return (
    <>
      <Card className="p-0 text-[13px]">
        <Line label="Microscope">
          <Dot colour={i.microscope?.connected ? "ok" : running ? "off" : "none"} />
          {i.microscope?.connected ? (
            "Connected"
          ) : (
            <span className={MUTED}>
              {i.microscope?.connecting ? "Connecting" : "Not connected"}
            </span>
          )}
          {i.microscope?.configuration && (
            <span className={MUTED}>· {i.microscope.configuration}</span>
          )}
        </Line>
        <Line
          label="App"
          actions={
            running ? (
              <button
                type="button"
                className={RED}
                disabled={stop.isPending || !!i.gateway.switching}
                onClick={() => stop.mutate()}
              >
                Stop
              </button>
            ) : (
              apps.length > 0 && (
                <>
                  <select
                    aria-label="App"
                    value={chosen}
                    onChange={(event) => setApp(event.target.value)}
                    className={FIELD}
                  >
                    {apps.map((a) => (
                      <option key={a.key} value={a.key}>
                        {a.name}
                      </option>
                    ))}
                  </select>
                  <Button
                    disabled={start.isPending || !!i.gateway.switching}
                    title={
                      i.gateway.last_experiment
                        ? `Opens ${leaf(i.gateway.last_experiment)}, without a window on its PC.`
                        : "Starts without a window on its PC."
                    }
                    onClick={() => start.mutate()}
                  >
                    Start
                  </Button>
                </>
              )
            )
          }
        >
          <Dot colour={running ? "ok" : "none"} />
          {running ? (
            <>
              {i.gateway.app ?? "Running"}
              <span className={`${MUTED} truncate`}>
                {[
                  leaf(i.gateway.experiment),
                  i.gateway.since && `up ${uptime(i.gateway.since)}`,
                ]
                  .filter(Boolean)
                  .map((part) => `· ${part}`)
                  .join(" ")}
              </span>
            </>
          ) : (
            <span className={MUTED}>
              {i.gateway.switching ? "Starting" : "Not running"}
            </span>
          )}
        </Line>
        {(stop.error ?? start.error) ? (
          <div className="border-t border-border px-4 py-2.5">
            <Failed error={stop.error ?? start.error} />
          </div>
        ) : null}
      </Card>
      <AgentCard instrument={i} />
    </>
  );
}

