import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Facts } from "@/components/ui/card";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { api, forgetToken } from "@/lib/api";
import * as fmt from "@/lib/format";
import { setNoticePrefs, systemNotices, useNoticePrefs } from "@/lib/notify";
import type { Capabilities, Pairing, PairingRole } from "@/lib/types";
import { cn } from "@/lib/utils";

const SECTIONS = { connections: "Connections", notifications: "Notifications", about: "About" } as const;
type Section = keyof typeof SECTIONS;

function thisComputer(): string | null {
  try {
    return localStorage.getItem("fibsem-paired-id");
  } catch {
    return null;
  }
}

const ROLE_NAMES: Record<PairingRole, string> = { operator: "Operator", agent: "Agent", hub: "Hub" };
// A hub may start and stop apps on a gateway only with this grant.
const SWITCH_MODES = "switch_modes";

function GrantToggle({ on, onChange }: { on: boolean; onChange: (on: boolean) => void }) {
  return (
    <label className="flex cursor-pointer items-center gap-1.5 text-[13px]">
      <input
        type="checkbox"
        className="accent-primary"
        checked={on}
        onChange={(event) => onChange(event.target.checked)}
      />
      May start and stop apps
    </label>
  );
}

/** The paired computers and pairing a new one: the App Server's operators and
 * agents, or a gateway's operators and hubs. */
export function Connections({ roles = ["operator", "agent"] }: { roles?: PairingRole[] }) {
  const client = useQueryClient();
  const pairing = useQuery({
    queryKey: ["pairing"],
    queryFn: () => api<Pairing>("/pairing"),
    // The code's time left counts down while it is open.
    refetchInterval: (query) => (query.state.data?.open ? 1000 : false),
  });
  const act = useMutation({
    mutationFn: ({ method, path, body }: { method: string; path: string; body?: unknown }) =>
      api<Pairing>(path, { method, body: body === undefined ? undefined : JSON.stringify(body) }),
    onSuccess: (answer, { method, path }) => {
      // Unpairing this browser ends its session now, not at its next refused call.
      if (method === "DELETE" && path.endsWith(`/${encodeURIComponent(thisComputer() ?? "")}`)) forgetToken("unpaired");
      else client.setQueryData(["pairing"], answer);
    },
  });
  const [unpairing, setUnpairing] = useState<string | null>(null);
  const me = thisComputer();
  const data = pairing.data;
  // One agent per microscope: pairing another replaces it.
  const pairedAgent = data?.computers.find((c) => c.role === "agent");
  const left = data ? Math.max(0, Math.round(data.expires_in_s)) : 0;

  if (pairing.error) return <p className="m-0 text-[13px] text-muted-foreground">{(pairing.error as Error).message}</p>;
  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-3">
        <h3 className="m-0 text-[13px] font-semibold text-muted-foreground">Paired computers</h3>
        <div className="flex-1" />
        {!data?.open && (
          <Button onClick={() => act.mutate({ method: "POST", path: "/pairing", body: { role: roles[0] } })}>
            Pair a computer…
          </Button>
        )}
      </div>
      {data?.open && (
        <div className="flex flex-wrap items-center gap-4 rounded-md border border-border p-3.5">
          <div className="flex flex-col gap-1.5">
            <span className="text-xs text-muted-foreground">Pair as</span>
            <span className="flex gap-1">
              {roles.map((role) => (
                <Button
                  key={role}
                  variant={data.role === role ? "default" : "ghost"}
                  onClick={() => act.mutate({ method: "POST", path: "/pairing", body: { role } })}
                >
                  {ROLE_NAMES[role]}
                </Button>
              ))}
            </span>
            {data.role === "hub" && (
              <GrantToggle
                on={data.grants.includes(SWITCH_MODES)}
                onChange={(on) =>
                  act.mutate({ method: "POST", path: "/pairing", body: { role: "hub", grants: on ? [SWITCH_MODES] : [] } })
                }
              />
            )}
          </div>
          <div className="flex flex-col gap-0.5">
            <span className="text-[32px] font-semibold tracking-[0.2em] tabular-nums">{data.code}</span>
            <span className="text-xs text-muted-foreground">
              {data.role === "agent"
                ? `Run fibsem-mcp pair ${data.code} on the agent's computer${pairedAgent ? `. It replaces ${pairedAgent.name}` : ""}`
                : data.role === "hub"
                  ? "Enter it in the hub's Add instrument"
                  : "Enter it in the other computer's browser"}{" "}
              · {Math.floor(left / 60)}:{String(left % 60).padStart(2, "0")} left
            </span>
          </div>
          <div className="flex-1" />
          <Button variant="ghost" onClick={() => act.mutate({ method: "DELETE", path: "/pairing" })}>
            Close pairing
          </Button>
        </div>
      )}
      {data && roles.includes("agent") && (
        <label className="flex items-center gap-2 text-[13px]">
          <input
            type="checkbox"
            className="accent-primary"
            checked={data.always_allow_local}
            onChange={(e) =>
              act.mutate({ method: "PUT", path: "/pairing/always_allow_local", body: { on: e.target.checked } })
            }
          />
          Always allow agents logged in to the microscope PC
          <span className="text-muted-foreground">· over SSH too; they act only at the level you set</span>
        </label>
      )}
      <div className="flex flex-col">
        {data?.computers.length === 0 && (
          <p className="m-0 border-t border-border py-2.5 text-[13px] text-muted-foreground">None yet.</p>
        )}
        {data?.computers.map((computer) => (
          <div
            key={computer.id}
            className="grid grid-cols-[minmax(0,1fr)_auto_auto] items-center gap-3 border-t border-border py-2.5"
          >
            <span className="flex min-w-0 flex-col gap-0.5">
              <span className="font-semibold">
                {computer.name}
                {computer.id === me && <span className="font-normal text-muted-foreground"> · this browser</span>}
              </span>
              <span className="text-xs text-muted-foreground">
                Paired {fmt.when(computer.paired_at)}
              </span>
              {computer.role === "hub" && (
                <GrantToggle
                  on={computer.grants.includes(SWITCH_MODES)}
                  onChange={(on) =>
                    act.mutate({
                      method: "PATCH",
                      path: `/pairing/computers/${encodeURIComponent(computer.id)}`,
                      body: { grants: on ? [SWITCH_MODES] : [] },
                    })
                  }
                />
              )}
            </span>
            <span className="text-[13px]">{ROLE_NAMES[computer.role]}</span>
            {unpairing === computer.id ? (
              <span className="flex gap-1">
                <Button
                  variant="danger"
                  onClick={() => act.mutate({ method: "DELETE", path: `/pairing/computers/${encodeURIComponent(computer.id)}` })}
                >
                  Unpair
                </Button>
                <Button variant="ghost" onClick={() => setUnpairing(null)}>
                  Cancel
                </Button>
              </span>
            ) : (
              <Button variant="danger" onClick={() => setUnpairing(computer.id)}>
                Unpair…
              </Button>
            )}
          </div>
        ))}
      </div>
      {act.error && <p className="m-0 text-[13px] text-destructive">{(act.error as Error).message}</p>}
    </div>
  );
}

/** How this browser tells the operator, beyond its toasts, that the agent needs an answer. */
function Notifications() {
  const prefs = useNoticePrefs();
  const can = systemNotices();
  const [permission, setPermission] = useState(() => (can ? Notification.permission : "denied"));
  const state = !can
    ? "Not available here: system notifications need localhost or the SSH tunnel."
    : permission === "granted"
      ? "Allowed in this browser. A system notification, while this tab is in the background."
      : permission === "denied"
        ? "Blocked in this browser: allow notifications for this site in its settings."
        : "The browser asks for permission when you turn it on.";
  return (
    <div className="flex flex-col gap-3 text-[13px]">
      <label className="flex items-start gap-2">
        <input
          type="checkbox"
          className="mt-0.5 accent-primary"
          checked={prefs.toasts}
          onChange={(e) => setNoticePrefs({ toasts: e.target.checked })}
        />
        <span className="flex flex-col gap-0.5">
          Show toasts when the agent needs an answer
          <span className="text-muted-foreground">
            On every page but the Agent page. The tab title counts them either way.
          </span>
        </span>
      </label>
      <label className="flex items-center gap-2">
        <input
          type="checkbox"
          className="accent-primary"
          checked={prefs.errors}
          onChange={(e) => setNoticePrefs({ errors: e.target.checked })}
        />
        Also for an action of the agent's that failed
      </label>
      <label className="flex items-start gap-2">
        <input
          type="checkbox"
          className="mt-0.5 accent-primary"
          disabled={!can}
          checked={can && prefs.system && permission === "granted"}
          onChange={async (e) => {
            const on = e.target.checked;
            const allowed = on && permission !== "granted" ? await Notification.requestPermission() : permission;
            setPermission(allowed);
            setNoticePrefs({ system: on && allowed === "granted" });
          }}
        />
        <span className="flex flex-col gap-0.5">
          Tell me outside the browser when the agent needs an answer
          <span className="text-muted-foreground">{state}</span>
        </span>
      </label>
      <label className="flex items-center gap-2">
        <input
          type="checkbox"
          className="accent-primary"
          checked={prefs.sound}
          onChange={(e) => setNoticePrefs({ sound: e.target.checked })}
        />
        Play a sound when the agent asks
      </label>
    </div>
  );
}

function About() {
  const caps = useQuery({ queryKey: ["capabilities"], queryFn: () => api<Capabilities>("/capabilities") });
  return (
    <Facts
      className="max-w-md [&_dd]:text-left"
      rows={[
        ["App Server", location.host],
        ["App", caps.data?.app_name ?? "–"],
        ["fibsemOS", caps.data?.fibsem?.version ?? "–"],
        ["API", caps.data?.api_version ?? "–"],
      ]}
    />
  );
}

/** The gear: settings that are not a page. */
export function Settings({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const [section, setSection] = useState<Section>("connections");
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent title="Settings">
        <div className="flex min-h-[420px] flex-col overflow-auto sm:flex-row">
          <nav
            aria-label="Settings sections"
            className="flex shrink-0 gap-0.5 border-b border-border p-2.5 sm:w-[170px] sm:flex-col sm:border-r sm:border-b-0"
          >
            {(Object.keys(SECTIONS) as Section[]).map((key) => (
              <button
                key={key}
                onClick={() => setSection(key)}
                className={cn(
                  "cursor-pointer rounded border-0 bg-transparent px-2.5 py-2 text-left text-[13px] text-muted-foreground hover:bg-accent",
                  section === key && "bg-accent text-foreground",
                )}
              >
                {SECTIONS[key]}
              </button>
            ))}
          </nav>
          <div className="min-w-0 flex-1 px-5 py-4">
            {section === "connections" ? <Connections /> : section === "notifications" ? <Notifications /> : <About />}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
