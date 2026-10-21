import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Facts } from "@/components/ui/card";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { api } from "@/lib/api";
import type { Capabilities, Pairing } from "@/lib/types";
import { cn } from "@/lib/utils";

const SECTIONS = { connections: "Connections", about: "About" } as const;
type Section = keyof typeof SECTIONS;

function thisComputer(): string | null {
  try {
    return localStorage.getItem("fibsem-paired-id");
  } catch {
    return null;
  }
}

function Connections() {
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
    onSuccess: (answer) => client.setQueryData(["pairing"], answer),
  });
  const [unpairing, setUnpairing] = useState<string | null>(null);
  const me = thisComputer();
  const data = pairing.data;
  const left = data ? Math.max(0, Math.round(data.expires_in_s)) : 0;

  if (pairing.error) return <p className="m-0 text-[13px] text-muted-foreground">{(pairing.error as Error).message}</p>;
  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-3">
        <h3 className="m-0 text-[13px] font-semibold text-muted-foreground">Paired computers</h3>
        <div className="flex-1" />
        {!data?.open && (
          <Button onClick={() => act.mutate({ method: "POST", path: "/pairing", body: { role: "operator" } })}>
            Pair a computer…
          </Button>
        )}
      </div>
      {data?.open && (
        <div className="flex flex-wrap items-center gap-4 rounded-md border border-border p-3.5">
          <div className="flex flex-col gap-1.5">
            <span className="text-xs text-muted-foreground">Pair as</span>
            <span className="flex gap-1">
              {(["operator", "agent"] as const).map((role) => (
                <Button
                  key={role}
                  variant={data.role === role ? "default" : "ghost"}
                  onClick={() => act.mutate({ method: "POST", path: "/pairing", body: { role } })}
                >
                  {role === "operator" ? "Operator" : "Agent"}
                </Button>
              ))}
            </span>
          </div>
          <div className="flex flex-col gap-0.5">
            <span className="text-[32px] font-semibold tracking-[0.2em] tabular-nums">{data.code}</span>
            <span className="text-xs text-muted-foreground">
              {data.role === "agent"
                ? `Run fibsem-mcp pair ${data.code} on the agent's computer`
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
                Paired {new Date(computer.paired_at).toLocaleString([], { dateStyle: "medium", timeStyle: "short" })}
              </span>
            </span>
            <span className="text-[13px]">{computer.role === "agent" ? "Agent" : "Operator"}</span>
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
          <div className="min-w-0 flex-1 px-5 py-4">{section === "connections" ? <Connections /> : <About />}</div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
