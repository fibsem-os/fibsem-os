import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Copy, Info } from "lucide-react";
import { type ReactNode, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { HoverCard } from "@/components/ui/hover-card";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import {
  type Configuration,
  ConfigurationFacts,
  dollars,
  FIELD,
  LEVELS,
  type Playbook,
  title,
} from "@/pages/gateway/AgentCard";

/** What the page fills in for the next start; unset is the page's own
 * choice. */
export interface Defaults {
  configuration: string | null;
  connect: boolean;
  app: string | null;
  window: boolean;
  playbook: string | null;
  level: string | null;
  budget_usd: number | null;
}

interface Secret {
  name: string;
  about: string;
  set: boolean;
  source: string | null;
}

const WARN = "text-[var(--fibsem-semantic-warning-color)]";

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="grid grid-cols-1 items-center gap-1 py-1.5 text-[13px] sm:grid-cols-[190px_minmax(0,1fr)] sm:gap-3">
      <span className="text-muted-foreground">{label}</span>
      <span className="min-w-0">{children}</span>
    </div>
  );
}

function Dot({ on }: { on: boolean }) {
  return (
    <span
      aria-hidden
      className={cn(
        "mr-2 inline-block size-2 rounded-full",
        on ? "bg-[var(--fibsem-ok-color)]" : "bg-[var(--fibsem-pending-color)]",
      )}
    />
  );
}

function CopyCommand({ command }: { command: string }) {
  const [copied, setCopied] = useState<"yes" | "no" | null>(null);
  useEffect(() => {
    if (!copied) return;
    const t = setTimeout(() => setCopied(null), 2000);
    return () => clearTimeout(t);
  }, [copied]);
  return (
    <Button
      title={command}
      onClick={() =>
        navigator.clipboard
          .writeText(command)
          .then(() => setCopied("yes"))
          .catch(() => setCopied("no"))
      }
    >
      <Copy size={13} />
      {copied === "yes"
        ? "Copied"
        : copied === "no"
          ? "Couldn't copy"
          : "Copy command"}
    </Button>
  );
}

/** Which keys this PC has, and where from. Never the key: it is set on the
 * PC itself. */
function Secrets() {
  const secrets = useQuery({
    queryKey: ["gateway/secrets"],
    queryFn: () => api<Secret[]>("/gateway/secrets"),
  });
  return (
    <Card className="flex flex-col gap-1">
      <CardTitle className="mb-1.5">Secrets</CardTitle>
      {secrets.error && (
        <p className="m-0 text-[13px] text-destructive">
          {(secrets.error as Error).message}
        </p>
      )}
      {secrets.data?.map((s) => {
        const command = `fibsem-cli key set ${s.name}`;
        return (
          <div
            key={s.name}
            className="grid grid-cols-1 items-center gap-1 border-t border-border py-2 text-[13px] sm:grid-cols-[190px_minmax(0,1fr)_auto] sm:gap-3"
          >
            <span className="font-mono text-xs text-muted-foreground">
              {s.name}
            </span>
            <span>
              <Dot on={s.set} />
              {s.set ? (
                `Set · from ${s.source}`
              ) : (
                <span className="text-muted-foreground">Not set</span>
              )}
            </span>
            <span className="flex items-center gap-2">
              <CopyCommand command={command} />
              <HoverCard
                trigger={
                  <button
                    aria-label={`About ${s.name}`}
                    className="flex cursor-pointer border-0 bg-transparent p-1 text-muted-foreground hover:text-foreground"
                  >
                    <Info size={15} />
                  </button>
                }
              >
                <p className="m-0 text-[13px]">
                  {s.about.charAt(0).toUpperCase() + s.about.slice(1)}. Set or
                  replace it on this PC with{" "}
                  <span className="font-mono text-xs">{command}</span>, then
                  restart the gateway.
                </p>
              </HoverCard>
            </span>
          </div>
        );
      })}
    </Card>
  );
}

function SubHeading({ first, children }: { first?: boolean; children: string }) {
  return (
    <h3
      className={cn(
        "m-0 pb-1 text-sm font-semibold text-foreground",
        !first && "mt-2 border-t border-border pt-3",
      )}
    >
      {children}
    </h3>
  );
}

function DefaultsCard({
  apps,
  configurations,
}: {
  apps: { key: string; name: string }[];
  configurations: { name: string; default: boolean }[];
}) {
  const client = useQueryClient();
  const saved = useQuery({
    queryKey: ["gateway/defaults"],
    queryFn: () => api<Defaults>("/gateway/defaults"),
  });
  const playbooks = useQuery({
    queryKey: ["gateway/agent/playbooks"],
    queryFn: () => api<Playbook[]>("/gateway/agent/playbooks"),
    staleTime: 30_000,
    retry: false,
  });
  const [edit, setEdit] = useState<Defaults | null>(null);
  const [budget, setBudget] = useState<string | null>(null);
  const save = useMutation({
    mutationFn: (body: unknown) =>
      api<Defaults>("/gateway/defaults", {
        method: "PUT",
        body: JSON.stringify(body),
      }),
    onSuccess: (answer) => {
      client.setQueryData(["gateway/defaults"], answer);
      setEdit(null);
      setBudget(null);
    },
  });
  const d = edit ?? saved.data;
  if (!d)
    return saved.error ? (
      <Card>
        <p className="m-0 text-[13px] text-destructive">
          {(saved.error as Error).message}
        </p>
      </Card>
    ) : null;
  const set = (change: Partial<Defaults>) => setEdit({ ...d, ...change });
  const shownBudget = budget ?? (d.budget_usd != null ? String(d.budget_usd) : "");
  const changed = edit !== null || budget !== null;
  const usable = (playbooks.data ?? []).filter((p) => !p.error);
  const select = (
    value: string | null,
    onChange: (value: string | null) => void,
    options: [string, string][],
    unset: string,
    label: string,
  ) => (
    <select
      value={value ?? ""}
      onChange={(event) => onChange(event.target.value || null)}
      aria-label={label}
      className={cn(FIELD, "w-full sm:w-72")}
    >
      <option value="">{unset}</option>
      {options.map(([v, text]) => (
        <option key={v} value={v}>
          {text}
        </option>
      ))}
    </select>
  );
  return (
    <Card className="flex flex-col gap-1">
      <CardTitle>Defaults</CardTitle>
      <p className="m-0 pb-2 text-[13px] text-muted-foreground">
        What Overview fills in. You can still change each start.
      </p>
      <SubHeading first>Microscope</SubHeading>
      <Row label="Configuration">
        {select(
          d.configuration,
          (configuration) => set({ configuration }),
          configurations.map((c) => [c.name, c.name]),
          `The default (${configurations.find((c) => c.default)?.name ?? configurations[0]?.name ?? "none"})`,
          "Default configuration",
        )}
      </Row>
      <Row label="Connect when the app starts">
        <input
          type="checkbox"
          checked={d.connect}
          onChange={(event) => set({ connect: event.target.checked })}
          aria-label="Connect when the app starts"
        />
      </Row>
      <SubHeading>App</SubHeading>
      <Row label="App">
        {select(
          d.app,
          (app) => set({ app }),
          apps.map((a) => [a.key, a.name]),
          `The first (${apps[0]?.name ?? "none"})`,
          "Default app",
        )}
      </Row>
      <Row label="Show the window on this PC">
        <input
          type="checkbox"
          checked={d.window}
          onChange={(event) => set({ window: event.target.checked })}
          aria-label="Show the window on this PC"
        />
      </Row>
      <SubHeading>Agent</SubHeading>
      <Row label="Playbook">
        {select(
          d.playbook,
          (playbook) => set({ playbook }),
          usable.map((p) => [p.name, p.name]),
          "The usual (fibsem-agent)",
          "Default playbook",
        )}
      </Row>
      <Row label="Level">
        {select(
          d.level,
          (level) => set({ level }),
          LEVELS.map((l) => [l, title(l)]),
          "The playbook's, or Watch",
          "Default level",
        )}
      </Row>
      <Row label="Budget ($)">
        <input
          value={shownBudget}
          onChange={(event) => setBudget(event.target.value)}
          inputMode="decimal"
          placeholder="The playbook's"
          aria-label="Default budget"
          className={cn(FIELD, "w-full sm:w-32")}
        />
      </Row>
      {save.error && (
        <p className="m-0 text-[13px] text-destructive">
          {(save.error as Error).message}
        </p>
      )}
      <div className="flex justify-end gap-2.5 pt-2">
        {changed && (
          <Button
            onClick={() => {
              setEdit(null);
              setBudget(null);
              save.reset();
            }}
          >
            Cancel
          </Button>
        )}
        <Button
          variant="primary"
          disabled={!changed || save.isPending}
          onClick={() =>
            save.mutate({
              ...d,
              budget_usd: budget !== null ? budget.trim() || null : d.budget_usd,
            })
          }
        >
          {save.isPending ? "Saving…" : "Save"}
        </Button>
      </div>
    </Card>
  );
}

function PlaybookRow({ playbook }: { playbook: Playbook }) {
  const [open, setOpen] = useState(false);
  const config = useQuery({
    queryKey: ["gateway/agent/playbooks", playbook.name],
    queryFn: () =>
      api<Configuration>(
        `/gateway/agent/playbooks/${encodeURIComponent(playbook.name)}`,
      ),
    enabled: !playbook.error,
    staleTime: 30_000,
  });
  const c = config.data;
  return (
    <div className="flex flex-col gap-2 border-t border-border py-2 text-[13px]">
      <div className="grid grid-cols-1 gap-1 sm:grid-cols-[190px_minmax(0,1fr)_auto] sm:gap-3">
        <span className="text-muted-foreground">{playbook.name}</span>
        {playbook.error ? (
          <span className="text-destructive">{playbook.error}</span>
        ) : (
          <span>
            {c
              ? [
                  title(c.level ?? "watch"),
                  c.max_budget_usd != null
                    ? dollars(c.max_budget_usd)
                    : "no budget",
                ].join(" · ")
              : "…"}
            {playbook.hash && (
              <span className="text-muted-foreground"> · {playbook.hash}</span>
            )}
          </span>
        )}
        {!playbook.error && (
          <button
            onClick={() => setOpen(!open)}
            className="w-fit cursor-pointer border-0 bg-transparent p-0 text-xs text-[var(--primary)] hover:underline"
          >
            {open ? "Hide configuration" : "Show configuration"}
          </button>
        )}
      </div>
      {open && <ConfigurationFacts name={playbook.name} />}
    </div>
  );
}

function Playbooks() {
  const playbooks = useQuery({
    queryKey: ["gateway/agent/playbooks"],
    queryFn: () => api<Playbook[]>("/gateway/agent/playbooks"),
    staleTime: 30_000,
    retry: false,
  });
  return (
    <Card className="flex flex-col gap-1">
      <CardTitle>Playbooks</CardTitle>
      <p className="m-0 pb-1 text-[13px] text-muted-foreground">
        On this PC. Read-only.
      </p>
      {playbooks.error && (
        <p className="m-0 text-[13px] text-destructive">
          {(playbooks.error as Error).message}
        </p>
      )}
      {playbooks.data?.map((p) => <PlaybookRow key={p.name} playbook={p} />)}
    </Card>
  );
}

type Ask = "stop" | "restart" | null;

/** The gateway's version, and stopping or restarting it. Stopping ends the
 * app and any run, and nothing remote starts the gateway again. */
function GatewayCard({
  version,
  newer,
  running,
  onQuit,
}: {
  version: string;
  newer: { text: string; detail: string } | null;
  /** The app's name while it runs, which stops with the gateway. */
  running: string | null;
  onQuit: (restart: boolean) => void;
}) {
  const [ask, setAsk] = useState<Ask>(null);
  const quit = useMutation({
    mutationFn: (restart: boolean) =>
      api<{ stopping: boolean; restart: boolean }>("/gateway/quit", {
        method: "POST",
        body: JSON.stringify({ restart }),
      }),
    onSuccess: (answer) => {
      setAsk(null);
      onQuit(answer.restart);
    },
  });
  const stops = running
    ? `This stops ${running} and any agent run.`
    : "Nothing is running.";
  return (
    <Card className="flex flex-col gap-1">
      <CardTitle className="mb-1.5">Gateway</CardTitle>
      <Row label="Version">
        fibsemOS {version}
        {newer && (
          <span title={newer.detail} className="text-muted-foreground">
            {" "}
            · {newer.text}
          </span>
        )}
      </Row>
      <div className="flex justify-end gap-2.5 pt-2">
        <Button onClick={() => setAsk("restart")}>Restart</Button>
        <Button variant="danger" onClick={() => setAsk("stop")}>
          Stop
        </Button>
      </div>
      <Dialog
        open={ask !== null}
        onOpenChange={(open) => {
          if (!open) {
            setAsk(null);
            quit.reset();
          }
        }}
      >
        <DialogContent
          title={ask === "stop" ? "Stop the gateway?" : "Restart the gateway?"}
          className="w-[min(460px,calc(100vw-32px))]"
        >
          <div className="flex flex-col gap-2 px-4 py-3.5 text-[13px]">
            <p className="m-0">{stops}</p>
            {ask === "stop" ? (
              <p className={cn("m-0", WARN)}>
                You need to start it again at the computer.
              </p>
            ) : (
              <p className="m-0 text-muted-foreground">
                The gateway starts again on its own, and this page reloads.
              </p>
            )}
            {quit.error && (
              <p className="m-0 text-destructive">
                {(quit.error as Error).message}
              </p>
            )}
            <div className="flex justify-end gap-2.5 pt-2">
              <Button onClick={() => setAsk(null)}>Cancel</Button>
              <Button
                variant={ask === "stop" ? "danger" : "primary"}
                disabled={quit.isPending}
                onClick={() => quit.mutate(ask === "restart")}
              >
                {ask === "stop" ? "Stop the gateway" : "Restart the gateway"}
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </Card>
  );
}

export function Settings({
  apps,
  configurations,
  version,
  newer,
  running,
  onQuit,
}: {
  apps: { key: string; name: string }[];
  configurations: { name: string; default: boolean }[];
  version: string;
  newer: { text: string; detail: string } | null;
  running: string | null;
  onQuit: (restart: boolean) => void;
}) {
  return (
    <div className="flex flex-col gap-3">
      <Secrets />
      <DefaultsCard apps={apps} configurations={configurations} />
      <Playbooks />
      <GatewayCard
        version={version}
        newer={newer}
        running={running}
        onQuit={onQuit}
      />
    </div>
  );
}
