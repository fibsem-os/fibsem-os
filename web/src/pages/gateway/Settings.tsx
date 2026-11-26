import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Copy, Info } from "lucide-react";
import { type ReactNode, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { HoverCard } from "@/components/ui/hover-card";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import { FIELD, LEVELS, type Playbook, title } from "@/pages/gateway/AgentCard";

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
  /** The app's own start settings, by name. */
  settings?: Record<string, string>;
  /** Each app's own choices, by app key. */
  apps?: Record<string, Defaults>;
}

/** An app the gateway can start, and what its start takes. */
export interface AppFacts {
  key: string;
  name: string;
  headless: boolean;
  opens_experiments?: boolean;
  connects_microscope?: boolean;
  has_window?: boolean;
  settings?: { name: string; label: string; default: string }[];
}

/** *key*'s own saved choices: its entry, else the saved ones when they are its. */
export function forApp(
  d: Defaults | undefined,
  key: string | null | undefined,
): Defaults | undefined {
  if (!d || !key) return undefined;
  return d.apps?.[key] ?? (d.app === key ? d : undefined);
}

export interface Secret {
  name: string;
  about: string;
  set: boolean;
  source: string | null;
  /** Whether this caller may set it: the operator at this PC only. */
  can_set?: boolean;
  /** The connectors that use it, for a connector's token. */
  connectors?: string[];
  /** Where the stored token may go, and where its connectors point now. */
  sent_to?: string[] | null;
  goes_to?: string[];
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
      type="button"
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

/** One key: whether it is set and where from, and for the operator at this
 * PC, Set or Replace (pasted once, never shown again) and Remove. */
/** One key: its state and Set/Replace/Remove. *inline* is the short form
 * that sits beside a connector's token name in its editor. */
export function KeyRow({ s, inline = false }: { s: Secret; inline?: boolean }) {
  const client = useQueryClient();
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState("");
  const [confirm, setConfirm] = useState(false);
  const done = () => {
    setEditing(false);
    setValue("");
    setConfirm(false);
    client.invalidateQueries({ queryKey: ["gateway/secrets"] });
  };
  const save = useMutation({
    mutationFn: () =>
      api<Secret>(`/gateway/secrets/${encodeURIComponent(s.name)}`, {
        method: "PUT",
        body: JSON.stringify({ value }),
      }),
    onSuccess: done,
  });
  const remove = useMutation({
    mutationFn: () =>
      api<Secret>(`/gateway/secrets/${encodeURIComponent(s.name)}`, {
        method: "DELETE",
      }),
    onSuccess: done,
  });
  const command = `fibsem-cli key set ${s.name}`;
  const error = (save.error ?? remove.error) as Error | null;
  const about = s.about.charAt(0).toUpperCase() + s.about.slice(1);
  return (
    <div
      className={cn(
        "flex flex-col gap-1.5 text-[13px]",
        !inline && "border-t border-border py-2",
      )}
    >
      <div
        className={cn(
          "items-center gap-1",
          inline
            ? "flex flex-wrap gap-3"
            : "grid grid-cols-1 sm:grid-cols-[190px_minmax(0,1fr)_auto] sm:gap-3",
        )}
      >
        {!inline && (
          <span
            className="font-mono text-xs text-muted-foreground"
            title={about}
          >
            {s.name}
          </span>
        )}
        <span className="min-w-0" title={inline ? about : undefined}>
          <Dot on={s.set} />
          {s.set ? (
            `Set · from ${s.source}`
          ) : (
            <span className="text-muted-foreground">Not set</span>
          )}
          <span
            className={cn(
              "block text-xs text-muted-foreground",
              inline && "hidden",
            )}
          >
            {about}
            {s.sent_to?.length ? ` · sent only to ${s.sent_to.join(", ")}` : ""}
          </span>
          {s.set &&
          s.goes_to?.length &&
          s.sent_to?.length &&
          s.goes_to.some((place) => !s.sent_to?.includes(place)) ? (
            <span className={cn("block text-xs", WARN)}>
              Its connector now points at {s.goes_to.join(", ")}: set the token
              again to send it there.
            </span>
          ) : null}
        </span>
        <span className="flex items-center gap-2">
          {s.can_set ? (
            s.source === "environment" ? (
              <span className="text-xs text-muted-foreground">
                Set in the environment
              </span>
            ) : (
              <>
                <Button type="button" onClick={() => setEditing(!editing)}>
                  {s.set ? "Replace…" : "Set…"}
                </Button>
                {s.set &&
                  (confirm ? (
                    <Button
                      type="button"
                      variant="danger"
                      disabled={remove.isPending}
                      onClick={() => remove.mutate()}
                    >
                      Remove it
                    </Button>
                  ) : (
                    <Button type="button" onClick={() => setConfirm(true)}>
                      Remove
                    </Button>
                  ))}
              </>
            )
          ) : (
            <>
              <CopyCommand command={command} />
              <HoverCard
                trigger={
                  <button
                    type="button"
                    aria-label={`How to set ${s.name}`}
                    className="flex cursor-pointer border-0 bg-transparent p-1 text-muted-foreground hover:text-foreground"
                  >
                    <Info size={15} />
                  </button>
                }
              >
                <p className="m-0 text-[13px]">
                  Keys are set only at this PC: on the gateway's own page there,
                  or with <span className="font-mono text-xs">{command}</span>.
                </p>
              </HoverCard>
            </>
          )}
        </span>
      </div>
      {confirm && (
        <p className="m-0 text-xs text-muted-foreground">
          This PC stops using it. Turn it off where it was issued too.{" "}
          <button
            type="button"
            className="cursor-pointer border-0 bg-transparent p-0 underline"
            onClick={() => setConfirm(false)}
          >
            Keep it
          </button>
        </p>
      )}
      {editing && (
        // Not a <form>: inside a connector's editor it would be a form in a
        // form, which the browser submits itself (reloading the page, and
        // offering to save the key as a password) instead of storing it.
        <div className="flex flex-wrap items-center gap-2">
          <input
            type="password"
            autoComplete="off"
            value={value}
            onChange={(event) => setValue(event.target.value)}
            onKeyDown={(event) => {
              if (event.key !== "Enter") return;
              event.preventDefault();
              if (value.trim()) save.mutate();
            }}
            placeholder={`Paste ${s.name}`}
            aria-label={`New ${s.name}`}
            className={cn(FIELD, "w-full sm:w-80")}
            autoFocus
          />
          <Button
            variant="primary"
            type="button"
            disabled={!value.trim() || save.isPending}
            onClick={() => save.mutate()}
          >
            Save
          </Button>
          <Button
            type="button"
            onClick={() => {
              setEditing(false);
              setValue("");
            }}
          >
            Cancel
          </Button>
          <span className="w-full text-xs text-muted-foreground">
            Stored in this PC's key store; it isn't shown again.
            {s.goes_to?.length
              ? ` It is sent only to ${s.goes_to.join(", ")}.`
              : ""}
          </span>
        </div>
      )}
      {error && <p className="m-0 text-xs text-destructive">{error.message}</p>}
    </div>
  );
}

/** This PC's keys: the ones fibsemOS reads, and each connector's token.
 * Write-only: whether each is set and where from, never the key. */
function Secrets() {
  const secrets = useQuery({
    queryKey: ["gateway/secrets"],
    queryFn: () => api<Secret[]>("/gateway/secrets"),
  });
  return (
    <Card className="flex flex-col gap-1">
      <CardTitle className="mb-1.5">Keys</CardTitle>
      {secrets.error && (
        <p className="m-0 text-[13px] text-destructive">
          {(secrets.error as Error).message}
        </p>
      )}
      {secrets.data?.map((s) => (
        <KeyRow key={s.name} s={s} />
      ))}
    </Card>
  );
}

function SubHeading({
  first,
  children,
}: {
  first?: boolean;
  children: string;
}) {
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
  apps: AppFacts[];
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
  // Choosing an app shows its own saved choices.
  const pick = (app: string | null) =>
    setEdit({
      ...d,
      ...(forApp(d, app) ?? { settings: {} }),
      apps: d.apps,
      app,
    });
  const facts = apps.find((a) => a.key === (d.app ?? apps[0]?.key));
  const shownBudget =
    budget ?? (d.budget_usd != null ? String(d.budget_usd) : "");
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
        What Overview fills in, for each app. You can still change each start.
      </p>
      {facts?.connects_microscope !== false && (
        <SubHeading first>Microscope</SubHeading>
      )}
      {facts?.connects_microscope !== false && (
        <Row label="Configuration">
          {select(
            d.configuration,
            (configuration) => set({ configuration }),
            configurations.map((c) => [c.name, c.name]),
            `The default (${configurations.find((c) => c.default)?.name ?? configurations[0]?.name ?? "none"})`,
            "Default configuration",
          )}
        </Row>
      )}
      {facts?.connects_microscope !== false && (
        <Row label="Connect when the app starts">
          <input
            type="checkbox"
            checked={d.connect}
            onChange={(event) => set({ connect: event.target.checked })}
            aria-label="Connect when the app starts"
          />
        </Row>
      )}
      <SubHeading first={facts?.connects_microscope === false}>App</SubHeading>
      <Row label="App">
        {select(
          d.app,
          pick,
          apps.map((a) => [a.key, a.name]),
          `The first (${apps[0]?.name ?? "none"})`,
          "Default app",
        )}
      </Row>
      {facts?.has_window !== false && (
        <Row label="Show the window on this PC">
          <input
            type="checkbox"
            checked={d.window}
            onChange={(event) => set({ window: event.target.checked })}
            aria-label="Show the window on this PC"
          />
        </Row>
      )}
      {(facts?.settings ?? []).map((setting) => (
        <Row key={setting.name} label={setting.label}>
          <input
            value={d.settings?.[setting.name] ?? ""}
            onChange={(event) =>
              set({
                settings: { ...d.settings, [setting.name]: event.target.value },
              })
            }
            placeholder={setting.default || "Unset"}
            aria-label={setting.label}
            className={cn(FIELD, "w-full sm:w-72")}
          />
        </Row>
      ))}
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
              budget_usd:
                budget !== null ? budget.trim() || null : d.budget_usd,
            })
          }
        >
          {save.isPending ? "Saving…" : "Save"}
        </Button>
      </div>
    </Card>
  );
}

interface ThisPc {
  microscope: boolean;
  allow_local_agents: boolean;
  port: number;
  update_check: boolean;
  /** Whether this caller is the operator at this PC. */
  at_this_pc: boolean;
}

/** This PC's install settings. Each switch saves at once; the port is used
 * from the gateway's next start. Agents connecting without a click is the
 * operator's at this PC only. */
function ThisPcCard() {
  const client = useQueryClient();
  const shown = useQuery({
    queryKey: ["gateway/this-pc"],
    queryFn: () => api<ThisPc>("/gateway/this-pc"),
  });
  const [port, setPort] = useState<string | null>(null);
  const [askAgents, setAskAgents] = useState(false);
  const save = useMutation({
    mutationFn: (change: Partial<ThisPc>) =>
      api<ThisPc>("/gateway/this-pc", {
        method: "PUT",
        body: JSON.stringify(change),
      }),
    onSuccess: (answer) => {
      client.setQueryData(["gateway/this-pc"], answer);
      setPort(null);
      setAskAgents(false);
    },
  });
  const d = shown.data;
  if (!d)
    return shown.error ? (
      <Card>
        <p className="m-0 text-[13px] text-destructive">
          {(shown.error as Error).message}
        </p>
      </Card>
    ) : null;
  const toggle = (name: keyof ThisPc, label: string, disabled = false) => (
    <input
      type="checkbox"
      checked={Boolean(d[name])}
      disabled={disabled || save.isPending}
      onChange={(event) => save.mutate({ [name]: event.target.checked })}
      aria-label={label}
    />
  );
  const portChanged = port !== null && port.trim() !== String(d.port);
  return (
    <Card className="flex flex-col gap-1">
      <CardTitle className="mb-1.5">This PC</CardTitle>
      <Row label="Agents may use the microscope">
        {toggle("microscope", "Agents may use the microscope")}
      </Row>
      <p className="m-0 pb-1 text-xs text-muted-foreground">
        Off: agents see the app and the experiment, but not the stage, beams or
        milling. Used from the app's next start; the agent's level still decides
        what it may do.
      </p>
      <Row label="Agents on this PC connect without a click">
        <span className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={d.allow_local_agents}
            disabled={!d.at_this_pc || save.isPending}
            onChange={(event) =>
              event.target.checked
                ? setAskAgents(true)
                : save.mutate({ allow_local_agents: false })
            }
            aria-label="Agents on this PC connect without a click"
          />
          {!d.at_this_pc && (
            <span className="text-xs text-muted-foreground">
              Set at this PC only
            </span>
          )}
        </span>
      </Row>
      {askAgents && (
        <div className="flex flex-col gap-2 rounded-md border border-border px-3 py-2 text-[13px]">
          <p className={cn("m-0", WARN)}>
            Any program on this PC that asks to connect is then let in at once,
            with no Allow from you.
          </p>
          <div className="flex justify-end gap-2.5">
            <Button onClick={() => setAskAgents(false)}>Cancel</Button>
            <Button
              variant="primary"
              disabled={save.isPending}
              onClick={() => save.mutate({ allow_local_agents: true })}
            >
              Let them connect
            </Button>
          </div>
        </div>
      )}
      <Row label="Gateway port">
        <span className="flex items-center gap-2">
          <input
            value={port ?? String(d.port)}
            onChange={(event) => setPort(event.target.value)}
            inputMode="numeric"
            aria-label="Gateway port"
            className={cn(FIELD, "w-24")}
          />
          {portChanged && (
            <Button
              disabled={save.isPending}
              onClick={() => save.mutate({ port: Number(port!.trim()) })}
            >
              Save
            </Button>
          )}
        </span>
      </Row>
      <p className="m-0 pb-1 text-xs text-muted-foreground">
        A new port is used once the gateway restarts (Gateway, below). This page
        then opens at the new port, and your computer's SSH forward must
        name it.
      </p>
      <Row label="Check for a newer fibsemOS">
        {toggle("update_check", "Check for a newer fibsemOS")}
      </Row>
      {save.error && (
        <p className="m-0 text-[13px] text-destructive">
          {(save.error as Error).message}
        </p>
      )}
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
  apps: AppFacts[];
  configurations: { name: string; default: boolean }[];
  version: string;
  newer: { text: string; detail: string } | null;
  running: string | null;
  onQuit: (restart: boolean) => void;
}) {
  return (
    <div className="flex flex-col gap-3">
      <Secrets />
      <ThisPcCard />
      <DefaultsCard apps={apps} configurations={configurations} />
      <GatewayCard
        version={version}
        newer={newer}
        running={running}
        onQuit={onQuit}
      />
    </div>
  );
}
