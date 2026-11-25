import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import { FIELD } from "@/pages/gateway/AgentCard";
import { KeyRow, type Secret } from "@/pages/gateway/Settings";

/** One connector as mcp.yaml has it (`fibsem.ai.agent.extra_mcp`). */
type Tool = string | Record<string, string>;
interface Entry {
  url?: string;
  command?: string[];
  token?: string;
  about?: string;
  level?: "watch" | "drive";
  tools?: Tool[] | "all";
  subagents?: string[];
}
interface Connector {
  name: string;
  entry: Entry;
  problem: string | null;
}
interface Connectors {
  file: string;
  connectors: Connector[];
  problem: string | null;
  at_this_pc: boolean;
}

const WARN = "text-[var(--fibsem-semantic-warning-color)]";

/** The form's text for an entry, and the entry from the form's text. */
interface Form {
  name: string;
  kind: "url" | "command";
  url: string;
  command: string;
  token: string;
  about: string;
  level: "watch" | "drive";
  all: boolean;
  tools: string;
  subagents: string;
}

function toForm(name: string, e: Entry): Form {
  const tools = Array.isArray(e.tools)
    ? e.tools
        .map((t) =>
          typeof t === "string"
            ? t
            : Object.entries(t).map(([k, v]) => `${k}: ${v}`)[0],
        )
        .join("\n")
    : "";
  return {
    name,
    kind: e.command ? "command" : "url",
    url: e.url ?? "",
    command: (e.command ?? []).join("\n"),
    token: e.token ?? "",
    about: e.about ?? "",
    level: e.level ?? "drive",
    all: e.tools === "all",
    tools,
    subagents: (e.subagents ?? []).join(", "),
  };
}

function lines(text: string): string[] {
  return text
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);
}

function toEntry(f: Form): Entry {
  const entry: Entry = {};
  if (f.kind === "url") entry.url = f.url.trim();
  else entry.command = lines(f.command);
  if (f.token.trim()) entry.token = f.token.trim();
  if (f.about.trim()) entry.about = f.about.trim();
  if (f.level !== "drive") entry.level = f.level;
  entry.tools = f.all
    ? "all"
    : lines(f.tools).map((t) => {
        const [name, mark] = t.split(":").map((p) => p.trim());
        return mark ? { [name]: mark } : name;
      });
  const subagents = f.subagents
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  if (subagents.length) entry.subagents = subagents;
  return entry;
}

function Field({
  label,
  children,
  hint,
}: {
  label: string;
  children: React.ReactNode;
  hint?: string;
}) {
  return (
    <label className="grid grid-cols-1 items-start gap-1 py-1 text-[13px] sm:grid-cols-[150px_minmax(0,1fr)] sm:gap-3">
      <span className="pt-1.5 text-muted-foreground">{label}</span>
      <span className="flex min-w-0 flex-col gap-1">
        {children}
        {hint && <span className="text-xs text-muted-foreground">{hint}</span>}
      </span>
    </label>
  );
}

/** Add or edit one connector. A command connector asks to confirm its
 * command line before it is saved (the server answers 428 until then). */
function ConnectorForm({
  start,
  isNew,
  atThisPc,
  onDone,
}: {
  start: Form;
  isNew: boolean;
  atThisPc: boolean;
  onDone: () => void;
}) {
  const client = useQueryClient();
  const [f, setF] = useState<Form>(start);
  const [confirm, setConfirm] = useState<string | null>(null);
  const set = (change: Partial<Form>) => {
    setF({ ...f, ...change });
    setConfirm(null);
  };
  const save = useMutation({
    mutationFn: (confirmed?: string) =>
      api(`/gateway/connectors/${encodeURIComponent(f.name.trim())}`, {
        method: "PUT",
        body: JSON.stringify({ entry: toEntry(f), confirm: confirmed }),
      }),
    onSuccess: () => {
      client.invalidateQueries({ queryKey: ["gateway/connectors"] });
      client.invalidateQueries({ queryKey: ["gateway/secrets"] });
      onDone();
    },
    onError: (error: Error & { status?: number }) => {
      if (error.message.includes("runs this on this PC"))
        setConfirm(lines(f.command).join(" "));
    },
  });
  const commandBlocked = f.kind === "command" && !atThisPc;
  return (
    <form
      className="flex flex-col gap-0.5 border-t border-border pt-2"
      onSubmit={(event) => {
        event.preventDefault();
        save.mutate(undefined);
      }}
    >
      {isNew && (
        <Field
          label="Name"
          hint="Letters, digits, _ and -: the agent sees its tools as mcp__<name>__<tool>."
        >
          <input
            value={f.name}
            onChange={(e) => set({ name: e.target.value })}
            className={cn(FIELD, "sm:w-64")}
          />
        </Field>
      )}
      <Field label="Runs as">
        <span className="flex gap-3">
          {(["url", "command"] as const).map((kind) => (
            <label key={kind} className="flex items-center gap-1.5">
              <input
                type="radio"
                checked={f.kind === kind}
                onChange={() => set({ kind })}
              />
              {kind === "url"
                ? "A server at an address"
                : "A program on this PC"}
            </label>
          ))}
        </span>
      </Field>
      {f.kind === "url" ? (
        <Field label="Address">
          <input
            value={f.url}
            onChange={(e) => set({ url: e.target.value })}
            placeholder="https://notebook.example/mcp"
            className={cn(FIELD, "w-full")}
          />
        </Field>
      ) : (
        <Field
          label="Command"
          hint={
            atThisPc
              ? "One argument per line. It runs as this Windows user at each agent start."
              : "A program on this PC is changed only at this PC."
          }
        >
          <textarea
            value={f.command}
            onChange={(e) => set({ command: e.target.value })}
            rows={3}
            disabled={!atThisPc}
            className={cn(FIELD, "w-full font-mono text-xs")}
          />
        </Field>
      )}
      <Field
        label="Token name"
        hint="Where its token is stored, such as NOTEBOOK_TOKEN. Set the token itself under Keys."
      >
        <input
          value={f.token}
          onChange={(e) => set({ token: e.target.value.toUpperCase() })}
          className={cn(FIELD, "font-mono text-xs sm:w-64")}
        />
      </Field>
      <Field label="About" hint="What the agent is told it is.">
        <input
          value={f.about}
          onChange={(e) => set({ about: e.target.value })}
          className={cn(FIELD, "w-full")}
        />
      </Field>
      <Field label="Needs the agent at">
        <select
          value={f.level}
          onChange={(e) => set({ level: e.target.value as Form["level"] })}
          className={cn(FIELD, "sm:w-40")}
        >
          <option value="watch">Watch (reads)</option>
          <option value="drive">Drive</option>
        </select>
      </Field>
      <Field
        label="Tools"
        hint="One tool per line; add ': watch' or ': drive' to set its own level. The agent sees no others."
      >
        <label className="flex items-center gap-1.5">
          <input
            type="checkbox"
            checked={f.all}
            onChange={(e) => set({ all: e.target.checked })}
          />
          Every tool it offers
        </label>
        {!f.all && (
          <textarea
            value={f.tools}
            onChange={(e) => set({ tools: e.target.value })}
            rows={4}
            className={cn(FIELD, "w-full font-mono text-xs")}
          />
        )}
      </Field>
      <Field label="Subagents may use" hint="Tool names, comma separated.">
        <input
          value={f.subagents}
          onChange={(e) => set({ subagents: e.target.value })}
          className={cn(FIELD, "w-full")}
        />
      </Field>
      {confirm && (
        <div
          className={cn(
            "flex flex-col gap-1 rounded border border-border p-2 text-[13px]",
            WARN,
          )}
        >
          <span>Saving this runs, on this PC at each agent start:</span>
          <span className="font-mono text-xs text-foreground">{confirm}</span>
        </div>
      )}
      {save.error && !confirm && (
        <p className="m-0 text-xs text-destructive">
          {(save.error as Error).message}
        </p>
      )}
      <div className="flex justify-end gap-2 pt-1">
        <Button type="button" onClick={onDone}>
          Cancel
        </Button>
        {confirm ? (
          <Button
            variant="danger"
            type="button"
            disabled={save.isPending}
            onClick={() => save.mutate(confirm)}
          >
            Run it, and save
          </Button>
        ) : (
          <Button
            variant="primary"
            type="submit"
            disabled={save.isPending || commandBlocked || !f.name.trim()}
          >
            Save
          </Button>
        )}
      </div>
    </form>
  );
}

function ConnectorRow({
  c,
  keys,
  atThisPc,
}: {
  c: Connector;
  keys: Secret[];
  atThisPc: boolean;
}) {
  const client = useQueryClient();
  const [editing, setEditing] = useState(false);
  const [confirm, setConfirm] = useState(false);
  const test = useMutation({
    mutationFn: () =>
      api<{ tools: string[] }>(
        `/gateway/connectors/${encodeURIComponent(c.name)}/test`,
        { method: "POST" },
      ),
  });
  const remove = useMutation({
    mutationFn: () =>
      api(`/gateway/connectors/${encodeURIComponent(c.name)}`, {
        method: "DELETE",
      }),
    onSuccess: () =>
      client.invalidateQueries({ queryKey: ["gateway/connectors"] }),
  });
  const isCommand = !!c.entry.command;
  const local = !isCommand || atThisPc;
  const key = keys.find((k) => k.name === c.entry.token);
  const tools =
    c.entry.tools === "all"
      ? "every tool"
      : `${(c.entry.tools ?? []).length} tools`;
  return (
    <div className="flex flex-col gap-1.5 border-t border-border py-2.5 text-[13px]">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-medium text-foreground">{c.name}</span>
        <span className="min-w-0 flex-1 truncate font-mono text-xs text-muted-foreground">
          {isCommand ? (c.entry.command ?? []).join(" ") : c.entry.url}
        </span>
        {local && (
          <>
            <Button disabled={test.isPending} onClick={() => test.mutate()}>
              {test.isPending ? "Testing…" : "Test"}
            </Button>
            <Button onClick={() => setEditing(!editing)}>Edit</Button>
            {confirm ? (
              <Button variant="danger" onClick={() => remove.mutate()}>
                Remove it
              </Button>
            ) : (
              <Button onClick={() => setConfirm(true)}>Remove</Button>
            )}
          </>
        )}
      </div>
      <span className="text-xs text-muted-foreground">
        {c.entry.about ? `${c.entry.about} · ` : ""}
        {tools} · needs {c.entry.level === "watch" ? "Watch" : "Drive"}
        {isCommand ? " · a program on this PC" : ""}
        {!local ? " · changed only at this PC" : ""}
      </span>
      {c.problem && (
        <span className={cn("text-xs", WARN)}>Left off: {c.problem}</span>
      )}
      {test.data && (
        <span className="text-xs text-muted-foreground">
          Offers {test.data.tools.length} tools:{" "}
          {test.data.tools.join(", ") || "none"}
        </span>
      )}
      {(test.error ?? remove.error) && (
        <span className="text-xs text-destructive">
          {((test.error ?? remove.error) as Error).message}
        </span>
      )}
      {key && <KeyRow s={key} />}
      {editing && (
        <ConnectorForm
          start={toForm(c.name, c.entry)}
          isNew={false}
          atThisPc={atThisPc}
          onDone={() => setEditing(false)}
        />
      )}
    </div>
  );
}

/** The lab's connectors: MCP servers the agent may use, in mcp.yaml. A
 * playbook turns each on by naming it. */
function ConnectorsCard() {
  const found = useQuery({
    queryKey: ["gateway/connectors"],
    queryFn: () => api<Connectors>("/gateway/connectors"),
  });
  const keys = useQuery({
    queryKey: ["gateway/secrets"],
    queryFn: () => api<Secret[]>("/gateway/secrets"),
  });
  const [adding, setAdding] = useState(false);
  const data = found.data;
  return (
    <Card className="flex flex-col gap-1">
      <div className="mb-1 flex items-center justify-between gap-2">
        <CardTitle>Connectors</CardTitle>
        <Button onClick={() => setAdding(!adding)}>Add</Button>
      </div>
      <p className="m-0 text-xs text-muted-foreground">
        MCP servers the agent may use, such as the lab notebook. A playbook
        turns one on by naming it. Changes apply at the next agent start. Kept
        in <span className="font-mono">{data?.file ?? "mcp.yaml"}</span>.
      </p>
      {found.error && (
        <p className="m-0 text-[13px] text-destructive">
          {(found.error as Error).message}
        </p>
      )}
      {data?.problem && (
        <p className={cn("m-0 text-[13px]", WARN)}>{data.problem}</p>
      )}
      {adding && (
        <ConnectorForm
          start={toForm("", { tools: [] })}
          isNew
          atThisPc={!!data?.at_this_pc}
          onDone={() => setAdding(false)}
        />
      )}
      {data?.connectors.length === 0 && !adding && (
        <p className="m-0 border-t border-border py-2 text-[13px] text-muted-foreground">
          No connectors yet.
        </p>
      )}
      {data?.connectors.map((c) => (
        <ConnectorRow
          key={c.name}
          c={c}
          keys={keys.data ?? []}
          atThisPc={!!data.at_this_pc}
        />
      ))}
    </Card>
  );
}

/** What shapes the agent's work on this PC: its connectors, and (to come)
 * its playbooks and skills. */
export function Customise() {
  return (
    <div className="flex flex-col gap-4">
      <ConnectorsCard />
    </div>
  );
}
