import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CircleAlert, Pencil, Plus, Trash2 } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
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
interface ConnectorFile {
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
  keyRow,
  onDone,
}: {
  start: Form;
  isNew: boolean;
  atThisPc: boolean;
  keyRow?: Secret;
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
        hint="Where its token is stored, such as NOTEBOOK_TOKEN."
      >
        <input
          value={f.token}
          onChange={(e) => set({ token: e.target.value.toUpperCase() })}
          className={cn(FIELD, "font-mono text-xs sm:w-64")}
        />
      </Field>
      {keyRow && keyRow.name === f.token && <KeyRow s={keyRow} />}
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
      : `${(c.entry.tools ?? []).length} tool${(c.entry.tools ?? []).length === 1 ? "" : "s"}`;
  const facts = [
    c.entry.about,
    tools,
    `needs ${c.entry.level === "watch" ? "Watch" : "Drive"}`,
    isCommand ? "a program on this PC" : null,
    !local ? "changed only at this PC" : null,
  ]
    .filter(Boolean)
    .join(" · ");
  const small = "px-2 py-1 text-[12px]";
  return (
    <div className="flex flex-col gap-1 border-t border-border py-2 text-[13px]">
      <div className="flex items-center gap-2" title={facts}>
        <Dot on={!c.problem} />
        <span className="font-medium text-foreground">{c.name}</span>
        {key && !key.set && (
          <CircleAlert
            size={13}
            className="shrink-0 text-amber-500"
            aria-label={`${key.name} isn't set`}
          >
            <title>{`${key.name} isn't set: set it under Edit`}</title>
          </CircleAlert>
        )}
        <span className="min-w-0 flex-1 truncate font-mono text-xs text-muted-foreground">
          {isCommand ? (c.entry.command ?? []).join(" ") : c.entry.url}
        </span>
        {local && (
          <>
            <Button
              variant="ghost"
              className={small}
              disabled={test.isPending}
              onClick={() => test.mutate()}
            >
              {test.isPending ? "Testing…" : "Test"}
            </Button>
            <Button
              variant="ghost"
              className={small}
              title="Edit"
              aria-label={`Edit ${c.name}`}
              onClick={() => setEditing(!editing)}
            >
              <Pencil size={13} />
            </Button>
            {confirm ? (
              <Button
                variant="danger"
                className={small}
                onClick={() => remove.mutate()}
              >
                Remove
              </Button>
            ) : (
              <Button
                variant="ghost"
                className={small}
                title="Remove"
                aria-label={`Remove ${c.name}`}
                onClick={() => setConfirm(true)}
              >
                <Trash2 size={13} />
              </Button>
            )}
          </>
        )}
      </div>
      {c.problem && (
        <span className={cn("text-xs", WARN)}>Left off: {c.problem}</span>
      )}
      {test.data && (
        <span className="text-xs text-muted-foreground">
          {test.data.tools.join(", ") || "No tools"}
        </span>
      )}
      {(test.error ?? remove.error) && (
        <span className="text-xs text-destructive">
          {((test.error ?? remove.error) as Error).message}
        </span>
      )}
      {editing && (
        <ConnectorForm
          start={toForm(c.name, c.entry)}
          isNew={false}
          atThisPc={atThisPc}
          keyRow={key}
          onDone={() => setEditing(false)}
        />
      )}
    </div>
  );
}

/** fibsem's own connector: started by the agent (the fibsem-mcp sidecar),
 * not in mcp.yaml, on under every playbook of its instrument kind. */
function BuiltIn() {
  return (
    <div
      className="flex items-center gap-2 border-t border-border py-2 text-[13px]"
      title="Started by the agent itself; every playbook for this instrument turns it on"
    >
      <Dot on />
      <span className="font-medium text-foreground">fibsem</span>
      <span className="flex-1" />
      <span className="rounded border border-border px-1.5 py-0.5 text-[11px] text-muted-foreground">
        Built In
      </span>
    </div>
  );
}

function Dot({ on }: { on: boolean }) {
  return (
    <span
      className={cn(
        "inline-block size-1.5 shrink-0 rounded-full",
        on
          ? "bg-[var(--primary)]"
          : "bg-[var(--fibsem-semantic-warning-color)]",
      )}
    />
  );
}

/** The lab's connectors: MCP servers the agent may use, in mcp.yaml. A
 * playbook turns each on by naming it. The gateway keeps them: on a page
 * not opened through it, they can't be read here. */
export function Connectors({ enabled }: { enabled: boolean }) {
  const found = useQuery({
    queryKey: ["gateway/connectors"],
    queryFn: () => api<ConnectorFile>("/gateway/connectors"),
    enabled,
    retry: false,
  });
  const keys = useQuery({
    queryKey: ["gateway/secrets"],
    queryFn: () => api<Secret[]>("/gateway/secrets"),
    enabled,
    retry: false,
  });
  const [adding, setAdding] = useState(false);
  const data = found.data;
  return (
    <div className="flex h-[min(560px,76vh)] flex-col">
      <div className="flex items-center gap-3 border-b border-border px-4 py-1.5">
        <p className="m-0 flex-1 text-[12px] text-muted-foreground">
          MCP servers the agent may use, such as the lab notebook. A playbook
          turns one on by naming it.
        </p>
        <Button
          variant="ghost"
          className="shrink-0 px-2 py-1 text-[12px]"
          title={`Kept in ${data?.file ?? "mcp.yaml"}; used from the next agent start`}
          onClick={() => setAdding(!adding)}
        >
          <Plus size={14} />
          New
        </Button>
      </div>
      <div className="flex min-h-0 flex-1 flex-col gap-1 overflow-y-auto px-4 py-1">
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
        <BuiltIn />
        {data?.connectors.map((c) => (
          <ConnectorRow
            key={c.name}
            c={c}
            keys={keys.data ?? []}
            atThisPc={!!data.at_this_pc}
          />
        ))}
      </div>
    </div>
  );
}
