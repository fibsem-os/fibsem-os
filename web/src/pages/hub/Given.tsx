import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Info, X } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Field, FIELD } from "@/pages/gateway/AgentCard";

/** What an agent on the hub PC is given beyond its own folder, as its
 * session's record has it (`fibsem.hub.agent_settings`). */
export interface Given {
  folder?: string;
  connectors?: string[];
  code?: boolean;
  network?: boolean;
  sites?: string[];
}

/** The Hub's settings for it: the two lists, the usable connectors, whether
 * this hub PC has a sandbox, and whether the start form offers any of it. */
interface AgentSettings {
  offered: boolean;
  data_folders: string[];
  allowed_sites: string[];
  connectors: string[];
  code_mode: boolean;
  at_this_pc: boolean;
}

const KEY = ["hub/agent-settings"];
const useAgentSettings = () =>
  useQuery({ queryKey: KEY, queryFn: () => api<AgentSettings>("/hub/agent-settings"), staleTime: 10_000 });

/** *given* in a few words, for a row or a page's header. */
export function givenWords(given: Given | null | undefined): string[] {
  if (!given) return [];
  return [
    given.code && (given.network ? "runs code, sandboxed, with network" : "runs code, sandboxed"),
    given.folder && `reads ${given.folder}`,
    given.connectors?.length && `connectors: ${given.connectors.join(", ")}`,
  ].filter((words): words is string => !!words);
}

function Hint({ children }: { children: string }) {
  return (
    <span title={children} className="inline-flex text-muted-foreground">
      <Info size={13} aria-label={children} />
    </span>
  );
}

function Tick({
  on,
  onChange,
  disabled,
  children,
  hint,
}: {
  on: boolean;
  onChange: (on: boolean) => void;
  disabled?: boolean;
  children: string;
  hint?: string;
}) {
  return (
    <span className="flex items-center gap-1.5 text-[13px] text-foreground">
      <label className={cn("flex items-center gap-1.5", disabled ? "opacity-60" : "cursor-pointer")}>
        <input
          type="checkbox"
          className="m-0 accent-[var(--primary)]"
          checked={on}
          disabled={disabled}
          onChange={(event) => onChange(event.target.checked)}
        />
        {children}
      </label>
      {hint && <Hint>{hint}</Hint>}
    </span>
  );
}

/** The start form's Folder, Connectors, Code mode and Network, for an agent
 * that runs on the hub PC. Nothing until the hub PC has switched them on. */
export function GivenFields({ value, onChange }: { value: Given; onChange: (value: Given) => void }) {
  const settings = useAgentSettings();
  const s = settings.data;
  if (!s?.offered) return null;
  const connectors = value.connectors ?? [];
  return (
    <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-3">
      <Field
        label={
          <span className="flex items-center gap-1.5">
            Folder
            <Hint>
              One data folder on the hub PC the agent may read, from the folders listed in the Hub's Settings. It never writes there: what it makes stays in its own work folder, in its session.
            </Hint>
          </span>
        }
      >
        <select
          value={value.folder ?? ""}
          onChange={(event) => onChange({ ...value, folder: event.target.value || undefined })}
          className={FIELD}
        >
          <option value="">None</option>
          {s.data_folders.map((folder) => (
            <option key={folder} value={folder}>
              {folder}
            </option>
          ))}
        </select>
      </Field>
      <div className="flex flex-col gap-1 text-xs text-muted-foreground sm:col-span-2">
        <span className="flex items-center gap-1.5">
          Connectors
          <Hint>
            Which of this Hub's connectors the agent may use. Their tokens stay in the hub PC's key store and are never shown to the agent or on this page.
          </Hint>
        </span>
        <div className="flex min-h-[31px] flex-wrap items-center gap-x-4 gap-y-1">
          {s.connectors.length === 0 && <span className="text-[13px]">None set up on this hub PC.</span>}
          {s.connectors.map((name) => (
            <Tick
              key={name}
              on={connectors.includes(name)}
              onChange={(on) =>
                onChange({
                  ...value,
                  connectors: on ? [...connectors, name] : connectors.filter((c) => c !== name),
                })
              }
            >
              {name}
            </Tick>
          ))}
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-x-5 gap-y-1 sm:col-span-3">
        <Tick
          on={!!value.code}
          disabled={!s.code_mode}
          onChange={(code) => onChange({ ...value, code, network: code && value.network })}
          hint={
            s.code_mode
              ? "The agent writes and runs its own code on the hub PC, always inside the sandbox: it writes only in its own work folder. There is no unsandboxed choice."
              : "Code mode needs the sandbox, which Windows doesn't have: the hub PC must run macOS or Linux."
          }
        >
          Code mode
        </Tick>
        <Tick
          on={!!value.network}
          disabled={!value.code || s.allowed_sites.length === 0}
          onChange={(network) => onChange({ ...value, network })}
          hint={
            s.allowed_sites.length
              ? `Off, the sandbox has no network. Ticked, its code reaches the Hub's allowed sites and nothing else, never a local address: ${s.allowed_sites.join(", ")}.`
              : "No site is allowed yet. The allowed sites are listed in the Hub's Settings, at the hub PC."
          }
        >
          Network
        </Tick>
      </div>
    </div>
  );
}

/** One of the two lists: its entries, each with a remove, and a box to add one. */
function Listed({
  label,
  hint,
  entries,
  placeholder,
  here,
  busy,
  onSave,
}: {
  label: string;
  hint: string;
  entries: string[];
  placeholder: string;
  here: boolean;
  busy: boolean;
  onSave: (entries: string[], done: () => void) => void;
}) {
  const [typed, setTyped] = useState("");
  const add = () => typed.trim() && onSave([...entries, typed.trim()], () => setTyped(""));
  return (
    <div className="flex flex-col gap-1.5 border-t border-border pt-2.5 text-[13px]">
      <span className="flex items-center gap-1.5">
        {label}
        <Hint>{hint}</Hint>
      </span>
      {entries.length === 0 && <span className="text-muted-foreground">None listed.</span>}
      {entries.map((entry) => (
        <span key={entry} className="flex items-center gap-2">
          <span className="min-w-0 flex-1 break-all font-mono text-xs">{entry}</span>
          <Button
            variant="ghost"
            aria-label={`Remove ${entry}`}
            disabled={!here || busy}
            onClick={() => onSave(entries.filter((e) => e !== entry), () => undefined)}
          >
            <X size={13} />
          </Button>
        </span>
      ))}
      <div className="flex items-center gap-2.5">
        <input
          value={typed}
          onChange={(event) => setTyped(event.target.value)}
          onKeyDown={(event) => event.key === "Enter" && add()}
          placeholder={placeholder}
          aria-label={label}
          disabled={!here}
          className={cn(FIELD, "min-w-0 flex-1")}
        />
        <Button disabled={!here || busy || !typed.trim()} onClick={add}>
          Add
        </Button>
      </div>
    </div>
  );
}

/** The Hub's settings for what an agent on this hub PC may be given: changed
 * only at the hub PC, on the Hub's own page. */
export function AgentSettingsCard() {
  const client = useQueryClient();
  const settings = useAgentSettings();
  const save = useMutation({
    mutationFn: (body: Partial<{ offered: boolean; data_folders: string[]; allowed_sites: string[] }>) =>
      api<AgentSettings>("/hub/agent-settings", { method: "PUT", body: JSON.stringify(body) }),
    onSuccess: () => client.invalidateQueries({ queryKey: KEY }),
  });
  const s = settings.data;
  if (!s) return null;
  const here = s.at_this_pc;
  return (
    <Card className="flex flex-col gap-2.5">
      <div className="flex items-center gap-2">
        <CardTitle>What an agent on this hub PC may be given</CardTitle>
        <Hint>
          A data folder to read, connectors, and code mode in the sandbox. Each start is checked against these lists. Changed only at this PC, on the Hub's own page.
        </Hint>
      </div>
      <Tick
        on={s.offered}
        disabled={!here || save.isPending}
        onChange={(offered) => save.mutate({ offered })}
        hint="Off, the start form offers none of it and a start that asks for it is refused. Switch it on once code mode has been tried on this hub PC."
      >
        Offer a folder, connectors and code mode in the start form
      </Tick>
      {!s.code_mode && (
        <p className="m-0 text-[13px] text-muted-foreground">
          Code mode needs the sandbox, which Windows doesn't have. A folder and connectors can still be given.
        </p>
      )}
      <Listed
        label="Data folders"
        hint="The folders on this hub PC an agent may be given to read, one per start. An agent never writes there."
        entries={s.data_folders}
        placeholder="/data/papers"
        here={here}
        busy={save.isPending}
        onSave={(data_folders, done) => save.mutate({ data_folders }, { onSuccess: done })}
      />
      <Listed
        label="Allowed sites"
        hint="The sites a code-mode sandbox may reach when Network is ticked, by host name. A local address is never allowed."
        entries={s.allowed_sites}
        placeholder="pypi.org"
        here={here}
        busy={save.isPending}
        onSave={(allowed_sites, done) => save.mutate({ allowed_sites }, { onSuccess: done })}
      />
      {save.error && <p className="m-0 text-[13px] text-destructive">{(save.error as Error).message}</p>}
    </Card>
  );
}
