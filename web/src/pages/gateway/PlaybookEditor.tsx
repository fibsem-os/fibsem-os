import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { X } from "lucide-react";
import { type ReactNode, useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import { FIELD, type Playbook } from "@/pages/gateway/AgentCard";

/** What the editor's pickers offer (`fibsem.gateway.playbook_editor.options`). */
interface Options {
  playbooks: string[];
  instruments: string[];
  skills: string[];
  agents: string[];
  mcp: string[];
  tools: string[];
  wakes: string[];
  hooks: {
    when: string;
    label: string;
    instrument: string | null;
    by_playbook: boolean;
  }[];
  efforts: string[];
  modes: string[];
  levels: string[];
  session: string[];
  numbers: Record<string, "int" | "float">;
  plan: string[];
}

/** A playbook as the agent loads it (`Playbook.summary`). */
interface Summary {
  name: string;
  description: string | null;
  extends: string[];
  instrument: string | null;
  mcp: string[];
  tools: string[];
  wakes: string[] | null;
  hooks: string[];
  skills: string[];
  optional: { skill: string; when: string; agents?: string[] }[];
  reminders: { when: string; label: string; say: string }[];
  rules: string[];
  agents: string[];
  plan: string[];
  session: Record<string, unknown>;
}

type Data = { [key: string]: unknown };

interface PlaybookFile {
  name: string;
  source: "lab" | "shipped" | "plugin";
  path: string;
  text: string;
  data: Data;
  lab: boolean;
  shipped: boolean;
  inherited: Summary | null;
  error: string | null;
}

interface Checked {
  text: string;
  data: Data | null;
  inherited: Summary | null;
  playbook: Summary | null;
  error: string | null;
  diff: string;
}

const WARN = "text-[var(--fibsem-semantic-warning-color)]";
const LINK =
  "w-fit cursor-pointer border-0 bg-transparent p-0 text-xs text-[var(--primary)] hover:underline";

function list(value: unknown): string[] {
  return Array.isArray(value) ? value.map(String) : [];
}

function mapping(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="grid grid-cols-1 items-start gap-1 py-1.5 text-[13px] sm:grid-cols-[150px_minmax(0,1fr)] sm:gap-3">
      <span className="pt-1 text-muted-foreground">{label}</span>
      <div className="flex min-w-0 flex-col gap-1">
        {children}
        {hint && <span className="text-xs text-muted-foreground">{hint}</span>}
      </div>
    </div>
  );
}

function Chip({
  children,
  onRemove,
  faint,
}: {
  children: ReactNode;
  onRemove?: () => void;
  faint?: boolean;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded border border-border px-1.5 py-0.5 font-mono text-xs",
        faint && "border-dashed text-muted-foreground",
      )}
    >
      {children}
      {onRemove && (
        <button
          onClick={onRemove}
          aria-label="Remove"
          className="cursor-pointer border-0 bg-transparent p-0 text-muted-foreground hover:text-foreground"
        >
          <X size={12} />
        </button>
      )}
    </span>
  );
}

/** Names picked from *options*, with what comes from below shown faint. */
function Picks({
  value,
  options,
  inherited,
  onChange,
  label,
}: {
  value: string[];
  options: string[];
  inherited: string[];
  onChange: (value: string[]) => void;
  label: string;
}) {
  const left = options.filter((o) => !value.includes(o));
  return (
    <>
      <div className="flex flex-wrap gap-1.5">
        {inherited.map((name) => (
          <Chip key={`below-${name}`} faint>
            {name}
          </Chip>
        ))}
        {value.map((name) => (
          <Chip
            key={name}
            onRemove={() => onChange(value.filter((v) => v !== name))}
          >
            {name}
          </Chip>
        ))}
        {left.length > 0 && (
          <select
            value=""
            onChange={(event) =>
              event.target.value && onChange([...value, event.target.value])
            }
            aria-label={`Add to ${label}`}
            className={cn(FIELD, "w-40 py-0.5 text-xs")}
          >
            <option value="">Add…</option>
            {left.map((o) => (
              <option key={o} value={o}>
                {o}
              </option>
            ))}
          </select>
        )}
      </div>
    </>
  );
}

/** One line each: rules, plan headings. */
function Lines({
  value,
  onChange,
  add,
  label,
  long,
}: {
  value: string[];
  onChange: (value: string[]) => void;
  add: string;
  label: string;
  long?: boolean;
}) {
  const set = (i: number, text: string) =>
    onChange(value.map((v, j) => (j === i ? text : v)));
  return (
    <div className="flex flex-col gap-1.5">
      {value.map((v, i) => (
        <div key={i} className="flex items-start gap-1.5">
          {long ? (
            <textarea
              value={v}
              rows={Math.min(4, Math.max(1, Math.ceil(v.length / 80)))}
              onChange={(event) => set(i, event.target.value)}
              aria-label={`${label} ${i + 1}`}
              className={cn(FIELD, "w-full resize-y")}
            />
          ) : (
            <input
              value={v}
              onChange={(event) => set(i, event.target.value)}
              aria-label={`${label} ${i + 1}`}
              className={cn(FIELD, "w-full sm:w-72")}
            />
          )}
          <button
            onClick={() => onChange(value.filter((_, j) => j !== i))}
            aria-label={`Remove ${label} ${i + 1}`}
            className="mt-1.5 cursor-pointer border-0 bg-transparent p-0 text-muted-foreground hover:text-foreground"
          >
            <X size={14} />
          </button>
        </div>
      ))}
      <button className={LINK} onClick={() => onChange([...value, ""])}>
        {add}
      </button>
    </div>
  );
}

/** Reminders: what the agent is told at a hook, by the hook's name. */
function Reminders({
  value,
  hooks,
  onChange,
}: {
  value: Record<string, unknown>;
  hooks: Options["hooks"];
  onChange: (value: Record<string, string>) => void;
}) {
  const rows = Object.entries(value).map(([when, say]) => [when, String(say)]);
  const put = (next: string[][]) =>
    onChange(Object.fromEntries(next.filter(([when]) => when)));
  const free = hooks.filter((h) => !(h.when in value));
  return (
    <div className="flex flex-col gap-1.5">
      {rows.map(([when, say], i) => (
        <div
          key={when}
          className="flex flex-col gap-1 sm:flex-row sm:items-start"
        >
          <span className="w-40 shrink-0 pt-1 text-xs">
            {hooks.find((h) => h.when === when)?.label ?? when}
          </span>
          <textarea
            value={say}
            rows={Math.min(4, Math.max(1, Math.ceil(say.length / 70)))}
            onChange={(event) =>
              put(
                rows.map((r, j) => (j === i ? [when, event.target.value] : r)),
              )
            }
            aria-label={`Reminder ${when}`}
            className={cn(FIELD, "w-full resize-y")}
          />
          <button
            onClick={() => put(rows.filter((_, j) => j !== i))}
            aria-label={`Remove reminder ${when}`}
            className="mt-1.5 cursor-pointer border-0 bg-transparent p-0 text-muted-foreground hover:text-foreground"
          >
            <X size={14} />
          </button>
        </div>
      ))}
      {free.length > 0 && (
        <select
          value=""
          onChange={(event) =>
            event.target.value && put([...rows, [event.target.value, ""]])
          }
          aria-label="Add a reminder"
          className={cn(FIELD, "w-56 py-0.5 text-xs")}
        >
          <option value="">Add a reminder at…</option>
          {free.map((h) => (
            <option key={h.when} value={h.when}>
              {h.label}
              {h.by_playbook ? " (an app's hook)" : ""}
            </option>
          ))}
        </select>
      )}
    </div>
  );
}

/** Optional skills: read only when the job needs them. */
function OptionalSkills({
  value,
  options,
  onChange,
  inherited,
}: {
  value: unknown[];
  options: Options;
  onChange: (value: Data[]) => void;
  inherited: Summary["optional"];
}) {
  const rows = value.map(mapping);
  const set = (i: number, change: Data) =>
    onChange(rows.map((r, j) => (j === i ? { ...r, ...change } : r)));
  return (
    <div className="flex flex-col gap-2">
      {inherited.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {inherited.map((o) => (
            <Chip key={o.skill} faint>
              {o.skill}
            </Chip>
          ))}
        </div>
      )}
      {rows.map((row, i) => (
        <div
          key={i}
          className="flex flex-col gap-1.5 rounded-md border border-border p-2"
        >
          <div className="flex items-center gap-1.5">
            <select
              value={String(row.skill ?? "")}
              onChange={(event) => set(i, { skill: event.target.value })}
              aria-label={`Optional skill ${i + 1}`}
              className={cn(FIELD, "w-56")}
            >
              <option value="">Pick a skill</option>
              {options.skills.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
            <span className="flex-1" />
            <button
              onClick={() => onChange(rows.filter((_, j) => j !== i))}
              aria-label={`Remove optional skill ${i + 1}`}
              className="cursor-pointer border-0 bg-transparent p-0 text-muted-foreground hover:text-foreground"
            >
              <X size={14} />
            </button>
          </div>
          <input
            value={String(row.when ?? "")}
            onChange={(event) => set(i, { when: event.target.value })}
            placeholder="When it is read: before the first polishing pass"
            aria-label={`When optional skill ${i + 1} is read`}
            className={cn(FIELD, "w-full")}
          />
          <Picks
            value={list(row.agents)}
            options={options.agents}
            inherited={[]}
            onChange={(agents) => set(i, { agents })}
            label="its subagents"
          />
          <Reminders
            value={mapping(row.reminders)}
            hooks={options.hooks}
            onChange={(reminders) => set(i, { reminders })}
          />
        </div>
      ))}
      <button
        className={LINK}
        onClick={() => onChange([...rows, { skill: "", when: "" }])}
      >
        Add an optional skill
      </button>
    </div>
  );
}

const SESSION_LABELS: Record<string, string> = {
  level: "Level it starts at",
  model: "Model",
  effort: "Effort",
  max_budget_usd: "Budget ($)",
  max_turns: "Turn limit",
  mode: "Mode",
  folder: "Folder",
};
const WAKE_ON = ["all", "digest"];

function SessionFields({
  value,
  options,
  inherited,
  onChange,
}: {
  value: Record<string, unknown>;
  options: Options;
  inherited: Record<string, unknown>;
  onChange: (value: Data) => void;
}) {
  const [more, setMore] = useState(false);
  const main = Object.keys(SESSION_LABELS);
  const shown = options.session.filter(
    (k) => main.includes(k) || more || value[k] != null,
  );
  const below = (k: string) =>
    inherited[k] != null ? `From below: ${String(inherited[k])}` : "Unset";
  const set = (k: string, v: unknown) =>
    onChange({ ...value, [k]: v === "" ? undefined : v });
  const choices: Record<string, string[]> = {
    level: options.levels,
    effort: options.efforts,
    mode: options.modes,
    wake_on_watch: WAKE_ON,
    wake_on_assist: WAKE_ON,
    wake_on_drive: WAKE_ON,
  };
  return (
    <div className="flex flex-col">
      {shown.map((k) => {
        const label = SESSION_LABELS[k] ?? k.replace(/_/g, " ");
        const v = value[k];
        let control: ReactNode;
        if (choices[k]) {
          control = (
            <select
              value={v == null ? "" : String(v)}
              onChange={(event) => set(k, event.target.value)}
              aria-label={label}
              className={cn(FIELD, "w-full sm:w-48")}
            >
              <option value="">{below(k)}</option>
              {choices[k].map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          );
        } else if (k === "context_marks") {
          control = (
            <input
              value={list(v).join(", ")}
              onChange={(event) =>
                set(
                  k,
                  event.target.value
                    .split(",")
                    .map((s) => s.trim())
                    .filter(Boolean)
                    .map(Number),
                )
              }
              placeholder={below(k)}
              aria-label={label}
              className={cn(FIELD, "w-full sm:w-72")}
            />
          );
        } else {
          const number = options.numbers[k];
          control = (
            <input
              value={v == null ? "" : String(v)}
              onChange={(event) => {
                const text = event.target.value;
                const n = Number(text);
                set(k, number && text.trim() && !isNaN(n) ? n : text);
              }}
              inputMode={number ? "decimal" : undefined}
              placeholder={below(k)}
              aria-label={label}
              className={cn(
                FIELD,
                number ? "w-full sm:w-32" : "w-full sm:w-72",
              )}
            />
          );
        }
        return (
          <Field key={k} label={label}>
            {control}
          </Field>
        );
      })}
      <button className={cn(LINK, "mt-1")} onClick={() => setMore(!more)}>
        {more ? "Fewer session settings" : "More session settings"}
      </button>
    </div>
  );
}

function SubHeading({ children }: { children: ReactNode }) {
  return (
    <h3 className="m-0 mt-3 border-t border-border pt-3 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
      {children}
    </h3>
  );
}

/** The form over a draft's fields: each list shows, faint, what comes from
 * below (what it extends, or the shipped base). */
function PlaybookForm({
  data,
  options,
  inherited,
  onChange,
}: {
  data: Data;
  options: Options;
  inherited: Summary | null;
  onChange: (data: Data) => void;
}) {
  const set = (key: string, value: unknown) =>
    onChange({ ...data, [key]: value });
  const from = (key: keyof Summary) => (inherited ? list(inherited[key]) : []);
  const wakes = data.wakes;
  return (
    <div className="flex flex-col">
      <Field label="Description">
        <textarea
          value={String(data.description ?? "")}
          rows={2}
          onChange={(event) => set("description", event.target.value)}
          aria-label="Description"
          className={cn(FIELD, "w-full resize-y")}
        />
      </Field>
      <Field
        label="Extends"
        hint="Its skills, rules and reminders come first; its settings hold where this one sets none."
      >
        <select
          value={String(data.extends ?? "")}
          onChange={(event) => set("extends", event.target.value || undefined)}
          aria-label="Extends"
          className={cn(FIELD, "w-full sm:w-72")}
        >
          <option value="">Nothing (the shipped base only)</option>
          {options.playbooks.map((p) => (
            <option key={p} value={p}>
              {p}
            </option>
          ))}
        </select>
      </Field>
      <Field label="Instrument">
        <select
          value={String(data.instrument ?? "")}
          onChange={(event) =>
            set("instrument", event.target.value || undefined)
          }
          aria-label="Instrument"
          className={cn(FIELD, "w-full sm:w-48")}
        >
          <option value="">
            {inherited?.instrument
              ? `From below: ${inherited.instrument}`
              : "Unset"}
          </option>
          {options.instruments.map((k) => (
            <option key={k} value={k}>
              {k}
            </option>
          ))}
        </select>
      </Field>

      <SubHeading>Session</SubHeading>
      <SessionFields
        value={mapping(data.session)}
        options={options}
        inherited={inherited?.session ?? {}}
        onChange={(session) => set("session", session)}
      />

      <SubHeading>What it reads and uses</SubHeading>
      <Field label="Skills" hint="Read at the start of the job.">
        <Picks
          value={list(data.skills)}
          options={options.skills}
          inherited={from("skills")}
          onChange={(v) => set("skills", v)}
          label="skills"
        />
      </Field>
      <Field label="Optional skills" hint="Read only when the job needs them.">
        <OptionalSkills
          value={Array.isArray(data.optional) ? data.optional : []}
          options={options}
          inherited={inherited?.optional ?? []}
          onChange={(v) => set("optional", v)}
        />
      </Field>
      <Field label="Subagents">
        <Picks
          value={list(data.agents)}
          options={options.agents}
          inherited={from("agents")}
          onChange={(v) => set("agents", v)}
          label="subagents"
        />
      </Field>
      <Field
        label="Connectors"
        hint="MCP servers, by name: fibsem, or one in Connectors below."
      >
        <Picks
          value={list(data.mcp)}
          options={options.mcp}
          inherited={from("mcp")}
          onChange={(v) => set("mcp", v)}
          label="connectors"
        />
      </Field>
      <Field
        label="Tools loaded at once"
        hint="Each costs context from the first turn: keep it short."
      >
        <Picks
          value={list(data.tools)}
          options={options.tools}
          inherited={from("tools")}
          onChange={(v) => set("tools", v)}
          label="tools"
        />
      </Field>
      <Field
        label="App hooks"
        hint="An app's hooks it switches on, for reminders."
      >
        <Picks
          value={list(data.hooks)}
          options={options.hooks
            .filter((h) => h.by_playbook)
            .map((h) => h.when)}
          inherited={from("hooks")}
          onChange={(v) => set("hooks", v)}
          label="app hooks"
        />
      </Field>

      <SubHeading>What it is told</SubHeading>
      <Field label="Rules">
        {from("rules").length > 0 && (
          <ul className="m-0 mb-1 pl-4 text-xs text-muted-foreground">
            {from("rules").map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
        )}
        <Lines
          value={list(data.rules)}
          onChange={(v) => set("rules", v)}
          add="Add a rule"
          label="Rule"
          long
        />
      </Field>
      <Field label="Reminders">
        {(inherited?.reminders ?? []).length > 0 && (
          <ul className="m-0 mb-1 pl-4 text-xs text-muted-foreground">
            {inherited!.reminders.map((r) => (
              <li key={`${r.when}-${r.say}`}>
                {r.label}: {r.say}
              </li>
            ))}
          </ul>
        )}
        <Reminders
          value={mapping(data.reminders)}
          hooks={options.hooks}
          onChange={(v) => set("reminders", v)}
        />
      </Field>

      <SubHeading>Waking and the plan</SubHeading>
      <Field
        label="What wakes it"
        hint="The operator's messages, the level and Stop always do."
      >
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={wakes == null}
            onChange={(event) =>
              set("wakes", event.target.checked ? undefined : [])
            }
          />
          {inherited?.wakes
            ? `As below (${inherited.wakes.join(", ")})`
            : "Every line"}
        </label>
        {wakes != null &&
          options.wakes.map((g) => (
            <label key={g} className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={list(wakes).includes(g)}
                onChange={(event) =>
                  set(
                    "wakes",
                    event.target.checked
                      ? [...list(wakes), g]
                      : list(wakes).filter((w) => w !== g),
                  )
                }
              />
              {g}
            </label>
          ))}
      </Field>
      <Field
        label="Plan headings"
        hint={`Unset: ${(inherited?.plan ?? options.plan).join(", ")}`}
      >
        <Lines
          value={list(data.plan)}
          onChange={(v) => set("plan", v)}
          add="Add a heading"
          label="Heading"
        />
      </Field>
    </div>
  );
}

/** The playbook the draft loads as: what the agent would run with. */
function Preview({ checked }: { checked: Checked | null }) {
  if (!checked)
    return <p className="m-0 text-xs text-muted-foreground">Checking…</p>;
  if (checked.error)
    return (
      <p className="m-0 break-words text-[13px] text-destructive [overflow-wrap:anywhere]">
        It wouldn't load: {checked.error}
      </p>
    );
  const p = checked.playbook!;
  const session = Object.entries(p.session)
    .map(([k, v]) => `${k} ${Array.isArray(v) ? v.join(",") : String(v)}`)
    .join(" · ");
  const rows: [string, string][] = [
    ["Runs", p.extends.concat(p.name).join(" → ")],
    ["Instrument", p.instrument ?? "—"],
    ["Skills", p.skills.join(", ") || "none"],
    ["Optional", p.optional.map((o) => o.skill).join(", ") || "none"],
    ["Subagents", p.agents.join(", ") || "none"],
    ["Connectors", p.mcp.join(", ") || "none"],
    ["Tools at once", p.tools.length ? `${p.tools.length}` : "none"],
    ["Rules", String(p.rules.length)],
    ["Reminders", p.reminders.map((r) => r.label).join(", ") || "none"],
    [
      "Wakes on",
      p.wakes ? p.wakes.join(", ") || "nothing routine" : "every line",
    ],
    ["Session", session || "the defaults"],
  ];
  return (
    <div className="flex flex-col gap-0.5 text-xs">
      {rows.map(([k, v]) => (
        <div key={k} className="grid grid-cols-[100px_minmax(0,1fr)] gap-2">
          <span className="text-muted-foreground">{k}</span>
          <span className="break-words">{v}</span>
        </div>
      ))}
    </div>
  );
}

function Diff({ diff }: { diff: string }) {
  if (!diff)
    return <p className="m-0 text-xs text-muted-foreground">No changes.</p>;
  return (
    <pre className="m-0 max-h-96 overflow-auto rounded-md border border-border p-2 font-mono text-xs leading-5">
      {diff.split("\n").map((line, i) => (
        <div
          key={i}
          className={cn(
            line.startsWith("+") &&
              !line.startsWith("+++") &&
              "text-[var(--primary)]",
            line.startsWith("-") &&
              !line.startsWith("---") &&
              "text-destructive",
            line.startsWith("@@") && "text-muted-foreground",
          )}
        >
          {line || " "}
        </div>
      ))}
    </pre>
  );
}

type Tab = "form" | "yaml" | "changes";

/** Edit one playbook: the form or its YAML, checked as the agent loads it
 * as you type, with what it would run and the change from the file. Saving
 * writes the lab's copy. */
function Editor({
  name,
  start,
  options,
  onClose,
}: {
  name: string;
  /** A new playbook's first fields; unset to edit the file there is. */
  start?: Data;
  options: Options;
  onClose: () => void;
}) {
  const client = useQueryClient();
  const file = useQuery({
    queryKey: ["gateway/playbooks", name],
    queryFn: () =>
      api<PlaybookFile>(`/gateway/playbooks/${encodeURIComponent(name)}`),
    enabled: start === undefined,
    retry: false,
  });
  const [tab, setTab] = useState<Tab>("form");
  const [data, setData] = useState<Data | null>(start ?? null);
  const [text, setText] = useState<string | null>(null);
  const [from, setFrom] = useState<"form" | "yaml">("form");
  const [checked, setChecked] = useState<Checked | null>(null);
  const [askRemove, setAskRemove] = useState(false);
  const asked = useRef(0);
  useEffect(() => {
    if (file.data && data === null) {
      setData(file.data.data);
      setText(file.data.text);
      setFrom("yaml");
    }
  }, [file.data, data]);
  useEffect(
    () => {
      if (data === null) return;
      const body = from === "yaml" && text !== null ? { text } : { data };
      const mine = ++asked.current;
      const timer = setTimeout(() => {
        api<Checked>(`/gateway/playbooks/${encodeURIComponent(name)}/check`, {
          method: "POST",
          body: JSON.stringify(body),
        })
          .then((answer) => {
            if (mine !== asked.current) return;
            setChecked(answer);
            if (from === "form") setText(answer.text);
            else if (answer.data) setData(answer.data);
          })
          .catch((error: Error) => {
            if (mine !== asked.current) return;
            setChecked({
              text: text ?? "",
              data: null,
              inherited: null,
              playbook: null,
              error: error.message,
              diff: "",
            });
          });
      }, 400);
      return () => clearTimeout(timer);
      // `text` follows `data` from the form, and the reverse: ask only on
      // the side being edited.
      // eslint-disable-next-line react-hooks/exhaustive-deps
    },
    from === "form" ? [data, name] : [text, name],
  );
  const done = () => {
    client.invalidateQueries({ queryKey: ["gateway/agent/playbooks"] });
    client.invalidateQueries({ queryKey: ["gateway/playbooks", name] });
    client.invalidateQueries({ queryKey: ["gateway/playbook-options"] });
  };
  const save = useMutation({
    mutationFn: () =>
      api<PlaybookFile>(`/gateway/playbooks/${encodeURIComponent(name)}`, {
        method: "PUT",
        body: JSON.stringify(
          from === "yaml" && text !== null ? { text } : { data },
        ),
      }),
    onSuccess: () => {
      done();
      onClose();
    },
  });
  const remove = useMutation({
    mutationFn: () =>
      api(`/gateway/playbooks/${encodeURIComponent(name)}`, {
        method: "DELETE",
      }),
    onSuccess: () => {
      done();
      onClose();
    },
  });
  if (file.error)
    return (
      <p className="m-0 text-[13px] text-destructive">
        {(file.error as Error).message}
      </p>
    );
  if (data === null)
    return <p className="m-0 text-xs text-muted-foreground">Reading…</p>;
  const f = file.data;
  const inherited = checked ? checked.inherited : (f?.inherited ?? null);
  const changed = Boolean(checked?.diff);
  const tabs: [Tab, string][] = [
    ["form", "Form"],
    ["yaml", "YAML"],
    ["changes", changed ? "Changes •" : "Changes"],
  ];
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-baseline gap-2">
        <span className="font-mono text-sm">{name}</span>
        <span className="text-xs text-muted-foreground">
          {start !== undefined
            ? "new, the lab's"
            : f?.source === "lab"
              ? f.shipped
                ? "the lab's copy, used in place of the shipped one"
                : "the lab's"
              : `${f?.source}: saving makes the lab's copy, used in place of it`}
        </span>
        <span className="flex-1" />
        <button className={LINK} onClick={onClose}>
          Back to playbooks
        </button>
      </div>
      <div className="flex gap-1 border-b border-border">
        {tabs.map(([t, label]) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={cn(
              "cursor-pointer border-0 border-b-2 bg-transparent px-2.5 py-1.5 text-[13px]",
              tab === t
                ? "border-[var(--primary)] text-foreground"
                : "border-transparent text-muted-foreground",
            )}
          >
            {label}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_280px]">
        <div className="min-w-0">
          {tab === "form" && (
            <>
              {f?.text.includes("#") && from === "form" && (
                <p className={cn("m-0 pb-1 text-xs", WARN)}>
                  Saving from the form keeps the file's opening comments but not
                  the others. The YAML tab keeps them.
                </p>
              )}
              <PlaybookForm
                data={data}
                options={options}
                inherited={inherited}
                onChange={(next) => {
                  setFrom("form");
                  setData(next);
                }}
              />
            </>
          )}
          {tab === "yaml" && (
            <textarea
              value={text ?? ""}
              onChange={(event) => {
                setFrom("yaml");
                setText(event.target.value);
              }}
              spellCheck={false}
              aria-label="The playbook's YAML"
              className={cn(
                FIELD,
                "h-[480px] w-full resize-y font-mono text-xs",
              )}
            />
          )}
          {tab === "changes" && <Diff diff={checked?.diff ?? ""} />}
        </div>
        <div className="flex flex-col gap-2 lg:border-l lg:border-border lg:pl-4">
          <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            What the agent would run
          </span>
          <Preview checked={checked} />
        </div>
      </div>
      {save.error && (
        <p className="m-0 text-[13px] text-destructive">
          {(save.error as Error).message}
        </p>
      )}
      {askRemove && (
        <div className="flex flex-col gap-2 rounded-md border border-border px-3 py-2 text-[13px]">
          <p className={cn("m-0", WARN)}>
            {f?.shipped
              ? "Remove the lab's copy? This PC goes back to the shipped playbook."
              : "Delete this playbook? A session can no longer start on it."}{" "}
            The file is kept as {name}.yaml.bak.
          </p>
          <div className="flex justify-end gap-2.5">
            <Button onClick={() => setAskRemove(false)}>Cancel</Button>
            <Button
              variant="danger"
              disabled={remove.isPending}
              onClick={() => remove.mutate()}
            >
              {f?.shipped ? "Use the shipped one" : "Delete it"}
            </Button>
          </div>
        </div>
      )}
      <div className="flex flex-wrap items-center gap-2.5 pt-1">
        {f?.lab && !askRemove && (
          <Button variant="danger" onClick={() => setAskRemove(true)}>
            {f.shipped ? "Use the shipped one again" : "Delete"}
          </Button>
        )}
        <span className="flex-1 text-xs text-muted-foreground">
          A running session keeps the playbook it started on: this counts from
          the next one. Each save is in History.
        </span>
        <Button onClick={onClose}>Cancel</Button>
        <Button
          variant="primary"
          disabled={
            save.isPending ||
            !checked ||
            Boolean(checked.error) ||
            (!changed && start === undefined)
          }
          onClick={() => save.mutate()}
        >
          {save.isPending ? "Saving…" : "Save"}
        </Button>
      </div>
    </div>
  );
}

/** Start a new playbook: its name, and what it extends. */
function NewPlaybook({
  options,
  onStart,
  onCancel,
}: {
  options: Options;
  onStart: (name: string, extendsName: string) => void;
  onCancel: () => void;
}) {
  const [name, setName] = useState("");
  const [base, setBase] = useState(options.playbooks[0] ?? "");
  const taken = options.playbooks.includes(name.trim());
  const ok = /^[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(name.trim()) && !taken;
  return (
    <div className="flex flex-col gap-1 rounded-md border border-border p-3">
      <Field
        label="Name"
        hint={
          taken
            ? "There is a playbook of that name: edit it instead."
            : undefined
        }
      >
        <input
          value={name}
          onChange={(event) => setName(event.target.value)}
          placeholder="my-lamellae"
          aria-label="New playbook's name"
          className={cn(FIELD, "w-full sm:w-72")}
        />
      </Field>
      <Field label="Extends">
        <select
          value={base}
          onChange={(event) => setBase(event.target.value)}
          aria-label="New playbook extends"
          className={cn(FIELD, "w-full sm:w-72")}
        >
          <option value="">Nothing (the shipped base only)</option>
          {options.playbooks.map((p) => (
            <option key={p} value={p}>
              {p}
            </option>
          ))}
        </select>
      </Field>
      <div className="flex justify-end gap-2.5 pt-1">
        <Button onClick={onCancel}>Cancel</Button>
        <Button
          variant="primary"
          disabled={!ok}
          onClick={() => onStart(name.trim(), base)}
        >
          Start it
        </Button>
      </div>
    </div>
  );
}

/** The playbooks on this PC, each opened in the editor. Any operator edits
 * them, at this PC or paired. */
export function PlaybooksCard() {
  const playbooks = useQuery({
    queryKey: ["gateway/agent/playbooks"],
    queryFn: () => api<Playbook[]>("/gateway/agent/playbooks"),
    staleTime: 30_000,
    retry: false,
  });
  const options = useQuery({
    queryKey: ["gateway/playbook-options"],
    queryFn: () => api<Options>("/gateway/playbook-options"),
    staleTime: 30_000,
    retry: false,
  });
  const [open, setOpen] = useState<{ name: string; start?: Data } | null>(null);
  const [adding, setAdding] = useState(false);
  const error = playbooks.error ?? options.error;
  return (
    <Card className="flex flex-col gap-1">
      <CardTitle>Playbooks</CardTitle>
      <p className="m-0 pb-1 text-[13px] text-muted-foreground">
        How each kind of job runs. An edit is saved as the lab's copy, checked
        first as the agent loads it.
      </p>
      {error && (
        <p className="m-0 text-[13px] text-destructive">
          {(error as Error).message}
        </p>
      )}
      {open && options.data ? (
        <Editor
          key={open.name}
          name={open.name}
          start={open.start}
          options={options.data}
          onClose={() => setOpen(null)}
        />
      ) : (
        <>
          {playbooks.data?.map((p) => (
            <div
              key={p.name}
              className="grid grid-cols-1 gap-1 border-t border-border py-2 text-[13px] sm:grid-cols-[190px_minmax(0,1fr)_auto] sm:gap-3"
            >
              <span className="font-mono text-xs">{p.name}</span>
              {p.error ? (
                <span className="text-destructive">{p.error}</span>
              ) : (
                <span className="text-muted-foreground">
                  {(p as Playbook & { description?: string }).description ?? ""}
                </span>
              )}
              <button
                className={LINK}
                disabled={!options.data}
                onClick={() => setOpen({ name: p.name })}
              >
                Edit
              </button>
            </div>
          ))}
          {adding && options.data ? (
            <NewPlaybook
              options={options.data}
              onCancel={() => setAdding(false)}
              onStart={(name, base) => {
                setAdding(false);
                setOpen({
                  name,
                  start: base
                    ? { description: "", extends: base }
                    : { description: "" },
                });
              }}
            />
          ) : (
            <div className="flex justify-end pt-1">
              <Button disabled={!options.data} onClick={() => setAdding(true)}>
                New playbook
              </Button>
            </div>
          )}
        </>
      )}
    </Card>
  );
}
