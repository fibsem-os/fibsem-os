import { useQuery } from "@tanstack/react-query";
import { Plus, X } from "lucide-react";
import type { ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { Markdown } from "@/lib/markdown";
import { cn } from "@/lib/utils";
import { PickSummary } from "./Picker";

/** What the playbook and subagent forms offer to pick from (GET /app/library/options). */
export interface LibraryOptions {
  levels: string[];
  modes: string[];
  efforts: string[];
  reminders: { when: string; label: string }[];
  skills: string[];
  playbooks: string[];
  agents: string[];
  tools: string[];
}

export type Data = Record<string, unknown>;
type Reminder = { when: string; say: string };

const MODES: Record<string, string> = { none: "No files or code", files: "Files", code: "Code" };
const LEVELS: Record<string, string> = { watch: "Watch", assist: "Assist", drive: "Drive" };

const field = "rounded border border-input bg-background px-2 py-1.5 text-[13px] text-foreground";
const text = (v: unknown) => (typeof v === "string" ? v : "");
const list = (v: unknown) => (Array.isArray(v) ? (v as unknown[]) : []);
const strings = (v: unknown) => list(v).filter((x): x is string => typeof x === "string");

function Row({ label, form = false, children }: { label: string; form?: boolean; children: ReactNode }) {
  return (
    <div className={cn("grid grid-cols-[132px_1fr] gap-3 py-1.5", form ? "items-start" : "items-baseline")}>
      <span className={cn("text-[12px] text-muted-foreground", form && "pt-[7px]")}>{label}</span>
      <div className="min-w-0 text-[13px]">{children}</div>
    </div>
  );
}

function Choice({
  value,
  options,
  empty,
  labels = {},
  onChange,
}: {
  value: string;
  options: string[];
  empty?: string;
  labels?: Record<string, string>;
  onChange: (value: string) => void;
}) {
  return (
    <select className={cn(field, "py-1")} value={value} onChange={(e) => onChange(e.target.value)}>
      {empty !== undefined && <option value="">{empty}</option>}
      {options.map((o) => (
        <option key={o} value={o}>
          {labels[o] ?? o}
        </option>
      ))}
    </select>
  );
}

function Lines({ value, rows = 2, onChange }: { value: string; rows?: number; onChange: (value: string) => void }) {
  return (
    <textarea
      rows={rows}
      className={cn(field, "w-full resize-y leading-[1.5]")}
      value={value}
      onChange={(e) => onChange(e.target.value)}
    />
  );
}

function Add({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <Button variant="ghost" className="mt-1 px-1.5 py-0.5 text-[12px]" onClick={onClick}>
      <Plus size={13} />
      {label}
    </Button>
  );
}

function Remove({ onClick }: { onClick: () => void }) {
  return (
    <Button variant="ghost" className="px-1.5" aria-label="Remove" title="Remove" onClick={onClick}>
      <X size={14} />
    </Button>
  );
}

function Reminders({
  value,
  options,
  onChange,
}: {
  value: Reminder[];
  options: LibraryOptions;
  onChange: (value: Reminder[]) => void;
}) {
  const set = (i: number, r: Partial<Reminder>) => onChange(value.map((v, j) => (j === i ? { ...v, ...r } : v)));
  return (
    <div className="flex flex-col gap-2">
      {value.map((r, i) => (
        <div key={i} className="flex items-start gap-1.5">
          <div className="flex min-w-0 flex-1 flex-col gap-1">
            <Choice
              value={r.when}
              options={options.reminders.map((o) => o.when)}
              labels={Object.fromEntries(options.reminders.map((o) => [o.when, o.label]))}
              onChange={(when) => set(i, { when })}
            />
            <Lines value={r.say} onChange={(say) => set(i, { say })} />
          </div>
          <Remove onClick={() => onChange(value.filter((_, j) => j !== i))} />
        </div>
      ))}
      <div>
        <Add label="Add reminder" onClick={() => onChange([...value, { when: options.reminders[0].when, say: "" }])} />
      </div>
    </div>
  );
}

function TextList({ value, add, onChange }: { value: string[]; add: string; onChange: (value: string[]) => void }) {
  return (
    <div className="flex flex-col gap-1.5">
      {value.map((v, i) => (
        <div key={i} className="flex items-start gap-1.5">
          <Lines value={v} onChange={(t) => onChange(value.map((x, j) => (j === i ? t : x)))} />
          <Remove onClick={() => onChange(value.filter((_, j) => j !== i))} />
        </div>
      ))}
      <div>
        <Add label={add} onClick={() => onChange([...value, ""])} />
      </div>
    </div>
  );
}

/** A reminder's label, as the options name it. */
function label(options: LibraryOptions | undefined, when: string) {
  return options?.reminders.find((o) => o.when === when)?.label ?? when;
}

function ReminderList({ value, options }: { value: Reminder[]; options?: LibraryOptions }) {
  return (
    <ul className="m-0 flex list-none flex-col gap-1.5 p-0">
      {value.map((r, i) => (
        <li key={i}>
          <span className="text-muted-foreground">{label(options, r.when)}: </span>
          {r.say}
        </li>
      ))}
    </ul>
  );
}

const playbookReminders = (value: unknown): Reminder[] =>
  value && typeof value === "object" && !Array.isArray(value)
    ? Object.entries(value as Data).map(([when, say]) => ({ when, say: text(say) }))
    : [];

const asMapping = (value: Reminder[]) => Object.fromEntries(value.map((r) => [r.when, r.say]));

/** A playbook's session settings in one line: what a session started on it runs with. */
function sessionLine(session: Data) {
  return [
    session.level ? LEVELS[text(session.level)] : undefined,
    session.mode ? (MODES[text(session.mode)] ?? text(session.mode)) : undefined,
    session.model ? text(session.model) : undefined,
    session.effort ? `${text(session.effort)} effort` : undefined,
    session.max_budget_usd != null ? `US$${session.max_budget_usd}` : undefined,
    session.max_turns != null ? `${session.max_turns} turns` : undefined,
  ]
    .filter(Boolean)
    .join(" · ");
}

type Inherited = Record<string, { value: string | number; source: string }>;

/** The session settings of the playbook *base*, with where each comes from. */
function useInherited(base: string): Inherited {
  const parent = useQuery({
    queryKey: ["library", "playbooks", base],
    queryFn: () => api<{ session?: Inherited | null }>(`/app/library/playbooks/${encodeURIComponent(base)}`),
    enabled: !!base,
  });
  return (base && parent.data?.session) || {};
}

/** Where an empty setting's value comes from, beside it. */
function From({ found }: { found?: { source: string } }) {
  return found ? <span className="shrink-0 text-[12px] text-muted-foreground">from {found.source}</span> : null;
}

/** The session settings as form rows. An empty one shows the value of the playbook it builds on,
 * and where that comes from. */
function SessionForm({
  value,
  base,
  options,
  onChange,
}: {
  value: Data;
  base: string;
  options: LibraryOptions;
  onChange: (value: Data) => void;
}) {
  const inherited = useInherited(base);
  const set = (key: string, v: unknown) => onChange({ ...value, [key]: v });
  const number = (key: string, v: string) => set(key, v === "" ? null : Number(v));
  // Set here, or else the inherited value and where it comes from.
  const own = (key: string) => value[key] != null && value[key] !== "";
  const from = (key: string) => (own(key) ? undefined : inherited[key]);
  const shown = (key: string, labels: Record<string, string> = {}, fallback = "") => {
    const found = inherited[key];
    return found ? (labels[String(found.value)] ?? String(found.value)) : fallback;
  };
  const line = (key: string, control: ReactNode) => (
    <div className="flex items-center gap-2">
      {control}
      <From found={from(key)} />
    </div>
  );
  return (
    <>
      <Row form label="Level">
        {line(
          "level",
          <Choice
            value={text(value.level)}
            options={options.levels}
            labels={LEVELS}
            empty={shown("level", LEVELS, "Watch")}
            onChange={(v) => set("level", v)}
          />,
        )}
      </Row>
      <Row form label="Files and code">
        {line(
          "mode",
          <Choice
            value={text(value.mode)}
            options={options.modes}
            labels={MODES}
            empty={shown("mode", MODES, MODES.none)}
            onChange={(v) => set("mode", v)}
          />,
        )}
      </Row>
      <Row form label="Model">
        {line(
          "model",
          <input
            className={cn(field, "min-w-0 flex-1")}
            placeholder={shown("model", {}, "Default")}
            value={text(value.model)}
            onChange={(e) => set("model", e.target.value)}
          />,
        )}
      </Row>
      <Row form label="Effort">
        {line(
          "effort",
          <Choice
            value={text(value.effort)}
            options={options.efforts}
            empty={shown("effort", {}, "Default")}
            onChange={(v) => set("effort", v)}
          />,
        )}
      </Row>
      <Row form label="Budget (US$)">
        {line(
          "max_budget_usd",
          <input
            type="number"
            min={0}
            className={cn(field, "w-28")}
            placeholder={shown("max_budget_usd")}
            value={value.max_budget_usd == null ? "" : String(value.max_budget_usd)}
            onChange={(e) => number("max_budget_usd", e.target.value)}
          />,
        )}
      </Row>
      <Row form label="Turns">
        {line(
          "max_turns",
          <input
            type="number"
            min={1}
            className={cn(field, "w-28")}
            placeholder={shown("max_turns")}
            value={value.max_turns == null ? "" : String(value.max_turns)}
            onChange={(e) => number("max_turns", e.target.value)}
          />,
        )}
      </Row>
    </>
  );
}

type Optional_ = Data & { skill: string; when: string };

/** A playbook's settings: read, or as a form when *onChange* is given. */
export function PlaybookSettings({
  name,
  data,
  options,
  onChange,
}: {
  name: string;
  data: Data;
  options?: LibraryOptions;
  onChange?: (data: Data) => void;
}) {
  const set = (key: string, value: unknown) => onChange?.({ ...data, [key]: value });
  const reminders = playbookReminders(data.reminders);
  // Each optional skill keeps its own reminders, which the form doesn't show.
  const optional = list(data.optional).map((o) => {
    const entry = (o ?? {}) as Data;
    return { ...entry, skill: text(entry.skill), when: text(entry.when) } as Optional_;
  });
  const rules = strings(data.rules);
  const session = (data.session && typeof data.session === "object" ? data.session : {}) as Data;
  // What a session started on it runs with, its own settings and those it builds on.
  const resolved = useInherited(name);
  if (!onChange || !options) {
    const values = Object.fromEntries(Object.entries(resolved).map(([k, v]) => [k, v.value]));
    const elsewhere = [...new Set(Object.values(resolved).map((v) => v.source))].filter((n) => n !== name);
    const rows: [string, ReactNode][] = [
      ["Builds on", text(data.extends)],
      [
        "Session",
        sessionLine(values) ? (
          <>
            {sessionLine(values)}
            {elsewhere.length > 0 && <span className="text-muted-foreground"> · from {elsewhere.join(", ")}</span>}
          </>
        ) : undefined,
      ],
      ["Skills", strings(data.skills).join(", ")],
      ["Subagents", strings(data.agents).join(", ")],
      [
        "Optional skills",
        optional.length ? (
          <ul className="m-0 flex list-none flex-col gap-1.5 p-0">
            {optional.map((o, i) => (
              <li key={i}>
                {o.skill} <span className="text-muted-foreground">when {o.when}</span>
                {strings(o.agents).length > 0 && (
                  <span className="text-muted-foreground"> · subagents {strings(o.agents).join(", ")}</span>
                )}
              </li>
            ))}
          </ul>
        ) : undefined,
      ],
      ["Reminders", reminders.length ? <ReminderList value={reminders} options={options} /> : undefined],
      [
        "Rules",
        rules.length ? (
          <ul className="m-0 flex list-disc flex-col gap-1 pl-4">
            {rules.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        ) : undefined,
      ],
    ];
    return (
      <div>
        {rows
          .filter(([, v]) => v)
          .map(([k, v]) => (
            <Row key={k} label={k}>
              {v}
            </Row>
          ))}
      </div>
    );
  }
  const setOptional = (i: number, o: Partial<Optional_>) =>
    set(
      "optional",
      optional.map((v, j) => (j === i ? { ...v, ...o } : v)),
    );
  return (
    <div>
      <Row form label="Description">
        <Lines value={text(data.description)} onChange={(v) => set("description", v)} />
      </Row>
      <Row form label="Builds on">
        <Choice
          value={text(data.extends)}
          options={options.playbooks.filter((p) => p !== name)}
          empty="Nothing"
          onChange={(v) => set("extends", v)}
        />
      </Row>
      <h4 className="mt-3 mb-0 text-[12px] font-semibold text-muted-foreground">A session started on this playbook</h4>
      <SessionForm value={session} base={text(data.extends)} options={options} onChange={(v) => set("session", v)} />
      <h4 className="mt-3 mb-0 text-[12px] font-semibold text-muted-foreground">The work</h4>
      <Row form label="Skills">
        <PickSummary
          kind="skills"
          value={strings(data.skills)}
          options={options.skills}
          onChange={(v) => set("skills", v)}
        />
      </Row>
      <Row form label="Subagents">
        <PickSummary
          kind="agents"
          value={strings(data.agents)}
          options={options.agents}
          onChange={(v) => set("agents", v)}
        />
      </Row>
      <Row form label="Optional skills">
        <div className="flex flex-col gap-2">
          {optional.map((o, i) => (
            <div key={i} className="flex items-start gap-1.5">
              <div className="flex min-w-0 flex-1 flex-col gap-1">
                <Choice value={o.skill} options={options.skills} onChange={(skill) => setOptional(i, { skill })} />
                <input
                  className={cn(field, "w-full")}
                  placeholder="When the agent reads it"
                  value={o.when}
                  onChange={(e) => setOptional(i, { when: e.target.value })}
                />
                <div className="text-[12px] text-muted-foreground">
                  Subagents:{" "}
                  <PickSummary
                    kind="agents"
                    inline
                    value={strings(o.agents)}
                    options={options.agents}
                    onChange={(agents) => setOptional(i, { agents })}
                  />
                </div>
              </div>
              <Remove
                onClick={() =>
                  set(
                    "optional",
                    optional.filter((_, j) => j !== i),
                  )
                }
              />
            </div>
          ))}
          <div>
            <Add
              label="Add optional skill"
              onClick={() => set("optional", [...optional, { skill: options.skills[0], when: "" }])}
            />
          </div>
        </div>
      </Row>
      <Row form label="Reminders">
        <Reminders value={reminders} options={options} onChange={(v) => set("reminders", asMapping(v))} />
      </Row>
      <Row form label="Rules">
        <TextList value={rules} add="Add rule" onChange={(v) => set("rules", v)} />
      </Row>
    </div>
  );
}

/** A subagent's settings: read, or as a form when *onChange* is given. */
export function SubagentSettings({
  data,
  options,
  onChange,
}: {
  data: Data;
  options?: LibraryOptions;
  onChange?: (data: Data) => void;
}) {
  const set = (key: string, value: unknown) => onChange?.({ ...data, [key]: value });
  if (!onChange || !options) {
    const rows: [string, ReactNode][] = [
      ["Model", text(data.model)],
      ["Effort", text(data.effort)],
      ["Skills", strings(data.skills).join(", ")],
      ["Tools", strings(data.tools).length ? <PickSummary kind="tools" value={strings(data.tools)} /> : undefined],
      ["Instructions", data.prompt ? <Markdown text={text(data.prompt)} /> : undefined],
    ];
    return (
      <div>
        {rows
          .filter(([, v]) => v)
          .map(([k, v]) => (
            <Row key={k} label={k}>
              {v}
            </Row>
          ))}
      </div>
    );
  }
  return (
    <div>
      <Row form label="Description">
        <Lines value={text(data.description)} onChange={(v) => set("description", v)} />
      </Row>
      <Row form label="Model">
        <input
          className={cn(field, "w-full")}
          placeholder="Same as the agent"
          value={text(data.model)}
          onChange={(e) => set("model", e.target.value)}
        />
      </Row>
      <Row form label="Effort">
        <Choice
          value={text(data.effort)}
          options={options.efforts}
          empty="Default"
          onChange={(v) => set("effort", v)}
        />
      </Row>
      <Row form label="Skills">
        <PickSummary
          kind="skills"
          value={strings(data.skills)}
          options={options.skills}
          onChange={(v) => set("skills", v)}
        />
      </Row>
      <Row form label="Tools">
        <PickSummary
          kind="tools"
          value={strings(data.tools)}
          options={options.tools}
          onChange={(v) => set("tools", v)}
        />
      </Row>
      <Row form label="Instructions">
        <Lines value={text(data.prompt)} rows={8} onChange={(v) => set("prompt", v)} />
      </Row>
    </div>
  );
}
