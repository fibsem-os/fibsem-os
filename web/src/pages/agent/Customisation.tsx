import { useQuery } from "@tanstack/react-query";
import { ListFilter, SlidersHorizontal } from "lucide-react";
import { useMemo, useState, type ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { Facts } from "@/components/ui/card";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { Popup } from "@/components/ui/popup";
import { api } from "@/lib/api";
import type { AgentCustomisation, ProfileRow, ReportedSession, SkillRow, ToolRow } from "@/lib/types";
import { cn } from "@/lib/utils";

const SECTIONS = {
  tools: "Tools",
  skills: "Skills",
  specialists: "Specialists",
  profiles: "Profiles & reminders",
  never: "Never touches",
} as const;
type Section = keyof typeof SECTIONS;

const LEVEL_WORD: Record<ToolRow["level"], string> = { watch: "Reads", assist: "Asks", drive: "Acts" };
const acts = (tool: ToolRow) => tool.level !== "watch";

interface Filter {
  reads: boolean;
  acts: boolean;
  hardware: boolean;
  categories: string[];
}
const FILTER_LABEL = "Filter the tools";
const NO_FILTER: Filter = { reads: false, acts: false, hardware: false, categories: [] };

function shown(tool: ToolRow, filter: Filter, text: string): boolean {
  if (filter.reads !== filter.acts && acts(tool) !== filter.acts) return false;
  if (filter.hardware && !tool.hardware) return false;
  if (filter.categories.length && !filter.categories.includes(tool.category)) return false;
  return !text || tool.name.includes(text) || tool.description.toLowerCase().includes(text);
}

function Check({ on, onChange, label, count }: { on: boolean; onChange: () => void; label: string; count: number }) {
  return (
    <label className="flex cursor-pointer items-center gap-2 py-1 text-[13px]">
      <input type="checkbox" className="m-0 accent-[var(--primary)]" checked={on} onChange={onChange} />
      <span className="flex-1">{label}</span>
      <span className="text-muted-foreground tabular-nums">{count}</span>
    </label>
  );
}

function FilterMenu({
  tools,
  categories,
  filter,
  set,
}: {
  tools: ToolRow[];
  categories: string[];
  filter: Filter;
  set: (f: Filter) => void;
}) {
  const on = Number(filter.reads) + Number(filter.acts) + Number(filter.hardware) + filter.categories.length;
  const flip = (key: "reads" | "acts" | "hardware") => set({ ...filter, [key]: !filter[key] });
  const flipCategory = (name: string) =>
    set({
      ...filter,
      categories: filter.categories.includes(name)
        ? filter.categories.filter((c) => c !== name)
        : [...filter.categories, name],
    });
  return (
    <Popup
      label={FILTER_LABEL}
      align="right"
      side="below"
      className="w-72 px-3.5 py-2.5"
      trigger={(open, toggle) => (
        <Button variant="ghost" aria-expanded={open} onClick={toggle}>
          <ListFilter size={14} />
          {on ? `Filter · ${on}` : "Filter"}
        </Button>
      )}
    >
      {() => (
        <div className="flex flex-col">
          <div className="mb-0.5 text-[12px] font-semibold text-muted-foreground">Level</div>
          <Check
            on={filter.reads}
            onChange={() => flip("reads")}
            label="Reads"
            count={tools.filter((t) => !acts(t)).length}
          />
          <Check on={filter.acts} onChange={() => flip("acts")} label="Acts" count={tools.filter(acts).length} />
          <Check
            on={filter.hardware}
            onChange={() => flip("hardware")}
            label="Moves the microscope"
            count={tools.filter((t) => t.hardware).length}
          />
          <div className="mt-2 mb-0.5 text-[12px] font-semibold text-muted-foreground">Category</div>
          {categories.map((name) => (
            <Check
              key={name}
              on={filter.categories.includes(name)}
              onChange={() => flipCategory(name)}
              label={name}
              count={tools.filter((t) => t.category === name).length}
            />
          ))}
          <button
            className="mt-2 cursor-pointer border-0 border-t border-border bg-transparent px-0 pt-2 text-left text-[12px] text-muted-foreground hover:text-foreground disabled:cursor-default disabled:opacity-50"
            disabled={!on}
            onClick={() => set(NO_FILTER)}
          >
            Clear filters
          </button>
        </div>
      )}
    </Popup>
  );
}

function Tool({ tool }: { tool: ToolRow }) {
  const [open, setOpen] = useState(false);
  return (
    <li
      className="grid cursor-pointer grid-cols-1 gap-x-3 gap-y-0.5 border-b border-border/60 py-1.5 sm:grid-cols-[240px_minmax(0,1fr)_44px_76px]"
      onClick={() => setOpen(!open)}
    >
      <span className="min-w-0 font-mono text-[12px] break-all">{tool.name}</span>
      <span className={cn("text-muted-foreground", !open && "line-clamp-2")}>{tool.description}</span>
      <span className={acts(tool) ? "text-[var(--fibsem-warn-color)]" : "text-muted-foreground"}>
        {LEVEL_WORD[tool.level]}
      </span>
      <span>
        {tool.hardware && (
          <span className="rounded-sm border border-input px-1 text-[11px] text-muted-foreground">microscope</span>
        )}
      </span>
    </li>
  );
}

function Tools({ data }: { data: AgentCustomisation }) {
  const [text, setText] = useState("");
  const [filter, setFilter] = useState<Filter>(NO_FILTER);
  const groups = useMemo(() => {
    const found = new Map<string, ToolRow[]>();
    const words = text.trim().toLowerCase();
    for (const tool of data.tools) {
      if (shown(tool, filter, words)) found.set(tool.group, [...(found.get(tool.group) ?? []), tool]);
    }
    return [...found];
  }, [data.tools, filter, text]);
  return (
    <>
      <p className="m-0 mb-3 text-muted-foreground">
        What the agent can call. Reads work at every level; Acts needs Drive, or your approval at Assist. What it may
        never touch is refused at every level. Click a tool for its whole description.
      </p>
      <div className="mb-1 flex items-center gap-2">
        <input
          type="search"
          aria-label="Search the tools"
          placeholder="Search tools"
          className="min-w-0 flex-1 rounded border border-input bg-background px-2 py-1 text-[13px] text-foreground outline-none focus:border-primary"
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <FilterMenu tools={data.tools} categories={data.categories} filter={filter} set={setFilter} />
      </div>
      {groups.length ? (
        groups.map(([group, tools]) => (
          <section key={group}>
            <h4 className="mt-4 mb-1 text-[12px] font-semibold text-muted-foreground">
              {group} <span className="font-normal">· {tools.length}</span>
            </h4>
            <ul className="m-0 list-none p-0">
              {tools.map((tool) => (
                <Tool key={tool.name} tool={tool} />
              ))}
            </ul>
          </section>
        ))
      ) : (
        <p className="mt-4 text-muted-foreground">No tool matches.</p>
      )}
    </>
  );
}

function Skills({ rows, read, intro }: { rows: SkillRow[]; read?: string[]; intro: string }) {
  return (
    <>
      <p className="m-0 mb-3 text-muted-foreground">{intro}</p>
      <ul className="m-0 flex list-none flex-col gap-3 p-0">
        {rows.map((row) => (
          <li key={row.name}>
            <div className="font-mono text-[12px]">
              {row.name}
              {read?.includes(row.name) && (
                <span className="font-sans text-muted-foreground"> · read this session</span>
              )}
            </div>
            <div className="text-muted-foreground">{row.description || "No description."}</div>
          </li>
        ))}
      </ul>
    </>
  );
}

function Profile({ profile }: { profile: ProfileRow }) {
  const rows: [string, ReactNode][] = [];
  if (profile.model) rows.push(["Model", profile.model]);
  if (profile.effort) rows.push(["Effort", profile.effort]);
  if (profile.max_budget_usd != null) rows.push(["Budget", `$${profile.max_budget_usd}`]);
  if (profile.max_turns != null) rows.push(["Turns", profile.max_turns]);
  rows.push(["Mode", profile.mode]);
  if (profile.skills.length) rows.push(["Extra skills", profile.skills.join(", ")]);
  return (
    <section className="border-b border-border/60 pb-3.5 last:border-b-0">
      <div className="font-semibold">{profile.name}</div>
      <div className="mb-2 font-mono text-[11px] break-all text-muted-foreground">
        {profile.path}
        {profile.hash && ` · ${profile.hash.slice(0, 7)}`}
      </div>
      {profile.error ? (
        <p className="m-0 text-destructive">{profile.error}</p>
      ) : (
        <>
          <Facts className="max-w-sm [&_dd]:text-left" rows={rows} />
          {profile.prompt && <p className="m-0 mt-2 whitespace-pre-wrap">{profile.prompt}</p>}
          {profile.reminders.length ? (
            <>
              <h4 className="mt-3 mb-1 text-[12px] font-semibold text-muted-foreground">Reminders</h4>
              {profile.reminders.map((r, i) => (
                <p key={i} className="m-0 mt-1 break-words">
                  <span className="text-muted-foreground">{r.label}:</span> {r.say}
                </p>
              ))}
            </>
          ) : null}
        </>
      )}
    </section>
  );
}

function Profiles({ data }: { data: AgentCustomisation }) {
  return (
    <>
      <p className="m-0 mb-3 text-muted-foreground">
        The profiles on this PC. A reminder is said to the agent when it makes a matching call:{" "}
        {data.reminder_options.join(", ").toLowerCase()}.
      </p>
      {data.profiles.length ? (
        <div className="flex flex-col gap-3.5">
          {data.profiles.map((p) => (
            <Profile key={p.name} profile={p} />
          ))}
        </div>
      ) : (
        <p className="text-muted-foreground">No profiles on this PC.</p>
      )}
    </>
  );
}

function Never({ data }: { data: AgentCustomisation }) {
  return (
    <>
      <p className="m-0 mb-3 text-muted-foreground">
        Refused at every level, an approved request included. Their values stay readable.
      </p>
      <ul className="m-0 list-none p-0">
        {data.never.map((row) => (
          <li
            key={row.device}
            className="grid grid-cols-1 gap-x-3 border-b border-border/60 py-1.5 sm:grid-cols-[130px_1fr]"
          >
            <span>{row.device}</span>
            <span>
              {row.blocked}
              <span className="block text-muted-foreground">{row.why}</span>
            </span>
          </li>
        ))}
      </ul>
    </>
  );
}

function Body({ section, data }: { section: Section; data: AgentCustomisation }) {
  const session = useQuery({
    queryKey: ["app/agent", "session"],
    queryFn: () => api<{ session: ReportedSession | null }>("/app/agent/session"),
  });
  if (section === "tools") return <Tools data={data} />;
  if (section === "skills")
    return (
      <Skills
        rows={data.skills}
        read={session.data?.session?.skills}
        intro="What the agent reads: fibsem-agent first, then the one for what you want done."
      />
    );
  if (section === "specialists")
    return <Skills rows={data.specialists} intro="The checkers a Run Lead calls, each in a fresh context." />;
  if (section === "profiles") return <Profiles data={data} />;
  return <Never data={data} />;
}

function counts(data: AgentCustomisation | undefined): Partial<Record<Section, number>> {
  if (!data) return {};
  return {
    tools: data.tools.length,
    skills: data.skills.length,
    specialists: data.specialists.length,
    profiles: data.profiles.length,
  };
}

/** The agent row's Customise: what the agent runs with, read-only. */
export function Customise() {
  const [open, setOpen] = useState(false);
  const [section, setSection] = useState<Section>("tools");
  const data = useQuery({
    queryKey: ["app/agent", "customisation"],
    queryFn: () => api<AgentCustomisation>("/app/agent/customisation"),
    enabled: open,
    staleTime: 60000,
  });
  const count = counts(data.data);
  return (
    <>
      <button
        className="flex cursor-pointer items-center gap-1.5 rounded border-0 bg-transparent px-2 py-1 text-[13px] text-foreground hover:bg-accent"
        onClick={() => setOpen(true)}
      >
        <SlidersHorizontal size={14} />
        Customise
      </button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent
          title="Agent customisation"
          className="w-[min(1100px,calc(100vw-32px))]"
          // Escape closes the open filter menu first, not the whole dialog.
          onEscapeKeyDown={(e) => document.querySelector(`[aria-label="${FILTER_LABEL}"]`) && e.preventDefault()}
        >
          <div className="flex min-h-0 flex-col sm:flex-row">
            <nav
              aria-label="Customisation sections"
              className="flex shrink-0 gap-0.5 overflow-x-auto border-b border-border p-2.5 sm:w-[190px] sm:flex-col sm:border-r sm:border-b-0"
            >
              {(Object.keys(SECTIONS) as Section[]).map((key) => (
                <button
                  key={key}
                  onClick={() => setSection(key)}
                  className={cn(
                    "flex shrink-0 cursor-pointer gap-3 rounded border-0 bg-transparent px-2.5 py-2 text-left text-[13px] whitespace-nowrap text-muted-foreground hover:bg-accent",
                    section === key && "bg-accent text-foreground",
                  )}
                >
                  <span className="flex-1">{SECTIONS[key]}</span>
                  {count[key] != null && <span className="text-muted-foreground tabular-nums">{count[key]}</span>}
                </button>
              ))}
            </nav>
            <div className="min-h-0 min-w-0 flex-1 overflow-auto px-5 py-4 text-[13px] leading-[1.45]">
              {data.data ? (
                <Body section={section} data={data.data} />
              ) : (
                <p className="m-0 text-muted-foreground">
                  {data.error ? String(data.error) : "Reading what the agent runs with…"}
                </p>
              )}
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
