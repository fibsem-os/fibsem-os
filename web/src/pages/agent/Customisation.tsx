import { useQuery } from "@tanstack/react-query";
import { ListFilter } from "lucide-react";
import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Popup } from "@/components/ui/popup";
import { api } from "@/lib/api";
import type { AgentCustomisation, ToolRow } from "@/lib/types";
import { cn } from "@/lib/utils";

export const LEVEL_WORD: Record<ToolRow["level"], string> = { watch: "Reads", assist: "Asks", drive: "Acts" };
export const acts = (tool: ToolRow) => tool.level !== "watch";

export interface Filter {
  reads: boolean;
  acts: boolean;
  hardware: boolean;
  categories: string[];
}
export const FILTER_LABEL = "Filter the tools";
export const NO_FILTER: Filter = { reads: false, acts: false, hardware: false, categories: [] };

export function shown(tool: ToolRow, filter: Filter, text: string): boolean {
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

export function FilterMenu({
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
      <span className={acts(tool) ? "text-foreground" : "text-muted-foreground"}>{LEVEL_WORD[tool.level]}</span>
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

/** What the agent can call and what it may never touch, read-only: two tabs of the
 * knowledge window. */
export function AgentRules({ section, enabled }: { section: "tools" | "never"; enabled: boolean }) {
  const data = useQuery({
    queryKey: ["app/agent", "customisation"],
    queryFn: () => api<AgentCustomisation>("/app/agent/customisation"),
    enabled,
    staleTime: 60000,
  });
  return (
    <div className="h-[min(560px,76vh)] min-h-0 overflow-auto px-5 py-4 text-[13px] leading-[1.45]">
      {data.data ? (
        section === "tools" ? (
          <Tools data={data.data} />
        ) : (
          <Never data={data.data} />
        )
      ) : (
        <p className="m-0 text-muted-foreground">{data.error ? String(data.error) : "Reading…"}</p>
      )}
    </div>
  );
}
