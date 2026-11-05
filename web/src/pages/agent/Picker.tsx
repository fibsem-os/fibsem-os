import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { api } from "@/lib/api";
import type { AgentCustomisation, ToolRow } from "@/lib/types";
import { cn } from "@/lib/utils";
import { acts, FILTER_LABEL, type Filter, FilterMenu, LEVEL_WORD, NO_FILTER, shown } from "./Customisation";

export type PickKind = "tools" | "skills" | "agents";

const NOUN: Record<PickKind, [string, string]> = {
  tools: ["tool", "tools"],
  skills: ["skill", "skills"],
  agents: ["subagent", "subagents"],
};
const FIBSEM_TOOL = "mcp__fibsem__";
const bare = (tool: string) => (tool.startsWith(FIBSEM_TOOL) ? tool.slice(FIBSEM_TOOL.length) : tool);

/** "5 tools", which opens the list to see them or, with *onChange*, to choose them. */
export function PickSummary({
  kind,
  value,
  options = [],
  inline = false,
  onChange,
}: {
  kind: PickKind;
  /** Inside a line of its own, not a form row. */
  inline?: boolean;
  value: string[];
  /** Every name there is to choose from. */
  options?: string[];
  onChange?: (value: string[]) => void;
}) {
  const [open, setOpen] = useState(false);
  const [one, many] = NOUN[kind];
  const n = value.length;
  if (!onChange && !n) return null;
  return (
    <>
      <button
        className={cn(
          "cursor-pointer rounded border-0 bg-transparent p-0 text-left text-[13px] text-foreground hover:underline",
          onChange && !inline && "pt-[7px]",
        )}
        onClick={() => setOpen(true)}
      >
        {n ? `${n} ${n === 1 ? one : many}` : `No ${many}`}
        {onChange && <span className="text-muted-foreground"> · Choose</span>}
      </button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent
          title={onChange ? `Choose ${many}` : `${n} ${n === 1 ? one : many}`}
          className="w-[min(900px,calc(100vw-32px))]"
          onEscapeKeyDown={(e) => document.querySelector(`[aria-label="${FILTER_LABEL}"]`) && e.preventDefault()}
        >
          <PickList kind={kind} value={value} options={onChange ? options : value} onChange={onChange} />
          {onChange && (
            <div className="flex justify-end border-t border-border px-4 py-2.5">
              <Button variant="primary" onClick={() => setOpen(false)}>
                Done
              </Button>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}

function PickList({
  kind,
  value,
  options,
  onChange,
}: {
  kind: PickKind;
  value: string[];
  options: string[];
  onChange?: (value: string[]) => void;
}) {
  const [text, setText] = useState("");
  const [filter, setFilter] = useState<Filter>(NO_FILTER);
  // Chosen when it opened: listed first, and kept in place while ticking.
  const [first] = useState(value);
  const tools = useQuery({
    queryKey: ["app/agent", "customisation"],
    queryFn: () => api<AgentCustomisation>("/app/agent/customisation"),
    enabled: kind === "tools",
    staleTime: 60000,
  });
  const library = useQuery({
    queryKey: ["library", kind],
    queryFn: () =>
      api<{ items: { name: string; description: string | null }[] }>(`/app/library/${kind}`).then((a) => a.items),
    enabled: kind !== "tools",
  });
  const rows = useMemo(() => {
    const known = new Map<string, ToolRow>();
    for (const t of tools.data?.tools ?? []) known.set(FIBSEM_TOOL + t.name, t);
    for (const i of library.data ?? [])
      known.set(i.name, {
        name: i.name,
        description: i.description ?? "",
        level: "watch",
        category: "",
        group: "",
        hardware: false,
      });
    return options.map(
      (name): ToolRow =>
        known.get(name) ?? {
          name: bare(name),
          description: "",
          level: "watch",
          category: "",
          group: kind === "tools" ? "Files and other" : "",
          hardware: false,
        },
    );
  }, [tools.data, library.data, options, kind]);
  const full = (row: ToolRow) =>
    kind === "tools" && row.group !== "Files and other" ? FIBSEM_TOOL + row.name : row.name;
  const groups = useMemo(() => {
    const found = new Map<string, ToolRow[]>(first.length ? [["Chosen", []]] : []);
    const words = text.trim().toLowerCase();
    for (const row of rows) {
      if (!shown(row, filter, words)) continue;
      const group = first.includes(full(row)) ? "Chosen" : row.group || "Not chosen";
      found.set(group, [...(found.get(group) ?? []), row]);
    }
    return [...found].filter(([, list]) => list.length);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rows, filter, text, first]);
  const tickAll = (names: string[], on: boolean) =>
    onChange?.(on ? [...value, ...names.filter((n) => !value.includes(n))] : value.filter((v) => !names.includes(v)));
  const flip = (name: string) => onChange?.(value.includes(name) ? value.filter((v) => v !== name) : [...value, name]);
  return (
    <div className="flex min-h-0 flex-col px-5 py-4 text-[13px] leading-[1.45]">
      <div className="mb-1 flex items-center gap-2">
        <input
          type="search"
          aria-label="Search"
          placeholder="Search"
          className="min-w-0 flex-1 rounded border border-input bg-background px-2 py-1 text-[13px] text-foreground outline-none focus:border-primary"
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        {onChange && <span className="shrink-0 px-1 text-muted-foreground">{value.length} chosen</span>}
        {kind === "tools" && tools.data && (
          <FilterMenu tools={tools.data.tools} categories={tools.data.categories} filter={filter} set={setFilter} />
        )}
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto">
        {groups.length ? (
          groups.map(([group, list]) => (
            <section key={group}>
              {group && (
                <h4 className="mt-4 mb-1 text-[12px] font-semibold text-muted-foreground">
                  {onChange && group !== "Chosen" ? (
                    <label className="grid cursor-pointer grid-cols-[20px_1fr] items-baseline gap-x-3">
                      <input
                        type="checkbox"
                        aria-label={`All of ${group}`}
                        title="Tick or untick the whole group"
                        className="m-0 accent-[var(--primary)]"
                        checked={list.every((row) => value.includes(full(row)))}
                        onChange={(e) => tickAll(list.map(full), e.target.checked)}
                      />
                      <span>
                        {group} <span className="font-normal">· {list.length}</span>
                      </span>
                    </label>
                  ) : (
                    <>
                      {group} <span className="font-normal">· {list.length}</span>
                    </>
                  )}
                </h4>
              )}
              <ul className="m-0 list-none p-0">
                {list.map((row) => {
                  const name = full(row);
                  const on = value.includes(name);
                  return (
                    <li key={name} className="border-b border-border/60">
                      <label
                        className={cn(
                          "grid grid-cols-[20px_220px_minmax(0,1fr)_44px] items-baseline gap-x-3 py-1.5",
                          onChange && "cursor-pointer",
                          kind !== "tools" && "grid-cols-[20px_220px_minmax(0,1fr)]",
                        )}
                      >
                        {onChange ? (
                          <input
                            type="checkbox"
                            className="m-0 accent-[var(--primary)]"
                            checked={on}
                            onChange={() => flip(name)}
                          />
                        ) : (
                          <span />
                        )}
                        <span className="min-w-0 font-mono text-[12px] break-all">{row.name}</span>
                        <span className="line-clamp-2 text-muted-foreground">{row.description}</span>
                        {kind === "tools" && (
                          <span className={acts(row) ? "text-[var(--fibsem-warn-color)]" : "text-muted-foreground"}>
                            {row.group === "Files and other" ? "" : LEVEL_WORD[row.level]}
                          </span>
                        )}
                      </label>
                    </li>
                  );
                })}
              </ul>
            </section>
          ))
        ) : (
          <p className="mt-4 text-muted-foreground">Nothing matches.</p>
        )}
      </div>
    </div>
  );
}
