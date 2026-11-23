import * as DialogPrimitive from "@radix-ui/react-dialog";
import { Search } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { cn } from "@/lib/utils";

/** One thing the palette finds or runs. *group* heads its section; *find*
 * entries show only once something is typed, so an empty search lists the
 * commands, panels and pages. */
export interface PaletteEntry {
  id: string;
  group: string;
  label: string;
  detail?: string;
  icon?: ReactNode;
  /** More words it answers to, not shown. */
  words?: string;
  find?: boolean;
  run: () => void;
}

// The most of one section found at once: type more to narrow it. With nothing
// typed, every command, panel and page shows.
const PER_GROUP = 8;

/** ⌘K on a Mac, Ctrl+K elsewhere. */
export const PALETTE_KEY = /Mac|iPhone|iPad/.test(navigator.platform) ? "⌘K" : "Ctrl+K";

/** Cmd/Ctrl+K opens it, wherever the focus is; pressed again, it closes. */
export function usePaletteKey(toggle: () => void) {
  const latest = useRef(toggle);
  latest.current = toggle;
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() !== "k" || !(e.metaKey || e.ctrlKey) || e.altKey || e.shiftKey) return;
      e.preventDefault();
      latest.current();
    };
    addEventListener("keydown", key);
    return () => removeEventListener("keydown", key);
  }, []);
}

/** Every word typed is somewhere in the entry, in any order. */
function matches(entry: PaletteEntry, words: string[]) {
  const text = `${entry.group} ${entry.label} ${entry.detail ?? ""} ${entry.words ?? ""}`.toLowerCase();
  return words.every((w) => text.includes(w));
}

/** The quick search: type to find, arrows to move, Enter to run, Esc to close. */
export function Palette({
  open,
  onOpenChange,
  entries,
  placeholder = "Search commands, the chat, review items, lamellae, pages…",
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  entries: PaletteEntry[];
  placeholder?: string;
}) {
  const [query, setQuery] = useState("");
  const [at, setAt] = useState(0);
  const list = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (open) {
      setQuery("");
      setAt(0);
    }
  }, [open]);
  const shown = useMemo(() => {
    const words = query.toLowerCase().split(/\s+/).filter(Boolean);
    const out: PaletteEntry[] = [];
    const counts = new Map<string, number>();
    for (const e of entries) {
      if (words.length ? !matches(e, words) : e.find) continue;
      const n = counts.get(e.group) ?? 0;
      if (words.length && n >= PER_GROUP) continue;
      counts.set(e.group, n + 1);
      out.push(e);
    }
    // Sections in the order they first appear, each one's entries together.
    const groups = [...new Set(out.map((e) => e.group))];
    return groups.flatMap((g) => out.filter((e) => e.group === g));
  }, [entries, query]);
  const current = Math.min(at, shown.length - 1);
  useEffect(() => {
    list.current?.querySelector(`[data-at="${current}"]`)?.scrollIntoView({ block: "nearest" });
  }, [current]);
  const run = (entry: PaletteEntry | undefined) => {
    if (!entry) return;
    onOpenChange(false);
    // After the dialog gives the focus back, so an entry that focuses something keeps it.
    setTimeout(entry.run, 0);
  };
  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-40 bg-black/40" />
        <DialogPrimitive.Content
          aria-describedby={undefined}
          className="fixed top-[12vh] left-1/2 z-50 flex max-h-[70vh] w-[min(600px,calc(100vw-32px))] -translate-x-1/2 flex-col overflow-hidden rounded-lg border border-input bg-popover shadow-2xl focus:outline-none"
        >
          <DialogPrimitive.Title className="sr-only">Quick search</DialogPrimitive.Title>
          <div className="flex items-center gap-2 border-b border-border px-3.5">
            <Search size={16} className="shrink-0 text-muted-foreground" aria-hidden />
            <input
              autoFocus
              role="combobox"
              aria-expanded
              aria-controls="palette-list"
              aria-activedescendant={shown.length ? `palette-${current}` : undefined}
              aria-label="Search"
              placeholder={placeholder}
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setAt(0);
              }}
              onKeyDown={(e) => {
                if (e.key === "ArrowDown") {
                  e.preventDefault();
                  setAt((n) => (shown.length ? (Math.min(n, shown.length - 1) + 1) % shown.length : 0));
                } else if (e.key === "ArrowUp") {
                  e.preventDefault();
                  setAt((n) => (shown.length ? (Math.min(n, shown.length - 1) - 1 + shown.length) % shown.length : 0));
                } else if (e.key === "Enter") {
                  e.preventDefault();
                  run(shown[current]);
                }
              }}
              className="h-12 min-w-0 flex-1 border-0 bg-transparent font-[inherit] text-[15px] text-foreground outline-none"
            />
            <kbd className="rounded border border-border px-1.5 py-0.5 text-[11px] text-muted-foreground">Esc</kbd>
          </div>
          <div ref={list} id="palette-list" role="listbox" aria-label="Results" className="overflow-y-auto py-1.5">
            {shown.length === 0 && (
              <div className="px-4 py-6 text-center text-[13px] text-muted-foreground">Nothing matches “{query}”</div>
            )}
            {shown.map((e, i) => (
              <div key={e.id}>
                {(i === 0 || shown[i - 1].group !== e.group) && (
                  <div className="px-4 pt-2 pb-1 text-[11px] font-semibold tracking-wide text-muted-foreground uppercase">
                    {e.group}
                  </div>
                )}
                <div
                  id={`palette-${i}`}
                  data-at={i}
                  role="option"
                  aria-selected={i === current}
                  className={cn(
                    "mx-1.5 flex cursor-pointer items-center gap-2.5 rounded px-2.5 py-1.5 text-[13.5px]",
                    i === current && "bg-accent",
                  )}
                  onMouseMove={() => i !== current && setAt(i)}
                  onClick={() => run(e)}
                >
                  <span className="flex w-4 shrink-0 justify-center text-muted-foreground">{e.icon}</span>
                  <span className="shrink-0 text-foreground">{e.label}</span>
                  {e.detail && <span className="min-w-0 truncate text-muted-foreground">{e.detail}</span>}
                </div>
              </div>
            ))}
          </div>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}
