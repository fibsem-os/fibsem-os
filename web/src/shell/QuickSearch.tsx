import { useQuery } from "@tanstack/react-query";
import { Compass, Layers, Settings as Gear } from "lucide-react";
import { useState } from "react";
import { api } from "@/lib/api";
import type { Go, Page } from "@/lib/route";
import type { ItemSummary } from "@/lib/types";
import { Palette, type PaletteEntry, usePaletteKey } from "@/pages/agent/Palette";

const NAMES: [Page, string][] = [
  ["microscope", "Microscope"],
  ["experiment", "Experiment"],
  ["lamellae", "Lamellae"],
  ["grids", "Grids"],
  ["positions", "Positions"],
  ["agent", "Agent"],
  ["events", "Events"],
];

/** Cmd/Ctrl+K on the pages other than the Agent page, which has its own: the
 * app's pages, Settings, and the open experiment's items. */
export function QuickSearch({
  page,
  go,
  agent,
  volume,
  openSettings,
}: {
  page: Page;
  go: Go;
  agent: boolean;
  volume: boolean | undefined;
  openSettings: () => void;
}) {
  const [open, setOpen] = useState(false);
  usePaletteKey(() => setOpen((on) => !on));
  const items = useQuery({
    queryKey: ["app/experiment"],
    queryFn: () => api<{ items: ItemSummary[] }>("/app/experiment"),
    enabled: open,
    retry: false,
  });
  // Volume's items are positions; the others' are lamellae.
  const itemPage: Page = volume ? "positions" : "lamellae";
  const entries: PaletteEntry[] = [];
  for (const [key, name] of NAMES) {
    if (key === page || (key === "agent" && !agent)) continue;
    if ((key === "positions" && !volume) || (key === "lamellae" && volume)) continue;
    entries.push({
      id: `page-${key}`,
      group: "Go to",
      label: name,
      detail: "Page",
      icon: <Compass size={14} />,
      run: () => go(key),
    });
  }
  entries.push({ id: "settings", group: "Go to", label: "Settings", icon: <Gear size={14} />, run: openSettings });
  for (const item of items.data?.items ?? [])
    entries.push({
      id: `item-${item.item_name}`,
      group: volume ? "Positions" : "Lamellae",
      label: item.item_name,
      detail: item.last_completed_task ? `Last: ${item.last_completed_task}` : undefined,
      words: "lamella item position",
      icon: <Layers size={14} />,
      find: true,
      run: () => go(itemPage, item.item_name),
    });
  return (
    <Palette
      open={open}
      onOpenChange={setOpen}
      entries={entries}
      placeholder={volume ? "Search pages and positions…" : "Search pages and lamellae…"}
    />
  );
}
