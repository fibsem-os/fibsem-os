import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Card } from "@/components/ui/card";
import { api, ApiError, imageUrl } from "@/lib/api";
import type { HistoryFilter, ItemSummary } from "@/lib/types";
import { cn } from "@/lib/utils";
import { History, NO_FILTER } from "./lamellae/History";
import { ItemCard, ItemListPage, type TaskRun, useItemList } from "./lamellae/ItemList";
import { Scrubber, useStack } from "./positions/Scrubber";

const itemPath = (name: string) => `/app/experiment/items/${encodeURIComponent(name)}`;

function slicesLine(item: ItemSummary) {
  const cut = item.slices_cut ?? 0;
  return item.iterations ? `${cut} of ${item.iterations} slices` : `${cut} ${cut === 1 ? "slice" : "slices"}`;
}

function PositionCard(props: {
  item: ItemSummary;
  running: string | null;
  last: TaskRun | undefined;
  picked: boolean;
  onPick: () => void;
}) {
  const name = props.item.item_name;
  const newest = useStack(name, "slice").data?.slices.slice(-1)[0];
  const image = useQuery({
    queryKey: ["slice image", name, newest?.filename, 640],
    queryFn: () => imageUrl(`${itemPath(name)}/outputs/${encodeURIComponent(newest!.filename)}?width=640`),
    enabled: !!newest,
    staleTime: Infinity,
    // The last slice stays up while the next one loads.
    placeholderData: (previous, last) => (last?.queryKey[1] === name ? previous : undefined),
  });
  return (
    <ItemCard
      {...props}
      picture={
        image.data ? (
          <img className="block size-full object-contain" alt="" src={image.data} />
        ) : (
          <div className="flex size-full items-center justify-center text-[12px] text-muted-foreground">
            {newest ? "Loading slice…" : "No slices yet"}
          </div>
        )
      }
      badge={<span className="shrink-0 text-[12px] text-muted-foreground tabular-nums">{slicesLine(props.item)}</span>}
    />
  );
}

const TABS = [
  ["slices", "Slices"],
  ["history", "History"],
] as const;

/** Whether this app serves an item's History; until it does, the tab stays hidden. */
function useHistoryServed(name: string | null) {
  return useQuery({
    queryKey: ["history served"],
    queryFn: () =>
      api(`${itemPath(name!)}/history`).then(
        () => true,
        (e) => !(e instanceof ApiError && e.status === 404),
      ),
    enabled: !!name,
    staleTime: Infinity,
  }).data;
}

export function PositionsPage() {
  const { closed, loaded, list, lastRun, name, setPicked, runningOn } = useItemList();
  const [tab, setTab] = useState<"slices" | "history">("slices");
  const [filter, setFilter] = useState<HistoryFilter>(NO_FILTER);
  const historyServed = useHistoryServed(name);
  if (closed) return <div className="p-5 text-muted-foreground">No experiment is open.</div>;
  const item = list.find((i) => i.item_name === name);
  const shown = historyServed ? tab : "slices";
  return (
    <ItemListPage
      empty={loaded && !list.length && <div className="text-muted-foreground">No positions yet.</div>}
      cards={list.map((i) => (
        <PositionCard
          key={i.item_id}
          item={i}
          running={runningOn(i.item_name)}
          last={lastRun.get(i.item_name)}
          picked={i.item_name === name}
          onPick={() => setPicked(i.item_name)}
        />
      ))}
      detail={
        name && (
          <Card className="flex min-w-0 flex-col gap-3">
            <div className="flex flex-wrap items-center gap-3">
              <span className="min-w-0 truncate text-[15px] font-semibold">{name}</span>
              {historyServed && (
                <div role="tablist" aria-label="Position" className="flex gap-1">
                  {TABS.map(([key, label]) => (
                    <button
                      key={key}
                      role="tab"
                      aria-selected={shown === key}
                      onClick={() => setTab(key)}
                      className={cn(
                        "cursor-pointer border-0 bg-transparent px-2.5 py-1 text-[13px]",
                        shown === key
                          ? "text-foreground shadow-[inset_0_-2px_0_var(--primary)]"
                          : "text-muted-foreground hover:text-foreground",
                      )}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              )}
            </div>
            {shown === "slices" ? (
              <Scrubber key={name} name={name} planned={item?.iterations} running={!!runningOn(name)} />
            ) : (
              <History name={name} filter={filter} setFilter={setFilter} />
            )}
          </Card>
        )
      }
    />
  );
}
