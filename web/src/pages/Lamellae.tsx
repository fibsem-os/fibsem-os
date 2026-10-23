import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Card } from "@/components/ui/card";
import { api } from "@/lib/api";
import { useFileRead } from "@/lib/files";
import type { HistoryFilter, ItemDetail, ItemSummary } from "@/lib/types";
import { cn } from "@/lib/utils";
import { History, NO_FILTER } from "./lamellae/History";
import { ItemCard, ItemListPage, type TaskRun, useItemList } from "./lamellae/ItemList";

// The defect verdict's mark, as the Qt card draws it.
const VERDICT: Record<string, [string, string]> = {
  GOOD: ["Good", "bg-[var(--fibsem-ok-color)]"],
  REWORK: ["Rework", "bg-[var(--fibsem-defect-orange-color)]"],
  FAILED: ["Failed", "bg-[var(--fibsem-defect-red-color)]"],
};

function LamellaCard(props: {
  item: ItemSummary;
  running: string | null;
  last: TaskRun | undefined;
  picked: boolean;
  onPick: () => void;
}) {
  const name = props.item.item_name;
  const thumb = useFileRead(`${name}/thumbnail.png`, 640);
  const detail = useQuery({
    queryKey: ["app/experiment/items", name],
    queryFn: () => api<ItemDetail>(`/app/experiment/items/${encodeURIComponent(name)}`),
  });
  const p = thumb.data?.preview;
  const defect = detail.data?.defect;
  const verdict = defect ? VERDICT[defect.verdict] : undefined;
  return (
    <ItemCard
      {...props}
      picture={
        <>
          {p && <img className="block size-full object-contain" alt="" src={`data:image/jpeg;base64,${p.image_b64_jpeg}`} />}
          {thumb.isError && <div className="flex size-full items-center justify-center text-[12px] text-muted-foreground">No thumbnail yet</div>}
        </>
      }
      badge={
        verdict && (
          <span className="flex items-center gap-1.5 text-[12px] text-muted-foreground" title={defect?.reason || undefined}>
            <span className={cn("inline-block size-2.5 rounded-full", verdict[1])} />
            {verdict[0]}
          </span>
        )
      }
    >
      {verdict && defect?.reason && <div className="text-[12px] text-muted-foreground">{defect.reason}</div>}
    </ItemCard>
  );
}

export function LamellaePage() {
  const { closed, loaded, list, lastRun, name, setPicked, runningOn } = useItemList();
  const [filter, setFilter] = useState<HistoryFilter>(NO_FILTER);
  if (closed) return <div className="p-5 text-muted-foreground">No experiment is open.</div>;
  return (
    <ItemListPage
      empty={loaded && !list.length && <div className="text-muted-foreground">No lamellae yet.</div>}
      cards={list.map((item) => (
        <LamellaCard
          key={item.item_id}
          item={item}
          running={runningOn(item.item_name)}
          last={lastRun.get(item.item_name)}
          picked={item.item_name === name}
          onPick={() => setPicked(item.item_name)}
        />
      ))}
      detail={
        name && (
          <Card>
            <History name={name} filter={filter} setFilter={setFilter} />
          </Card>
        )
      }
    />
  );
}
