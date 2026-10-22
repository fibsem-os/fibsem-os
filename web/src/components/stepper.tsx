import { ChevronLeft, ChevronRight } from "lucide-react";
import type { ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

/** ‹ › and a slider through *count* steps of a stack: a position's slices, or a
 * fluorescence stack's planes. *between* sits between the arrows (Play), *after*
 * after the slider (Follow). */
export function Stepper({
  at,
  count,
  noun,
  go,
  between,
  after,
  compact,
}: {
  at: number;
  count: number;
  noun: string;
  go: (index: number) => void;
  between?: ReactNode;
  after?: ReactNode;
  compact?: boolean;
}) {
  const size = compact ? "size-7" : "size-11";
  return (
    <div className="flex items-center gap-2">
      <Button className={cn(size, "shrink-0 p-0")} aria-label={`Previous ${noun}`} onClick={() => go(at - 1)} disabled={at <= 0}>
        <ChevronLeft size={compact ? 14 : 16} className="mx-auto" />
      </Button>
      {between}
      <Button className={cn(size, "shrink-0 p-0")} aria-label={`Next ${noun}`} onClick={() => go(at + 1)} disabled={at >= count - 1}>
        <ChevronRight size={compact ? 14 : 16} className="mx-auto" />
      </Button>
      <input
        type="range"
        aria-label={noun[0].toUpperCase() + noun.slice(1)}
        className={cn("mx-1.5 min-w-20 flex-1 accent-[var(--primary)]", compact ? "h-7" : "h-11")}
        min={0}
        max={count - 1}
        value={at}
        onChange={(e) => go(Number(e.target.value))}
      />
      {after}
    </div>
  );
}
