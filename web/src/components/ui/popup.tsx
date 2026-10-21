import { useEffect, useRef, useState, type ReactNode } from "react";
import { cn } from "@/lib/utils";

/** A button with a panel above or below it; a click elsewhere or Escape closes it.
 * The panel opens on the side with room and scrolls within the window. */
export function Popup({
  trigger,
  children,
  label,
  align = "left",
  side = "above",
  className,
}: {
  trigger: (open: boolean, toggle: () => void) => ReactNode;
  children: (close: () => void) => ReactNode;
  label: string;
  align?: "left" | "right";
  side?: "above" | "below";
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  const [fit, setFit] = useState<{ side: "above" | "below"; maxHeight: number }>({ side, maxHeight: 0 });
  const ref = useRef<HTMLDivElement>(null);
  const toggle = () => {
    const box = ref.current?.getBoundingClientRect();
    if (!open && box) {
      const room = { above: box.top - 16, below: innerHeight - box.bottom - 16 };
      const other = side === "above" ? "below" : "above";
      const chosen = room[side] >= 240 || room[side] >= room[other] ? side : other;
      setFit({ side: chosen, maxHeight: room[chosen] });
    }
    setOpen(!open);
  };
  useEffect(() => {
    if (!open) return;
    const click = (e: MouseEvent) => ref.current?.contains(e.target as Node) || setOpen(false);
    const key = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    addEventListener("mousedown", click);
    addEventListener("keydown", key);
    return () => {
      removeEventListener("mousedown", click);
      removeEventListener("keydown", key);
    };
  }, [open]);
  return (
    <div ref={ref} className="relative flex">
      {trigger(open, toggle)}
      {open && (
        <div
          role="dialog"
          aria-label={label}
          className={cn(
            "absolute z-30 rounded-md border border-input bg-popover p-1 shadow-[0_6px_20px_rgba(0,0,0,0.45)]",
            align === "left" ? "left-0" : "right-0",
            fit.side === "above" ? "bottom-[calc(100%+8px)]" : "top-[calc(100%+8px)]",
            "overflow-auto",
            className,
          )}
          style={{ maxHeight: fit.maxHeight }}
        >
          {children(() => setOpen(false))}
        </div>
      )}
    </div>
  );
}
