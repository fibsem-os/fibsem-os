import { useEffect, useRef, useState, type ReactNode } from "react";
import { cn } from "@/lib/utils";

/** A button with a panel above it; a click elsewhere or Escape closes it. */
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
  const ref = useRef<HTMLDivElement>(null);
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
      {trigger(open, () => setOpen(!open))}
      {open && (
        <div
          role="dialog"
          aria-label={label}
          className={cn(
            "absolute z-30 rounded-md border border-input bg-popover p-1 shadow-[0_6px_20px_rgba(0,0,0,0.45)]",
            align === "left" ? "left-0" : "right-0",
            side === "above" ? "bottom-[calc(100%+8px)]" : "top-[calc(100%+8px)]",
            className,
          )}
        >
          {children(() => setOpen(false))}
        </div>
      )}
    </div>
  );
}
