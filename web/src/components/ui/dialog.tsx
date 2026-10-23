import * as DialogPrimitive from "@radix-ui/react-dialog";
import { X } from "lucide-react";
import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

export const Dialog = DialogPrimitive.Root;

export function DialogContent({
  title,
  className,
  actions,
  onEscapeKeyDown,
  children,
}: {
  title: string;
  className?: string;
  /** Buttons in the header, before Close. */
  actions?: ReactNode;
  onEscapeKeyDown?: (e: KeyboardEvent) => void;
  children: ReactNode;
}) {
  return (
    <DialogPrimitive.Portal>
      <DialogPrimitive.Overlay className="fixed inset-0 z-40 bg-black/55" />
      <DialogPrimitive.Content
        aria-describedby={undefined}
        onEscapeKeyDown={onEscapeKeyDown}
        className={cn(
          "fixed top-[8vh] left-1/2 z-50 flex max-h-[84vh] w-[min(880px,calc(100vw-32px))] -translate-x-1/2 flex-col rounded-lg border border-input bg-card shadow-2xl focus:outline-none",
          className,
        )}
      >
        <div className="flex items-center border-b border-border px-4 py-3">
          <DialogPrimitive.Title className="m-0 text-sm font-semibold">{title}</DialogPrimitive.Title>
          {actions && <div className="ml-auto flex items-center gap-2">{actions}</div>}
          <DialogPrimitive.Close
            className={cn(
              "cursor-pointer rounded p-1 text-muted-foreground hover:bg-accent hover:text-foreground",
              actions ? "ml-2" : "ml-auto",
            )}
            aria-label="Close"
          >
            <X size={16} />
          </DialogPrimitive.Close>
        </div>
        {children}
      </DialogPrimitive.Content>
    </DialogPrimitive.Portal>
  );
}
