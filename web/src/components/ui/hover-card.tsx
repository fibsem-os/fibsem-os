import * as HoverCardPrimitive from "@radix-ui/react-hover-card";
import { useState, type ReactNode } from "react";

/** A card on hover or focus; a tap opens and closes it too, for a touch screen. */
export function HoverCard({
  trigger,
  children,
  align = "end",
}: {
  trigger: ReactNode;
  children: ReactNode;
  align?: "start" | "end";
}) {
  const [open, setOpen] = useState(false);
  return (
    <HoverCardPrimitive.Root open={open} onOpenChange={setOpen} openDelay={150} closeDelay={100}>
      <HoverCardPrimitive.Trigger asChild onClick={() => setOpen(true)}>
        {trigger}
      </HoverCardPrimitive.Trigger>
      <HoverCardPrimitive.Portal>
        <HoverCardPrimitive.Content
          align={align}
          sideOffset={8}
          className="z-50 w-80 rounded-md border border-input bg-popover px-3.5 py-3 shadow-xl"
        >
          {children}
        </HoverCardPrimitive.Content>
      </HoverCardPrimitive.Portal>
    </HoverCardPrimitive.Root>
  );
}
