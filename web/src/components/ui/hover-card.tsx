import * as HoverCardPrimitive from "@radix-ui/react-hover-card";
import type { ReactNode } from "react";

export function HoverCard({ trigger, children }: { trigger: ReactNode; children: ReactNode }) {
  return (
    <HoverCardPrimitive.Root openDelay={150} closeDelay={100}>
      <HoverCardPrimitive.Trigger asChild>{trigger}</HoverCardPrimitive.Trigger>
      <HoverCardPrimitive.Portal>
        <HoverCardPrimitive.Content
          align="end"
          sideOffset={8}
          className="z-50 w-80 rounded-md border border-input bg-popover px-3.5 py-3 shadow-xl"
        >
          {children}
        </HoverCardPrimitive.Content>
      </HoverCardPrimitive.Portal>
    </HoverCardPrimitive.Root>
  );
}
