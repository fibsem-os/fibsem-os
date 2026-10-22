import { useQuery } from "@tanstack/react-query";
import { FolderOpen, Grid3x3, Layers, MessageSquare, Microscope, Settings as Gear } from "lucide-react";
import type { ReactNode } from "react";
import { api } from "@/lib/api";
import { useVolume } from "@/lib/experiment";
import type { Page } from "@/lib/route";
import type { AgentState } from "@/lib/types";
import { cn } from "@/lib/utils";

const tab =
  "relative flex min-w-0 flex-1 cursor-pointer flex-col items-center justify-center gap-1 rounded-md border-0 bg-transparent px-0.5 py-2.5 text-[10.5px] text-muted-foreground min-[400px]:px-1 min-[400px]:text-xs no-underline hover:bg-accent hover:text-foreground md:flex-none";

function Tab({
  on,
  icon,
  label,
  ...props
}: { on?: boolean; icon: ReactNode; label: ReactNode } & (
  | { href: string; onClick?: undefined }
  | { onClick: () => void; href?: undefined }
)) {
  const className = cn(tab, on && "bg-accent text-foreground");
  const body = (
    <>
      {icon}
      <span className="flex items-center gap-1">{label}</span>
    </>
  );
  return props.href ? (
    <a className={className} href={props.href}>
      {body}
    </a>
  ) : (
    <button className={className} onClick={props.onClick} aria-current={on ? "page" : undefined}>
      {body}
    </button>
  );
}

/** The pages down the left; along the bottom at phone width. */
export function Rail({ page, go, openSettings }: { page: Page; go: (page: Page) => void; openSettings: () => void }) {
  const agent = useQuery({
    queryKey: ["app/agent"],
    queryFn: () => api<AgentState>("/app/agent"),
  });
  const waiting = agent.data?.request?.status === "pending" || agent.data?.suggestion || agent.data?.asking ? 1 : 0;
  const volume = useVolume();
  return (
    <nav
      aria-label="Pages"
      className="order-last flex shrink-0 flex-row gap-1 border-t border-border bg-card p-1 md:order-first md:w-[92px] md:flex-col md:border-t-0 md:border-r md:px-2 md:py-3"
    >
      <Tab
        on={page === "microscope"}
        onClick={() => go("microscope")}
        icon={<Microscope size={22} />}
        label="Microscope"
      />
      <Tab
        on={page === "experiment"}
        onClick={() => go("experiment")}
        icon={<FolderOpen size={22} />}
        label="Experiment"
      />
      {volume ? (
        <Tab on={page === "positions"} onClick={() => go("positions")} icon={<Layers size={22} />} label="Positions" />
      ) : (
        <>
          <Tab on={page === "lamellae"} onClick={() => go("lamellae")} icon={<Layers size={22} />} label="Lamellae" />
          <Tab on={page === "grids"} onClick={() => go("grids")} icon={<Grid3x3 size={22} />} label="Grids" />
        </>
      )}
      <Tab
        on={page === "agent"}
        onClick={() => go("agent")}
        icon={<MessageSquare size={22} />}
        label={
          <>
            Agent
            {waiting > 0 && (
              <span
                className="rounded-full bg-[var(--fibsem-purple-color)] px-1.5 text-[11px] leading-4 text-white"
                aria-label={`${waiting} request waiting`}
              >
                {waiting}
              </span>
            )}
          </>
        }
      />
      <div className="hidden md:block md:flex-1" />
      <Tab onClick={openSettings} icon={<Gear size={22} />} label="Settings" />
    </nav>
  );
}
