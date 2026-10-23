import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { Go } from "@/lib/route";
import type { AgentLevel, AgentState } from "@/lib/types";
import { cn } from "@/lib/utils";

// How the chip draws each level; its words come from the server (`chip`).
const LOOKS: Record<AgentLevel, string> = {
  off: "border-transparent text-muted-foreground",
  watch: "border-[color:var(--fibsem-purple-color)] text-foreground",
  assist:
    "border-[color:var(--fibsem-purple-color)] bg-[color-mix(in_srgb,var(--fibsem-purple-color)_35%,transparent)] text-foreground",
  drive: "border-[color:var(--fibsem-purple-color)] bg-[var(--fibsem-purple-color)] text-white",
};
const ASKING =
  "border-[color:var(--fibsem-waiting-color)] bg-[color-mix(in_srgb,var(--fibsem-waiting-color)_30%,transparent)] text-foreground";

/** The agent's level, or what it asks for; pulsing while it acts on the
 * microscope. Clicking it opens the Agent page. */
export function AgentChip({ go }: { go: Go }) {
  const agent = useQuery({ queryKey: ["app/agent"], queryFn: () => api<AgentState>("/app/agent") });
  const a = agent.data;
  if (!a?.level) return null;
  let look = LOOKS[a.level];
  let tip = "The agent's level: click to open the Agent page";
  if (a.request) {
    look = ASKING;
    tip = "The agent is asking: click to see what, and decide";
  } else if (a.suggestion) {
    tip = "The agent suggests a decision on the proposal: click to see it";
  }
  const acting = !a.request && !!a.now;
  return (
    <button
      onClick={() => go("agent")}
      title={a.now ? `${tip}\nNow: ${a.now}` : tip}
      className={cn(
        "shrink-0 cursor-pointer rounded-[3px] border px-2.5 py-0.5 text-[12px] font-semibold whitespace-nowrap",
        look,
        acting && "animate-[agent-pulse_1.6s_ease-in-out_infinite]",
      )}
    >
      {a.chip}
    </button>
  );
}
