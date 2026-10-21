import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { Go } from "@/lib/route";
import type { AgentLevel, AgentRequest, AgentState } from "@/lib/types";
import { cn } from "@/lib/utils";

// As the Qt status bar's chip says and draws each level.
const LEVELS: Record<AgentLevel, [string, string]> = {
  off: ["Agent: Off", "border-transparent text-muted-foreground"],
  watch: ["Agent: Watching", "border-[color:var(--fibsem-purple-color)] text-foreground"],
  assist: [
    "Agent: Assisting",
    "border-[color:var(--fibsem-purple-color)] bg-[color-mix(in_srgb,var(--fibsem-purple-color)_35%,transparent)] text-foreground",
  ],
  drive: ["Agent: Driving", "border-[color:var(--fibsem-purple-color)] bg-[var(--fibsem-purple-color)] text-white"],
};
const ASKING =
  "border-[color:var(--fibsem-waiting-color)] bg-[color-mix(in_srgb,var(--fibsem-waiting-color)_30%,transparent)] text-foreground";

/** What a request's step is doing, in a word, as the Qt chip says it. */
function doing(tool = "") {
  if (tool.startsWith("milling_") || tool.startsWith("spot_burn")) return "milling";
  if (["stage_", "item_move_", "grid_load", "grid_unload"].some((p) => tool.startsWith(p))) return "moving";
  if (tool.includes("acquire") || tool.startsWith("fm_auto_focus") || tool.startsWith("beam_auto_")) return "imaging";
  return "running";
}

function requestText(request: AgentRequest) {
  if (request.status !== "applying") return "Agent asks";
  const total = request.actions.length;
  const step = Math.min((request.results ?? []).length + 1, total);
  return `Agent: ${total ? doing(request.actions[step - 1].tool) : "running"} ${step}/${total}`;
}

/** The agent's level, or what it asks for; pulsing while it acts on the
 * microscope. Clicking it opens the Agent page. */
export function AgentChip({ go }: { go: Go }) {
  const agent = useQuery({ queryKey: ["app/agent"], queryFn: () => api<AgentState>("/app/agent") });
  const a = agent.data;
  if (!a?.level) return null;
  let [text, look] = LEVELS[a.level];
  let tip = "The agent's level: click to open the Agent page";
  if (a.request) {
    [text, look] = [requestText(a.request), ASKING];
    tip = "The agent is asking: click to see what, and decide";
  } else if (a.suggestion) {
    text = "Agent suggests";
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
      {text}
    </button>
  );
}
