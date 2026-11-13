import { useEffect } from "react";
import { getToken } from "@/lib/api";
import { type HubState, Instruments, useHub } from "@/pages/gateway/Instruments";

/** What the hub looks after, in a line. */
function summary(state: HubState): string {
  const total = state.instruments.length;
  if (!total) return "";
  const down = state.instruments.filter((i) => i.reachable === false).length;
  const agents = state.instruments.filter((i) => i.session || i.agent).length;
  return [
    `${total} instrument${total === 1 ? "" : "s"}`,
    agents && `${agents} agent${agents === 1 ? "" : "s"} running`,
    down && `${down} not reachable`,
  ]
    .filter(Boolean)
    .join(" · ");
}

/** The lab hub's own page: the instruments this PC looks after. The same
 * program as the gateway serves it; this PC's own page is ``/ui/gateway``. */
export function HubPage() {
  useEffect(() => {
    document.title = "Lab Hub";
  }, []);
  const hub = useHub();
  return (
    <div className="h-full overflow-auto">
      <div className="mx-auto flex max-w-[960px] flex-col gap-3 px-4 py-5 sm:px-6">
        <div className="flex items-baseline gap-3">
          <h1 className="m-0 text-[17px] font-semibold">Lab Hub</h1>
          <span className="text-[13px] text-muted-foreground">
            {hub.data ? summary(hub.data) : ""}
          </span>
          <a
            href={`/ui/gateway#token=${getToken() ?? ""}`}
            className="ml-auto text-[13px] text-muted-foreground hover:text-foreground"
          >
            This PC's gateway
          </a>
        </div>
        {hub.error && (
          <p className="m-0 text-[13px] text-destructive">
            {(hub.error as Error).message}
          </p>
        )}
        <Instruments />
      </div>
    </div>
  );
}
