import { ChevronRight } from "lucide-react";
import type { ReactNode } from "react";
import * as fmt from "@/lib/format";
import {
  agentMeta,
  Dot,
  health,
  Hint,
  type Instrument,
  leaf,
} from "@/pages/hub/shared";

function Row({
  label,
  dot,
  children,
}: {
  label: string;
  dot: "ok" | "warn" | "off" | "none";
  children: ReactNode;
}) {
  return (
    <>
      <span className="text-[12.5px] leading-[18px] text-muted-foreground">
        {label}
      </span>
      <span className="flex min-w-0 items-start gap-2 leading-[18px]">
        <Dot colour={dot} className="mt-[5px]" />
        <span className="min-w-0">{children}</span>
      </span>
    </>
  );
}

const NONE = <span className="text-muted-foreground">None</span>;

/** Why nothing is read from it, where its rows would be. */
function Unread({ instrument: i }: { instrument: Instrument }) {
  const head =
    i.reachable === null
      ? "Checking"
      : !i.reachable
        ? `Not reachable since ${fmt.clock(i.since)}`
        : !i.paired
          ? "Not paired yet"
          : "Not read";
  return (
    <div className="flex flex-1 flex-col gap-1 border-t border-border px-4 py-3.5">
      <span className="flex items-center gap-2">
        {head}
        {i.why && (i.paired || !i.reachable) && <Hint>{i.why}</Hint>}
      </span>
      {i.reachable === false && (
        <span className="text-muted-foreground">
          Runs on its own PC carry on. Nothing new starts from here.
        </span>
      )}
    </div>
  );
}

function InstrumentCard({
  instrument: i,
  open,
}: {
  instrument: Instrument;
  open: () => void;
}) {
  const agent = i.session ?? i.agent;
  const running = i.gateway?.mode === "running";
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={open}
      onKeyDown={(event) => event.key === "Enter" && open()}
      className="flex min-h-40 cursor-pointer flex-col rounded-md border border-border bg-card text-[13px] hover:border-input hover:bg-accent/40 focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
    >
      <div className="flex items-center gap-2.5 px-4 pt-3.5 pb-3">
        <Dot colour={health(i)} />
        <b className="text-[15px] font-semibold">{i.name}</b>
        <ChevronRight size={16} className="ml-auto text-muted-foreground/60" />
      </div>
      {i.reachable && i.gateway ? (
        <div className="grid grid-cols-[86px_minmax(0,1fr)] items-start gap-y-[9px] border-t border-border px-4 pt-2.5 pb-3.5">
          <Row
            label="Microscope"
            dot={i.microscope?.connected ? "ok" : running ? "off" : "none"}
          >
            {i.microscope?.connected ? (
              "Connected"
            ) : (
              <span className="text-muted-foreground">
                {i.microscope?.connecting ? "Connecting" : "Not connected"}
              </span>
            )}
          </Row>
          <Row label="App" dot={running ? "ok" : "none"}>
            {running ? (
              <>
                {i.gateway.app ?? "Running"}
                {i.gateway.experiment && (
                  <span className="text-muted-foreground">
                    {" "}
                    · {leaf(i.gateway.experiment)}
                  </span>
                )}
              </>
            ) : (
              NONE
            )}
          </Row>
          <Row label="Agent" dot={agent ? "ok" : "none"}>
            {agent ? (
              <>
                <span className="block truncate">
                  {agent.goal || agent.playbook || "Running"}
                </span>
                <span className="text-xs text-muted-foreground">
                  {agentMeta(i)}
                </span>
              </>
            ) : (
              NONE
            )}
          </Row>
        </div>
      ) : (
        <Unread instrument={i} />
      )}
    </div>
  );
}

/** Each instrument's status on a card: nothing to press but the card. */
export function Overview({
  instruments,
  open,
}: {
  instruments: Instrument[];
  open: (name: string) => void;
}) {
  return (
    <div className="grid gap-3 md:grid-cols-2">
      {instruments.map((i) => (
        <InstrumentCard key={i.name} instrument={i} open={() => open(i.name)} />
      ))}
    </div>
  );
}
