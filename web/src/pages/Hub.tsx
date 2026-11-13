import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { api, getToken } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Agents, StartButton } from "@/pages/hub/Agents";
import { InstrumentView } from "@/pages/hub/InstrumentView";
import { AddButton, Instruments } from "@/pages/hub/Instruments";
import { Overview } from "@/pages/hub/Overview";
import { useHub } from "@/pages/hub/shared";

type Tab = "overview" | "instruments" | "agents";
const TABS: [Tab, string][] = [
  ["overview", "Overview"],
  ["instruments", "Instruments"],
  ["agents", "Agents"],
];

/** The lab's page: the instruments this gateway is the hub for. Overview is
 * their status and opens one; Instruments is the list and its upkeep; Agents
 * is every agent running. The page for this PC alone is the gateway's. */
export function HubPage() {
  const hub = useHub();
  const gateway = useQuery({
    queryKey: ["gateway"],
    queryFn: () => api<{ version: string }>("/gateway"),
    staleTime: 60_000,
  });
  const [tab, setTab] = useState<Tab>("overview");
  // The instrument opened from Overview or Agents, shown under Overview.
  const [opened, setOpened] = useState<string | null>(null);
  const [adding, setAdding] = useState(false);
  const [starting, setStarting] = useState(false);
  useEffect(() => {
    document.title = "Lab Hub";
  }, []);
  const instruments = hub.data?.instruments ?? [];
  const one =
    tab === "overview" ? instruments.find((i) => i.name === opened) : undefined;
  const go = (to: Tab) => {
    setTab(to);
    setOpened(null);
  };
  const open = (name: string) => {
    setTab("overview");
    setOpened(name);
  };
  return (
    <div className="flex h-full flex-col overflow-hidden sm:flex-row">
      <nav className="flex shrink-0 flex-col gap-0.5 border-b border-border px-3 py-3 sm:w-52 sm:border-r sm:border-b-0 sm:py-5">
        <div className="px-3 pb-3">
          <div className="text-[15px] font-semibold">Lab Hub</div>
          <div className="text-xs text-muted-foreground">{location.host}</div>
        </div>
        <div className="flex gap-0.5 overflow-x-auto sm:flex-col">
          {TABS.map(([key, label]) => (
            <button
              key={key}
              onClick={() => go(key)}
              aria-current={tab === key ? "page" : undefined}
              className={cn(
                "cursor-pointer rounded border-0 px-3 py-1.5 text-left text-[13px]",
                tab === key
                  ? "bg-accent text-foreground"
                  : "bg-transparent text-muted-foreground hover:bg-accent hover:text-foreground",
              )}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="mt-auto hidden flex-col gap-1.5 px-3 pt-4 text-xs text-muted-foreground sm:flex">
          <a
            href={`/ui/gateway#token=${getToken() ?? ""}`}
            className="text-muted-foreground no-underline hover:text-foreground"
          >
            This PC's gateway
          </a>
          {gateway.data && <span>fibsemOS {gateway.data.version}</span>}
        </div>
      </nav>
      <main className="min-w-0 flex-1 overflow-auto">
        <div className="flex max-w-[980px] flex-col gap-3 px-4 py-5 sm:px-6">
          <div className="flex min-h-[30px] items-center gap-3">
            {one ? (
              <>
                <h1 className="m-0 flex items-baseline gap-2 text-[17px] font-semibold">
                  <button
                    type="button"
                    onClick={() => setOpened(null)}
                    className="cursor-pointer border-0 bg-transparent p-0 text-[17px] font-normal text-muted-foreground hover:text-foreground"
                  >
                    Overview
                  </button>
                  <span className="text-[15px] font-normal text-muted-foreground/60">
                    /
                  </span>
                  {one.name}
                </h1>
                <span className="text-[12.5px] text-muted-foreground">
                  {[one.ssh ?? one.url, one.gateway?.version]
                    .filter(Boolean)
                    .join(" · ")}
                </span>
              </>
            ) : (
              <h1 className="m-0 text-[17px] font-semibold">
                {TABS.find(([key]) => key === tab)?.[1]}
              </h1>
            )}
            <div className="ml-auto flex gap-2">
              {tab === "instruments" && (
                <AddButton onClick={() => setAdding(true)} />
              )}
              {tab === "agents" && (
                <StartButton
                  instruments={instruments}
                  onClick={() => setStarting(true)}
                />
              )}
            </div>
          </div>
          {hub.error && (
            <p className="m-0 text-[13px] text-destructive">
              {(hub.error as Error).message}
            </p>
          )}
          {hub.data && tab === "overview" && one && (
            <InstrumentView key={one.name} instrument={one} />
          )}
          {hub.data &&
            tab === "overview" &&
            !one &&
            (instruments.length ? (
              <Overview instruments={instruments} open={open} />
            ) : (
              <p className="m-0 text-[13px] text-muted-foreground">
                No instruments yet. Add the first on{" "}
                <button
                  type="button"
                  onClick={() => go("instruments")}
                  className="cursor-pointer border-0 bg-transparent p-0 text-[13px] text-foreground underline"
                >
                  Instruments
                </button>
                .
              </p>
            ))}
          {hub.data && tab === "instruments" && (
            <Instruments
              instruments={instruments}
              adding={adding}
              setAdding={setAdding}
            />
          )}
          {hub.data && tab === "agents" && (
            <Agents
              instruments={instruments}
              open={open}
              starting={starting}
              setStarting={setStarting}
            />
          )}
        </div>
      </main>
    </div>
  );
}
