import { useEffect, useState } from "react";
import { AgentPage } from "@/pages/Agent";
import { ExperimentPage } from "@/pages/Experiment";
import { GridsPage } from "@/pages/Grids";
import { LamellaePage } from "@/pages/Lamellae";
import { MicroscopePage } from "@/pages/Microscope";
import { PositionsPage } from "@/pages/Positions";
import { Pair } from "@/shell/Pair";
import { Rail } from "@/shell/Rail";
import { Settings } from "@/shell/Settings";
import { TopLine } from "@/shell/TopLine";
import { forgotten, getToken, loadToken, onTokenChange } from "@/lib/api";
import { useEventLoop } from "@/lib/events";
import { useVolume } from "@/lib/experiment";
import { usePage, useSettingsPath } from "@/lib/route";

function Shell() {
  const [asked, go] = usePage();
  // Volume's items are positions: a link to lamellae or grids opens them there.
  const volume = useVolume();
  const page =
    volume === true && (asked === "lamellae" || asked === "grids")
      ? "positions"
      : volume === false && asked === "positions"
        ? "lamellae"
        : asked;
  const [settings, setSettings] = useSettingsPath(page);
  useEventLoop();
  return (
    <div className="flex h-full flex-col md:flex-row">
      <Rail page={page} go={go} openSettings={() => setSettings(true)} />
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <TopLine go={go} />
        <div className="min-h-0 flex-1 overflow-auto">
          {page === "microscope" && <MicroscopePage />}
          {page === "experiment" && <ExperimentPage />}
          {page === "lamellae" && <LamellaePage />}
          {page === "grids" && <GridsPage go={go} />}
          {page === "positions" && <PositionsPage />}
          {page === "agent" && <AgentPage />}
        </div>
      </div>
      <Settings open={settings} onOpenChange={setSettings} />
    </div>
  );
}

export function App() {
  const [token, setToken] = useState(loadToken);
  const [wasPaired] = useState(() => token !== null);
  useEffect(() => onTokenChange(() => setToken(getToken())), []);
  if (!token)
    return (
      <Pair
        note={
          forgotten === "unpaired"
            ? "This browser was unpaired. Pair it again to use it."
            : wasPaired
              ? "This browser's token was refused. It may have been unpaired: pair it again."
              : undefined
        }
      />
    );
  // Keyed on the token, so a new pairing starts with nothing cached.
  return <Shell key={token} />;
}
