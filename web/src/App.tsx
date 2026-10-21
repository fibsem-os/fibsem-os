import { useEffect, useState } from "react";
import { AgentPage } from "@/pages/Agent";
import { ExperimentPage } from "@/pages/Experiment";
import { MicroscopePage } from "@/pages/Microscope";
import { Pair } from "@/shell/Pair";
import { Rail } from "@/shell/Rail";
import { Settings } from "@/shell/Settings";
import { TopLine } from "@/shell/TopLine";
import { forgotten, getToken, loadToken, onTokenChange } from "@/lib/api";
import { useEventLoop } from "@/lib/events";
import { usePage, useSettingsPath } from "@/lib/route";

function Shell() {
  const [page, go] = usePage();
  const [settings, setSettings] = useSettingsPath(page);
  useEventLoop();
  return (
    <div className="flex h-full flex-col md:flex-row">
      <Rail page={page} go={go} openSettings={() => setSettings(true)} />
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <TopLine />
        <div className="min-h-0 flex-1 overflow-auto">
          {page === "microscope" && <MicroscopePage />}
          {page === "experiment" && <ExperimentPage />}
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
