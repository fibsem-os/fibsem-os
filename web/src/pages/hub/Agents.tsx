import { Plus } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { api } from "@/lib/api";
import * as fmt from "@/lib/format";
import {
  at,
  Dot,
  Failed,
  FIELD,
  type HubState,
  type Instrument,
  LABEL,
  RED,
  ready,
  StartForm,
  title,
  useHubAction,
} from "@/pages/hub/shared";

function Running({
  instrument: i,
  open,
}: {
  instrument: Instrument;
  open: () => void;
}) {
  const end = useHubAction(() =>
    api<HubState>(`${at(i.name)}/agent/end`, { method: "POST" }),
  );
  const agent = i.session ?? i.agent;
  if (!agent) return null;
  return (
    <tr className="border-t border-border first:border-t-0">
      <td className="py-[11px] pr-3 pl-4 whitespace-nowrap">
        <span className="flex items-center gap-2">
          <Dot colour="ok" />
          <b className="font-semibold">{i.name}</b>
        </span>
      </td>
      <td className="w-full max-w-0 px-3 py-[11px]">
        <span className="block truncate">
          {agent.goal || agent.playbook || "Running"}
        </span>
        <span className="text-xs text-muted-foreground">
          {[agent.playbook, agent.level && title(agent.level)]
            .filter(Boolean)
            .join(" · ")}
        </span>
        <Failed error={end.error} />
      </td>
      <td className="px-3 py-[11px] whitespace-nowrap text-muted-foreground">
        {agent.started && `since ${fmt.clock(agent.started)}`}
      </td>
      <td className="w-36 py-[11px] pr-4 pl-3 text-right whitespace-nowrap">
        <Button onClick={open}>Open</Button>
        {i.session && (
          <button
            type="button"
            className={`${RED} ml-1.5`}
            disabled={end.isPending}
            onClick={() => end.mutate()}
          >
            End
          </button>
        )}
      </td>
    </tr>
  );
}

function Start({
  instruments,
  onDone,
}: {
  instruments: Instrument[];
  onDone: () => void;
}) {
  const [name, setName] = useState(instruments[0]?.name ?? "");
  const chosen = instruments.find((i) => i.name === name) ?? instruments[0];
  return (
    <Dialog open onOpenChange={(open) => !open && onDone()}>
      <DialogContent
        title="Start an agent"
        className="w-[min(500px,calc(100vw-32px))]"
      >
        <div className="px-4 py-3.5 text-[13px]">
          <StartForm
            key={chosen?.name}
            instrument={chosen}
            onDone={onDone}
            onCancel={onDone}
          >
            <label className={LABEL}>
              Instrument
              <select
                value={chosen?.name ?? ""}
                onChange={(event) => setName(event.target.value)}
                className={FIELD}
              >
                {instruments.map((i) => (
                  <option key={i.name} value={i.name}>
                    {i.name} · {i.gateway?.app ?? "App"} up
                  </option>
                ))}
              </select>
            </label>
          </StartForm>
        </div>
      </DialogContent>
    </Dialog>
  );
}

/** The agents running across the lab, and starting one more. */
export function Agents({
  instruments,
  open,
  starting,
  setStarting,
}: {
  instruments: Instrument[];
  open: (name: string) => void;
  starting: boolean;
  setStarting: (starting: boolean) => void;
}) {
  const running = instruments.filter((i) => i.session || i.agent);
  return (
    <>
      <Card className="p-0 text-[13px]">
        <div className="px-4 pt-3 text-xs font-semibold text-muted-foreground">
          Running
        </div>
        {running.length ? (
          <table className="w-full border-collapse">
            <tbody>
              {running.map((i) => (
                <Running key={i.name} instrument={i} open={() => open(i.name)} />
              ))}
            </tbody>
          </table>
        ) : (
          <p className="m-0 px-4 py-3 text-muted-foreground">None</p>
        )}
      </Card>
      {starting && (
        <Start
          instruments={instruments.filter(ready)}
          onDone={() => setStarting(false)}
        />
      )}
    </>
  );
}

export function StartButton({
  instruments,
  onClick,
}: {
  instruments: Instrument[];
  onClick: () => void;
}) {
  const any = instruments.some(ready);
  return (
    <Button
      variant="primary"
      disabled={!any}
      title={any ? undefined : "No instrument has its app up and no agent."}
      onClick={onClick}
    >
      <Plus size={14} />
      Start an agent
    </Button>
  );
}
