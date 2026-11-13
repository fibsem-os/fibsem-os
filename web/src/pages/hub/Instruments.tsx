import { Plus } from "lucide-react";
import { type ReactNode, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { api } from "@/lib/api";
import * as fmt from "@/lib/format";
import { cn } from "@/lib/utils";
import {
  at,
  brief,
  Dot,
  Failed,
  FIELD,
  health,
  Hint,
  type HubState,
  type Instrument,
  LABEL,
  RED,
  title,
  useHubAction,
} from "@/pages/hub/shared";

const SECTION = "flex flex-col gap-3 border-t border-border p-4 first:border-t-0";
const HEADING = "text-xs font-semibold text-muted-foreground";

function Add({ onDone }: { onDone: (name?: string) => void }) {
  const [name, setName] = useState("");
  const [ssh, setSsh] = useState("");
  const add = useHubAction(
    () =>
      api<HubState>("/hub/instruments", {
        method: "POST",
        body: JSON.stringify({ name: name.trim(), ssh: ssh.trim() }),
      }),
    () => onDone(name.trim()),
  );
  return (
    <Dialog open onOpenChange={(open) => !open && onDone()}>
      <DialogContent
        title="Add an instrument"
        className="w-[min(460px,calc(100vw-32px))]"
      >
        <div className="flex flex-col gap-3 px-4 py-3.5">
          <label className={LABEL}>
            Name
            <input
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="fib-5"
              autoFocus
              className={FIELD}
            />
          </label>
          <label className={LABEL}>
            <span className="flex items-center gap-1.5">
              SSH destination
              <Hint>
                Where ssh connects to its PC. This PC's ssh key must be set up
                there first.
              </Hint>
            </span>
            <input
              value={ssh}
              onChange={(event) => setSsh(event.target.value)}
              placeholder="user@microscope-pc"
              className={FIELD}
            />
          </label>
          <Failed error={add.error} />
        </div>
        <div className="flex justify-end gap-2 px-4 pt-1 pb-4">
          <Button variant="ghost" onClick={() => onDone()}>
            Cancel
          </Button>
          <Button
            variant="primary"
            disabled={!name.trim() || !ssh.trim() || add.isPending}
            onClick={() => add.mutate()}
          >
            Add
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function Pair({ instrument: i }: { instrument: Instrument }) {
  const [open, setOpen] = useState(!i.paired);
  const [code, setCode] = useState("");
  const pair = useHubAction(
    () =>
      api<HubState>(`${at(i.name)}/pair`, {
        method: "POST",
        body: JSON.stringify({ code: code.trim() }),
      }),
    () => {
      setOpen(false);
      setCode("");
    },
  );
  if (!open) return <Button onClick={() => setOpen(true)}>Pair again</Button>;
  return (
    <span className="flex flex-col gap-1.5">
      <span className="flex items-center gap-2">
        <input
          value={code}
          onChange={(event) => setCode(event.target.value)}
          inputMode="numeric"
          placeholder="Pairing code"
          aria-label="Pairing code"
          className={cn(FIELD, "w-32")}
        />
        <Button
          variant="primary"
          disabled={!code.trim() || pair.isPending || !i.reachable}
          onClick={() => pair.mutate()}
        >
          Pair
        </Button>
        {i.paired && (
          <Button variant="ghost" onClick={() => setOpen(false)}>
            Cancel
          </Button>
        )}
        <Hint>
          On the microscope PC, run <code>fibsem-gateway pair --hub</code> and
          type the code it shows here.
        </Hint>
      </span>
      <Failed error={pair.error} />
    </span>
  );
}

function Fact({ label, children }: { label: string; children: ReactNode }) {
  return (
    <>
      <span className="text-[12.5px] text-muted-foreground">{label}</span>
      <span className="flex min-w-0 items-center gap-1.5">{children}</span>
    </>
  );
}

function Detail({ instrument: i }: { instrument: Instrument }) {
  const [ssh, setSsh] = useState(i.ssh ?? i.url ?? "");
  const byUrl = !i.ssh && !!i.url;
  const save = useHubAction(() =>
    api<HubState>("/hub/instruments", {
      method: "POST",
      body: JSON.stringify({
        name: i.name,
        ...(byUrl ? { url: ssh.trim() } : { ssh: ssh.trim() }),
      }),
    }),
  );
  const remove = useHubAction(() =>
    api<HubState>(at(i.name), { method: "DELETE" }),
  );
  const state =
    i.reachable === null
      ? "Checking"
      : !i.reachable
        ? `Not reachable since ${fmt.clock(i.since)}`
        : i.paired
          ? "Reachable · paired"
          : "Reachable · not paired";
  return (
    <Card className="p-0 text-[13px]">
      <div className={cn(SECTION, "flex-row items-center gap-2.5")}>
        <Dot colour={health(i)} />
        <b className="text-[15px] font-semibold">{i.name}</b>
        <span className="text-muted-foreground">{state}</span>
        {i.why && (i.paired || !i.reachable) && <Hint>{i.why}</Hint>}
      </div>
      <div className={SECTION}>
        <span className={HEADING}>Connection</span>
        <div className="grid grid-cols-2 gap-3">
          <label className={LABEL}>
            Name
            <input
              value={i.name}
              disabled
              title="To rename it, remove it and add it again."
              className={FIELD}
            />
          </label>
          <label className={LABEL}>
            {byUrl ? "Gateway address" : "SSH destination"}
            <input
              value={ssh}
              onChange={(event) => setSsh(event.target.value)}
              className={FIELD}
            />
          </label>
        </div>
        <Failed error={save.error} />
        <div className="flex justify-end">
          <Button
            variant="primary"
            disabled={
              !ssh.trim() ||
              ssh.trim() === (i.ssh ?? i.url ?? "") ||
              save.isPending
            }
            onClick={() => save.mutate()}
          >
            Save
          </Button>
        </div>
      </div>
      <div className={SECTION}>
        <span className={HEADING}>Instrument</span>
        <div className="grid grid-cols-[150px_minmax(0,1fr)] items-center gap-x-3 gap-y-2.5">
          {i.microscope?.model && (
            <Fact label="Microscope">{i.microscope.model}</Fact>
          )}
          {i.gateway?.version && (
            <Fact label="fibsemOS">{i.gateway.version}</Fact>
          )}
          {i.gateway && (
            <Fact label="Agent level cap">
              {i.level_cap ? title(i.level_cap) : "None"}
              <Hint lock>
                Set on the microscope PC, in its gateway Settings. The hub
                can't raise it.
              </Hint>
            </Fact>
          )}
          <Fact label="Paired">{i.paired ? "Yes" : "No"}</Fact>
          {i.tunnel && (
            <Fact label="Tunnel">
              {i.tunnel.open ? "Open" : "Closed"}
              <span className="text-muted-foreground">
                · ports {i.tunnel.gateway_port}, {i.tunnel.app_port}
              </span>
            </Fact>
          )}
        </div>
      </div>
      <div className={cn(SECTION, "flex-row items-start gap-2")}>
        <Pair key={String(i.paired)} instrument={i} />
        <span className="ml-auto flex flex-col items-end gap-1.5 self-center">
          <button
            type="button"
            className={RED}
            disabled={!!i.session || remove.isPending}
            title={i.session ? "End its agent first." : undefined}
            onClick={() => remove.mutate()}
          >
            Remove
          </button>
          <Failed error={remove.error} />
        </span>
      </div>
    </Card>
  );
}

/** The list of instruments, and the one chosen in detail: how it is reached,
 * what it is, pairing it and taking it off the list. */
export function Instruments({
  instruments,
  adding,
  setAdding,
}: {
  instruments: Instrument[];
  adding: boolean;
  setAdding: (adding: boolean) => void;
}) {
  const [chosen, setChosen] = useState<string | null>(null);
  const shown =
    instruments.find((i) => i.name === chosen) ?? instruments[0] ?? null;
  return (
    <>
      {instruments.length ? (
        <div className="grid items-start gap-3 md:grid-cols-[260px_minmax(0,1fr)]">
          <Card className="flex flex-col p-0 text-[13px]">
            {instruments.map((i) => (
              <button
                key={i.name}
                type="button"
                aria-current={i === shown}
                onClick={() => setChosen(i.name)}
                className={cn(
                  "flex cursor-pointer items-center gap-2.5 border-0 border-t border-border px-3.5 py-[11px] text-left first:border-t-0",
                  i === shown ? "bg-accent" : "bg-transparent hover:bg-accent/50",
                )}
              >
                <Dot colour={health(i)} />
                <b className="font-semibold text-foreground">{i.name}</b>
                <span className="ml-auto text-xs text-muted-foreground">
                  {brief(i)}
                </span>
              </button>
            ))}
          </Card>
          {shown && <Detail key={shown.name} instrument={shown} />}
        </div>
      ) : (
        <Card className="text-[13px] text-muted-foreground">
          No instruments yet.
        </Card>
      )}
      {adding && (
        <Add
          onDone={(name) => {
            setAdding(false);
            if (name) setChosen(name);
          }}
        />
      )}
    </>
  );
}

export function AddButton({ onClick }: { onClick: () => void }) {
  return (
    <Button onClick={onClick}>
      <Plus size={14} />
      Add instrument
    </Button>
  );
}
