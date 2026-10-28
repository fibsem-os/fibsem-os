import { ChevronDown, Paperclip, Square } from "lucide-react";
import { useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Facts } from "@/components/ui/card";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { Popup } from "@/components/ui/popup";
import { api } from "@/lib/api";
import type { AgentLevel, AgentState } from "@/lib/types";
import { cn } from "@/lib/utils";
import { useAct } from "./act";
import { TAKEN, type Attachments } from "./attach";
import { AttachedChips } from "./Attachments";
import { Customise } from "./Customisation";

const LIMIT = 2000;

// Watch's mark is white (Patrick 2026-10-21).
export const LEVEL_COLOUR: Record<AgentLevel, string> = {
  off: "var(--fibsem-gray-secondary-color)",
  watch: "var(--fibsem-gray-text-color)",
  assist: "#a78bfa",
  drive: "var(--fibsem-purple-color)",
};

function Dot({ colour }: { colour: string }) {
  return <span className="inline-block size-2 shrink-0 rounded-full" style={{ background: colour }} />;
}

export function heard(seconds: number | null): string {
  if (seconds == null) return "not heard from yet";
  if (seconds < 60) return `${Math.round(seconds)} s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)} min`;
  return `${Math.round(seconds / 3600)} h`;
}

// Heard from in the last two minutes: the agent reads a message on its next turn.
const connected = (a: AgentState) => a.last_heard_s != null && a.last_heard_s < 120;

const plain = "flex items-center gap-1.5 rounded border-0 bg-transparent px-2 py-1 text-[13px] text-foreground hover:bg-accent cursor-pointer disabled:cursor-default disabled:opacity-50";

function AgentInfo({ agent }: { agent: AgentState }) {
  const name = agent.client?.split("/")[0] || "Agent";
  const level = agent.levels.find((l) => l.level === agent.level);
  return (
    <Popup
      label="Agent"
      className="w-[min(340px,calc(100vw-48px))] p-3.5"
      trigger={(open, toggle) => (
        <button className={plain} aria-expanded={open} onClick={toggle}>
          <Dot colour={connected(agent) ? "var(--fibsem-ok-color)" : "var(--fibsem-gray-secondary-color)"} />
          {name}
          <span className="text-muted-foreground">· {heard(agent.last_heard_s)}</span>
        </button>
      )}
    >
      {() => (
        <div className="flex flex-col gap-3">
          <b>{name}</b>
          <Facts
            className="[&_dd]:text-left"
            rows={[
              ["Client", agent.client ?? "Not named"],
              ["Last heard", agent.last_heard_s == null ? "Not yet" : `${heard(agent.last_heard_s)} ago`],
              ...(agent.session ? [["Session", agent.session] as [string, string]] : []),
              ...agent.facts,
              ["Level", `${level?.name ?? "Unknown"}${agent.stopped ? ", stopped by you" : ""}`],
            ]}
          />
        </div>
      )}
    </Popup>
  );
}

function Connect() {
  const [open, setOpen] = useState(false);
  const [text, setText] = useState<string | null | undefined>(undefined);
  const [copied, setCopied] = useState(false);
  const show = async () => {
    setOpen(true);
    setCopied(false);
    setText((await api<{ text: string | null }>("/app/agent/connect").catch(() => ({ text: null }))).text);
  };
  return (
    <>
      <button className={cn(plain, "text-link")} onClick={show}>
        Connect an agent…
      </button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent title="Connect an agent" className="w-[min(640px,calc(100vw-32px))]">
          <div className="flex flex-col gap-3 p-4 text-[13px]">
            {text === undefined ? (
              <p className="m-0 text-muted-foreground">Reading…</p>
            ) : text ? (
              <>
                <p className="m-0">
                  Paste this into Claude Code on the microscope PC. It registers the fibsem sidecar, checks the
                  connection and tells you its level.
                </p>
                <pre className="m-0 max-h-[40vh] overflow-auto rounded bg-background p-3 text-[12px] whitespace-pre-wrap">{text}</pre>
                <div className="flex justify-end">
                  <Button
                    variant="primary"
                    onClick={() => navigator.clipboard?.writeText(text).then(() => setCopied(true))}
                  >
                    {copied ? "Copied" : "Copy"}
                  </Button>
                </div>
              </>
            ) : (
              <p className="m-0">
                fibsem-mcp isn't installed on the microscope PC. Install it there with: pip install fibsem[mcp]
              </p>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}

function Level({ agent, act }: { agent: AgentState; act: ReturnType<typeof useAct>["act"] }) {
  const current = agent.levels.find((l) => l.level === agent.level);
  return (
    <Popup
      label="Agent level"
      className="w-56"
      trigger={(open, toggle) => (
        <button
          className={plain}
          aria-haspopup="menu"
          aria-expanded={open}
          aria-label={`Agent level: ${current?.name ?? "unknown"}`}
          disabled={agent.level == null}
          onClick={toggle}
        >
          <Dot colour={agent.level ? LEVEL_COLOUR[agent.level] : "transparent"} />
          {current?.name ?? "Level"}
          <ChevronDown size={14} className="text-muted-foreground" />
        </button>
      )}
    >
      {(close) => (
        <div role="menu" className="flex flex-col">
          {agent.levels.map((l) => (
            <button
              key={l.level}
              role="menuitemradio"
              aria-checked={l.level === agent.level}
              className={cn(plain, "py-1.5", l.level === agent.level && "bg-accent")}
              onClick={() => {
                close();
                act("/app/agent/level", { method: "PUT", body: JSON.stringify({ level: l.level }) });
              }}
            >
              <Dot colour={LEVEL_COLOUR[l.level]} />
              <span className="flex-1 text-left">{l.name}</span>
              <span className="text-[12px] text-muted-foreground">{l.tip}</span>
            </button>
          ))}
        </div>
      )}
    </Popup>
  );
}

function Stop({ agent, act }: { agent: AgentState; act: ReturnType<typeof useAct>["act"] }) {
  return (
    <Popup
      label="Stop the agent"
      align="right"
      className="w-64 p-3"
      trigger={(open, toggle) => (
        <button className={cn(plain, "px-2 py-1.5")} aria-label="Stop agent" aria-expanded={open} disabled={agent.level == null} onClick={toggle}>
          <Square size={14} fill="currentColor" />
        </button>
      )}
    >
      {(close) => (
        <div className="flex flex-col gap-2.5">
          <div className="font-semibold">Stop the agent?</div>
          <div className="text-[13px] text-muted-foreground">It drops to Watch and stops acting.</div>
          <div className="flex justify-end gap-2">
            <Button onClick={close}>Cancel</Button>
            <Button
              className="border-[var(--fibsem-semantic-error-color)] bg-[var(--fibsem-semantic-error-color)] text-white hover:bg-[var(--fibsem-semantic-error-hover-color)]"
              onClick={() => {
                close();
                act("/app/agent/stop");
              }}
            >
              Stop
            </Button>
          </div>
        </div>
      )}
    </Popup>
  );
}

/** The message box, and under it the agent, its level and Stop. */
function Attach({ agent, attach }: { agent: AgentState; attach: Attachments }) {
  const input = useRef<HTMLInputElement>(null);
  return (
    <>
      <button
        className={cn(plain, "px-1.5")}
        aria-label="Attach files"
        title={
          agent.experiment
            ? "Attach files: saved in the experiment's notes folder"
            : "Open an experiment to attach files"
        }
        disabled={!agent.experiment}
        onClick={() => input.current?.click()}
      >
        <Paperclip size={15} />
      </button>
      <input
        ref={input}
        type="file"
        multiple
        accept={TAKEN.join(",")}
        className="hidden"
        onChange={(e) => {
          attach.add(e.target.files ?? []);
          e.target.value = "";
        }}
      />
    </>
  );
}

export function Composer({ agent, attach, onSent }: { agent: AgentState; attach: Attachments; onSent: () => void }) {
  const [draft, setDraft] = useState("");
  const [said, setSaid] = useState<string | null>(null);
  const { act, busy, error, setError } = useAct();
  // Saved files go as lines at the message's end ("Attached notes/plan.md (4.0 KB)").
  const text = [draft.trim(), attach.lines.join("\n")].filter(Boolean).join("\n\n");
  const send = async () => {
    if (!text || attach.uploading) return;
    setSaid(null);
    const answer = await act("/app/agent/messages", { method: "POST", body: JSON.stringify({ text }) });
    if (answer === undefined) return;
    setDraft("");
    attach.clear();
    onSent();
    setSaid(connected(agent) ? "Sent · the agent reads it on its next turn" : "Sent · waiting for an agent");
  };
  return (
    <div className="flex flex-col rounded-md border border-input bg-card">
      <AttachedChips attach={attach} />
      <label htmlFor="msg" className="sr-only">
        Message
      </label>
      <textarea
        id="msg"
        rows={2}
        maxLength={LIMIT}
        value={draft}
        onChange={(e) => {
          setDraft(e.target.value);
          setSaid(null);
          setError(null);
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            send();
          }
        }}
        className="max-h-[150px] min-h-[52px] resize-none border-0 bg-transparent px-3 py-2.5 font-[inherit] text-foreground outline-none [field-sizing:content]"
      />
      <div className="flex flex-wrap items-center gap-2 px-2 pt-1.5 pb-2">
        <Attach agent={agent} attach={attach} />
        {agent.last_heard_s == null ? <Connect /> : <AgentInfo agent={agent} />}
        <Level agent={agent} act={act} />
        <Customise />
        <span className={cn("min-w-0 truncate text-[12px]", error ? "text-destructive" : "text-muted-foreground")}>
          {error ?? said}
        </span>
        <div className="flex-1" />
        <Stop agent={agent} act={act} />
        <Button variant="primary" onClick={send} disabled={!text || attach.uploading || busy}>
          Send
        </Button>
      </div>
    </div>
  );
}
