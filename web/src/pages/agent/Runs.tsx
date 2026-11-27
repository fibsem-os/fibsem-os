import { Loader } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import type { TranscriptLine } from "@/lib/types";
import { cn } from "@/lib/utils";
import { useAct } from "./act";
import { say } from "./steps";

const LIMIT = 2000;

/** Whether a run is the lead's in the background (it carried on) or one it
 * waits for: "Background", or "Lead waiting" while a blocking run goes on. */
export function RunTag({ call, running }: { call: TranscriptLine; running: boolean }) {
  if (!call.background && !running) return null;
  return (
    <span
      className={cn(
        "rounded-full px-2 py-px text-[11px] font-semibold",
        call.background ? "bg-primary/15 text-primary" : "bg-muted text-muted-foreground",
      )}
      title={
        call.background
          ? "The lead started it in the background and carries on; its answer comes on its own."
          : "The lead waits for its answer before it goes on."
      }
    >
      {call.background ? "Background" : "Lead waiting"}
    </span>
  );
}

/** What a running run does now, in words: its newest step ("Took an SEM
 * image"), or null when it has taken none. */
export function runNow(lines: TranscriptLine[]): string | null {
  const step = [...lines].reverse().find((l) => l.kind === "step");
  if (!step) return null;
  const said = say(step.verb ?? "", step.text);
  return [said.verb, said.words].filter(Boolean).join(" ");
}

/** The line under a run's name: what it does now while it runs. */
export function RunNow({ lines, running }: { lines: TranscriptLine[]; running: boolean }) {
  const now = running ? runNow(lines) : null;
  if (!now) return null;
  return (
    <div className="min-w-0 truncate text-[12px] text-muted-foreground" title={now}>
      Now: {now}
    </div>
  );
}

/** A message to a run while it goes on: the subagent reads it after its next
 * step, without a turn of the lead's, and the lead hears of it. */
export function RunMessage({ call, name }: { call: TranscriptLine; name: string }) {
  const [draft, setDraft] = useState("");
  const [sent, setSent] = useState(false);
  const { act, busy, error } = useAct();
  if (!call.call) return null;
  const send = async () => {
    const text = draft.trim();
    if (!text) return;
    const answer = await act("/app/agent/messages", {
      method: "POST",
      body: JSON.stringify({ text, run: call.call }),
    });
    if (answer !== undefined) {
      setDraft("");
      setSent(true);
    }
  };
  return (
    <div className="flex flex-col gap-1 border-t border-border pt-2.5">
      <div className="flex items-end gap-2">
        <textarea
          aria-label={`Message to the ${name}`}
          rows={2}
          maxLength={LIMIT}
          placeholder={`Write to the ${name}…`}
          className="min-w-0 flex-1 resize-none rounded-md border border-input bg-card px-2.5 py-1.5 text-[13px] text-foreground outline-none focus:border-primary"
          value={draft}
          onChange={(e) => {
            setDraft(e.target.value);
            setSent(false);
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void send();
            }
          }}
        />
        <Button variant="primary" disabled={busy || !draft.trim()} onClick={() => void send()}>
          Send
        </Button>
      </div>
      <span className="text-[12px] text-muted-foreground">
        {error ? (
          <span className="text-destructive">{error}</span>
        ) : sent ? (
          "Sent: it reads it after its next step. If it ends first, the lead reads it."
        ) : (
          "It reads it after its next step; the lead hears of it too."
        )}
      </span>
    </div>
  );
}

/** Under the live line: the blocking run the lead waits on, and how many run
 * in the background. Each opens its run, or the list. */
export function RunsLine({
  running,
  name,
  onOpen,
  onList,
}: {
  running: TranscriptLine[];
  name: (call: TranscriptLine) => string;
  onOpen: (call: TranscriptLine) => void;
  onList: () => void;
}) {
  // The lead's own calls: a run a subagent started is its parent's to wait on.
  const own = running.filter((c) => c.run == null);
  const waited = own.find((c) => !c.background);
  const background = own.filter((c) => c.background).length;
  if (!waited && !background) return null;
  const link =
    "cursor-pointer border-0 bg-transparent p-0 text-[13px] text-primary underline-offset-2 hover:underline";
  return (
    <div className="flex min-h-5 items-center gap-1.5 text-[13px] text-muted-foreground">
      <Loader size={12} className="shrink-0 animate-spin" aria-hidden />
      {waited && (
        <span>
          Waiting on{" "}
          <button className={link} onClick={() => onOpen(waited)}>
            {name(waited)}
          </button>
        </span>
      )}
      {waited && background > 0 && <span>·</span>}
      {background > 0 && (
        <button className={link} onClick={onList}>
          {background} in background
        </button>
      )}
    </div>
  );
}
