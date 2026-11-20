import { Loader } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Markdown } from "@/lib/markdown";
import type { AgentState, RecentRow } from "@/lib/types";
import { useAct } from "./act";

// The question as the operator sends it, and the mark the answer comes with
// (fibsem.ai.agent.aside: PREFIX, LEAD): the tab shows both without them.
const ASK = "/btw ";
const QUESTION = /^\s*\/btw\s*/i;
const ANSWER = /^Side answer \(\/btw, not added to the agent's conversation\):\s*/;
const LIMIT = 2000;

const connected = (a: AgentState) => a.last_heard_s != null && a.last_heard_s < 120;

/** A side question still waiting for its answer, among *rows* (the side
 * chat's, oldest first): the newest question with nothing after it. */
export function asking(rows: RecentRow[]): boolean {
  const last = rows[rows.length - 1];
  return last != null && last.kind === "you";
}

/** The side chat: quick questions the agent's session answers from its
 * conversation as it stands, without stopping or seeing them (the operator's
 * ``/btw`` messages, marked ``aside``). *rows*: those messages and the answers,
 * oldest first. */
export function SideChat({ rows, agent }: { rows: RecentRow[]; agent: AgentState }) {
  const [draft, setDraft] = useState("");
  const { act, busy, error } = useAct();
  const live = connected(agent);
  const send = async () => {
    const text = draft.trim();
    if (!text) return;
    const answer = await act("/app/agent/messages", {
      method: "POST",
      body: JSON.stringify({ text: ASK + text }),
    });
    if (answer !== undefined) setDraft("");
  };
  return (
    <div className="flex min-h-full flex-col text-[13px]">
      <div className="flex flex-1 flex-col gap-3">
        {rows.length === 0 && (
          <p className="m-0 text-muted-foreground">
            Ask a quick question about the run. It is answered from the agent's conversation as it stands; the agent
            carries on and doesn't see the question. To tell the agent something, write in the chat.
          </p>
        )}
        {rows.map((row) =>
          row.kind === "you" ? (
            <div key={`${row.at}|${row.message_id}`} className="self-end rounded-md bg-card px-2.5 py-1.5 whitespace-pre-wrap break-words">
              {row.text.replace(QUESTION, "")}
            </div>
          ) : (
            <div key={`${row.at}|${row.text.slice(0, 40)}`} className="break-words">
              <Markdown text={row.text.replace(ANSWER, "")} />
            </div>
          ),
        )}
        {asking(rows) && (
          <div className="flex items-center gap-1.5 text-muted-foreground">
            <Loader size={12} className="animate-spin" aria-hidden />
            {live ? "Answering…" : "Waiting for an agent to answer"}
          </div>
        )}
      </div>
      <div className="sticky bottom-0 mt-3 flex flex-col gap-1.5 bg-background pt-2">
        <textarea
          aria-label="Side question"
          rows={2}
          maxLength={LIMIT}
          placeholder="Ask on the side…"
          className="w-full resize-none rounded-md border border-input bg-card px-2.5 py-1.5 text-[13px] text-foreground outline-none focus:border-primary"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void send();
            }
          }}
        />
        <div className="flex items-center justify-between gap-2">
          <span className="min-w-0 text-[12px] text-destructive">{error}</span>
          <Button variant="primary" disabled={busy || !draft.trim()} onClick={() => void send()}>
            Ask
          </Button>
        </div>
      </div>
    </div>
  );
}
