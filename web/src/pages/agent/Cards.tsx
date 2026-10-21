import { Check, ChevronDown, ChevronRight, X } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import type { AgentRequest, AgentState, ConnectAsk, RequestAction, Suggestion } from "@/lib/types";
import { useAct } from "./act";
import { CardPicture, EvidenceImage } from "./Pictures";

const card = "flex flex-col gap-2.5 rounded-md border border-[var(--fibsem-purple-color)] bg-card p-3";
const title = "text-[13px] font-semibold text-[#c4b5fd]";

function Details({ details }: { details: NonNullable<RequestAction["details"]> }) {
  return (
    <div className="overflow-x-auto">
      <table className="mt-1 border-collapse text-[12px]">
        <thead>
          <tr>
            {(details.columns ?? []).map((c) => (
              <th key={c} className="border-b border-border px-2 py-1 text-left font-normal text-muted-foreground">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {(details.rows ?? []).map((row, i) => (
            <tr key={i}>
              {row.map((v, j) => (
                <td key={j} className="px-2 py-0.5 tabular-nums">
                  {v}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Reject…, Decline… or Deny… asks why first, then sends it with the refusal. */
function Decide({
  approve,
  refuse,
  refuseWord,
  approveWord = "Approve",
}: {
  approve: () => void;
  refuse: (reason: string) => void;
  refuseWord: string;
  approveWord?: string;
}) {
  const [asking, setAsking] = useState(false);
  const [reason, setReason] = useState("");
  if (asking)
    return (
      <div className="flex flex-wrap items-center gap-2">
        <input
          autoFocus
          aria-label="Tell the agent why (optional)"
          className="min-w-0 flex-1 rounded border border-input bg-background px-2 py-1.5 text-[13px] text-foreground"
          value={reason}
          onChange={(e) => setReason(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && refuse(reason)}
        />
        <Button onClick={() => setAsking(false)}>Back</Button>
        <Button variant="primary" onClick={() => refuse(reason)}>
          {refuseWord}
        </Button>
      </div>
    );
  return (
    <div className="flex flex-wrap gap-2">
      <Button variant="primary" onClick={approve}>
        {approveWord}
      </Button>
      <Button onClick={() => setAsking(true)}>{refuseWord}…</Button>
    </div>
  );
}

/** The card's title; clicking it folds the card to one line, keeping its buttons. */
function Header({ text, summary, folded, fold }: { text: string; summary: string; folded: boolean; fold: () => void }) {
  const Chevron = folded ? ChevronRight : ChevronDown;
  return (
    <button
      className="flex min-w-0 cursor-pointer items-center gap-1.5 border-0 bg-transparent p-0 text-left"
      aria-expanded={!folded}
      onClick={fold}
    >
      <Chevron size={14} className="shrink-0 text-[#c4b5fd]" />
      <span className={title}>{text}</span>
      {folded && <span className="min-w-0 truncate text-[13px] text-muted-foreground">{summary}</span>}
    </button>
  );
}

function Why({ reason }: { reason: string }) {
  return (
    <div className="text-[13px]">
      <span className="text-muted-foreground">Why: </span>
      {reason}
    </div>
  );
}

/** The agent's request, each action a numbered step; while it runs, how far each got. */
export function RequestCard({ request, agent }: { request: AgentRequest; agent: AgentState }) {
  const { act, error } = useAct();
  const [details, setDetails] = useState(false);
  const applying = request.status === "applying";
  const results = request.results ?? [];
  const n = request.actions.length;
  const id = encodeURIComponent(request.request_id);
  const picture = agent.pictures.request;
  const [folded, setFolded] = useState(false);
  const summary = request.actions.map((a) => a.what || a.tool).join(", ");
  return (
    <article aria-label="Waiting on you" className={card}>
      <Header text="Agent asks" summary={summary} folded={folded} fold={() => setFolded(!folded)} />
      {!folded && (
        <>
          <ol className="m-0 flex list-none flex-col gap-2 p-0">
            {request.actions.map((a, i) => (
              <li key={i} className="flex gap-2.5 text-[13px]">
                <span className="w-4 shrink-0 text-center text-muted-foreground">
                  {!applying ? (
                    i + 1
                  ) : i < results.length ? (
                    results[i].ok ? (
                      <Check size={14} className="text-[var(--fibsem-ok-color)]" />
                    ) : (
                      <X size={14} className="text-destructive" />
                    )
                  ) : i === results.length ? (
                    "…"
                  ) : (
                    "○"
                  )}
                </span>
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-baseline gap-2">
                    <span className="font-semibold">{a.what || a.tool}</span>
                    {i === 0 && request.actions.some((x) => x.details) && (
                      <button
                        className="cursor-pointer border-0 bg-transparent p-0 text-[12px] text-link"
                        onClick={() => setDetails(!details)}
                      >
                        {details ? "Hide details" : "Show details"}
                      </button>
                    )}
                  </div>
                  {a.change && <div className="font-mono text-[12px] text-[var(--fibsem-text-color)]">{a.change}</div>}
                  {details && a.details && <Details details={a.details} />}
                </div>
              </li>
            ))}
          </ol>
          <Why reason={request.reason} />
          {request.evidence ? (
            <EvidenceImage id={request.evidence.evidence_id} />
          ) : picture?.image ? (
            <CardPicture kind="request" version={picture.version} />
          ) : null}
          {picture?.caption ? (
            <div className="text-[12px] text-muted-foreground">{picture.caption}</div>
          ) : (
            !request.evidence && <div className="text-[12px] text-muted-foreground">No evidence</div>
          )}
        </>
      )}
      {applying ? (
        <div className="text-[13px] text-muted-foreground">
          Approved by you. Running {Math.min(results.length + 1, n)} of {n}…
        </div>
      ) : (
        <Decide
          refuseWord="Reject"
          approve={() => act(`/app/agent/requests/${id}/approve`)}
          refuse={(reason) =>
            act(`/app/agent/requests/${id}/reject`, {
              method: "POST",
              body: JSON.stringify({ reason }),
            })
          }
        />
      )}
      {error && <div className="text-[13px] text-destructive">{error}</div>}
    </article>
  );
}

/** The agent's suggestion on a pending proposal: Approve decides it as suggested. */
export function SuggestionCard({ suggestion, agent }: { suggestion: Suggestion; agent: AgentState }) {
  const { act, error } = useAct();
  const id = encodeURIComponent(suggestion.proposal_id);
  const picture = agent.pictures.suggestion;
  const [folded, setFolded] = useState(false);
  return (
    <article aria-label="Waiting on you" className={card}>
      <Header text="Agent suggests" summary={suggestion.decision} folded={folded} fold={() => setFolded(!folded)} />
      {!folded && (
        <>
          <div className="text-[13px] text-muted-foreground">{suggestion.proposal}</div>
          <div className="font-semibold">{suggestion.decision}</div>
          <Why reason={suggestion.reason} />
          {!suggestion.evidence && (
            <div className="text-[12px] text-muted-foreground">{suggestion.no_evidence || "No evidence"}</div>
          )}
          {suggestion.evidence ? (
            <EvidenceImage id={suggestion.evidence.evidence_id} />
          ) : picture?.image ? (
            <CardPicture kind="suggestion" version={picture.version} />
          ) : null}
          {picture?.caption && <div className="text-[12px] text-muted-foreground">{picture.caption}</div>}
        </>
      )}
      <Decide
        refuseWord="Decline"
        approve={() => act(`/app/agent/suggestions/${id}/approve`)}
        refuse={(reason) =>
          act(`/app/agent/suggestions/${id}/decline`, {
            method: "POST",
            body: JSON.stringify({ reason }),
          })
        }
      />
      {error && <div className="text-[13px] text-destructive">{error}</div>}
    </article>
  );
}

/** An agent on this PC asking to connect: until it is allowed it reaches nothing. */
export function ConnectCard({ ask }: { ask: ConnectAsk }) {
  const { act, error } = useAct();
  const id = encodeURIComponent(ask.id);
  const rows: [string, string][] = [
    ["Agent", `${ask.client}, as it names itself`],
    ["From", ask.origin],
    ["Runs as", ask.user],
  ];
  return (
    <article aria-label="Waiting on you" className={card}>
      <span className={title}>Agent wants to connect</span>
      <dl className="m-0 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-[13px]">
        {rows.map(([heading, value]) => (
          <div key={heading} className="contents">
            <dt className="text-muted-foreground">{heading}</dt>
            <dd className="m-0 min-w-0 break-words">{value}</dd>
          </div>
        ))}
      </dl>
      <div className="text-[13px]">
        <span className="text-muted-foreground">If you allow it: </span>
        {ask.if_allowed}
      </div>
      <Decide
        approveWord="Allow"
        refuseWord="Deny"
        approve={() => act(`/app/agent/connections/${id}/allow`)}
        refuse={(reason) =>
          act(`/app/agent/connections/${id}/deny`, {
            method: "POST",
            body: JSON.stringify({ reason }),
          })
        }
      />
      {error && <div className="text-[13px] text-destructive">{error}</div>}
    </article>
  );
}
