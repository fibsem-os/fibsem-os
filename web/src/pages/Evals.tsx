import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

// The eval review page, served by `python -m fibsem.ai.evals serve` on this
// computer: a run's results against the keys, and settling the disagreements.

interface RunListed {
  name: string;
  results: number;
  cases: number;
}

interface CaseRow {
  id: string;
  task: string;
  found: boolean;
  repeats: number;
  now: string[][];
  open: number;
}

interface RunSummary {
  name: string;
  ran_with: string;
  cases: CaseRow[];
}

interface Item {
  id: string;
  kind: string;
  title: string;
  repeats: number[];
  said: string[];
  actions: { action: string; label: string }[];
}

interface Repeat {
  repeat: number;
  then: string[];
  now: string[];
  status: string;
  cost_usd: number;
  seconds: number;
  reason: string;
}

interface CaseDetail {
  id: string;
  task: string;
  brief: string;
  figures: string[];
  repeats: Repeat[];
  live: { part: string | null; live: string; same: number; eval: string[] }[];
  items: Item[];
  settled: { at: string; run: string; id: string; action: string }[];
  can_settle: boolean;
}

const COLOURS: Record<string, string> = {
  hit: "text-[#3fb950]",
  miss: "text-[#f85149]",
  early: "text-[#f0883e]",
  late: "text-[#f0883e]",
  unsafe: "text-[#bc8cff]",
  invalid: "text-muted-foreground",
};

async function get<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) throw new Error((await response.json().catch(() => null))?.detail ?? response.statusText);
  return response.json();
}

const enc = encodeURIComponent;
const runPath = (run: string) => `/evals/runs/${enc(run)}`;
const casePath = (run: string, id: string) => `${runPath(run)}/cases/${enc(id)}`;

function Results({ results }: { results: string[] }) {
  return (
    <span className="inline-flex flex-wrap gap-1.5">
      {results.map((r, i) => (
        <span key={i} className={cn("text-[12px] font-semibold", COLOURS[r])}>
          {r}
        </span>
      ))}
    </span>
  );
}

function CaseList({ run, picked, pick }: { run: string; picked: string | null; pick: (id: string) => void }) {
  const summary = useQuery({
    queryKey: ["evals-run", run],
    queryFn: () => get<RunSummary>(runPath(run)),
  });
  if (summary.isError) return <p className="text-[13px] text-destructive">{String(summary.error)}</p>;
  if (!summary.data) return <p className="text-[13px] text-muted-foreground">Reading the run…</p>;
  return (
    <div className="flex flex-col gap-1.5">
      {summary.data.cases.map((c) => (
        <button
          key={c.id}
          onClick={() => pick(c.id)}
          className={cn(
            "flex cursor-pointer flex-col gap-1 rounded border px-3 py-2 text-left",
            c.id === picked ? "border-primary bg-accent" : "border-border bg-transparent hover:bg-accent",
          )}
        >
          <span className="flex items-center gap-2 text-[13px]">
            <span className="min-w-0 flex-1 truncate">{c.id}</span>
            {c.open > 0 && (
              <span className="rounded-full bg-[#f0883e33] px-2 text-[11px] font-semibold text-[#f0883e]">
                {c.open}
              </span>
            )}
          </span>
          <Results results={c.now.flat()} />
        </button>
      ))}
    </div>
  );
}

function Settle({ run, detail }: { run: string; detail: CaseDetail }) {
  const client = useQueryClient();
  const item = detail.items[0];
  const settle = useMutation({
    mutationFn: async (action: string) => {
      const response = await fetch(`${casePath(run, detail.id)}/settle`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ item: item.id, action }),
      });
      if (!response.ok) throw new Error((await response.json().catch(() => null))?.detail ?? response.statusText);
      return response.json();
    },
    onSuccess: (next: CaseDetail) => {
      client.setQueryData(["evals-case", run, detail.id], next);
      client.invalidateQueries({ queryKey: ["evals-run", run] });
    },
  });
  if (!detail.can_settle)
    return <p className="text-[13px] text-muted-foreground">Nothing to settle for this task here yet.</p>;
  if (!item) return <p className="text-[13px] text-[#3fb950]">Nothing left to settle.</p>;
  return (
    <div className="flex flex-col gap-3">
      <div className="text-[12px] text-muted-foreground">To settle: {detail.items.length} left</div>
      <div className="text-[14px] font-semibold">{item.title}</div>
      <div className="text-[13px] text-muted-foreground">
        {item.repeats.length} of {detail.repeats.length} repeats
      </div>
      <img
        key={item.id}
        src={`${casePath(run, detail.id)}/items/${enc(item.id)}.png`}
        alt={item.title}
        className="w-full rounded border border-border"
        onError={(e) => (e.currentTarget.style.display = "none")}
      />
      {item.said.map((s, i) => (
        <p key={i} className="m-0 text-[13px] text-muted-foreground">
          “{s}”
        </p>
      ))}
      {item.actions.map((a, i) => (
        <Button
          key={a.action}
          variant={i === 0 ? "primary" : "default"}
          disabled={settle.isPending}
          onClick={() => settle.mutate(a.action)}
        >
          {a.label}
        </Button>
      ))}
      <Button disabled={settle.isPending} onClick={() => settle.mutate("keep")}>
        Keep the key
      </Button>
      {settle.isError && <p className="m-0 text-[13px] text-destructive">{String(settle.error)}</p>}
    </div>
  );
}

function CaseView({ run, id }: { run: string; id: string }) {
  const detail = useQuery({
    queryKey: ["evals-case", run, id],
    queryFn: () => get<CaseDetail>(casePath(run, id)),
  });
  if (detail.isError) return <p className="p-4 text-[13px] text-destructive">{String(detail.error)}</p>;
  if (!detail.data) return <p className="p-4 text-[13px] text-muted-foreground">Drawing the case…</p>;
  const d = detail.data;
  // The figures are redrawn with the key as it is after each choice.
  const version = d.settled.length;
  return (
    <div className="flex flex-wrap items-stretch">
      <main className="flex min-w-0 flex-[999_1_560px] flex-col gap-3 p-4">
        <p className="m-0 text-[13px] text-muted-foreground">{d.brief}</p>
        {d.figures.map((caption, n) => (
          <figure key={`${n}-${version}`} className="m-0 flex flex-col gap-1">
            <img
              src={`${casePath(run, id)}/figures/${n}.png?v=${version}`}
              alt={caption}
              className="max-w-full rounded border border-border bg-white"
            />
            {d.figures.length > 1 && <figcaption className="text-[12px] text-muted-foreground">{caption}</figcaption>}
          </figure>
        ))}
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-[13px]">
            <thead>
              <tr className="text-left text-muted-foreground">
                <th className="px-2 py-1.5 font-medium">Repeat</th>
                <th className="px-2 py-1.5 font-medium">Now</th>
                <th className="px-2 py-1.5 font-medium">When run</th>
                <th className="px-2 py-1.5 font-medium">Cost</th>
                <th className="px-2 py-1.5 font-medium">Reason</th>
              </tr>
            </thead>
            <tbody>
              {d.repeats.map((r) => (
                <tr key={r.repeat} className="border-t border-border align-top">
                  <td className="px-2 py-1.5">r{r.repeat}</td>
                  <td className="px-2 py-1.5">
                    <Results results={r.now} />
                  </td>
                  <td className="px-2 py-1.5">
                    <Results results={r.then} />
                  </td>
                  <td className="px-2 py-1.5 tabular-nums">${r.cost_usd.toFixed(2)}</td>
                  <td className="px-2 py-1.5 text-muted-foreground">{r.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {d.live.map((row) => (
          <p key={row.part ?? ""} className="m-0 text-[13px]">
            {row.part ? `Part ${row.part}: ` : ""}Real run <Results results={[row.live]} />, same as the real run in{" "}
            {row.same} of {row.eval.length} repeats
          </p>
        ))}
      </main>
      <aside className="flex max-w-[320px] min-w-[260px] flex-[1_1_280px] flex-col gap-3 border-l border-border p-4">
        <Settle run={run} detail={d} />
      </aside>
    </div>
  );
}

export function EvalsPage() {
  const runs = useQuery({
    queryKey: ["evals-runs"],
    queryFn: () => get<RunListed[]>("/evals/runs"),
  });
  const [run, setRun] = useState<string | null>(null);
  const [picked, setPicked] = useState<string | null>(null);
  useEffect(() => {
    if (!run && runs.data?.length) setRun(runs.data[0].name);
  }, [run, runs.data]);
  return (
    <div className="flex h-full flex-col bg-background text-foreground">
      <header className="flex flex-wrap items-center gap-3 border-b border-border px-4 py-3">
        <h1 className="m-0 text-[15px] font-semibold">Eval results</h1>
        {runs.data && runs.data.length > 0 && (
          <select
            aria-label="Run"
            value={run ?? ""}
            onChange={(e) => {
              setRun(e.target.value);
              setPicked(null);
            }}
            className="rounded border border-input bg-accent px-2 py-1 text-[13px] text-foreground"
          >
            {runs.data.map((r) => (
              <option key={r.name} value={r.name}>
                {r.name} · {r.cases} cases · {r.results} runs
              </option>
            ))}
          </select>
        )}
      </header>
      {runs.isError && (
        <p className="p-4 text-[13px] text-destructive">
          No eval server here. Start it with: python -m fibsem.ai.evals serve RUNS --cases CASES
        </p>
      )}
      {runs.data?.length === 0 && <p className="p-4 text-[13px] text-muted-foreground">No runs in this folder yet.</p>}
      {run && (
        <div className="flex min-h-0 flex-1 flex-wrap items-stretch overflow-auto">
          <aside className="flex max-w-[300px] min-w-[240px] flex-[1_1_260px] flex-col gap-2 border-r border-border p-4">
            <CaseList run={run} picked={picked} pick={setPicked} />
          </aside>
          <div className="min-w-0 flex-[999_1_640px]">
            {picked ? (
              <CaseView key={`${run}/${picked}`} run={run} id={picked} />
            ) : (
              <p className="p-4 text-[13px] text-muted-foreground">Pick a case.</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
