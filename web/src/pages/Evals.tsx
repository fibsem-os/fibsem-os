import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ListFilter } from "lucide-react";
import { type ReactNode, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Popup } from "@/components/ui/popup";
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

interface Answer {
  repeat: number;
  said: string;
  reason: string;
}

interface Item {
  id: string;
  kind: string;
  title: string;
  key_text: string;
  repeats: number[];
  answers: Answer[];
  actions: { action: string; label: string }[];
}

interface Repeat {
  repeat: number;
  then: string[];
  now: string[];
  did: string;
  status: string;
  cost_usd: number;
  seconds: number;
  reason: string;
}

interface CaseDetail {
  id: string;
  task: string;
  task_text: string;
  key_text: string;
  brief: string;
  figures: string[];
  repeats: Repeat[];
  live: { part: string | null; live: string; same: number; eval: string[] }[];
  items: Item[];
  settled: { at: string; run: string; id: string; action: string }[];
  can_settle: boolean;
}

const TASKS: Record<string, string> = {
  site_selection: "Sites",
  square_quality: "Square labels",
  square_finding: "Squares",
  polishing_stop: "Polishing",
  fm_target: "FM target",
  correlation: "Correlation",
  target_present: "Target present",
};
const taskName = (task: string) => TASKS[task] ?? "Case not found";

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

/** Repeats where every result (each part) is a hit, out of all repeats. */
function Score({ now }: { now: string[][] }) {
  if (!now.length) return null;
  const hits = now.filter((r) => r.length > 0 && r.every((x) => x === "hit")).length;
  const colour = hits === now.length ? "text-[#3fb950]" : hits === 0 ? "text-[#f85149]" : "text-[#d29922]";
  return (
    <span className={cn("text-[12px] font-semibold", colour)}>
      {hits}/{now.length}
    </span>
  );
}

function CaseList({ run, picked, pick }: { run: string; picked: string | null; pick: (id: string) => void }) {
  const summary = useQuery({ queryKey: ["evals-run", run], queryFn: () => get<RunSummary>(runPath(run)) });
  const [task, setTask] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  if (summary.isError) return <p className="text-[13px] text-destructive">{String(summary.error)}</p>;
  if (!summary.data) return <p className="text-[13px] text-muted-foreground">Reading the run…</p>;
  const cases = summary.data.cases;
  const tasks = [...new Set(cases.map((c) => c.task))];
  const shown = cases.filter((c) => (task === null || c.task === task) && (!open || c.open > 0));
  const item = (on: boolean) =>
    cn(
      "flex cursor-pointer items-center gap-2 rounded border-0 bg-transparent px-2 py-1.5 text-left text-[13px] text-foreground hover:bg-accent",
      on && "bg-accent",
    );
  const filtered = task !== null || open;
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <span className="text-[12px] text-muted-foreground">
          {filtered ? `${shown.length} of ${cases.length} cases` : `${cases.length} cases`}
        </span>
        <div className="ml-auto">
          <Popup
            label="Filter"
            align="right"
            className="top-[calc(100%+4px)] bottom-auto w-52"
            trigger={(isOpen, toggle) => (
              <button
                className={cn(
                  "flex shrink-0 cursor-pointer items-center gap-1.5 rounded border-0 bg-transparent px-2 py-1 text-[12px] hover:bg-accent hover:text-foreground",
                  filtered ? "text-primary" : "text-muted-foreground",
                )}
                aria-label="Filter"
                aria-haspopup="menu"
                aria-expanded={isOpen}
                onClick={toggle}
              >
                <ListFilter size={14} />
                {task !== null ? taskName(task) : "All"}
              </button>
            )}
          >
            {(close) => (
              <div role="menu" className="flex flex-col">
                {[null, ...tasks].map((t) => (
                  <button
                    key={t ?? "all"}
                    role="menuitemradio"
                    aria-checked={task === t}
                    className={item(task === t)}
                    onClick={() => {
                      setTask(t);
                      close();
                    }}
                  >
                    {t === null ? "All tasks" : taskName(t)}
                    <span className="ml-auto text-[12px] text-muted-foreground">
                      {t === null ? cases.length : cases.filter((c) => c.task === t).length}
                    </span>
                  </button>
                ))}
                <div className="my-1 border-t border-border" />
                <label className={item(false)}>
                  <input type="checkbox" checked={open} onChange={(e) => setOpen(e.target.checked)} />
                  Only cases to settle
                </label>
              </div>
            )}
          </Popup>
        </div>
      </div>
      {shown.map((c) => (
        <button
          key={c.id}
          onClick={() => pick(c.id)}
          className={cn(
            "flex cursor-pointer flex-col gap-1 rounded border px-3 py-2 text-left",
            c.id === picked ? "border-primary bg-accent" : "border-border bg-transparent hover:bg-accent",
          )}
        >
          <span className="flex items-center gap-2">
            <span className="text-[12px] text-muted-foreground">{taskName(c.task)}</span>
            {c.open > 0 && (
              <span className="ml-auto rounded-full bg-[#f0883e33] px-2 text-[11px] font-semibold text-[#f0883e]">
                {c.open} to settle
              </span>
            )}
          </span>
          <span className="truncate text-[13px]">{c.id}</span>
          <Score now={c.now} />
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
  if (!detail.can_settle) return null;
  if (!item)
    return (
      <section className="rounded-md border border-border bg-card px-4 py-3 text-[13px] text-[#3fb950]">
        Nothing to settle: every answer is one the key accepts.
      </section>
    );
  return (
    <section className="flex flex-wrap gap-4 rounded-md border border-border bg-card p-4">
      <img
        key={item.id}
        src={`${casePath(run, detail.id)}/items/${enc(item.id)}.png`}
        alt={item.title}
        className="w-full max-w-[460px] flex-[1_1_320px] self-start rounded border border-border bg-white"
        onError={(e) => (e.currentTarget.style.display = "none")}
      />
      <div className="flex min-w-0 flex-[1_1_320px] flex-col gap-3">
        <div className="text-[12px] text-muted-foreground">To settle: 1 of {detail.items.length}</div>
        <div className="text-[15px] font-semibold">{item.title}</div>
        <div className="text-[13px]">{item.key_text}</div>
        <table className="w-full border-collapse text-[13px]">
          <tbody>
            {item.answers.map((a) => (
              <tr key={a.repeat} className="border-t border-border align-top">
                <td className="py-1.5 pr-3 text-muted-foreground">r{a.repeat}</td>
                <td className="py-1.5 pr-3 font-medium whitespace-nowrap">{a.said}</td>
                <td className="py-1.5 text-muted-foreground">{a.reason}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="flex flex-wrap gap-2">
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
        </div>
        {settle.isError && <p className="m-0 text-[13px] text-destructive">{String(settle.error)}</p>}
      </div>
    </section>
  );
}

function Fold({ title, open, children }: { title: string; open?: boolean; children: ReactNode }) {
  return (
    <details open={open} className="rounded-md border border-border bg-card px-4 py-2">
      <summary className="cursor-pointer text-[13px] font-semibold text-muted-foreground">{title}</summary>
      <div className="mt-2 flex flex-col gap-2">{children}</div>
    </details>
  );
}

function CaseView({ run, id }: { run: string; id: string }) {
  const detail = useQuery({ queryKey: ["evals-case", run, id], queryFn: () => get<CaseDetail>(casePath(run, id)) });
  if (detail.isError) return <p className="p-4 text-[13px] text-destructive">{String(detail.error)}</p>;
  if (!detail.data) return <p className="p-4 text-[13px] text-muted-foreground">Drawing the case…</p>;
  const d = detail.data;
  // The figures are redrawn with the key as it is after each choice.
  const version = d.settled.length;
  return (
    <div className="flex flex-col gap-3 p-4">
      <header className="flex flex-col gap-1">
        <h2 className="m-0 text-[16px] font-semibold">{d.task_text}</h2>
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-[13px] text-muted-foreground">
          <span>{d.id}</span>
          {d.key_text && <span>Key: {d.key_text}</span>}
        </div>
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-[13px]">
          {d.repeats.map((r) => (
            <span key={r.repeat}>
              r{r.repeat} <Results results={r.now} /> <span className="text-muted-foreground">{r.did}</span>
            </span>
          ))}
        </div>
        {d.live.map((row) => (
          <div key={row.part ?? ""} className="text-[13px] text-muted-foreground">
            {row.part ? `Part ${row.part}: ` : ""}The real run scored <Results results={[row.live]} />; {row.same} of{" "}
            {row.eval.length} repeats did the same.
          </div>
        ))}
      </header>
      <Settle run={run} detail={d} />
      <Fold title="Whole case: the key and every repeat" open={!d.items.length}>
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
      </Fold>
      <Fold title="What the agent was asked">
        <p className="m-0 text-[13px]">{d.brief}</p>
      </Fold>
      <Fold title="Each repeat's reasons">
        {d.repeats.map((r) => (
          <div key={r.repeat} className="text-[13px]">
            <div className="font-medium">
              r{r.repeat} · ${r.cost_usd.toFixed(2)} · {Math.round(r.seconds)} s · when run <Results results={r.then} />
            </div>
            <ul className="m-0 pl-5 text-muted-foreground">
              {r.reason.split(" / ").map((line, i) => (
                <li key={i}>{line.replace(/^\d+\. /, "")}</li>
              ))}
            </ul>
          </div>
        ))}
      </Fold>
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
