import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import { KnowledgeDialog, type KnowledgeTab } from "@/pages/agent/Knowledge";

const OPENS: [KnowledgeTab, string, string][] = [
  ["playbooks", "Playbooks", "How each kind of job runs."],
  ["skills", "Skills", "How to do a kind of task."],
  [
    "agents",
    "Subagents",
    "Second opinions and analysis, each in a fresh context.",
  ],
  [
    "connectors",
    "Connectors",
    "MCP servers the agent may use, such as the lab notebook.",
  ],
  [
    "tools",
    "Tools",
    "What the agent may do at each level, and what it never touches.",
  ],
  [
    "entries",
    "Entries",
    "What the lab knows about its instrument, samples and runs (with the app running).",
  ],
];

/** What shapes the agent's work on this PC: the Agent page's Knowledge
 * window, opened on the part chosen. Through the gateway, its playbooks,
 * skills, subagents and connectors work with no app running. */
export function Customise() {
  const [open, setOpen] = useState<KnowledgeTab | null>(null);
  return (
    <Card className="flex flex-col gap-1">
      <CardTitle className="mb-1.5">The agent's setup</CardTitle>
      {OPENS.map(([tab, label, about]) => (
        <div
          key={tab}
          className="grid grid-cols-1 items-center gap-1 border-t border-border py-2 text-[13px] sm:grid-cols-[160px_minmax(0,1fr)_auto] sm:gap-3"
        >
          <span>{label}</span>
          <span className="text-muted-foreground">{about}</span>
          <Button onClick={() => setOpen(tab)}>Open</Button>
        </div>
      ))}
      <KnowledgeDialog
        open={open !== null}
        onOpenChange={(o) => !o && setOpen(null)}
        start={open ?? "playbooks"}
      />
    </Card>
  );
}
