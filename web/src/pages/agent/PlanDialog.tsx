import { Dialog, DialogContent } from "@/components/ui/dialog";
import { useFileRead } from "@/lib/files";
import { ago } from "@/lib/format";
import { Markdown } from "@/lib/markdown";
import type { AgentPlan } from "@/lib/types";

/** A picture in the plan, a path in the experiment folder. */
function Picture({ path, caption }: { path: string; caption: string }) {
  const read = useFileRead(path, 480);
  const p = read.data?.preview;
  return (
    <figure className="m-0 flex w-[200px] flex-col gap-1">
      <div className="flex aspect-[3/2] items-center justify-center overflow-hidden rounded bg-black" title={path}>
        {p ? (
          <img
            className="block size-full object-contain"
            alt={caption || path}
            src={`data:image/jpeg;base64,${p.image_b64_jpeg}`}
          />
        ) : (
          <span className="text-[12px] text-muted-foreground">{read.error ? "Not readable" : ""}</span>
        )}
      </div>
      <figcaption className="text-[12px] text-muted-foreground">{caption || path.split("/").pop()}</figcaption>
    </figure>
  );
}

/** plan.md, drawn: headings, ticked steps, progress with its pictures. Read only. */
export function PlanDialog({
  plan,
  open,
  onOpenChange,
}: {
  plan: AgentPlan | null | undefined;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const by = plan?.updated_by === "operator" ? "by you" : plan?.updated_by === "agent" ? "by the agent" : "";
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent title="Plan" className="w-[min(760px,calc(100vw-32px))]">
        {plan?.updated && (
          <p className="m-0 border-b border-border px-4 py-2 text-[12px] text-muted-foreground">
            plan.md · updated {by} {ago(plan.updated)}
          </p>
        )}
        <div className="min-h-0 overflow-y-auto px-6 pt-2 pb-6 text-[13.5px] leading-[1.55]">
          {plan?.markdown ? (
            <Markdown
              doc
              // The file's own "# Plan" is the dialog's title.
              text={plan.markdown.replace(/^# Plan\s*\n/, "")}
              picture={(path, caption) => <Picture path={path} caption={caption} />}
            />
          ) : (
            <p className="text-muted-foreground">The agent hasn't written its plan yet.</p>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
