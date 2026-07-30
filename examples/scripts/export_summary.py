"""Export a summary of every lamella to CSV."""

# No flags. This script only reads, so the runner withholds ctx.microscope and
# never saves the experiment -- there is nothing here that can damage anything.
# Read-only is the default; the other two examples opt out of it explicitly.


def run(ctx):
    experiment = ctx.experiment

    # Needs a protocol loaded: this dataframe asks the protocol whether each
    # lamella is complete, so it raises AttributeError on an experiment saved
    # without one. experiment.task_history_dataframe() has no such requirement.
    df = experiment.experiment_summary_dataframe()

    # ctx.path is the experiment directory. Writing there by default keeps
    # exports next to the data they came from.
    out = ctx.path / "lamella-summary.csv"
    df.to_csv(out, index=False)

    # ctx.log goes to the application log, tagged with this script's name.
    ctx.log(f"wrote {len(df)} rows to {out.name}")

    # What you return decides what the user sees:
    #   Path      -> a toast, with an offer to open the containing folder
    #   str       -> a toast with that message
    #   DataFrame -> a table dialog
    # Returning None is legal but looks like the script did nothing, because
    # there is no log console in the application to fall back on.
    return out
