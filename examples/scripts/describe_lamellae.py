"""Note the last completed task on every lamella."""

# The runner calls ctx.save() for you after this returns, and asks the user to
# confirm before it starts. Without this flag the loop below would still run and
# still change things in memory, but nothing would be written to disk -- so a
# script that means to persist has to say so.
writes = True


def run(ctx):
    lamellae = ctx.experiment.positions

    changed = 0
    for lamella in lamellae:
        task = lamella.last_completed_task
        note = f"last task: {task.name}" if task else "no tasks completed yet"

        if lamella.description == note:
            continue  # nothing to do, and no point dirtying the save

        # `description` is one of the fields the lamella list and card widgets
        # subscribe to (lamella.events.description), so the GUI updates as this
        # loop runs -- no refresh call and no signal needed.
        lamella.description = note
        changed += 1

    ctx.log(f"updated {changed} of {len(lamellae)}")
    return f"described {changed} of {len(lamellae)} lamellae"
