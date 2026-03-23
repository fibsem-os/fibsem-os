---
name: linear
description: Look up a Linear issue (FIB-NNN), plan the implementation, then implement and commit it
argument-hint: [issue-number]
disable-model-invocation: true
---

You are implementing Linear issue FIB-$ARGUMENTS in the fibsem repository.

## Step 1 — Fetch the issue
Use the `mcp__linear-server__get_issue` tool to look up issue FIB-$ARGUMENTS.
Read the title, description, and any comments carefully.

## Step 2 — Plan first
Enter plan mode (`EnterPlanMode`) and design a complete implementation plan before writing any code.
- Explore the relevant parts of the codebase to understand what needs to change
- Identify all files that need to be modified
- Consider edge cases and existing patterns in the codebase

Get the user's approval on the plan before proceeding.

## Step 3 — Implement
After plan approval, implement the changes following the existing code patterns and conventions.

## Step 4 — Commit
Create a git commit with a message that:
- Starts with the type tag matching the issue type: `[fix]`, `[feat]`, `[refactor]`, `[docs]`, etc.
- Includes `[FIB-$ARGUMENTS]`
- Has a short descriptive summary

Example: `[fix][FIB-$ARGUMENTS] correct milling stage drag-drop reorder`

Do NOT include "Claude" or "Co-Authored-By" in the commit message (per project guidelines).
