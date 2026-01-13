---
allowed-tools: Bash(git:*)
description: Commit all changes and push to remote
---

Commit and push the following changes:

## Git Status
!git status --short

## Git Diff (staged and unstaged)
!git diff HEAD --stat

## Instructions
1. Stage all changes with `git add -A`
2. Commit with a clear, descriptive message (use "$ARGUMENTS" if provided, otherwise generate one based on the diff)
3. Push to the current branch

Keep the commit message concise but informative.
