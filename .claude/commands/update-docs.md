---
allowed-tools: Read, Edit, Grep
description: Update CLAUDE.md after making changes
---

Update the documentation in examples/regret/CLAUDE.md based on recent changes.

## Current CLAUDE.md
!head -100 examples/regret/CLAUDE.md

## Recent git changes
!git diff --name-only HEAD~3

## Instructions

Review what changed and update CLAUDE.md accordingly:

1. **New intervention types** → Add to "Intervention Types" section with formula
2. **New sweep flags** → Add to "Launch Script Flags" section
3. **New train.py arguments** → Document in relevant section
4. **New findings/experiments** → Add to "Key Finding" or create new section
5. **Bug fixes** → Add to "Known Issues" if relevant
6. **New scripts** → Add to "Key Files" section

Context for update: $ARGUMENTS

Keep the documentation concise and practical. Focus on what users need to know to run experiments.
