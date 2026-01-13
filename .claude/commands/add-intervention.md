---
allowed-tools: Read, Edit, Write, Bash, Grep
description: Add a new intervention type to the sweep pipeline
---

Add support for a new intervention type: **$ARGUMENTS**

## Current intervention types
!grep -E "choices=.*intervention_type" examples/regret/train.py

## Steps

### 1. Check the intervention class exists in pyreft/interventions.py
- Verify it has `debug` logging (metrics dict with base_norm, diff_norm, b_norm, delta_base_ratio)
- Add debug logging if missing (see LoreftIntervention or DireftIntervention as examples)
- Note any special kwargs it requires (e.g., `add_bias` for NodireftIntervention)

### 2. Update examples/regret/train.py
- Add import from pyreft
- Add to `--intervention_type` choices in argparse
- Add elif branch in intervention selection logic (around line 316)
- Handle any special kwargs in `intervention_kwargs` dict

### 3. Update examples/regret/scripts/launch_sweep.sh
- Add `--with-{name}` flag to usage comments
- Add `WITH_{NAME}=false` variable
- Add case in arg parsing: `--with-{name}) WITH_{NAME}=true; echo "=== INCLUDING {NAME} EXPERIMENTS ===" ;;`
- Add job count calculation
- Add echo line for job summary
- Add sweep loop section (copy DiReFT or NoDiReFT pattern, change INTERVENTION_TYPE)

### 4. Update examples/regret/CLAUDE.md
- Add to "Intervention Types" section with formula
- Add to "Launch Script Flags" section
- Add test command to "Quick Start" section

### 5. Test
Run a quick Python test to verify the intervention works:
```python
from pyreft import {InterventionClass}
import torch
intervention = {InterventionClass}(embed_dim=2048, low_rank_dimension=4, debug=True, ...)
output = intervention(torch.randn(1, 10, 2048))
```
