# Agent prompts

One file per agent: `prompts/<agent_name>.md`, where the name matches a key in `config/agents.yaml`. The prompts arrive in Phase 4.

Each file starts with a front-matter header:

```
---
agent: supply_demand_analyst
version: 1
changed: 2026-09-27
---
```

Rules:
- Bump `version` whenever the text changes. Every run logs `agent`, `version`, model and effort in its scratchpad, so results can be compared across versions.
- Keep prompts stable byte-for-byte within a version. They are prompt-cached, so no timestamps or per-run values go in the system prompt.
- Prompts describe how to reason and which tools to call. They never contain numbers the agent should repeat, because every number must come from a tool result.
