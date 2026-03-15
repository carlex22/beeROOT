# 🐝 beeROOT

**Four independent tools for distributed LLM processing.**

```python
pip install beeroot
```

---

## The Four Modules

| Module | Does | Standalone |
|--------|------|-----------|
| `Balancer` | Adaptive rate-limit gate for LLM APIs | ✅ |
| `Flow` | YAML-driven multi-agent workflow | ✅ |
| `Chunks` | Git-backed chunk storage (tar.gz/JSONL) | ✅ |
| `Endpoint` | Local/Git data I/O adapter | ✅ |

Use any combination. Each works without the others.

---

## Quick Start

```python
from beeroot import Balancer, Flow, Chunks, Endpoint
```

### Balancer — rate limit without 429s

```python
balancer = Balancer.from_yaml("config.yaml")
result   = balancer.call([{"role": "user", "content": "Hello"}])
print(result.content)
```

The **gate** serializes LLM call starts — 1 green light at a time.
The **pressure** tracks 429/498/timeout errors and adapts the delay automatically.
Multiple instances self-balance without coordination.

### Flow — multi-agent YAML workflow

```python
flow   = Flow.from_yaml("workflow.yaml", balancer=balancer)
result, stats = flow.run({"id": "doc-001", "text": "Your document..."})
```

The workflow is declared in YAML — no Python needed for agent logic.
Tasks, transitions, and conditions are all in the YAML file.
Supports custom document builders for any input format.

### Chunks — Git-backed batch processing

```python
chunks = Chunks.from_yaml("config.yaml")

for chunk_id, records in chunks.iter_pending():
    chunks.mark_started(chunk_id)
    results = [flow.run(r)[0] for r in records if flow.run(r)[0]]
    chunks.write(results, chunk_id)

chunks.stop()  # flush Git pushes
```

Reads from and writes to a Git repository (GitHub or HuggingFace Dataset).
Output is batched to respect the HuggingFace 30 push/min limit.

### Endpoint — data I/O

```python
ep = Endpoint.from_yaml("config.yaml")

records = ep.read()                         # from .jsonl, .json, .tar.gz
for batch in ep.iter_batches(size=100):     # memory-efficient iteration
    results = process(batch)
    ep.write(results)
```

---

## Config YAML

One YAML file configures all modules:

```yaml
balancer:
  provider:  openrouter
  api_key:   ${API_KEY}
  model:     openai/gpt-oss-120b
  delay_factor: 1.0
  params:
    reasoning_effort: high
    max_completion_tokens: 16000

chunks:
  git_token:  ${GIT_TOKEN}
  repo_slug:  your-org/your-data-repo
  input_dir:  chunks_input
  output_dir: chunks_output

endpoint:
  input:
    path: ./data/input.jsonl
  output:
    path: ./data/output.jsonl
```

See `examples/config.yaml` for the full reference.

---

## Workflow YAML

```yaml
workflow:
  id:          my_flow
  wrapper_key: output
  start:       STEP1
  tasks:
    - id: STEP1
      task_type: reasoning
      model:     openai/gpt-oss-120b
      params:
        reasoning_effort: high
      prompt: "Analyze the document."
      transitions:
        - target: STEP2
          condition: "not phase_failed"
        - target: END_FAILURE

    - id: STEP2
      task_type: final
      prompt: "Serialize as JSON."
      transitions:
        - target: END_SUCCESS
          condition: "json_valid"
        - target: END_FAILURE

    - id: END_SUCCESS
      task_type: end
    - id: END_FAILURE
      task_type: end
```

---

## Architecture

```
                    API errors (429/498/timeout)
                           ↓
    ┌──────────────────────────────────────┐
    │           Pressure                   │  ← tracks error weight
    │  stagger_delay = factor × Σ(errors)  │    in sliding window
    └──────────────┬───────────────────────┘
                   ↓
    ┌──────────────────────────────────────┐
    │              Gate                    │  ← 1 green light at a time
    │   FIFO queue of waiting callers      │    timer starts after prev passes
    └──────────────┬───────────────────────┘
                   ↓
    ┌──────────────────────────────────────┐
    │            Pasture                   │  ← active LLM calls
    │   Unlimited concurrent calls         │    self-regulated by pressure
    └──────────────────────────────────────┘
```

Multiple `Balancer` instances (different processes/machines) share the same
pressure signal from the API — they self-balance without coordination.

---
