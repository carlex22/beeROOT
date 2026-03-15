"""
beeROOT — example usage
========================
Complete pipeline in ~30 lines.
Each module is independent — use only what you need.
"""

# ── Example 1: Just the balancer ─────────────────────────────────
from beeroot import Balancer

balancer = Balancer.from_yaml("config.yaml")
result = balancer.call([
    {"role": "user", "content": "Summarize this document: ..."}
])
print(result.content)


# ── Example 2: Balancer + Flow ───────────────────────────────────
from beeroot import Balancer, Flow

balancer = Balancer.from_yaml("config.yaml")
flow     = Flow.from_yaml(
    "workflow_example.yaml",
    schema_json    = "schema.json",   # optional structured output
    system_role_md = "system_role.md",
    balancer       = balancer,
)

result, stats = flow.run({"id": "doc-001", "text": "Your document here..."})
print(result)     # {"output": {...}}
print(stats)      # FlowStats(api_calls=3, total_tokens=12500, ...)


# ── Example 3: Chunks (Git-backed batch processing) ───────────────
from beeroot import Balancer, Flow, Chunks

balancer = Balancer.from_yaml("config.yaml")
flow     = Flow.from_yaml("workflow_example.yaml", balancer=balancer)
chunks   = Chunks.from_yaml("config.yaml")

for chunk_id, records in chunks.iter_pending():
    chunks.mark_started(chunk_id)
    results, errors = [], []

    for record in records:
        output, stats = flow.run(record)
        if output:
            results.append(output)
        else:
            errors.append(record.get("id"))

    chunks.write(results, chunk_id, errors=errors)

chunks.stop()  # flush remaining Git pushes


# ── Example 4: Endpoint (local I/O) ──────────────────────────────
from beeroot import Balancer, Flow, Endpoint

balancer = Balancer.from_yaml("config.yaml")
flow     = Flow.from_yaml("workflow_example.yaml", balancer=balancer)
ep       = Endpoint.from_yaml("config.yaml")

for batch in ep.iter_batches(size=50):
    for record in batch:
        output, stats = flow.run(record)
        if output:
            ep.write_result(output)


# ── Example 5: Custom document builder ───────────────────────────
from beeroot import Flow, Balancer

def my_builder(record: dict) -> str:
    """Format a record into the document string sent to the LLM."""
    return f"## {record['title']}\n\n{record['body']}"

balancer = Balancer.from_yaml("config.yaml")
flow     = Flow.from_yaml(
    "workflow_example.yaml",
    balancer       = balancer,
    build_document = my_builder,   # plug your domain builder here
)

result, stats = flow.run({"id": "1", "title": "My Doc", "body": "Content..."})
