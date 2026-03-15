"""
beeROOT — Distributed LLM processing toolkit
==============================================
Four independent modules, each usable standalone:

    Balancer  — adaptive rate-limit gate for LLM APIs
    Flow      — YAML-driven multi-agent workflow
    Chunks    — Git-backed chunk storage (tar.gz/JSONL)
    Endpoint  — local/git data I/O adapter

Usage:
    from beeroot import Balancer, Flow, Chunks, Endpoint

    # Use any module independently:
    balancer = Balancer.from_yaml("config.yaml")
    result   = balancer.call(messages)

    flow   = Flow.from_yaml("workflow.yaml", balancer=balancer)
    result = flow.run(document)

    chunks = Chunks.from_yaml("config.yaml")
    chunk_id, records = chunks.read_next()
    chunks.write(results, chunk_id)

    ep      = Endpoint.from_yaml("config.yaml")
    records = ep.read()
    ep.write(results)
"""

__version__ = "1.0.0"

from .balancer import Balancer, Pressure, Gate, CallResult
from .flow     import Flow, FlowStats, WorkflowEngine
from .chunks   import Chunks
from .endpoint import Endpoint

__all__ = [
    "Balancer", "Pressure", "Gate", "CallResult",
    "Flow", "FlowStats", "WorkflowEngine",
    "Chunks",
    "Endpoint",
]
