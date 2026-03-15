"""
beeROOT — Flow
===============
YAML-driven multi-agent LLM workflow engine.

Executes a declarative workflow (tasks + transitions) against a document.
Zero retry logic — returns result dict or None. The caller decides what
to do with None (retry, skip, log).

Public API:
    flow   = Flow.from_yaml("workflow.yaml")
    result = flow.run(document_str, record_id="doc-001")
    stats  = flow.last_stats   # FlowStats after last run

Standalone — zero dependency on Balancer, Chunks or Endpoint.
Pass a Balancer instance to reuse an existing gate/pressure tracker,
or let Flow create its own internal one.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger("beeroot.flow")


# ── Flow stats ────────────────────────────────────────────────────
@dataclass
class FlowStats:
    api_calls:         int = 0
    total_tokens:      int = 0
    errors_429:        int = 0
    errors_498:        int = 0
    errors_other:      int = 0
    loop_count:        int = 0
    last_raw_response: str = ""
    last_reasoning:    str = ""


# ── Workflow engine (YAWL-inspired) ──────────────────────────────
@dataclass
class Transition:
    target:    str
    condition: Optional[str] = None

    _SAFE_BUILTINS = {
        "len": len, "int": int, "float": float, "bool": bool,
        "str": str, "min": min, "max": max, "abs": abs, "round": round,
        "True": True, "False": False, "None": None,
    }

    def evaluate(self, ctx: Dict) -> bool:
        if self.condition is None:
            return True
        try:
            return bool(eval(self.condition, {"__builtins__": {}},
                             {**self._SAFE_BUILTINS, **ctx}))
        except Exception:
            return False


@dataclass
class TaskDef:
    id:           str
    task_type:    str           # reasoning | final | end
    provider:     str           = "openrouter"
    model:        str           = ""
    params:       Dict          = field(default_factory=dict)
    prompt:       str           = ""
    transitions:  List[Transition] = field(default_factory=list)
    stop_tokens:  bool          = False

    @property
    def is_terminal(self) -> bool:
        return self.task_type == "end"

    @property
    def is_final(self) -> bool:
        return self.task_type == "final"


class WorkflowEngine:
    def __init__(self, tasks: Dict[str, TaskDef], start_id: str,
                 wrapper_key: str = "output", id_key: str = "id"):
        self._tasks      = tasks
        self._current    = start_id
        self.wrapper_key = wrapper_key
        self.id_key      = id_key

    @classmethod
    def from_yaml(cls, path: str | Path) -> "WorkflowEngine":
        cfg   = yaml.safe_load(Path(path).read_text())
        wf    = cfg.get("workflow", cfg)
        tasks = {}
        for t in wf.get("tasks", []):
            transitions = [
                Transition(target=tr["target"], condition=tr.get("condition"))
                for tr in t.get("transitions", [])
            ]
            tasks[t["id"]] = TaskDef(
                id          = t["id"],
                task_type   = t.get("task_type", "reasoning"),
                provider    = t.get("provider", "openrouter"),
                model       = t.get("model", ""),
                params      = t.get("params", {}),
                prompt      = t.get("prompt", ""),
                transitions = transitions,
                stop_tokens = t.get("stop_tokens", False),
            )
        return cls(
            tasks       = tasks,
            start_id    = wf.get("start", next(iter(tasks))),
            wrapper_key = wf.get("wrapper_key", "output"),
            id_key      = wf.get("id_key", "id"),
        )

    def start(self) -> TaskDef:
        self._current = list(self._tasks.keys())[0]
        return self._tasks[self._current]

    def advance(self, ctx: Dict) -> TaskDef:
        task = self._tasks[self._current]
        for tr in task.transitions:
            if tr.evaluate(ctx):
                self._current = tr.target
                return self._tasks[tr.target]
        raise KeyError(f"No transition fired from '{self._current}' with ctx={ctx}")


# ── Conversation memory ───────────────────────────────────────────
class Memory:
    def __init__(self, record_id: str):
        self.record_id = record_id
        self._entries: List[Dict] = []
        self._extraction: Optional[Dict] = None

    def add(self, role: str, content: str) -> None:
        self._entries.append({"role": role, "content": content})

    def to_messages(self) -> List[Dict]:
        return list(self._entries)

    def set_extraction(self, data: Dict) -> None:
        self._extraction = data

    def extraction(self) -> Optional[Dict]:
        return self._extraction


# ── Loop detector ─────────────────────────────────────────────────
class LoopDetected(Exception):
    def __init__(self, safe_content: str = ""):
        self.safe_content = safe_content
        super().__init__("Loop detected in LLM output")


def _detect_loop(text: str, ngram: int = 8, threshold: float = 0.4) -> bool:
    if not text or len(text) < 200:
        return False
    words = text.split()
    if len(words) < ngram * 2:
        return False
    ngrams = [" ".join(words[i:i+ngram]) for i in range(len(words)-ngram+1)]
    unique = len(set(ngrams))
    ratio  = 1 - (unique / len(ngrams))
    return ratio >= threshold


# ── JSON extractor ────────────────────────────────────────────────
def _extract_json(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r'\{.*\}', text, re.DOTALL)
    return m.group(0) if m else None


# ── Flow ──────────────────────────────────────────────────────────
class Flow:
    """
    YAML-driven multi-agent flow.

    Config YAML:
        workflow:
          id: my_flow
          wrapper_key: output
          id_key: id
          start: STEP1

          tasks:
            - id: STEP1
              task_type: reasoning
              provider: openrouter
              model: openai/gpt-oss-120b
              params:
                reasoning_effort: high
                max_completion_tokens: 16000
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
    """

    def __init__(
        self,
        engine:        WorkflowEngine,
        balancer:      Optional[Any] = None,   # beeroot.Balancer or None
        schema:        Optional[Dict] = None,
        system_prompt: str = "",
        param_overrides: Optional[Dict] = None,
        build_document:  Optional[Callable[[Dict], str]] = None,
    ):
        self._engine        = engine
        self._balancer      = balancer
        self._schema        = schema
        self._system_prompt = system_prompt
        self._overrides     = param_overrides or {}
        self._build_doc     = build_document or (lambda r: json.dumps(r, ensure_ascii=False, indent=2))
        self.last_stats     = FlowStats()

    @classmethod
    def from_yaml(
        cls,
        workflow_yaml:  str | Path,
        schema_json:    Optional[str | Path] = None,
        system_role_md: Optional[str | Path] = None,
        balancer:       Optional[Any] = None,
        build_document: Optional[Callable] = None,
    ) -> "Flow":
        engine  = WorkflowEngine.from_yaml(workflow_yaml)
        schema  = None
        if schema_json:
            schema = json.loads(Path(schema_json).read_text())
        system  = ""
        if system_role_md:
            system = Path(system_role_md).read_text()
        return cls(engine=engine, balancer=balancer, schema=schema,
                   system_prompt=system, build_document=build_document)

    def run(
        self,
        document:        str | Dict,
        record_id:       str = "record",
        param_overrides: Optional[Dict] = None,
    ) -> Tuple[Optional[Dict], FlowStats]:
        """
        Execute the workflow on a document.
        Returns (result_dict_or_None, FlowStats).
        One shot — no retry. Caller decides what to do with None.
        """
        if isinstance(document, dict):
            record_id = str(document.get(self._engine.id_key, record_id))
            document  = self._build_doc(document)

        stats   = FlowStats()
        memory  = Memory(record_id)
        overrides = {**self._overrides, **(param_overrides or {})}

        ctx = {
            "phase_failed": False, "json_valid": False,
            "audit_passed": False, "loop_count": 0,
            "reasoning_chars": 0,  "current_phase": "",
        }

        task = self._engine.start()
        while not task.is_terminal:
            ctx["current_phase"] = task.id
            result, updated_ctx = self._execute_task(
                task, document, memory, stats, overrides
            )
            ctx.update(updated_ctx)

            try:
                task = self._engine.advance(ctx)
            except KeyError:
                self.last_stats = stats
                return None, stats

        self.last_stats = stats
        if task.id.endswith("FAILURE") or task.id.endswith("failure"):
            return None, stats

        extraction = memory.extraction()
        if not extraction:
            return None, stats

        return {self._engine.wrapper_key: extraction}, stats

    # ── Task execution ────────────────────────────────────────────
    def _execute_task(
        self,
        task:      TaskDef,
        document:  str,
        memory:    Memory,
        stats:     FlowStats,
        overrides: Dict,
    ) -> Tuple[Optional[str], Dict]:

        if task.task_type == "end":
            return None, {}

        if task.task_type in ("reasoning", "reasoning_content"):
            return self._run_reasoning(task, document, memory, stats, overrides)

        if task.task_type == "final":
            return self._run_final(task, memory, stats, overrides)

        return None, {"phase_failed": True}

    def _run_reasoning(self, task, document, memory, stats, overrides):
        if not memory.to_messages():
            # First message — include document
            if self._system_prompt:
                memory.add("system", self._system_prompt)
            memory.add("user", f"## DOCUMENT\n{document}\n\n## INSTRUCTION\n{task.prompt}")
        else:
            memory.add("user", task.prompt)

        content, reasoning = self._call(task, memory, stats, overrides)
        if content is None and reasoning is None:
            return None, {"phase_failed": True}

        text = reasoning or content or ""
        if _detect_loop(text):
            stats.loop_count += 1
            stats.last_reasoning = text[:8000]
            return None, {"phase_failed": True, "loop_count": stats.loop_count}

        stats.last_reasoning = text[:8000] if text else ""
        memory.add("assistant", text)
        return text, {"phase_failed": False, "reasoning_chars": len(text)}

    def _run_final(self, task, memory, stats, overrides):
        memory.add("user", task.prompt or "Serialize your analysis as valid JSON.")
        content, _ = self._call(task, memory, stats, overrides,
                                schema=self._schema)
        if content is None:
            return None, {"phase_failed": True, "json_valid": False}

        if _detect_loop(content):
            stats.loop_count += 1
            stats.last_reasoning = content[:8000]
            return None, {"phase_failed": True, "json_valid": False,
                          "loop_count": stats.loop_count}

        # Try to parse JSON
        raw = _extract_json(content) or content
        try:
            parsed = json.loads(raw)
            memory.set_extraction(parsed)
            memory.add("assistant", content)
            return content, {"json_valid": True, "phase_failed": False}
        except json.JSONDecodeError:
            return None, {"json_valid": False, "phase_failed": True}

    def _call(
        self,
        task:      TaskDef,
        memory:    Memory,
        stats:     FlowStats,
        overrides: Dict,
        schema:    Optional[Dict] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Single LLM call — through gate if balancer available."""
        if self._balancer is None:
            raise RuntimeError(
                "Flow has no Balancer. Pass a Balancer via Flow(..., balancer=b) "
                "or Flow.from_yaml(..., balancer=b)."
            )

        params  = {**task.params, **overrides}
        payload = {
            "model":    task.model or self._balancer._model,
            "messages": memory.to_messages(),
            **params,
        }
        if schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": schema},
            }

        result = self._balancer.call(memory.to_messages(),
                                     param_overrides=params)
        stats.api_calls    += 1
        stats.total_tokens += result.tokens

        if result.error:
            if "429" in (result.error or ""):
                stats.errors_429 += 1
            elif "498" in (result.error or ""):
                stats.errors_498 += 1
            else:
                stats.errors_other += 1

        if result.content:
            stats.last_raw_response = result.content

        if not result.ok:
            return None, None

        return result.content, None
