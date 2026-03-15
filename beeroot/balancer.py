"""
beeROOT — Balancer
===================
Adaptive LLM call balancer with backpressure and sequential gate.

The gate serializes LLM call *starts* — only one call begins at a time,
with a delay derived from recent API error signals (429, 498, timeout).
Calls already in flight are not interrupted — the pasture is unlimited.

Public API:
    balancer = Balancer.from_yaml("config.yaml")
    result   = balancer.call(messages, **override_params)
    result   = await balancer.acall(messages, **override_params)  # async

Standalone — zero dependency on Flow, Chunks or Endpoint.
"""
from __future__ import annotations

import queue
import threading
import time
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
import requests

logger = logging.getLogger("beeroot.balancer")


# ── Pressure tracker ─────────────────────────────────────────────
@dataclass
class _PressureEvent:
    ts: float
    weight: float


class Pressure:
    """
    Sliding-window error pressure tracker.
    Each API error type has a weight. Stagger delay = factor × total_weight.
    Decays on clean success.
    """
    ERROR_WEIGHTS: Dict[str, float] = {
        "429":        2.0,
        "498":        1.5,
        "503":        1.0,
        "timeout":    0.3,
        "conn_error": 0.3,
        "other":      0.2,
    }

    def __init__(self, window_sec: int = 120):
        self._lock   = threading.Lock()
        self._events: deque = deque()
        self._window = window_sec

    def add(self, error_type: str) -> None:
        weight = self.ERROR_WEIGHTS.get(error_type, 0.2)
        with self._lock:
            self._events.append(_PressureEvent(time.time(), weight))

    def decay(self, factor: float = 0.75) -> None:
        with self._lock:
            for e in self._events:
                e.weight *= factor

    def total(self) -> float:
        cutoff = time.time() - self._window
        with self._lock:
            self._events = deque(e for e in self._events if e.ts >= cutoff)
            return sum(e.weight for e in self._events)

    def stagger_delay(self, factor: float = 1.0) -> float:
        return factor * self.total()


# ── Gate dispatcher ───────────────────────────────────────────────
class Gate:
    """
    Sequential call gate — 1 green light at a time.
    Workers queue here; the delay timer for each only starts
    after the previous one has passed through.
    """
    def __init__(self, pressure: Pressure, delay_factor: float = 1.0):
        self._queue       = queue.Queue()
        self._pressure    = pressure
        self._delay_factor = delay_factor
        self._stop        = threading.Event()
        self._thread      = threading.Thread(
            target=self._dispatcher, daemon=True, name="beeroot-gate"
        )
        self._thread.start()

    def acquire(self, stop_event: Optional[threading.Event] = None) -> bool:
        """Block until this caller receives the green light. Returns False if stopped."""
        go = threading.Event()
        killed = stop_event or threading.Event()
        self._queue.put((go, killed))
        while not go.is_set():
            if killed.is_set() or self._stop.is_set():
                return False
            go.wait(timeout=0.5)
        return not killed.is_set()

    def stop(self) -> None:
        self._stop.set()

    def _dispatcher(self) -> None:
        while not self._stop.is_set():
            try:
                go, killed = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if killed.is_set() or self._stop.is_set():
                go.set()
                self._queue.task_done()
                continue

            delay = self._pressure.stagger_delay(self._delay_factor)
            if delay > 0.05:
                deadline = time.time() + delay
                while time.time() < deadline:
                    if killed.is_set() or self._stop.is_set():
                        break
                    time.sleep(min(0.2, deadline - time.time()))

            go.set()
            self._queue.task_done()


# ── Provider adapters ─────────────────────────────────────────────
@dataclass
class CallResult:
    content:    Optional[str]
    ok:         bool
    status:     int
    error:      Optional[str]
    tokens:     int
    latency_ms: int


class _OpenRouterAdapter:
    BASE = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str):
        self._key = api_key

    def complete(self, payload: Dict, timeout: int = 300, pressure: Optional[Pressure] = None) -> CallResult:
        t0 = time.time()
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
        }
        try:
            r = requests.post(self.BASE, json=payload, headers=headers,
                              timeout=timeout, stream=False)
            ms = int((time.time() - t0) * 1000)
            if r.status_code == 200:
                data    = r.json()
                content = data["choices"][0]["message"]["content"]
                tokens  = data.get("usage", {}).get("total_tokens", 0)
                return CallResult(content=content, ok=True, status=200,
                                  error=None, tokens=tokens, latency_ms=ms)
            else:
                err = f"HTTP {r.status_code}"
                if pressure:
                    pressure.add("429" if r.status_code == 429 else
                                 "498" if r.status_code == 498 else "other")
                return CallResult(content=None, ok=False, status=r.status_code,
                                  error=err, tokens=0, latency_ms=ms)
        except requests.Timeout:
            if pressure:
                pressure.add("timeout")
            return CallResult(content=None, ok=False, status=0,
                              error="Timeout", tokens=0, latency_ms=int((time.time()-t0)*1000))
        except Exception as e:
            if pressure:
                pressure.add("conn_error")
            return CallResult(content=None, ok=False, status=0,
                              error=str(e), tokens=0, latency_ms=int((time.time()-t0)*1000))


class _GroqAdapter:
    BASE = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str):
        self._key = api_key

    def complete(self, payload: Dict, timeout: int = 300, pressure: Optional[Pressure] = None) -> CallResult:
        t0 = time.time()
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
        }
        # Groq doesn't support reasoning_effort
        p = {k: v for k, v in payload.items() if k != "reasoning_effort"}
        try:
            r = requests.post(self.BASE, json=p, headers=headers, timeout=timeout)
            ms = int((time.time() - t0) * 1000)
            if r.status_code == 200:
                data    = r.json()
                content = data["choices"][0]["message"]["content"]
                tokens  = data.get("usage", {}).get("total_tokens", 0)
                return CallResult(content=content, ok=True, status=200,
                                  error=None, tokens=tokens, latency_ms=ms)
            else:
                if pressure:
                    pressure.add("429" if r.status_code == 429 else "other")
                return CallResult(content=None, ok=False, status=r.status_code,
                                  error=f"HTTP {r.status_code}", tokens=0, latency_ms=ms)
        except requests.Timeout:
            if pressure:
                pressure.add("timeout")
            return CallResult(content=None, ok=False, status=0,
                              error="Timeout", tokens=0, latency_ms=int((time.time()-t0)*1000))
        except Exception as e:
            if pressure:
                pressure.add("conn_error")
            return CallResult(content=None, ok=False, status=0,
                              error=str(e), tokens=0, latency_ms=int((time.time()-t0)*1000))


_ADAPTERS = {
    "openrouter": _OpenRouterAdapter,
    "groq":       _GroqAdapter,
}


# ── Balancer ──────────────────────────────────────────────────────
class Balancer:
    """
    Adaptive LLM call balancer.

    Config YAML:
        provider:      openrouter          # openrouter | groq
        api_key:       ${API_KEY}          # or set via env
        model:         openai/gpt-oss-120b
        delay_factor:  1.0                 # stagger multiplier
        pressure_window_sec: 120           # error tracking window
        timeout:       300                 # per-call HTTP timeout (s)
        params:                            # default payload params
          reasoning_effort: high
          max_completion_tokens: 16000
    """

    def __init__(
        self,
        provider:     str,
        api_key:      str,
        model:        str,
        delay_factor: float = 1.0,
        pressure_window_sec: int = 120,
        timeout:      int   = 300,
        default_params: Optional[Dict] = None,
    ):
        adapter_cls = _ADAPTERS.get(provider)
        if not adapter_cls:
            raise ValueError(f"Unknown provider '{provider}'. Supported: {list(_ADAPTERS)}")

        self._adapter     = adapter_cls(api_key)
        self._model       = model
        self._timeout     = timeout
        self._default_params = default_params or {}
        self.pressure     = Pressure(pressure_window_sec)
        self.gate         = Gate(self.pressure, delay_factor)

        logger.info(f"Balancer ready — provider={provider} model={model} "
                    f"delay_factor={delay_factor}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Balancer":
        """Load Balancer from a YAML config file."""
        import os
        cfg = yaml.safe_load(Path(path).read_text())
        b   = cfg.get("balancer", cfg)  # supports top-level or nested

        # Resolve env vars in api_key
        api_key = b.get("api_key", "")
        if api_key.startswith("${") and api_key.endswith("}"):
            api_key = os.environ.get(api_key[2:-1], "")

        return cls(
            provider     = b.get("provider", "openrouter"),
            api_key      = api_key,
            model        = b.get("model", "openai/gpt-oss-120b"),
            delay_factor = float(b.get("delay_factor", 1.0)),
            pressure_window_sec = int(b.get("pressure_window_sec", 120)),
            timeout      = int(b.get("timeout", 300)),
            default_params = b.get("params", {}),
        )

    def call(
        self,
        messages:      List[Dict],
        stop_event:    Optional[threading.Event] = None,
        param_overrides: Optional[Dict] = None,
    ) -> CallResult:
        """
        Make a single LLM call through the gate.
        Blocks at the gate until it's this caller's turn.
        """
        if not self.gate.acquire(stop_event):
            return CallResult(content=None, ok=False, status=0,
                              error="cancelled", tokens=0, latency_ms=0)

        params = {**self._default_params, **(param_overrides or {})}
        payload = {"model": self._model, "messages": messages, **params}

        result = self._adapter.complete(payload, timeout=self._timeout,
                                        pressure=self.pressure)

        if result.ok:
            self.pressure.decay(0.75)
        else:
            logger.warning(f"Call failed: {result.error} (status={result.status})")

        return result

    def stop(self) -> None:
        """Shut down the gate dispatcher thread."""
        self.gate.stop()
