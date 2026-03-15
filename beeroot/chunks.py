"""
beeROOT — Chunks
=================
Git-backed chunk storage for large JSONL datasets.

Reads and writes data as tar.gz files containing JSONL, synced to a
Git repository (GitHub or HuggingFace Dataset repo).

Public API:
    chunks = Chunks.from_yaml("config.yaml")

    # Read next unprocessed chunk
    records = chunks.read_next()           # → List[dict] or []

    # Write results back
    chunks.write(results, chunk_id)        # → tar.gz → git push (batched)

    # Iterate all pending chunks
    for chunk_id, records in chunks.iter_pending():
        ...

Standalone — zero dependency on Balancer, Flow or Endpoint.
"""
from __future__ import annotations

import io
import json
import logging
import shutil
import tarfile
import threading
import time
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import pandas as pd
import yaml

logger = logging.getLogger("beeroot.chunks")


# ── Git manager ───────────────────────────────────────────────────
class _GitManager:
    def __init__(self, token: str, repo_slug: str, branch: str, repo_path: Path):
        from git import Repo, GitCommandError
        self._GitCommandError = GitCommandError
        self.repo_slug  = repo_slug
        self.branch     = branch
        self.repo_path  = Path(repo_path)
        self._lock      = threading.Lock()

        if "huggingface.co" in repo_slug or "/" not in repo_slug.split(".")[0]:
            self._url = f"https://user:{token}@huggingface.co/{repo_slug}.git"
        else:
            self._url = f"https://oauth2:{token}@github.com/{repo_slug}.git"

        self.repo_path.mkdir(parents=True, exist_ok=True)
        if not (self.repo_path / ".git").exists():
            logger.info(f"Cloning {repo_slug}...")
            self.repo = Repo.clone_from(
                self._url, self.repo_path, branch=branch, depth=1, single_branch=True
            )
        else:
            self.repo = Repo(self.repo_path)
            self.repo.remotes.origin.set_url(self._url)

        with self.repo.config_writer() as cfg:
            cfg.set_value("user", "email", "beeroot-bot@automation.com")
            cfg.set_value("user", "name",  "beeROOT")
            cfg.set_value("pull", "rebase", "true")

    def pull(self) -> bool:
        with self._lock:
            try:
                self.repo.git.reset("--hard", "HEAD")
                self.repo.git.clean("-fd")
                self.repo.remotes.origin.pull()
                return True
            except self._GitCommandError as e:
                logger.error(f"Pull failed: {e}")
                return False

    def push(self, files: List[Path], message: str) -> bool:
        with self._lock:
            try:
                self.repo.git.reset("--hard", "HEAD")
                self.repo.git.clean("-fd")
                self.repo.remotes.origin.pull()

                for src in files:
                    dst = self.repo_path / src.name
                    shutil.copy2(src, dst)

                self.repo.git.add(A=True)
                if not self.repo.is_dirty(untracked_files=True):
                    return True

                self.repo.index.commit(message)
                try:
                    self.repo.remotes.origin.push()
                except self._GitCommandError:
                    # Rebase and retry once
                    self.repo.git.fetch("origin")
                    self.repo.git.rebase(f"origin/{self.branch}")
                    self.repo.remotes.origin.push()

                logger.info(f"Pushed: {message}")
                return True
            except self._GitCommandError as e:
                logger.error(f"Push failed: {e}")
                try:
                    self.repo.git.rebase("--abort")
                except Exception:
                    pass
                return False


# ── Chunk I/O ─────────────────────────────────────────────────────
def _read_jsonl_from_tar(path: Path) -> List[Dict]:
    records = []
    with tarfile.open(path, "r:gz") as tar:
        member = next((m for m in tar.getmembers() if m.name.endswith(".jsonl")), None)
        if not member:
            return records
        f = tar.extractfile(member)
        for line in f:
            try:
                records.append(json.loads(line.decode("utf-8")))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
    return records


def _write_jsonl_to_tar(records: List[Dict], path: Path, inner_name: str = "data.jsonl") -> None:
    data  = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
    data_b = data.encode("utf-8")
    with tarfile.open(path, "w:gz") as tar:
        info      = tarfile.TarInfo(name=inner_name)
        info.size = len(data_b)
        tar.addfile(info, io.BytesIO(data_b))


# ── State tracker ─────────────────────────────────────────────────
class _StateManager:
    COLS = ["chunk_id", "status", "total_ok", "errors", "started_at", "finished_at"]

    def __init__(self, csv_path: Path):
        self._path = csv_path
        if csv_path.exists():
            self._df = pd.read_csv(csv_path, dtype=str)
        else:
            self._df = pd.DataFrame(columns=self.COLS)

    def sync(self, input_dir: Path, pattern: str = "chunk_input_*.tar.gz") -> None:
        known = set(self._df["chunk_id"].dropna())
        for p in sorted(input_dir.glob(pattern)):
            cid = p.stem.replace("chunk_input_", "")
            if cid not in known:
                self._df = pd.concat([
                    self._df,
                    pd.DataFrame([{"chunk_id": cid, "status": "pending",
                                   "total_ok": "0", "errors": "[]"}])
                ], ignore_index=True)
        self._save()

    def next_pending(self) -> Optional[str]:
        rows = self._df[self._df["status"] == "pending"]
        if rows.empty:
            return None
        return str(rows.sort_values("chunk_id").iloc[0]["chunk_id"])

    def all_pending(self) -> List[str]:
        rows = self._df[self._df["status"] == "pending"]
        return list(rows.sort_values("chunk_id")["chunk_id"])

    def mark_started(self, cid: str) -> None:
        self._update(cid, status="processing", started_at=time.strftime("%Y-%m-%dT%H:%M:%S"))

    def mark_done(self, cid: str, total_ok: int, errors: List) -> None:
        self._update(cid, status="done", total_ok=str(total_ok),
                     errors=json.dumps(errors),
                     finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"))

    def mark_error(self, cid: str, reason: str) -> None:
        self._update(cid, status="error", errors=json.dumps([reason]))

    def _update(self, cid: str, **kwargs) -> None:
        idx = self._df.index[self._df["chunk_id"] == cid]
        if not idx.empty:
            for k, v in kwargs.items():
                self._df.loc[idx, k] = v
            self._save()

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._df.to_csv(self._path, index=False)


# ── Git batch writer ──────────────────────────────────────────────
class _GitBatcher:
    """
    Accumulates finished chunks and pushes them in batches.
    Respects HF/GitHub push rate limits (default: max 25/min).
    """
    def __init__(self, git: _GitManager, max_rpm: int = 25,
                 batch_every_sec: int = 30, batch_every_n: int = 5):
        self._git       = git
        self._interval  = max(60 / max_rpm, batch_every_sec / batch_every_n)
        self._batch_sec = batch_every_sec
        self._batch_n   = batch_every_n
        self._pending:  List[Tuple[Path, str]] = []  # (file, csv_path)
        self._lock      = threading.Lock()
        self._last_push = 0.0
        self._stop      = threading.Event()
        self._thread    = threading.Thread(target=self._loop, daemon=True, name="beeroot-git-batcher")
        self._thread.start()

    def enqueue(self, output_file: Path, state_csv: Path) -> None:
        with self._lock:
            self._pending.append((output_file, state_csv))

    def flush(self) -> bool:
        with self._lock:
            if not self._pending:
                return True
            files  = [f for f, _ in self._pending]
            csvs   = list({str(c) for _, c in self._pending})
            msg    = f"beeROOT: {len(files)} chunk(s) processed"
            self._pending.clear()
        all_files = files + [Path(c) for c in csvs]
        return self._git.push(all_files, msg)

    def stop(self) -> None:
        self._stop.set()
        self.flush()

    def _loop(self) -> None:
        while not self._stop.is_set():
            time.sleep(1)
            with self._lock:
                n = len(self._pending)
            elapsed = time.time() - self._last_push
            if n >= self._batch_n or (n > 0 and elapsed >= self._batch_sec):
                self.flush()
                self._last_push = time.time()


# ── Chunks ────────────────────────────────────────────────────────
class Chunks:
    """
    Git-backed chunk reader/writer.

    Config YAML:
        chunks:
          git_token:    ${GIT_TOKEN}
          repo_slug:    your-org/your-data-repo
          branch:       main
          local_path:   ./repo_cache
          input_dir:    chunks_input       # folder inside repo
          output_dir:   chunks_output      # folder inside repo
          state_csv:    tracking.csv       # inside repo
          input_pattern: "chunk_input_*.tar.gz"
          min_text_len: 0                  # filter short records (0 = no filter)
          text_fields:  []                 # fields to sum for length check
          batch_every_sec: 30
          batch_every_n:   5
          max_push_rpm:    25
    """

    def __init__(
        self,
        git_token:       str,
        repo_slug:       str,
        branch:          str        = "main",
        local_path:      str | Path = "./repo_cache",
        input_dir:       str        = "chunks_input",
        output_dir:      str        = "chunks_output",
        state_csv:       str        = "tracking.csv",
        input_pattern:   str        = "chunk_input_*.tar.gz",
        min_text_len:    int        = 0,
        text_fields:     List[str]  = None,
        batch_every_sec: int        = 30,
        batch_every_n:   int        = 5,
        max_push_rpm:    int        = 25,
    ):
        self._local      = Path(local_path)
        self._in_dir     = input_dir
        self._out_dir    = output_dir
        self._pattern    = input_pattern
        self._min_len    = min_text_len
        self._txt_fields = text_fields or []

        self._git     = _GitManager(git_token, repo_slug, branch, self._local)
        self._state   = _StateManager(self._local / state_csv)
        self._batcher = _GitBatcher(
            self._git,
            max_rpm          = max_push_rpm,
            batch_every_sec  = batch_every_sec,
            batch_every_n    = batch_every_n,
        )
        self._sync_state()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Chunks":
        import os
        cfg = yaml.safe_load(Path(path).read_text())
        c   = cfg.get("chunks", cfg)

        def env(val):
            if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
                return os.environ.get(val[2:-1], "")
            return val or ""

        return cls(
            git_token       = env(c.get("git_token", "")),
            repo_slug       = c.get("repo_slug", ""),
            branch          = c.get("branch", "main"),
            local_path      = c.get("local_path", "./repo_cache"),
            input_dir       = c.get("input_dir", "chunks_input"),
            output_dir      = c.get("output_dir", "chunks_output"),
            state_csv       = c.get("state_csv", "tracking.csv"),
            input_pattern   = c.get("input_pattern", "chunk_input_*.tar.gz"),
            min_text_len    = int(c.get("min_text_len", 0)),
            text_fields     = c.get("text_fields", []),
            batch_every_sec = int(c.get("batch_every_sec", 30)),
            batch_every_n   = int(c.get("batch_every_n", 5)),
            max_push_rpm    = int(c.get("max_push_rpm", 25)),
        )

    # ── Public API ────────────────────────────────────────────────
    def read_next(self) -> Tuple[Optional[str], List[Dict]]:
        """
        Pull latest from Git, return (chunk_id, records) for the next
        pending chunk. Returns (None, []) if nothing pending.
        """
        self._git.pull()
        self._sync_state()
        cid = self._state.next_pending()
        if not cid:
            return None, []
        return cid, self._load(cid)

    def iter_pending(self) -> Generator[Tuple[str, List[Dict]], None, None]:
        """Yield (chunk_id, records) for every pending chunk."""
        self._git.pull()
        self._sync_state()
        for cid in self._state.all_pending():
            records = self._load(cid)
            if records:
                yield cid, records

    def mark_started(self, chunk_id: str) -> None:
        self._state.mark_started(chunk_id)

    def write(self, results: List[Dict], chunk_id: str,
              errors: Optional[List] = None) -> None:
        """
        Write processed results as output chunk tar.gz.
        Enqueues for batched Git push.
        """
        out_dir = self._local / self._out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"chunk_output_{chunk_id}.tar.gz"
        _write_jsonl_to_tar(results, out_file, f"output_{chunk_id}.jsonl")

        errs = errors or []
        self._state.mark_done(chunk_id, total_ok=len(results), errors=errs)
        self._batcher.enqueue(
            out_file,
            self._local / "tracking.csv",
        )
        logger.info(f"Chunk {chunk_id}: {len(results)} results queued for push")

    def flush(self) -> bool:
        """Force immediate Git push of all pending output."""
        return self._batcher.flush()

    def stop(self) -> None:
        """Flush and stop background batcher."""
        self._batcher.stop()

    # ── Internals ─────────────────────────────────────────────────
    def _sync_state(self) -> None:
        in_dir = self._local / self._in_dir
        if in_dir.exists():
            self._state.sync(in_dir, self._pattern)

    def _load(self, chunk_id: str) -> List[Dict]:
        path = self._local / self._in_dir / f"chunk_input_{chunk_id}.tar.gz"
        if not path.exists():
            # Try original naming convention
            for p in (self._local / self._in_dir).glob(f"*{chunk_id}*.tar.gz"):
                path = p
                break
        if not path.exists():
            return []

        records = _read_jsonl_from_tar(path)

        # Apply min_text_len filter
        if self._min_len > 0 and self._txt_fields:
            before = len(records)
            records = [
                r for r in records
                if sum(len(str(r.get(f, ""))) for f in self._txt_fields) >= self._min_len
            ]
            skipped = before - len(records)
            if skipped:
                logger.info(f"Chunk {chunk_id}: {skipped} records filtered (min_text_len={self._min_len})")

        return records
