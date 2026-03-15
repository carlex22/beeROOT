"""
beeROOT — Endpoint
===================
Storage adapter — reads and writes JSONL/tar.gz data.
Local filesystem or Git-backed. Zero processing logic.

The Endpoint is the boundary between your data and the pipeline.
It does not call LLMs, does not know about workflows, and does not
care about the content of the records — only their structure.

Public API:
    ep = Endpoint.from_yaml("config.yaml")

    records = ep.read()               # → List[dict]
    ep.write(records)                 # local or git
    ep.write_result(record)           # append single result

    # Batch iteration:
    for batch in ep.iter_batches(size=100):
        results = process(batch)
        ep.write(results)

Standalone — zero dependency on Balancer, Flow or Chunks.
"""
from __future__ import annotations

import io
import json
import logging
import tarfile
from pathlib import Path
from typing import Dict, Generator, Iterator, List, Optional

import yaml

logger = logging.getLogger("beeroot.endpoint")


# ── Readers ───────────────────────────────────────────────────────
def _read_jsonl(path: Path) -> List[Dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line: {e}")
    return records


def _read_tar_gz(path: Path) -> List[Dict]:
    records = []
    with tarfile.open(path, "r:gz") as tar:
        member = next((m for m in tar.getmembers() if m.name.endswith(".jsonl")), None)
        if not member:
            logger.warning(f"No .jsonl found inside {path.name}")
            return records
        f = tar.extractfile(member)
        for line in f:
            try:
                records.append(json.loads(line.decode("utf-8")))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
    return records


def _read_json(path: Path) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else [data]


def _read_file(path: Path) -> List[Dict]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _read_jsonl(path)
    if path.name.endswith(".tar.gz") or suffix == ".gz":
        return _read_tar_gz(path)
    if suffix == ".json":
        return _read_json(path)
    raise ValueError(f"Unsupported format: {path.name}. Use .jsonl, .json or .tar.gz")


# ── Writers ───────────────────────────────────────────────────────
def _write_jsonl(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_tar_gz(records: List[Dict], path: Path, inner_name: str = "output.jsonl") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data   = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
    data_b = data.encode("utf-8")
    with tarfile.open(path, "w:gz") as tar:
        info      = tarfile.TarInfo(name=inner_name)
        info.size = len(data_b)
        tar.addfile(info, io.BytesIO(data_b))


# ── Endpoint ──────────────────────────────────────────────────────
class Endpoint:
    """
    Storage adapter for JSONL / tar.gz data.

    Config YAML:
        endpoint:
          input:
            path: ./data/input.jsonl      # or .tar.gz or .json
            # Optional filters:
            id_field: id
            limit: 1000                   # max records to read

          output:
            path: ./data/output.jsonl     # or .tar.gz
            mode: append                  # append | overwrite
            format: jsonl                 # jsonl | tar_gz

          # Optional Git push after write:
          git:
            token:     ${GIT_TOKEN}
            repo_slug: your-org/your-repo
            branch:    main
            local_path: ./repo_cache
    """

    def __init__(
        self,
        input_path:   Optional[str | Path] = None,
        output_path:  Optional[str | Path] = None,
        output_mode:  str = "append",
        output_format: str = "jsonl",
        id_field:     str = "id",
        limit:        Optional[int] = None,
        git_config:   Optional[Dict] = None,
    ):
        self._input   = Path(input_path)  if input_path  else None
        self._output  = Path(output_path) if output_path else None
        self._mode    = output_mode
        self._format  = output_format
        self._id      = id_field
        self._limit   = limit
        self._git_cfg = git_config
        self._git     = None

        if git_config:
            from beeroot.chunks import _GitManager
            import os
            token = git_config.get("token", "")
            if token.startswith("${") and token.endswith("}"):
                token = os.environ.get(token[2:-1], "")
            self._git = _GitManager(
                token     = token,
                repo_slug = git_config["repo_slug"],
                branch    = git_config.get("branch", "main"),
                repo_path = Path(git_config.get("local_path", "./repo_cache")),
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Endpoint":
        cfg = yaml.safe_load(Path(path).read_text())
        c   = cfg.get("endpoint", cfg)

        inp = c.get("input", {})
        out = c.get("output", {})
        git = c.get("git", None)

        return cls(
            input_path    = inp.get("path"),
            output_path   = out.get("path"),
            output_mode   = out.get("mode", "append"),
            output_format = out.get("format", "jsonl"),
            id_field      = inp.get("id_field", "id"),
            limit         = inp.get("limit"),
            git_config    = git,
        )

    # ── Read ──────────────────────────────────────────────────────
    def read(self, path: Optional[str | Path] = None) -> List[Dict]:
        """
        Read all records from input.
        Optional path overrides the configured input path.
        """
        src = Path(path) if path else self._input
        if not src:
            raise ValueError("No input path configured.")
        if not src.exists():
            raise FileNotFoundError(f"Input not found: {src}")

        records = _read_file(src)

        if self._limit:
            records = records[:self._limit]

        logger.info(f"Read {len(records)} records from {src.name}")
        return records

    def iter_batches(
        self,
        size: int = 100,
        path: Optional[str | Path] = None,
    ) -> Iterator[List[Dict]]:
        """
        Iterate over records in batches of `size`.
        Useful for large files to avoid loading everything into memory.
        """
        records = self.read(path)
        for i in range(0, len(records), size):
            yield records[i : i + size]

    # ── Write ─────────────────────────────────────────────────────
    def write(
        self,
        records: List[Dict],
        path: Optional[str | Path] = None,
    ) -> Path:
        """
        Write records to output.
        Returns the path where data was written.
        """
        dst = Path(path) if path else self._output
        if not dst:
            raise ValueError("No output path configured.")

        # Overwrite mode — clear existing
        if self._mode == "overwrite" and dst.exists():
            dst.unlink()

        fmt = self._format
        if str(dst).endswith(".tar.gz"):
            fmt = "tar_gz"
        elif dst.suffix == ".jsonl":
            fmt = "jsonl"

        if fmt == "tar_gz":
            _write_tar_gz(records, dst, inner_name=dst.stem + ".jsonl")
        else:
            _write_jsonl(records, dst)

        logger.info(f"Wrote {len(records)} records to {dst.name}")

        if self._git:
            self._git.push([dst], f"beeROOT: write {dst.name}")

        return dst

    def write_result(self, record: Dict, path: Optional[str | Path] = None) -> None:
        """Append a single result record to output."""
        self.write([record], path)

    # ── Utils ─────────────────────────────────────────────────────
    def count(self, path: Optional[str | Path] = None) -> int:
        """Count records without loading all into memory."""
        return len(self.read(path))

    def ids(self, path: Optional[str | Path] = None) -> List[str]:
        """Return list of record IDs."""
        return [str(r.get(self._id, "")) for r in self.read(path)]
