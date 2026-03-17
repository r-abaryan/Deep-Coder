from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def ensure_parent_dir(path: str | Path) -> Path:
    p = path if isinstance(path, Path) else Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    """Write rows to a JSONL file (overwrite)."""
    p = ensure_parent_dir(path)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return p


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    """Append rows to an existing JSONL file (create if missing)."""
    p = ensure_parent_dir(path)
    with p.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return p


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = path if isinstance(path, Path) else Path(path)
    out: list[dict[str, Any]] = []
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out