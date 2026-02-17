from __future__ import annotations

import argparse
from collections import Counter

from distil_glm5.config import load_config
from distil_glm5.io_utils import read_jsonl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rows = read_jsonl(cfg.paths.curated_jsonl)
    if not rows:
        raise SystemExit(f"No curated dataset found at {cfg.paths.curated_jsonl}. Run filter_and_dedup.py first.")

    task = Counter(r.get("task_type", "unknown") for r in rows)
    lang = Counter(r.get("language", "unknown") for r in rows)

    print(f"run_name: {cfg.run_name}")
    print(f"rows: {len(rows)}")
    print(f"teacher: {cfg.teacher.model_id}")
    print("\nTask breakdown:")
    for k, v in task.most_common():
        print(f"  {k}: {v}")
    print("\nLanguage breakdown:")
    for k, v in lang.most_common():
        print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

