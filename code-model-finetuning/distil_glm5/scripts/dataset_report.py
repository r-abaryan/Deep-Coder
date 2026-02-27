from __future__ import annotations

import argparse
import logging
from collections import Counter

from src.distil_glm5.config import load_config
from src.distil_glm5.io_utils import read_jsonl

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rows = read_jsonl(cfg.paths.curated_jsonl)
    if not rows:
        raise SystemExit(
            f"No curated dataset found at {cfg.paths.curated_jsonl}. Run filter_and_dedup.py first."
        )

    task = Counter(r.get("task_type", "unknown") for r in rows)
    lang = Counter(r.get("language", "unknown") for r in rows)

    # CoT detection: check if system prompt contains <think> instruction
    cot_count = sum(
        1 for r in rows
        if any("<think>" in m.get("content", "") for m in r.get("messages", []))
    )

    # OSS-seeded detection
    oss_count = sum(
        1 for r in rows
        if (r.get("provenance") or {}).get("source", "").startswith("oss_seeded")
    )

    # Judge stats
    judged = [r for r in rows if (r.get("quality") or {}).get("judge_passed") is not None]

    # Difficulty breakdown
    difficulty = Counter(
        (r.get("quality") or {}).get("difficulty", {}).get("level", "unknown")
        for r in rows
    )

    print(f"run_name: {cfg.run_name}")
    print(f"rows: {len(rows)}")
    print(f"teacher: {cfg.teacher.model_id}")
    print(f"\nCoT prompts: {cot_count} ({100 * cot_count / max(len(rows), 1):.1f}%)")
    print(f"OSS-seeded: {oss_count} ({100 * oss_count / max(len(rows), 1):.1f}%)")
    if judged:
        print(f"Judge-passed: {len(judged)} (judge: {cfg.judge.model_id})")
    print("\nTask breakdown:")
    for k, v in task.most_common():
        print(f"  {k}: {v}")
    print("\nLanguage breakdown:")
    for k, v in lang.most_common():
        print(f"  {k}: {v}")
    print("\nDifficulty breakdown:")
    for k, v in difficulty.most_common():
        print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())