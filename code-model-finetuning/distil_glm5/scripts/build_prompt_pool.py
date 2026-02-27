from __future__ import annotations

import argparse
import logging

from src.distil_glm5.config import load_config
from src.distil_glm5.io_utils import write_jsonl
from src.distil_glm5.prompts import build_prompt_pool

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
    pool = build_prompt_pool(
        total=cfg.prompt_pool.total_prompts,
        task_mix=cfg.prompt_pool.task_mix,
        languages=cfg.prompt_pool.languages,
        seed=cfg.generation.seed,
        cot_ratio=cfg.prompt_pool.cot_ratio,
        oss_seed_path=cfg.prompt_pool.oss_seed_path,
        oss_ratio=cfg.prompt_pool.oss_ratio,
    )

    rows = [ex.to_json() for ex in pool]
    write_jsonl(cfg.paths.prompts_jsonl, rows)
    logger.info("Wrote %d prompts to %s", len(rows), cfg.paths.prompts_jsonl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())