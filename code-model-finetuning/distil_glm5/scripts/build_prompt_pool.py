from __future__ import annotations

import argparse

from distil_glm5.config import load_config
from distil_glm5.io_utils import write_jsonl
from distil_glm5.prompts import build_prompt_pool


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    pool = build_prompt_pool(
        total=cfg.prompt_pool.total_prompts,
        task_mix=cfg.prompt_pool.task_mix,
        languages=cfg.prompt_pool.languages,
        seed=cfg.generation.seed,
    )

    rows = [ex.to_json() for ex in pool]
    write_jsonl(cfg.paths.prompts_jsonl, rows)
    print(f"Wrote {len(rows)} prompts to {cfg.paths.prompts_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

