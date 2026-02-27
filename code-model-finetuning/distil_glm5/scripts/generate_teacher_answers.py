from __future__ import annotations

import argparse
import concurrent.futures as cf
import logging
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.distil_glm5.config import load_config
from src.distil_glm5.io_utils import read_jsonl, write_jsonl, append_jsonl
from src.distil_glm5.teacher_client import OpenAICompatChatClient, build_chat_payload

logger = logging.getLogger(__name__)

# How often to flush in-memory results to disk (number of completed rows).
_CHECKPOINT_EVERY = 200


def _one(
    *,
    client: OpenAICompatChatClient,
    teacher_model: str,
    prompt_row: dict[str, Any],
    gen: dict[str, Any],
) -> dict[str, Any]:
    payload = build_chat_payload(
        model=teacher_model,
        messages=prompt_row["messages"],
        temperature=gen["temperature"],
        top_p=gen["top_p"],
        max_tokens=gen["max_tokens"],
        n=gen["n"],
        stop=gen["stop"],
        seed=gen["seed"],
    )
    resp = client.chat_completions(payload)
    return {
        "schema_version": 1,
        "id": prompt_row["id"],
        "task_type": prompt_row["task_type"],
        "language": prompt_row["language"],
        "messages": prompt_row["messages"],
        "teacher": teacher_model,
        "gen": gen,
        "output_text": resp.content,
        "raw": resp.raw,
        "meta": prompt_row.get("meta", {}),
        "ts": time.time(),
    }


def _load_completed_ids(raw_path: Path) -> set[str]:
    """Load IDs of already-completed rows from a previous (partial) run."""
    if not raw_path.exists():
        return set()
    rows = read_jsonl(raw_path)
    ids = {r["id"] for r in rows if "id" in r}
    if ids:
        logger.info("Resuming: found %d already-completed rows in %s", len(ids), raw_path)
    return ids


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from existing partial output (skip already-completed IDs)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    prompts = read_jsonl(cfg.paths.prompts_jsonl)
    if not prompts:
        raise SystemExit(
            f"No prompts found at {cfg.paths.prompts_jsonl}. Run build_prompt_pool.py first."
        )

    raw_path = Path(cfg.paths.raw_jsonl)

    # Resume support: skip prompts whose IDs already appear in the output file.
    completed_ids: set[str] = set()
    if args.resume:
        completed_ids = _load_completed_ids(raw_path)
        prompts = [p for p in prompts if p["id"] not in completed_ids]
        if not prompts:
            logger.info("All prompts already completed. Nothing to do.")
            return 0
        logger.info("Remaining prompts after resume filtering: %d", len(prompts))

    client = OpenAICompatChatClient(
        base_url=cfg.teacher.base_url,
        api_key=cfg.teacher.api_key,
        timeout_s=cfg.teacher.timeout_s,
        max_retries=cfg.teacher.max_retries,
    )

    gen = {
        "seed": cfg.generation.seed,
        "temperature": cfg.generation.temperature,
        "top_p": cfg.generation.top_p,
        "max_tokens": cfg.generation.max_tokens,
        "n": cfg.generation.n,
        "stop": cfg.generation.stop,
    }

    # Temperature strategies for diversity
    temperatures = [0.0, 0.6, 1.0]

    succeeded = 0
    failed = 0
    pending_rows: list[dict[str, Any]] = []

    # If not resuming, start fresh; if resuming, we append.
    write_mode = "a" if args.resume and completed_ids else "w"
    if write_mode == "w":
        # Ensure parent dir exists and create empty file
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text("")

    with cf.ThreadPoolExecutor(max_workers=cfg.teacher.concurrency) as ex:
        futs: dict[cf.Future, dict[str, Any]] = {}
        for i, p in enumerate(prompts):
            row_gen = gen.copy()
            row_gen["temperature"] = temperatures[i % len(temperatures)]
            fut = ex.submit(
                _one, client=client, teacher_model=cfg.teacher.model_id,
                prompt_row=p, gen=row_gen,
            )
            futs[fut] = p

        for fut in tqdm(cf.as_completed(futs), total=len(futs), desc="Generating"):
            prompt_row = futs[fut]
            try:
                result = fut.result()
                pending_rows.append(result)
                succeeded += 1
            except Exception as e:
                failed += 1
                logger.warning(
                    "Generation failed for prompt %s: %s", prompt_row.get("id", "?"), e,
                )
                continue

            # Incremental checkpoint
            if len(pending_rows) >= _CHECKPOINT_EVERY:
                append_jsonl(raw_path, pending_rows)
                logger.info("Checkpointed %d rows to %s", len(pending_rows), raw_path)
                pending_rows = []

    # Flush remaining rows
    if pending_rows:
        append_jsonl(raw_path, pending_rows)

    logger.info(
        "Done. succeeded=%d, failed=%d. Output: %s",
        succeeded, failed, raw_path,
    )
    if failed:
        logger.warning(
            "%d prompts failed. Re-run with --resume to retry only the missing ones.", failed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())