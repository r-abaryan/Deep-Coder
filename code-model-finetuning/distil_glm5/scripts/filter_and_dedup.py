from __future__ import annotations

import argparse
import hashlib
import logging
from typing import Any

from distil_glm5.config import load_config
from distil_glm5.filters import (
    build_curated_row,
    build_minhash,
    filter_example,
    normalize_for_hash,
    redact_obvious_secrets,
)
from distil_glm5.io_utils import append_jsonl, read_jsonl
from distil_glm5.judge import get_instruction_from_row, judge_keep
from distil_glm5.teacher_client import OpenAICompatChatClient

logger = logging.getLogger(__name__)

_CHECKPOINT_EVERY = 200


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw_rows = read_jsonl(cfg.paths.raw_jsonl)
    if not raw_rows:
        raise SystemExit(
            f"No raw generations found at {cfg.paths.raw_jsonl}. "
            f"Run generate_teacher_answers.py first."
        )

    seen_exact: set[str] = set()
    lsh = None
    if cfg.dedup.enable_near:
        try:
            from datasketch import MinHashLSH  # type: ignore
            lsh = MinHashLSH(threshold=cfg.dedup.near_jaccard_threshold, num_perm=128)
            logger.info("Enabled near-dedup with threshold %.2f", cfg.dedup.near_jaccard_threshold)
        except ImportError:
            logger.warning(
                "enable_near is true but datasketch is not installed. Ignoring near-dedup."
            )

    # Stats
    stats: dict[str, int] = {
        "total": len(raw_rows),
        "filtered": 0,
        "exact_dedup": 0,
        "near_dedup": 0,
        "judge_rejected": 0,
        "judge_error": 0,
        "kept": 0,
    }

    judge_client = None
    if cfg.judge.enabled:
        logger.info("Judge enabled: %s at %s", cfg.judge.model_id, cfg.judge.base_url)
        judge_client = OpenAICompatChatClient(
            base_url=cfg.judge.base_url,
            api_key=cfg.judge.api_key,
            timeout_s=cfg.judge.timeout_s,
            max_retries=cfg.judge.max_retries,
        )

    # Prepare output file (overwrite)
    from pathlib import Path
    curated_path = Path(cfg.paths.curated_jsonl)
    curated_path.parent.mkdir(parents=True, exist_ok=True)
    curated_path.write_text("")

    to_judge: list[tuple[dict, str, str, Any, bool]] = []
    pending_curated: list[dict[str, Any]] = []

    for row in raw_rows:
        out_text = row.get("output_text", "") or ""

        out_text, redacted = redact_obvious_secrets(out_text)
        fr = filter_example(
            task_type=row.get("task_type", "unknown"),
            language=row.get("language", "python"),
            output_text=out_text,
            min_chars=cfg.filters.min_output_chars,
            max_chars=cfg.filters.max_output_chars,
            drop_refusals=cfg.filters.drop_refusals,
            require_python_syntax_valid=cfg.filters.require_python_syntax_valid,
            require_ts_js_syntax_valid=cfg.filters.require_ts_js_syntax_valid,
        )

        if not fr.keep:
            stats["filtered"] += 1
            continue

        if cfg.dedup.enable_exact:
            h = _hash_text(normalize_for_hash(out_text))
            if h in seen_exact:
                stats["exact_dedup"] += 1
                continue
            seen_exact.add(h)

        if lsh is not None:
            m = build_minhash(out_text)
            if m is not None:
                if len(lsh.query(m)) > 0:
                    stats["near_dedup"] += 1
                    continue
                lsh.insert(row["id"], m)

        if cfg.judge.enabled:
            instruction = get_instruction_from_row(row)
            to_judge.append((row, instruction, out_text, fr, redacted))
        else:
            pending_curated.append(
                build_curated_row(
                    prompt_row=row,
                    teacher_model=row.get("teacher", cfg.teacher.model_id),
                    gen_params=row.get("gen", {}),
                    output_text=out_text.strip(),
                    raw_response=row.get("raw", {}),
                    filter_reasons=fr.reasons,
                    redacted=redacted,
                    judge_passed=None,
                    judge_reasoning=None,
                )
            )
            # Checkpoint
            if len(pending_curated) >= _CHECKPOINT_EVERY:
                append_jsonl(curated_path, pending_curated)
                stats["kept"] += len(pending_curated)
                pending_curated = []

    # Flush non-judged pending rows
    if pending_curated and not cfg.judge.enabled:
        append_jsonl(curated_path, pending_curated)
        stats["kept"] += len(pending_curated)
        pending_curated = []

    # --- Judge pass (concurrent) ---
    if cfg.judge.enabled and to_judge:
        import concurrent.futures as cf_mod
        from tqdm import tqdm

        def _judge_one(item: tuple) -> tuple[tuple, bool, str]:
            row, instruction, out_text, fr, redacted = item
            keep, reasoning = judge_keep(
                instruction=instruction,
                output=out_text.strip(),
                judge_config=cfg.judge,
                client=judge_client,
                task_type=row.get("task_type", "unknown"),
                language=row.get("language", "unknown"),
            )
            return item, keep, reasoning

        judged_curated: list[dict[str, Any]] = []

        with cf_mod.ThreadPoolExecutor(
            max_workers=min(32, cfg.teacher.concurrency)
        ) as ex:
            futs = [ex.submit(_judge_one, item) for item in to_judge]
            for fut in tqdm(cf_mod.as_completed(futs), total=len(futs), desc="Judging"):
                try:
                    item, passed, reasoning = fut.result()
                except Exception as e:
                    logger.warning("Judge future failed: %s", e)
                    stats["judge_error"] += 1
                    continue

                row, instruction, out_text, fr, redacted = item
                if passed:
                    judged_curated.append(
                        build_curated_row(
                            prompt_row=row,
                            teacher_model=row.get("teacher", cfg.teacher.model_id),
                            gen_params=row.get("gen", {}),
                            output_text=out_text.strip(),
                            raw_response=row.get("raw", {}),
                            filter_reasons=fr.reasons,
                            redacted=redacted,
                            judge_passed=True,
                            judge_reasoning=reasoning,
                        )
                    )
                    # Checkpoint judged rows
                    if len(judged_curated) >= _CHECKPOINT_EVERY:
                        append_jsonl(curated_path, judged_curated)
                        stats["kept"] += len(judged_curated)
                        judged_curated = []
                else:
                    stats["judge_rejected"] += 1
                    logger.debug(
                        "Judge rejected %s: %s",
                        row.get("id", "?"),
                        reasoning[:120] if reasoning else "no_reason",
                    )

        # Flush remaining judged rows
        if judged_curated:
            append_jsonl(curated_path, judged_curated)
            stats["kept"] += len(judged_curated)

    # --- Report ---
    logger.info("=== Filter & Dedup Report ===")
    logger.info("  Total raw rows:     %d", stats["total"])
    logger.info("  Filtered out:       %d", stats["filtered"])
    logger.info("  Exact dedup:        %d", stats["exact_dedup"])
    logger.info("  Near dedup:         %d", stats["near_dedup"])
    logger.info("  Judge rejected:     %d", stats["judge_rejected"])
    logger.info("  Judge errors:       %d", stats["judge_error"])
    logger.info("  Kept (curated):     %d", stats["kept"])
    logger.info("  Output: %s", cfg.paths.curated_jsonl)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())