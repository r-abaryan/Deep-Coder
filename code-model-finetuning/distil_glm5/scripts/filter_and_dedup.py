from __future__ import annotations

import argparse
import hashlib
from typing import Any

from distil_glm5.config import load_config
from distil_glm5.filters import (
    build_curated_row,
    build_minhash,
    filter_example,
    normalize_for_hash,
    redact_obvious_secrets,
)
from distil_glm5.io_utils import read_jsonl, write_jsonl
from distil_glm5.judge import get_instruction_from_row, judge_keep
from distil_glm5.teacher_client import OpenAICompatChatClient


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw_rows = read_jsonl(cfg.paths.raw_jsonl)
    if not raw_rows:
        raise SystemExit(f"No raw generations found at {cfg.paths.raw_jsonl}. Run generate_teacher_answers.py first.")

    seen_exact: set[str] = set()
    lsh = None
    if cfg.dedup.enable_near:
        try:
            from datasketch import MinHashLSH
            lsh = MinHashLSH(threshold=cfg.dedup.near_jaccard_threshold, num_perm=128)
            print(f"Enabled near-dedup with threshold {cfg.dedup.near_jaccard_threshold}")
        except ImportError:
            print("Warning: enable_near is true but datasketch is not installed. Ignoring near-dedup.")

    curated: list[dict[str, Any]] = []
    dropped = 0

    judge_client = None
    if cfg.judge.enabled:
        print(f"Judge enabled: {cfg.judge.model_id} at {cfg.judge.base_url}")
        judge_client = OpenAICompatChatClient(
            base_url=cfg.judge.base_url,
            api_key=cfg.judge.api_key,
            timeout_s=cfg.judge.timeout_s,
            max_retries=cfg.judge.max_retries,
        )

    to_judge = []
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
            dropped += 1
            continue

        if cfg.dedup.enable_exact:
            h = _hash_text(normalize_for_hash(out_text))
            if h in seen_exact:
                dropped += 1
                continue
            seen_exact.add(h)
            
        if lsh is not None:
            m = build_minhash(out_text)
            if m is not None:
                if len(lsh.query(m)) > 0:
                    # Found a near-duplicate
                    dropped += 1
                    continue
                lsh.insert(row["id"], m)

        if cfg.judge.enabled:
            instruction = get_instruction_from_row(row)
            to_judge.append((row, instruction, out_text, fr, redacted))
        else:
            curated.append(
                build_curated_row(
                    prompt_row=row,
                    teacher_model=row.get("teacher", cfg.teacher.model_id),
                    gen_params=row.get("gen", {}),
                    output_text=out_text.strip(),
                    raw_response=row.get("raw", {}),
                    filter_reasons=fr.reasons,
                    redacted=redacted,
                    judge_passed=None,
                )
            )

    if cfg.judge.enabled and to_judge:
        import concurrent.futures as cf
        from tqdm import tqdm

        def _judge_one(item):
            row, instruction, out_text, fr, redacted = item
            passed = judge_keep(
                instruction=instruction,
                output=out_text.strip(),
                judge_config=cfg.judge,
                client=judge_client,
            )
            return item, passed

        with cf.ThreadPoolExecutor(max_workers=min(32, cfg.teacher.concurrency)) as ex:
            futs = [ex.submit(_judge_one, item) for item in to_judge]
            for fut in tqdm(cf.as_completed(futs), total=len(futs), desc="Judging"):
                item, passed = fut.result()
                row, instruction, out_text, fr, redacted = item
                if passed:
                    curated.append(
                        build_curated_row(
                            prompt_row=row,
                            teacher_model=row.get("teacher", cfg.teacher.model_id),
                            gen_params=row.get("gen", {}),
                            output_text=out_text.strip(),
                            raw_response=row.get("raw", {}),
                            filter_reasons=fr.reasons,
                            redacted=redacted,
                            judge_passed=True,
                        )
                    )
                else:
                    dropped += 1

    write_jsonl(cfg.paths.curated_jsonl, curated)
    print(f"Curated {len(curated)} rows; dropped {dropped}. Output: {cfg.paths.curated_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

