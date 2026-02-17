from __future__ import annotations

import argparse
import hashlib
from typing import Any

from distil_glm5.config import load_config
from distil_glm5.filters import (
    build_curated_row,
    filter_example,
    normalize_for_hash,
    redact_obvious_secrets,
)
from distil_glm5.io_utils import read_jsonl, write_jsonl


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
    curated: list[dict[str, Any]] = []
    dropped = 0

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

        curated.append(
            build_curated_row(
                prompt_row=row,
                teacher_model=row.get("teacher", cfg.teacher.model_id),
                gen_params=row.get("gen", {}),
                output_text=out_text.strip(),
                raw_response=row.get("raw", {}),
                filter_reasons=fr.reasons,
                redacted=redacted,
            )
        )

    write_jsonl(cfg.paths.curated_jsonl, curated)
    print(f"Curated {len(curated)} rows; dropped {dropped}. Output: {cfg.paths.curated_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

