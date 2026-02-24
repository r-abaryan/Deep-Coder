from __future__ import annotations

import argparse
import concurrent.futures as cf
import time
from typing import Any

from tqdm import tqdm

from distil_glm5.config import load_config
from distil_glm5.io_utils import read_jsonl, write_jsonl
from distil_glm5.teacher_client import OpenAICompatChatClient, build_chat_payload


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    prompts = read_jsonl(cfg.paths.prompts_jsonl)
    if not prompts:
        raise SystemExit(f"No prompts found at {cfg.paths.prompts_jsonl}. Run build_prompt_pool.py first.")

    client = OpenAICompatChatClient(
        base_url=cfg.teacher.base_url,
        api_key=cfg.teacher.api_key,
        timeout_s=cfg.teacher.timeout_s,
        max_retries=cfg.teacher.max_retries,
    )

    gen = {
        "seed": cfg.generation.seed,
        "temperature": cfg.generation.temperature, # default fallback
        "top_p": cfg.generation.top_p,
        "max_tokens": cfg.generation.max_tokens,
        "n": cfg.generation.n,
        "stop": cfg.generation.stop,
    }

    # Temperature strategies for diversity
    # Greedy (0.0): highly canonical
    # Balanced (0.6): standard coding style
    # Creative (1.0): out-of-the-box approaches
    temperatures = [0.0, 0.6, 1.0]

    out_rows: list[dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=cfg.teacher.concurrency) as ex:
        futs = []
        for i, p in enumerate(prompts):
            # Rotate strategies across the prompt pool
            row_gen = gen.copy()
            row_gen["temperature"] = temperatures[i % len(temperatures)]
            
            futs.append(
                ex.submit(_one, client=client, teacher_model=cfg.teacher.model_id, prompt_row=p, gen=row_gen)
            )
            
        for fut in tqdm(cf.as_completed(futs), total=len(futs), desc="Generating"):
            out_rows.append(fut.result())

    write_jsonl(cfg.paths.raw_jsonl, out_rows)
    print(f"Wrote {len(out_rows)} raw generations to {cfg.paths.raw_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

