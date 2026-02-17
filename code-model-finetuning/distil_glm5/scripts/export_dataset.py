from __future__ import annotations

import argparse
import json
from pathlib import Path

from distil_glm5.config import load_config
from distil_glm5.io_utils import ensure_parent_dir, read_jsonl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--format", choices=["jsonl", "hf"], default="jsonl")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rows = read_jsonl(cfg.paths.curated_jsonl)
    if not rows:
        raise SystemExit(f"No curated dataset found at {cfg.paths.curated_jsonl}. Run filter_and_dedup.py first.")

    export_dir = ensure_parent_dir(Path(cfg.paths.export_dir) / "dummy").parent
    export_dir.mkdir(parents=True, exist_ok=True)

    if args.format == "jsonl":
        # Curated JSONL already exists; just copy minimal metadata.
        meta_path = export_dir / "meta.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(
                {"run_name": cfg.run_name, "count": len(rows), "teacher": cfg.teacher.model_id},
                f,
                indent=2,
            )
        print(f"Exported metadata to {meta_path}")
        return 0

    if args.format == "hf":
        try:
            from datasets import Dataset  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise SystemExit(
                "Missing optional dependency. Install with: python -m pip install -e \".[hf]\""
            ) from e

        ds = Dataset.from_list(rows)
        out_path = export_dir / "hf_dataset"
        ds.save_to_disk(str(out_path))
        print(f"Saved HF dataset to {out_path}")
        return 0

    raise SystemExit("Unknown format")


if __name__ == "__main__":
    raise SystemExit(main())

