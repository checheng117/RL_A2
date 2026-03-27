"""Split raw JSON/JSONL into train/val/test and build processed views."""
from __future__ import annotations

import argparse
import glob
import hashlib
import random
import time
from pathlib import Path

from src.data import formatters, io, preprocess, teacher_process_data
from src.utils.path_utils import ensure_dir, find_project_root, resolve_path
from src.utils.seed import set_seed


def _file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Split dataset 90/5/5 and build processed JSONL.")
    parser.add_argument(
        "--config",
        nargs="+",
        default=["configs/base.yaml", "configs/data.yaml"],
        help="YAML configs to merge (later overrides earlier).",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--raw_glob", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--base_model", type=str, default=None)
    args = parser.parse_args()

    import yaml  # local import for CLI-only dependency path

    merged: dict = {}
    root = find_project_root()
    for p in args.config:
        with open(resolve_path(p, root), encoding="utf-8") as f:
            merged = {**merged, **(yaml.safe_load(f) or {})}

    data_cfg = merged.get("data", {}) or {}
    split_cfg = merged.get("split", {}) or data_cfg.get("split", {}) or {}
    fields = merged.get("fields", {}) or data_cfg.get("fields", {}) or {}
    proc = merged.get("processed", {}) or data_cfg.get("processed", {}) or {}
    # Assignment default: English main task; legacy teacher_lang=zh only for exploration.
    main_task_lang = str(data_cfg.get("main_task_lang") or merged.get("main_task_lang") or "en")

    seed = args.seed if args.seed is not None else split_cfg.get("seed", 42)
    set_seed(seed)

    raw_glob = args.raw_glob or split_cfg.get("raw_glob", "data/raw/*.jsonl")
    sequential = bool(split_cfg.get("sequential", False))
    out_dir = resolve_path(args.output_dir or split_cfg.get("output_dir", "data/splits"), root)
    ensure_dir(out_dir)

    answer_key = fields.get("answer_en", "answer_en")
    chosen_key = fields.get("summary_chosen_en", "summary_en_chosen")
    rejected_key = fields.get("summary_rejected_en", "summary_en_rejected")

    raw_dir = resolve_path("data/raw", root)
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(glob.glob(str(resolve_path(raw_glob, root))))
    if not paths:
        ph = raw_dir / "_smoke_placeholder.jsonl"
        tmpl = []
        for i in range(24):
            struct = (
                f"[point] 核心观点{i}。\n[reason] 1. 理由A{i} 2. 理由B{i} 3. 理由C{i}\n[summary] 总结{i}。"
            )
            tmpl.append(
                {
                    "answer_zh": f"占位中文回答{i}，用于本地无数据时的烟测；请替换为真实 train.jsonl。",
                    "summary_zh_chosen": struct,
                    "summary_zh_rejected": f"[point] 弱{i}。\n[reason] 1. x\n[summary] y",
                    "answer_en": f"Smoke EN answer {i} for placeholder.",
                    "summary_en_chosen": struct,
                    "summary_en_rejected": "[point] weak\n[reason] 1. x\n[summary] y",
                }
            )
        io.write_jsonl(str(ph), tmpl)
        paths = sorted(glob.glob(str(resolve_path(raw_glob, root))))

    all_rows: list[dict] = []
    file_hashes: dict[str, str] = {}
    for p in paths:
        fp = Path(p)
        file_hashes[str(fp.relative_to(root) if str(fp).startswith(str(root)) else fp)] = _file_sha256(
            fp
        )
        for row in io.load_json_or_jsonl(fp):
            if main_task_lang == "en":
                if preprocess.is_valid_en_row(
                    row, answer_key, chosen_key, rejected_key=rejected_key
                ):
                    all_rows.append(row)
            else:
                if preprocess.is_valid_teacher_row(row, lang=main_task_lang):
                    all_rows.append(row)

    n = len(all_rows)
    if n == 0:
        raise SystemExit(
            "No valid rows after filtering. For English main task require "
            f"{answer_key}, {chosen_key}, {rejected_key} (non-trivial length)."
        )

    if not sequential:
        rng = random.Random(seed)
        rng.shuffle(all_rows)

    tr = float(split_cfg.get("train_ratio", 0.9))
    vr = float(split_cfg.get("val_ratio", 0.05))
    te = float(split_cfg.get("test_ratio", 0.05))
    if abs(tr + vr + te - 1.0) > 1e-6:
        raise SystemExit("train_ratio + val_ratio + test_ratio must sum to 1.0")

    n_train = int(n * tr)
    n_val = int(n * vr)
    n_test = n - n_train - n_val
    # Small n: int(ratio) can yield 0 val or 0 test; trainers need non-empty val when n>=3.
    if n >= 3:
        if n_val < 1:
            n_val = 1
            n_train = max(1, n_train - 1)
        if n_test < 1:
            n_test = 1
            n_train = max(1, n_train - 1)
        while n_train + n_val + n_test > n:
            n_train = max(1, n_train - 1)
        while n_train + n_val + n_test < n:
            n_train += 1
    train_rows = all_rows[:n_train]
    val_rows = all_rows[n_train : n_train + n_val]
    test_rows = all_rows[n_train + n_val : n_train + n_val + n_test]

    io.write_jsonl(str(out_dir / "train.jsonl"), train_rows)
    io.write_jsonl(str(out_dir / "val.jsonl"), val_rows)
    io.write_jsonl(str(out_dir / "test.jsonl"), test_rows)

    manifest = {
        "seed": seed,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counts": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows), "total": n},
        "ratios": {"train": tr, "val": vr, "test": te},
        "sequential_split": sequential,
        "main_task_lang": main_task_lang,
        "source_files": file_hashes,
        "fields": {"answer": answer_key, "chosen": chosen_key, "rejected": rejected_key},
    }
    manifest_path = out_dir / split_cfg.get("manifest_name", "split_manifest.json")
    io.write_json(str(manifest_path), manifest)

    summary_path = resolve_path(
        split_cfg.get("summary_path", "outputs/metrics/dataset_split_summary.json"), root
    )
    ensure_dir(summary_path.parent)
    io.write_json(
        str(summary_path),
        {
            "manifest": manifest,
            "train_stats": preprocess.basic_stats(train_rows, chosen_key, rejected_key),
            "val_stats": preprocess.basic_stats(val_rows, chosen_key, rejected_key),
            "test_stats": preprocess.basic_stats(test_rows, chosen_key, rejected_key),
        },
    )

    paths_cfg = merged.get("paths", {}) or {}
    proc_root = resolve_path(paths_cfg.get("processed_dir", "data/processed"), root)
    ensure_dir(proc_root)

    lang = main_task_lang if main_task_lang in ("zh", "en") else "en"

    def _merge_teacher_columns(rows_local: list[dict], ds_sft, ds_dpo, ds_grpo) -> tuple[list, list, list]:
        sft_out, dpo_out, grpo_out = [], [], []
        for i, row in enumerate(rows_local):
            sft_out.append(
                {
                    **row,
                    "prompt": ds_sft[i]["prompt"],
                    "response": ds_sft[i]["response"],
                }
            )
            dpo_out.append(
                {
                    **row,
                    "prompt": ds_dpo[i]["prompt"],
                    "chosen": ds_dpo[i]["chosen"],
                    "rejected": ds_dpo[i]["rejected"],
                }
            )
            grpo_out.append(
                {
                    **row,
                    "prompt": ds_grpo[i]["prompt"],
                    "reference": ds_grpo[i]["reference"],
                    "original_answer": ds_grpo[i]["original_answer"],
                }
            )
        return sft_out, dpo_out, grpo_out

    def build_split(name: str, rows: list[dict]) -> None:
        if main_task_lang == "en":
            sft = [formatters.english_sft_record(r, answer_key, chosen_key) for r in rows]
            dpo = [formatters.english_dpo_record(r, answer_key, chosen_key, rejected_key) for r in rows]
            grpo = [formatters.english_grpo_record(r, answer_key, chosen_key) for r in rows]
        else:
            ds_sft = teacher_process_data.process_for_sft(rows, lang=lang)  # type: ignore[arg-type]
            ds_dpo = teacher_process_data.process_for_dpo(rows, lang=lang)  # type: ignore[arg-type]
            ds_grpo = teacher_process_data.process_for_grpo(rows, lang=lang)  # type: ignore[arg-type]
            sft, dpo, grpo = _merge_teacher_columns(rows, ds_sft, ds_dpo, ds_grpo)
        io.write_jsonl(
            str(resolve_path(proc.get(f"sft_{name}", f"data/processed/sft_{name}.jsonl"), root)),
            sft,
        )
        io.write_jsonl(
            str(resolve_path(proc.get(f"dpo_{name}", f"data/processed/dpo_{name}.jsonl"), root)),
            dpo,
        )
        io.write_jsonl(
            str(resolve_path(proc.get(f"grpo_{name}", f"data/processed/grpo_{name}.jsonl"), root)),
            grpo,
        )

    build_split("train", train_rows)
    build_split("val", val_rows)
    build_split("test", test_rows)

    align_path = resolve_path(
        split_cfg.get("alignment_summary_path", "outputs/metrics/dataset_alignment_summary.json"), root
    )
    ensure_dir(align_path.parent)
    sample_keys = list(
        {
            **train_rows[0],
            "prompt": "",
            "response": "",
        }.keys()
    ) if train_rows else []
    prompt_excerpt = (
        formatters.SUMMARY_INSTRUCTION[:120] + "..."
        if main_task_lang == "en"
        else teacher_process_data.PROMPT_TEMPLATE[:80] + "..."
    )
    io.write_json(
        str(align_path),
        {
            "main_task_lang": main_task_lang,
            "english_prompt_source": "src/data/formatters.py SUMMARY_INSTRUCTION + answer_en",
            "teacher_reference_zh_exploration_only": "train_code_with_data/data/process_data.py",
            "prompt_template_excerpt": prompt_excerpt,
            "processed_sft_columns": ["prompt", "response", "answer_en", "summary_en_chosen", "..."],
            "sequential_split": sequential,
            "split_manifest": manifest,
            "sample_keys_first_train_row": sample_keys,
        },
    )

    print(f"Wrote splits to {out_dir}, manifest {manifest_path}, summary {summary_path}, alignment {align_path}")


if __name__ == "__main__":
    main()
