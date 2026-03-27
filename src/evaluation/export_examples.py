"""Export qualitative examples and reward-hacking templates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.path_utils import find_project_root, resolve_path


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export qualitative examples to markdown")
    p.add_argument("--predictions_jsonl", type=str, required=True)
    p.add_argument("--output_md", type=str, required=True)
    p.add_argument("--n_good", type=int, default=2)
    p.add_argument("--n_bad_format", type=int, default=2)
    return p


def main() -> None:
    args = _parser().parse_args()
    root = find_project_root()
    pred_path = resolve_path(args.predictions_jsonl, root)
    out_path = resolve_path(args.output_md, root)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    def has_format(t: str) -> bool:
        return "[point]" in t.lower() and "[reason]" in t.lower() and "[summary]" in t.lower()

    good = [r for r in rows if has_format(r.get("prediction", ""))][: args.n_good]
    bad_fmt = [r for r in rows if not has_format(r.get("prediction", ""))][: args.n_bad_format]

    lines = ["# Selected examples", ""]
    lines.append("## Likely format-compliant")
    for i, r in enumerate(good, 1):
        lines.append(f"### Example {i}")
        lines.append("**Prompt (excerpt)**")
        lines.append("```")
        lines.append((r.get("prompt", "")[:500] + "...") if len(r.get("prompt", "")) > 500 else r.get("prompt", ""))
        lines.append("```")
        lines.append("**Prediction**")
        lines.append("```")
        lines.append(r.get("prediction", ""))
        lines.append("```\n")

    lines.append("## Format failures")
    for i, r in enumerate(bad_fmt, 1):
        lines.append(f"### Failure {i}")
        lines.append("```")
        lines.append(r.get("prediction", ""))
        lines.append("```\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
