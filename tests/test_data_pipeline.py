from pathlib import Path

from src.data import io, preprocess
from src.data.formatters import build_user_prompt


def test_io_roundtrip(tmp_path: Path):
    p = tmp_path / "t.jsonl"
    rows = [{"a": 1, "answer_en": "x" * 20, "summary_en_chosen": "y" * 20, "summary_en_rejected": "z" * 20}]
    io.write_jsonl(p, rows)
    back = list(io.iter_jsonl(p))
    assert back[0]["a"] == 1


def test_preprocess_filter():
    row = {"answer_en": "short", "summary_en_chosen": "x" * 20, "summary_en_rejected": "y" * 20}
    assert not preprocess.is_valid_en_row(row, "answer_en", "summary_en_chosen", "summary_en_rejected")


def test_prompt_template():
    s = build_user_prompt("Hello answer")
    assert "[point]" in s
    assert "Hello answer" in s
