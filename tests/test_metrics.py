from src.metrics.format_adherence import batch_format_adherence
from src.metrics.rouge_metrics import rouge_l_f1


def test_format_batch():
    m = batch_format_adherence(
        ["[point] a\n[reason] 1. b 2. c\n[summary] d", "no structure"]
    )
    assert m["avg_score"] < 1.0
    assert m["avg_score"] > 0.0


def test_rouge():
    s = rouge_l_f1(["hello world"], ["hello world"])
    assert s > 0.9
