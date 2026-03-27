from pathlib import Path

from src.training.common import load_merged_config


def test_merge_configs():
    root = Path(__file__).resolve().parents[1]
    cfg = load_merged_config(
        [
            str(root / "configs/base.yaml"),
            str(root / "configs/data.yaml"),
            str(root / "configs/sft_lora_3090.yaml"),
        ],
        root,
    )
    assert cfg["project"]["base_model"]
    assert cfg["training"]["per_device_train_batch_size"] >= 1
    assert "split" in cfg or ("data" in cfg)
