"""Config parsing smoke tests for SFT YAML configs.

These tests only validate structure — they do NOT import unsloth/trl/torch
(those pull heavy deps; see the sft_qwen3_unsloth module's lazy imports).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "backend" / "training" / "configs"


REQUIRED_STRUCTURE = {
    "model": ["name_or_path", "dtype", "max_seq_length", "load_in_4bit"],
    "lora": ["r", "alpha", "dropout", "target_modules", "bias"],
    "data": ["train_path", "dev_path", "prompt_template", "completion_field"],
    "train": [
        "num_train_epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "lr_scheduler_type",
        "warmup_ratio",
        "logging_steps",
        "save_steps",
        "eval_steps",
        "save_total_limit",
        "seed",
    ],
    "output": ["dir", "wandb_project", "wandb_run_name"],
    "thinking_mode": ["disable"],
    "eval": ["l1_dev_size", "l1_every_n_steps", "probe_every_checkpoint"],
}


@pytest.mark.parametrize(
    "config_name",
    ["qwen3_sft.yaml", "qwen3_4b_sft.yaml"],
)
def test_sft_config_has_required_keys(config_name: str) -> None:
    path = CONFIG_DIR / config_name
    assert path.exists(), f"missing config: {path}"
    with path.open("r") as fh:
        cfg = yaml.safe_load(fh)
    assert isinstance(cfg, dict)

    for section, keys in REQUIRED_STRUCTURE.items():
        assert section in cfg, f"{config_name}: missing section {section}"
        for key in keys:
            assert key in cfg[section], (
                f"{config_name}: missing key {section}.{key}"
            )


@pytest.mark.parametrize(
    "config_name",
    ["qwen3_sft.yaml", "qwen3_4b_sft.yaml"],
)
def test_prompt_template_has_jp_placeholder(config_name: str) -> None:
    with (CONFIG_DIR / config_name).open("r") as fh:
        cfg = yaml.safe_load(fh)
    tmpl = cfg["data"]["prompt_template"]
    assert "{jp}" in tmpl, f"{config_name}: prompt_template missing {{jp}} placeholder"
    assert "English:" in tmpl, f"{config_name}: prompt_template missing 'English:' marker"


@pytest.mark.parametrize(
    "config_name",
    ["qwen3_sft.yaml", "qwen3_4b_sft.yaml"],
)
def test_lora_target_modules_is_list(config_name: str) -> None:
    with (CONFIG_DIR / config_name).open("r") as fh:
        cfg = yaml.safe_load(fh)
    tm = cfg["lora"]["target_modules"]
    assert isinstance(tm, list) and len(tm) > 0
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert name in tm, f"{config_name}: {name} should be in target_modules"


def test_1p7b_and_4b_differ_as_expected() -> None:
    with (CONFIG_DIR / "qwen3_sft.yaml").open("r") as fh:
        small = yaml.safe_load(fh)
    with (CONFIG_DIR / "qwen3_4b_sft.yaml").open("r") as fh:
        big = yaml.safe_load(fh)

    assert "1.7B" in small["model"]["name_or_path"]
    assert "4B" in big["model"]["name_or_path"]
    # Same effective batch target: 2*8 = 16 vs 1*16 = 16
    small_eff = (
        small["train"]["per_device_train_batch_size"]
        * small["train"]["gradient_accumulation_steps"]
    )
    big_eff = (
        big["train"]["per_device_train_batch_size"]
        * big["train"]["gradient_accumulation_steps"]
    )
    assert small_eff == big_eff == 16
    assert small["output"]["dir"] != big["output"]["dir"]
    assert small["output"]["wandb_run_name"] != big["output"]["wandb_run_name"]


def test_sft_config_loader_accepts_file() -> None:
    """Import the SFTConfig class and make sure it accepts the real YAML."""
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from backend.scripts.train.sft_qwen3_unsloth import SFTConfig  # noqa: PLC0415

    cfg = SFTConfig.load(CONFIG_DIR / "qwen3_sft.yaml")
    assert cfg.model["name_or_path"] == "Qwen/Qwen3-1.7B-Base"
    assert cfg.thinking_mode["disable"] is True
