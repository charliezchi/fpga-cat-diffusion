"""docs/reference 冻结配置是 F2 手写复现的输入契约：UNet + scheduler 配置都必须在库。"""

import json
from pathlib import Path

REF = Path(__file__).resolve().parent.parent / "docs" / "reference"

REQUIRED_SCHEDULER_FIELDS = {
    "num_train_timesteps": 1000,
    "beta_start": 0.0001,
    "beta_end": 0.02,
    "beta_schedule": "linear",
    "clip_sample": True,
}


def test_unet_configs_frozen():
    assert (REF / "unet-config-ddpm-cat-256.json").is_file()
    assert (REF / "unet-config-butterflies-64.json").is_file()


def test_scheduler_configs_frozen_with_required_fields():
    for name in ("scheduler-config-ddpm-cat-256.json", "scheduler-config-butterflies-64.json"):
        path = REF / name
        assert path.is_file(), f"缺少冻结的 scheduler 配置: {name}"
        cfg = json.loads(path.read_text(encoding="utf-8"))
        for key, expected in REQUIRED_SCHEDULER_FIELDS.items():
            assert cfg.get(key) == expected, f"{name}: {key}={cfg.get(key)!r}, 期望 {expected!r}"
