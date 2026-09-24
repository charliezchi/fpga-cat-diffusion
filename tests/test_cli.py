import json

from catdiff.baseline import cli


def _run(out, monkeypatch, tiny_unet, tiny_scheduler):
    monkeypatch.setattr(cli, "load_unet", lambda model_id: tiny_unet)
    monkeypatch.setattr(cli, "load_ddim", lambda model_id, steps: tiny_scheduler(2))
    rc = cli.main(
        [
            "--model-id", "tiny",
            "--num-samples", "3",
            "--image-size", "32",
            "--num-inference-steps", "2",
            "--seed", "5",
            "--out-dir", str(out),
        ]
    )
    assert rc == 0


def test_cli_resume_idempotent(tmp_path, monkeypatch, tiny_unet, tiny_scheduler):
    out = tmp_path / "run"
    _run(out, monkeypatch, tiny_unet, tiny_scheduler)
    first = {p.name: p.read_bytes() for p in sorted(out.glob("*.png"))}
    assert {"seed5_0000.png", "seed5_0001.png", "seed5_0002.png", "grid.png"} <= set(first)

    _run(out, monkeypatch, tiny_unet, tiny_scheduler)  # 全部命中续跑，不得重写
    second = {p.name: p.read_bytes() for p in sorted(out.glob("*.png"))}
    assert first == second


def test_cli_writes_metadata(tmp_path, monkeypatch, tiny_unet, tiny_scheduler):
    out = tmp_path / "run"
    _run(out, monkeypatch, tiny_unet, tiny_scheduler)
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert meta["model_id"] == "tiny"
    assert meta["eta"] == 0.0
    assert meta["seed"] == 5
    assert meta["num_inference_steps"] == 2
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["sample_size"] == 32
