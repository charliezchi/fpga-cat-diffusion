import json

import numpy as np
import torch

from catdiff.export.ddim_table import build_ddim_table
from catdiff.export.film_table import build_film_vectors
from catdiff.export.weights_export import iter_quantized_layers
from catdiff.model.ddim import DDIMScheduler
from catdiff.model.unet import UNet2D
from tests.test_unet import TINY_CONFIG


def test_ddim_table_matches_scheduler():
    table = build_ddim_table(num_inference_steps=20)
    assert len(table) == 20
    sched = DDIMScheduler(num_train_timesteps=1000)
    sched.set_timesteps(20)
    for row, t in zip(table, sched.timesteps.tolist()):
        assert row["t"] == t
    # 首行 t 最大，末行 alpha_prev == 1.0（set_alpha_to_one）
    assert table[0]["t"] > table[-1]["t"]
    assert abs(table[-1]["alpha_prev"] - 1.0) < 1e-6


def test_iter_quantized_layers_covers_all_param_modules():
    torch.manual_seed(0)
    model = UNet2D(TINY_CONFIG).eval()
    layers = list(iter_quantized_layers(model))
    names = {name for name, _, _, _ in layers}
    import torch.nn as nn
    expect = {n for n, m in model.named_modules()
              if isinstance(m, (nn.Conv2d, nn.Linear))}
    assert names == expect  # 含全部注意力的 to_q/to_k/to_v/to_out.0
    for name, w_q, scales, bias in layers:
        assert w_q.dtype == torch.int8
        assert scales.shape[0] == w_q.shape[0]


def test_film_vectors_match_resnet_injection():
    torch.manual_seed(0)
    model = UNet2D(TINY_CONFIG).eval()
    t_value = 250
    vecs = build_film_vectors(model, [t_value])  # {resnet_name: {t: vector}}
    # 任取一个 resnet，独立计算 time_emb_proj(silu(temb)) 比对
    from catdiff.model.time_embed import timestep_embedding
    name, table = next(iter(vecs.items()))
    resnet = dict(model.named_modules())[name]
    temb = model.time_embedding(
        model.time_proj(torch.tensor([float(t_value)])))
    expect = resnet.time_emb_proj(torch.nn.functional.silu(temb))[0]
    got = torch.tensor(table[t_value])
    assert torch.allclose(got, expect, atol=1e-5)
