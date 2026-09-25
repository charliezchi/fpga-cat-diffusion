"""FiLM 表导出（契约 §3）：预计算每个 ResnetBlock2D 的时间偏置向量。"""

import torch

from catdiff.model.resnet import ResnetBlock2D


@torch.no_grad()
def build_film_vectors(model, timesteps: list[int]) -> dict:
    """{resnet_name: {t: [out_channels] 列表}}，resnet 按 named_modules 顺序。"""
    out = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, ResnetBlock2D):
            continue
        table = {}
        for t in timesteps:
            temb = model.time_embedding(
                model.time_proj(torch.tensor([float(t)])))
            vec = mod.time_emb_proj(torch.nn.functional.silu(temb))[0]
            table[t] = vec.tolist()
        out[name] = table
    return out
