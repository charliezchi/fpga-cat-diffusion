import torch
from diffusers.models.embeddings import TimestepEmbedding as RefTimestepEmbedding
from diffusers.models.embeddings import get_timestep_embedding

from catdiff.model.time_embed import TimestepEmbedding, Timesteps, timestep_embedding

# cat-256 的非默认配置（docs/reference/unet-config-ddpm-cat-256.json）：
# flip_sin_to_cos=False（diffusers 默认 True），downscale_freq_shift=1（恰好是默认值）


def test_sinusoid_matches_diffusers():
    t = torch.tensor([0, 1, 7, 250, 999], dtype=torch.float32)
    ref = get_timestep_embedding(t, 128, flip_sin_to_cos=False, downscale_freq_shift=1)
    mine = timestep_embedding(t, 128)
    assert torch.allclose(mine, ref, atol=1e-6)


def test_timesteps_module_matches_diffusers():
    from diffusers.models.embeddings import Timesteps as RefTimesteps

    t = torch.tensor([3, 981], dtype=torch.float32)
    ref = RefTimesteps(num_channels=128, flip_sin_to_cos=False, downscale_freq_shift=1)
    assert torch.allclose(Timesteps(128)(t), ref(t), atol=1e-6)


def test_timestep_embedding_mlp_matches_diffusers():
    torch.manual_seed(0)
    ref = RefTimestepEmbedding(in_channels=128, time_embed_dim=512)
    mine = TimestepEmbedding(128, 512)
    mine.load_state_dict(ref.state_dict())
    x = torch.randn(4, 128)
    assert torch.allclose(mine(x), ref(x), atol=1e-6)
