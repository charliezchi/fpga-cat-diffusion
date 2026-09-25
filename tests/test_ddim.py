import torch
from diffusers import DDIMScheduler as RefDDIM

from catdiff.model.ddim import DDIMScheduler


def test_timesteps_match_diffusers():
    for n in (20, 50):
        ref = RefDDIM(num_train_timesteps=1000)
        ref.set_timesteps(n)
        mine = DDIMScheduler(num_train_timesteps=1000)
        mine.set_timesteps(n)
        assert torch.equal(mine.timesteps, ref.timesteps.cpu())


def test_full_trajectory_matches_diffusers():
    torch.manual_seed(0)
    sample = torch.randn(1, 3, 8, 8)
    ref = RefDDIM(num_train_timesteps=1000)
    ref.set_timesteps(20)
    mine = DDIMScheduler(num_train_timesteps=1000)
    mine.set_timesteps(20)
    x_ref, x_mine = sample.clone(), sample.clone()
    for t in ref.timesteps:  # 覆盖 prev_t<0 的末步（final_alpha_cumprod 路径）
        noise = torch.randn_like(x_ref)
        x_ref = ref.step(noise, t, x_ref, eta=0.0).prev_sample
        x_mine = mine.step(noise, t, x_mine, eta=0.0).prev_sample
        assert torch.allclose(x_mine, x_ref, rtol=1e-5, atol=1e-7), f"t={t} 不对齐"
