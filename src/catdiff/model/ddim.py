"""手写 DDIM(eta=0) 调度器，复刻 diffusers DDIMScheduler 的数值行为。

仅实现本项目所需子集：epsilon 预测、linear beta、eta=0、
use_clipped_model_output=False（方向项用原始 model_output；
clip_sample=True 时对 pred_x0 做 [-1,1] 截断，与 diffusers 默认行为一致）。
"""

import types

import torch


class DDIMScheduler:
    def __init__(self, num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02,
                 beta_schedule="linear", set_alpha_to_one=True,
                 prediction_type="epsilon", clip_sample=True,
                 clip_sample_range=1.0):
        assert beta_schedule == "linear", "本项目仅使用 linear beta"
        assert prediction_type == "epsilon", "本项目仅使用 epsilon 预测"
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps,
                               dtype=torch.float32)
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.final_alpha_cumprod = (
            torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
        )
        self.num_train_timesteps = num_train_timesteps
        # cat-256 checkpoint 的 scheduler_config clip_sample=true，且 diffusers
        # DDIMScheduler 默认即 True：参考管线每步将 pred_x0 截到 [-1,1]
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        self.timesteps = None
        self._step_ratio = None

    def set_timesteps(self, num_inference_steps: int):
        self._step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = (
            (torch.arange(0, num_inference_steps) * self._step_ratio)
            .round().flip(0).long()
        )

    def step(self, model_output, timestep, sample, eta: float = 0.0):
        assert eta == 0.0, "本项目仅支持确定性采样 eta=0"
        t = int(timestep)
        prev_t = t - self._step_ratio
        alpha_t = self.alphas_cumprod[t]
        alpha_prev = (
            self.alphas_cumprod[prev_t] if prev_t >= 0 else self.final_alpha_cumprod
        )
        pred_x0 = (sample - (1 - alpha_t).sqrt() * model_output) / alpha_t.sqrt()
        if self.clip_sample:
            pred_x0 = pred_x0.clamp(-self.clip_sample_range, self.clip_sample_range)
        prev_sample = (
            alpha_prev.sqrt() * pred_x0 + (1 - alpha_prev).sqrt() * model_output
        )
        return types.SimpleNamespace(prev_sample=prev_sample)
