"""DDIM 常数表导出（契约 §3）：硬件零运行时超越函数。"""

from catdiff.model.ddim import DDIMScheduler


def build_ddim_table(num_inference_steps: int = 20,
                     num_train_timesteps: int = 1000) -> list[dict]:
    sched = DDIMScheduler(num_train_timesteps=num_train_timesteps)
    sched.set_timesteps(num_inference_steps)
    rows = []
    for t in sched.timesteps.tolist():
        prev_t = t - sched._step_ratio
        alpha_t = sched.alphas_cumprod[t].item()
        alpha_prev = (sched.alphas_cumprod[prev_t].item() if prev_t >= 0
                      else sched.final_alpha_cumprod.item())
        rows.append({
            "t": t,
            "alpha_t": alpha_t,
            "alpha_prev": alpha_prev,
            "sqrt_alpha_t": alpha_t ** 0.5,
            "sqrt_one_minus_alpha_t": (1 - alpha_t) ** 0.5,
            "sqrt_alpha_prev": alpha_prev ** 0.5,
            "sqrt_one_minus_alpha_prev": (1 - alpha_prev) ** 0.5,
        })
    return rows
