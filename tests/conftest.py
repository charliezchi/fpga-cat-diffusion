import pytest
from diffusers import DDIMScheduler, UNet2DModel


@pytest.fixture
def tiny_unet():
    return UNet2DModel(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        layers_per_block=1,
        block_out_channels=(32, 64),
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D"),
    )


@pytest.fixture
def tiny_scheduler():
    def _make(steps: int = 3) -> DDIMScheduler:
        s = DDIMScheduler(num_train_timesteps=1000)
        s.set_timesteps(steps)
        return s

    return _make


@pytest.fixture
def tiny_handwritten_unet():
    from catdiff.model.unet import UNet2D

    return UNet2D(dict(
        in_channels=3, out_channels=3,
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        block_out_channels=(32, 64), layers_per_block=1, norm_num_groups=8,
    ))
