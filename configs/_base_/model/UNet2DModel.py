from diffusers import UNet2DModel


model = UNet2DModel(
    sample_size=128,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(128,128, 256, 512, 512),
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDwonBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "AttnDwonBlock2D", "UpBlock2D", "UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
)