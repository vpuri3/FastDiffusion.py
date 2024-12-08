#
import torch
from torch import nn
import diffusers

import mlutils

__all__ = [
    'unet',
]

#======================================================================#
class DiffuserModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args):
        return self.model(*args)['sample']

#======================================================================#
def unet():

    block_out_channels=(128, 128, 256, 256, 512, 512)

    down_block_types=( 
        "DownBlock2D",      # a regular ResNet downsampling block
        "DownBlock2D", 
        "DownBlock2D", 
        "DownBlock2D", 
        "AttnDownBlock2D",  # downsampling block with spatial self-attention
        "DownBlock2D",
    )

    up_block_types=(
        "UpBlock2D",      # regular ResNet upsampling block
        "AttnUpBlock2D",  # upsampling block with spatial self-attention
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D"  
    )

    model = diffusers.UNet2DModel(
        block_out_channels=block_out_channels,
        out_channels=3, in_channels=3,
        up_block_types=up_block_types,
        down_block_types=down_block_types,
        add_attention=True,
    )

    return DiffuserModelWrapper(model)

#======================================================================#
#
