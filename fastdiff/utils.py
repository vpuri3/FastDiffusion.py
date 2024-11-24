#
import torch
import numpy as np

import mlutils

__all__ = [
]

#======================================================================#
# DIFFUSION
#======================================================================#

from torch import nn

__all__.append('sample')
__all__.append('loss_flow_matching')
__all__.append('FlowMatchingLoss')

@torch.no_grad()
def sample(model, x0, N):
    xt = x0
    for t in range(N):
        t0 = t / N
        t1 = (t + 1) / N
        tt0_tensor = torch.tensor(t0, device=x0.device)

        vt = model(xt, t0_tensor)
        xt = xt + (t1 - t0) * vt

    return xt

def loss_flow_matching(x1, trainer: mlutils.Trainer):
    B = x1.size(0)
    device = x1.device

    x0 = torch.rand_like(x1)
    tt = torch.rand(B, device=device).view(-1, 1, 1, 1)

    xt = x0 * tt + (1 - tt) * x1
    vt = trainer.model(xt, tt.view(-1))

    return trainer.lossfun(vt, x1 - x0)

class FlowMatchingLoss(nn.Module):
    def __init__(self, lossfun=nn.MSELoss()):
        super().__init__()
        self.lossfun = lossfun
        self.generator = None

    def forward(self, x1, model):
        B = x1.size(0)
        device = x1.device

        x0 = torch.rand_like(x1)
        tt = torch.rand(B, device=device).view(-1, 1, 1, 1)

        xt = x0 * tt + (1 - tt) * x1
        vt = model(xt, tt.view(-1))

        return self.lossfun(vt, x1 - x0)

#======================================================================#
# visualize
#======================================================================#
class Visualizer:
    def __init__(self, ):
        return
    def visualize(self, trainer: mlutils.Trainer):
        """
        Use trained model to visualize forward and backward diffusion process

        1. Sample the first batch of images from the dataloader
        2. Visualize the forward diffusion process at 0%, 25%, 50%, 75%, 99% of the total timesteps
        3. Use the last image from the forward diffusion process to visualize the backward diffusion
            process at 0%, 25%, 50%, 75%, 99% of the total timesteps
            you can use (percent * self.model.num_timesteps) to get the timesteps
        4. Save the images in wandb
        """
        trainer.model.eval()
        trainer.model.to(trainer.device)

        data_1 = next(self.dl)
        data_2 = torch.randn_like(data_1)

        data_1, data_2 = data_1.to(self.device), data_2.to(self.device)

        percent = [0.0, 0.25, 0.50, 0.75, 0.99]
        t_visualize = [int(i * self.model.num_timesteps) for i in percent]

        # visualize backward diffusion process
        img_backward = []
        for t in range(self.model.num_timesteps - 1, -1, -1):
            img = self.model.p_sample(
                img,
                torch.full((img.shape[0],), t, device=img.device, dtype=torch.long),
                t,
            )
            if t in t_visualize:
                img_vis = torch.clamp(img, -1, 1)
                img_vis = (img_vis + 1) / 2
                img_backward.append(img_vis)

#======================================================================#
# models
#======================================================================#
import diffusers

__all__.append('unet')

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

class DiffuserModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args):
        return self.model(*args)['sample']

#======================================================================#
__all__.append('AFHQDataset')

import PIL
from pathlib import Path
from torchvision import transforms

class AFHQDataset(torch.utils.data.Dataset):
    def __init__(
        self, root, image_size, data_class, augment=False, exts=["jpg", "jpeg", "png"]
    ):
        super().__init__()
        self.root = root
        self.image_size = image_size
        if data_class == "all" or data_class == None:
            self.paths = [
                p for ext in exts for p in Path(f"{root}").glob(f"**/*.{ext}")
            ]
        else:
            self.paths = [
                p
                for ext in exts
                for p in Path(f"{root}/{data_class}").glob(f"*.{ext}")
            ]

        if augment:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((int(image_size * 1.12), int(image_size * 1.12))),
                    transforms.RandomCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda t: (t * 2) - 1),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((int(image_size * 1.12), int(image_size * 1.12))),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda t: (t * 2) - 1),
                ]
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = PIL.Image.open(path)
        img = img.convert("RGB")
        return self.transform(img)

#======================================================================#
#
