#
import torch
import numpy as np

import os
import shutil

import mlutils

__all__ = [
    'Visualizer',
]

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
#
