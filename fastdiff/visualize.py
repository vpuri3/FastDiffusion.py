#
import torch
import torchvision
from cleanfid import fid

import os
import shutil

import mlutils

__all__ = [
    'Visualizer',
]

#======================================================================#
class Visualizer:
    def __init__(self, sample_fun, out_dir, image_size, save_every, data_root=None, fid=False):

        self.sample_fun = sample_fun
        self.sample_steps = 8

        self.out_dir = out_dir
        self.image_size = image_size
        self.save_every = save_every

        self.fid = fid
        self.data_root = data_root

        return

    def save_dir(self, trainer):
        nsave = trainer.epoch // self.save_every
        save_dir = os.path.join(self.out_dir, f'sample{str(nsave).zfill(2)}')
        return nsave, save_dir

    def sample(self, trainer):
        nsave, save_dir = self.save_dir(trainer)

        shape = (64, 3, self.image_size, self.image_size)
        x0 = torch.randn(shape, device=trainer.device)

        for s in range(self.sample_steps):
            N = 2 ** s
            x1 = self.sample_fun(trainer.model, x0, N) * 0.5 + 0.5
            grid = torchvision.utils.make_grid(x1)

            grid_path = os.path.join(save_dir, f"steps{str(N).zfill(4)}.png")
            torchvision.utils.save_image(grid, grid_path, nrow=8)

        return

    def compute_fid(self, trainer):
        if not self.fid:
            return

        val_dir = os.path.join(self.data_root, 'test') # 'val'
        if not os.path.exists(val_dir):
            os.makedirs(val_dir)

        nsave, save_dir = self.save_dir(trainer)
        fid_score = fid.compute_fid(save_dir, val_dir)
        print(f'fid score: {fid_score}')

        return

    @torch.no_grad()
    def __call__(self, trainer: mlutils.Trainer):
        trainer.model.eval()

        if (trainer.epoch % self.save_every) != 0:
            return

        _, save_dir = self.save_dir(trainer)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        trainer.save(os.path.join(save_dir, 'model.pt'))
        self.sample(trainer)
        self.compute_fid(trainer)

        return

#======================================================================#
#
