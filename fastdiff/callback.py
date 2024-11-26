#
import torch
import torchvision
from cleanfid import fid

import os
import shutil

import mlutils

__all__ = [
    'Callback',
]

#======================================================================#
class Callback:
    def __init__(self, out_dir, image_size, save_every, data_root=None, fid=False):

        self.log_max_steps = 8

        self.out_dir = out_dir
        self.image_size = image_size
        self.save_every = save_every

        self.fid = fid
        self.data_root = data_root

        return

    def load(self, trainer):
        load_dir = sorted(os.listdir(self.out_dir))[-1]
        model_file = os.path.join(self.out_dir, load_dir, 'model.pt')

        snapshot = torch.load(model_file, weights_only=False, map_location='cpu')
        trainer.model.load_state_dict(snapshot['model_state']).to(trainer.device)

        return

    def save_dir(self, trainer, final=False):
        if not final:
            nsave = trainer.epoch // self.save_every
            save_dir = os.path.join(self.out_dir, f'sample{str(nsave).zfill(2)}')
        else:
            save_dir = os.path.join(self.out_dir, f'sample')
        return save_dir

    def sample(self, trainer, save_dir):
        shape = (64, 3, self.image_size, self.image_size)
        x0 = torch.randn(shape, device=trainer.device)

        if isinstance(trainer.model, torch.nn.parallel.DistributedDataParallel):
            sample_fun = trainer.model.module.sample
        else:
            sample_fun = trainer.model.sample

        for s in range(self.log_max_steps):
            N = 2 ** s
            # x1 = sample_fun(trainer.model, x0, N) * 0.5 + 0.5
            x1 = sample_fun(x0, N) * 0.5 + 0.5
            grid = torchvision.utils.make_grid(x1)

            grid_path = os.path.join(save_dir, f"steps{str(N).zfill(4)}.png")
            torchvision.utils.save_image(grid, grid_path, nrow=8)

        return

    @torch.no_grad()
    def __call__(self, trainer: mlutils.Trainer, final=False):
        trainer.model.eval()

        if not final:
            if (trainer.epoch % self.save_every) != 0:
                return

        save_dir = self.save_dir(trainer, final=final)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        trainer.save(os.path.join(save_dir, 'model.pt'))
        self.sample(trainer, save_dir)

        # if self.fid:
        #     self.compute_fid(trainer)

        return

    def compute_fid(self, trainer):
        val_dir = os.path.join(self.data_root, 'test') # 'val'
        if not os.path.exists(val_dir):
            os.makedirs(val_dir)

        save_dir = self.save_dir(trainer)
        fid_score = fid.compute_fid(save_dir, val_dir)
        print(f'fid score: {fid_score}')

        return

#======================================================================#
#
