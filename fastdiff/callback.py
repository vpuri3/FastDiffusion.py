#
import torch
import torchvision
from cleanfid import fid

import os
import math
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
        sample_dirs = [dir for dir in os.listdir(self.out_dir) if dir.startswith('sample')]
        load_dir = sorted(sample_dirs)[-1]
        model_file = os.path.join(self.out_dir, load_dir, 'model.pt')

        print("loading model from", model_file)

        snapshot = torch.load(model_file, weights_only=False, map_location='cpu')
        trainer.model.load_state_dict(snapshot['model_state'])
        trainer.model.to(trainer.device)

        return

    def get_save_dir(self, trainer, final=False):
        if not final:
            nsave = trainer.epoch // self.save_every
            save_dir = os.path.join(self.out_dir, f'sample{str(nsave).zfill(2)}')
        else:
            save_dir = os.path.join(self.out_dir, f'final')
        return save_dir

    def batch_shape(self, B: int):
        return (B, 3, self.image_size, self.image_size)

    def save_samples(
        self, model, save_dir, device, x0=None, mode='grid', nrow=None,
        schedule_type='default', **schedule_params,
    ):
        if x0 is None:
            x0 = torch.randn(self.batch_shape(64), device=device)

        if mode == 'collage':
            samples = []
            assert x0 is not None
            assert x0.size(0) == 1

        for s in range(self.log_max_steps):
            N = 2 ** s
            x1 = model.sample(x0, N, schedule_type, **schedule_params) * 0.5 + 0.5

            if mode == 'grid':
                nrow = int(math.sqrt(x0.size(0))) if nrow is None else nrow
                grid_imgs = torchvision.utils.make_grid(x1)
                grid_path = os.path.join(save_dir, f"sample.{str(N).zfill(4)}.png")
                torchvision.utils.save_image(grid_imgs, grid_path, nrow=nrow)
            elif mode == 'dir':
                B = x0.size(0)
                out_dir = os.path.join(save_dir, f'{schedule_type}.{str(N).zfill(4)}')
                os.makedirs(out_dir, exist_ok=True)

                for b in range(B):
                    image_path = os.path.join(out_dir, f'sample{str(b).zfill(4)}.png')
                    torchvision.utils.save_image(x1[b], image_path)
            elif mode == 'collage':
                samples.append(x1)

        if mode == 'collage':
            nrow = self.log_max_steps
            x1 = torch.cat(samples, dim=0)
            grid_imgs = torchvision.utils.make_grid(x1)
            grid_path = os.path.join(save_dir, f"{schedule_type}.png")
            torchvision.utils.save_image(grid_imgs, grid_path, nrow=nrow)

        return

    @torch.no_grad()
    def __call__(self, trainer: mlutils.Trainer, final=False):

        device = trainer.device

        model = trainer.model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        model.eval()

        save_dir = self.get_save_dir(trainer, final=final)
        print("saving samples to", save_dir)
        os.makedirs(save_dir, exist_ok=True)

        #------------------------#
        #  during training
        #------------------------#

        if not final:
            if (trainer.epoch % self.save_every) == 0:
                return

            trainer.save(os.path.join(save_dir, 'model.pt'))
            self.save_samples(model, save_dir, device, mode='grid')

            return

        #------------------------#
        # After training is done
        #------------------------#

        x0_grd = torch.randn(self.batch_shape( 64), device=device)
        x0_dir = torch.randn(self.batch_shape(500), device=device)
        x0_col = torch.randn(self.batch_shape(  1), device=device)

        for schedule_type in model.schedule_types:
            print(f'sampling with {schedule_type}')

            schedule_dir = os.path.join(save_dir, schedule_type)
            os.makedirs(schedule_dir, exist_ok=True)

            self.save_samples(model, schedule_dir, device, x0=x0_grd, mode='grid'   , schedule_type=schedule_type)
            self.save_samples(model, schedule_dir, device, x0=x0_dir, mode='dir'    , schedule_type=schedule_type)
            self.save_samples(model, schedule_dir, device, x0=x0_col, mode='collage', schedule_type=schedule_type)

        #------------------------#
        # fid scores
        #------------------------#

        if self.fid:
            fid_file = os.path.join(save_dir, 'fid.txt')

            with open(fid_file, 'w') as file:
                for schedule_type in model.schedule_types:
                    for s in range(self.log_max_steps):
                        N = 2 ** s
                        NN = str(N).zfill(4)
                        exp_dir = os.path.join(save_dir, schedule_type, f'{schedule_type}.{NN}')

                        fid_score = self.compute_fid(exp_dir)

                        print(f'FID score of {exp_dir}: {fid_score}')
                        file.write(f'{schedule_type}\t{NN}\t{fid_score}\n')

        return

    def compute_fid(self, exp_dir: str):

        val_dir = os.path.join(self.data_root, '..', 'test', 'cat')

        assert os.path.exists(exp_dir)
        assert os.path.exists(val_dir)

        fid_score = fid.compute_fid(exp_dir, val_dir)

        return fid_score

#======================================================================#
#
