#
import torch
from torch import nn
import mlutils
import math
import numpy as np

__all__ = [
    'Diffusion',
]

#======================================================================#
# DIFFUSION
#======================================================================#

# model: velocity v(x(t), t)
# x0: noise ~N(0, I)
# N: number of steps
# Output:
# x1: sample ~ data distribution

class Diffusion(nn.Module):

    schedule_types = [
        'default',
        'cosine',
        'laplace',
        'cauchy',
        # 'cosine_shifted',
        # 'cosine_scaled',
        'exponential',
        'quadratic'
    ]

    def __init__(self, model, mode: int):
        super().__init__()

        msg = "mode must be 0 for Flow Matching (FM) or 1 Shortcut Model (SM)"
        assert (mode == 0) or (mode == 1), msg

        self.mode = mode
        self.model = model
        self.lossfun = nn.MSELoss()
        self.log_max_steps = 8 # 2 ** N

    def noise_like(self, x):
        return torch.randn_like(x)
    
    def cosine_schedule(self, t):
        return 1 - np.cos(t * math.pi / 2)

    def exponential_schedule(self,t,beta=1.0):
        return 1 - np.exp(-beta * t)
    
    def quadratic_schedule(self, t):
        return t**2

    def laplace_schedule(self, t, mu=0, b=0.5):
#         return mu - b * np.sign(0.5 - t) * np.log(1 - 2 * np.abs(t - 0.5))
        return mu - b * np.sign(0.5 - t) * np.log(np.maximum(1 - 2 * np.abs(t - 0.5), 1e-6))

    def cauchy_schedule(self, t, mu=0, gamma=1):
        return mu + gamma * np.tan(math.pi / 2 * (1 - 2 * t))

    def cosine_shifted_schedule(self, t, mu=0):
        return mu + 2 * np.log(np.tan(math.pi * t / 2))

    def cosine_scaled_schedule(self, t, s=1):
        return 2 / s * np.log(np.tan(math.pi * t / 2))

    def apply_schedule(self, t, schedule_type='default', **schedule_params):
        if schedule_type == 'default':
            return t
        elif schedule_type == 'cosine':
            return self.cosine_schedule(t)
        elif schedule_type == 'laplace':
            return self.laplace_schedule(t, **schedule_params)
        elif schedule_type == 'cauchy':
            return self.cauchy_schedule(t, **schedule_params)
        elif schedule_type == 'cosine_shifted':
            return self.cosine_shifted_schedule(t, **schedule_params)
        elif schedule_type == 'cosine_scaled':
            return self.cosine_scaled_schedule(t, **schedule_params)
        elif schedule_type == 'exponential':
            return self.exponential_schedule(t)
        elif schedule_type == 'quadratic':
            return self.quadratic_schedule(t)

    @torch.no_grad()
    def sample(self, x0, N, schedule_type='default', **schedule_params):
        xt = x0
        d  = 1 / N
        dd = torch.full((x0.size(0),), d, device=x0.device) if self.mode == 1 else None

        for t in range(N):
            t  = t / N
            t  = self.apply_schedule(t, schedule_type, **schedule_params)
            tt = torch.full((x0.size(0),), t, device=x0.device)
            vt = self.predict_velocity(xt, tt, dd)
            xt = xt + d * vt

        return xt
    
    def predict_velocity(self, xt, t, d):
        if self.mode == 0:  # Flow Matching
            return self.model(xt, t)
        else:  # Shortcut Model
            return self.model(xt, t, d)

    def _train_sample(self, x1):
        B = x1.size(0)
        device = x1.device

        x0 = self.noise_like(x1)
        tt = torch.rand(B, device=device)
        dd = 1 / (2 ** torch.randint(self.log_max_steps, (B,), device=device))

        xt = x1 * tt.view(-1,1,1,1) + (1 - tt).view(-1,1,1,1) * x0
     
        return x0, xt, tt, dd
    
    #==================================#
    # losses
    #==================================#

    def loss_SM(self, x1):
        B = x1.size(0) // 4
        x1_FM, x1_CM = x1[:B], x1[B:]

        loss_FM = self.loss_FM(x1_FM, use_d=True)
        loss_CM = self.loss_CM(x1_CM)

        return loss_FM + loss_CM

    def loss_CM(self, x1):
        x0, xt, tt, dd = self._train_sample(x1)

        # vt_XY: v(t + X * d, Y * d)

        vt_01 = self.model(xt, tt, dd)
        vt_02 = self.model(xt, tt, dd * 2)
        vt_11 = self.model(xt + dd.view(-1,1,1,1) * vt_01, tt + dd, dd)
        v_avg = 0.5 * (vt_01 + vt_11)

        mask = ((tt + dd) > 1.0) * ((tt + 2 * dd) > 1.0)
        mask = mask.view(-1,1,1,1)

        return self.lossfun(vt_02, v_avg.detach())

    def loss_FM(self, x1, use_d: bool):
        x0, xt, tt, dd = self._train_sample(x1)

        if use_d:
            vt = self.model(xt, tt, dd)
        else:
            vt = self.model(xt, tt)
        target_velocity = x1 - x0
        
        return self.lossfun(vt, target_velocity)

    #==================================#

    def forward(self, x1):
        if self.mode == 0:
            return self.loss_FM(x1, use_d=False)
        else:
            return self.loss_SM(x1)

#======================================================================#
# schedules
#======================================================================#

#======================================================================#
#
