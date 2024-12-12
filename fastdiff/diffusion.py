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

class Diffusion(nn.Module):
    '''
    model: velocity v(x(t), t)
    x0: noise ~N(0, I)
    x1: sample ~ data distribution
    N: number of steps

    Flow Matching formulation

    xt = (1-t) x0 + t x1, t ~ [0, 1]
    vt = x1 - x0

    sampling: solve forward from 0 to 1

    Trig Flow formulation

    xt = cos(t pi/2) x0 + sin(t pi/2) x1
    vt = pi/2 cos(t pi/2) x1 -  pi/2 sin(t pi/2)

    '''

    schedule_types = [
        'default',
        'cosine',
        # 'laplace',
        # 'cauchy',
        # 'cosine_shifted',
        # 'cosine_scaled',
        'exponential',
        'quadratic'
    ]

    def __init__(self, model, mode: int, log_max_steps=6):
        super().__init__()

        msg = "mode must be 0 for Flow Matching (FM) or 1 Shortcut Model (SM)"
        assert (mode == 0) or (mode == 1), msg

        self.mode = mode
        self.model = model
        self.lossfun = nn.MSELoss()
        self.log_max_steps = log_max_steps # 2 ** N

    #==================================#
    # noise generator
    #==================================#

    def noise(self, *shape, device=None):
        return torch.randn(*shape, device=device)

    def noise_like(self, x):
        return torch.randn_like(x)

    #==================================#
    # sampler
    #==================================#
    
    @torch.no_grad()
    def sample(self, x0, N, schedule_type='default', **schedule_params):
        xt = x0
        d  = 1 / N
        dd = torch.full((x0.size(0),), d, device=x0.device) if self.mode == 1 else None

        for t in range(N):
            t  = t / N
            t  = apply_schedule(t, schedule_type, **schedule_params)
            tt = torch.full((x0.size(0),), t, device=x0.device)
            vt = self.query_model(xt, tt, dd)
            xt = xt + d * vt

        return xt

    def query_model(self, xt, t, d=None):
        if self.mode == 0:  # Flow Matching
            return self.model(xt, t)
        else:               # Shortcut Model
            assert d is not None
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

        x1_FM, x1_CS = x1[:B], x1[B:]

        loss_FM = self.loss_FM(x1_FM)
        loss_CS = self.loss_CS(x1_CS)

        return loss_FM + loss_CS

    def loss_CS(self, x1):
        # consistency loss

        x0, xt, tt, dd = self._train_sample(x1)

        # vt_XY: v(t + X * d, Y * d)

        vt_01 = self.model(xt, tt, dd)
        vt_02 = self.model(xt, tt, dd * 2)
        vt_11 = self.model(xt + dd.view(-1,1,1,1) * vt_01, tt + dd, dd)
        v_avg = 0.5 * (vt_01 + vt_11)

        # ignore infeasible situations
        mask = ((tt + dd) > 1.0) * ((tt + 2 * dd) > 1.0)
        mask = mask.view(-1,1,1,1)

        vt_02 = vt_02 * mask
        v_avg = v_avg * mask

        return self.lossfun(vt_02, v_avg.detach())

    def loss_FM(self, x1):
        x0, xt, tt, dd = self._train_sample(x1)
        vt = self.query_model(xt, tt, dd)
        target_velocity = x1 - x0

        return self.lossfun(vt, target_velocity)

    #==================================#

    def forward(self, x1):
        if self.mode == 0:
            return self.loss_FM(x1)
        else:
            return self.loss_SM(x1)
#

#======================================================================#
# Schedules
#======================================================================#
def cosine_schedule(t):
    return 1 - np.cos(t * math.pi / 2)

def exponential_schedule(t, beta=10.):
    return 1 - np.exp(-beta * t)

def quadratic_schedule(t):
    return t**2

def laplace_schedule(t, mu=0, b=0.5):
    # return mu - b * np.sign(0.5 - t) * np.log(1 - 2 * np.abs(t - 0.5))
    return mu - b * np.sign(0.5 - t) * np.log(np.maximum(1 - 2 * np.abs(t - 0.5), 1e-6))

def cauchy_schedule(t, mu=0, gamma=1):
    return mu + gamma * np.tan(math.pi / 2 * (1 - 2 * t))

def cosine_shifted_schedule(t, mu=0):
    return mu + 2 * np.log(np.tan(math.pi * t / 2))

def cosine_scaled_schedule(t, s=1):
    return 2 / s * np.log(np.tan(math.pi * t / 2))

def apply_schedule(t, schedule_type='default', **schedule_params):
    if schedule_type == 'default':
        return t
    elif schedule_type == 'cosine':
        return cosine_schedule(t)
    elif schedule_type == 'laplace':
        return laplace_schedule(t, **schedule_params)
    elif schedule_type == 'cauchy':
        return cauchy_schedule(t, **schedule_params)
    elif schedule_type == 'cosine_shifted':
        return cosine_shifted_schedule(t, **schedule_params)
    elif schedule_type == 'cosine_scaled':
        return cosine_scaled_schedule(t, **schedule_params)
    elif schedule_type == 'exponential':
        return exponential_schedule(t)
    elif schedule_type == 'quadratic':
        return quadratic_schedule(t)


#======================================================================#
#
