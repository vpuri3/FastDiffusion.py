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
    x1: sample ~ Data

    N : number of steps
    t : time [0 -> 1]
    d : step size = 1 / N

    Flow Matching formulation

    xt = (1-t) x0 + t x1, t ~ [0, 1]
    vt = x1 - x0

    sampling: solve forward from 0 to 1

    Trig Flow formulation

    xt = cos(t pi/2) x0 + sin(t pi/2) x1
    vt = pi/2 cos(t pi/2) x1 -  pi/2 sin(t pi/2)

    In general

    xt = a(t)  x0 + b(t)  x1
    vt = a'(t) x0 + b'(t) x1

    '''

    def __init__(self, model, mode: int, trig: bool, log_max_steps=6):
        super().__init__()

        msg = "mode must be 0 for Flow Matching (FM) or 1 Shortcut Model (SM)"
        assert (mode == 0) or (mode == 1), msg

        self.model = model
        self.mode = mode
        self.schedule = TrigSchedule() if trig else LinearSchedule()

        self.log_max_steps = log_max_steps # 2 ** N
        self.lossfun = nn.MSELoss()

    #==================================#
    # utilities
    #==================================#

    def noise(self, *shape, device=None):
        return torch.randn(*shape, device=device)

    def noise_like(self, x):
        return torch.randn_like(x)

    def velocity(self, xt, t, d=None):
        if self.mode == 0:  # Flow Matching
            return self.model(xt, t)
        else:               # Shortcut Model
            assert d is not None
            return self.model(xt, t, d)

    def _train_sample(self, x1, from_grid=False):
        B = x1.size(0)
        device = x1.device

        x0 = self.noise_like(x1)

        if from_grid:
            # [2, 4, 8, 16, 32]
            num_steps = 2 ** torch.randint(1, self.log_max_steps, (B,), device=device)

            # step size
            dd = 1 / num_steps

            # time - ensure t+2d <= 1
            tt = torch.rand(B, device=device) * (1 - 2 * dd)

        else:
            tt = torch.rand(B, device=device)
            dd = torch.rand(B, device=device)

        aa, bb = self.schedule(tt)
        xt = x0 * aa.view(-1,1,1,1) + x1 * bb.view(-1,1,1,1)

        return x0, xt, tt, dd

    #==================================#
    # sampler
    #==================================#
    
    @torch.no_grad()
    def sample(self, x0, N):
        xt = x0
        d  = 1 / N
        dd = torch.full((x0.size(0),), d, device=x0.device)

        for t in range(N):
            t  = t / N
            tt = torch.full((x0.size(0),), t, device=x0.device)
            vt = self.velocity(xt, tt, dd)
            xt = self.schedule.next(xt, vt, tt, dd)

        return xt

    #==================================#
    # losses
    #==================================#

    def loss_SM(self, x1):
        N = x1.size(0)
        B = N // 4
        r = B / N

        x1_FM, x1_CS = x1[:B], x1[B:]

        loss_FM = self.loss_FM(x1_FM)
        loss_CS = self.loss_CS(x1_CS)

        return loss_FM * r + loss_CS * (1 - r)

    def loss_CS(self, x1):
        # consistency loss

        _, xt, tt, dd = self._train_sample(x1, from_grid=True)

        # big step 1
        xB = self.schedule.next(xt, self.velocity(xt, tt, dd*2), tt, dd*2)
        # small step 1, 2
        xS = self.schedule.next(xt, self.velocity(xt, tt   , dd), tt   , dd)
        xS = self.schedule.next(xS, self.velocity(xS, tt+dd, dd), tt+dd, dd)

        return self.lossfun(xB, xS.detach()) # teach big to follow small

    def loss_FM(self, x1):
        x0, xt, tt, dd = self._train_sample(x1)
        vt = self.velocity(xt, tt, dd * 0) # FM doesn't use d and SM sets it to 0

        da, db = self.schedule.derv(tt)
        target = x0 * da.view(-1,1,1,1) + x1 * db.view(-1,1,1,1)

        return self.lossfun(vt, target)

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
class LinearSchedule:
    def __call__(self, t: torch.Tensor):
        return (1 - t), t

    def derv(self, t: torch.Tensor):
        o = torch.ones_like(t)
        return -o, o

    def next(self, xt, vt, t, d):
        d = d.view(-1,1,1,1)
        return xt + d * vt

class TrigSchedule:
    m = math.pi / 2

    def __call__(self, t: torch.Tensor):
        return torch.cos(self.m * t), torch.sin(self.m * t)

    def derv(self, t: torch.Tensor):
        return -self.m * torch.sin(self.m * t), self.m * torch.cos(self.m * t)

    def next(self, xt, vt, t, d):
        d = d.view(-1,1,1,1)
        return xt * torch.cos(self.m * d) + vt * torch.sin(self.m * d) / self.m

#======================================================================#
#
