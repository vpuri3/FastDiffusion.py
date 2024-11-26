#
import torch
from torch import nn
import mlutils

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

    @torch.no_grad()
    def sample(self, x0, N):
        xt = x0
        d  = 1 / N
        if self.mode == 1:
            dd = torch.full((x0.size(0),), d, device=x0.device)

        for t in range(N):
            t  = t / N
            tt = torch.full((x0.size(0),), t, device=x0.device)

            if self.mode == 0: # FM
                vt = self.model(xt, tt)
            else: # SM
                vt = self.model(xt, tt, dd)

            xt = xt + d * vt

        return xt

    def _sample(self, x1):
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
        x0, xt, tt, dd = self._sample(x1)

        # vt_XY: v(t + X * d, Y * d)

        vt_01 = self.model(xt, tt, dd)
        vt_02 = self.model(xt, tt, dd * 2)
        vt_11 = self.model(xt + dd.view(-1,1,1,1) * vt_01, tt + dd, dd)
        v_avg = 0.5 * (vt_01 + vt_11)

        return self.lossfun(vt_02, v_avg.detach())

    def loss_FM(self, x1, use_d: bool):
        x0, xt, tt, dd = self._sample(x1)

        if use_d:
            vt = self.model(xt, tt, dd)
        else:
            vt = self.model(xt, tt)

        return self.lossfun(vt, x1 - x0)

    #==================================#

    def forward(self, x1):
        if self.mode == 0:
            return self.loss_FM(x1, use_d=False)
        else:
            return self.loss_SM(x1)

#======================================================================#
#
