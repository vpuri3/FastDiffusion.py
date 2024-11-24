#
import torch
import mlutils

__all__ = [
    'sample',
    'loss_flow_matching',
]

#======================================================================#
# DIFFUSION
#======================================================================#

from torch import nn

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

#======================================================================#
#
