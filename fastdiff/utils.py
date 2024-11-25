#
import torch
import mlutils

__all__ = [
    'sample_shortcut',
    'sample_flow_matching',

    'loss_shortcut',
    'loss_flow_matching',
]

#======================================================================#
# DIFFUSION
#======================================================================#

@torch.no_grad()
def sample_shortcut(model, x0, N):
    return

@torch.no_grad()
def sample_flow_matching(model, x0, N):
    """
    Arguments:
    - model: velocity v(x, t)
    - x0: noise ~N(0, I)
    - N: number of steps

    Output:
    - x1: sample ~ data distribution
    """
    xt = x0
    for t in range(N):
        t0 = t / N
        t1 = (t + 1) / N
        tt0 = torch.full((x0.size(0),), t0, device=x0.device)

        vt = model(xt, tt0)
        xt = xt + (t1 - t0) * vt

    return xt

def loss_shortcut(x1, trainer: mlutils.Trainer):
    return

def loss_flow_matching(x1, trainer: mlutils.Trainer):
    B = x1.size(0)
    device = x1.device

    x0 = torch.randn_like(x1)
    tt = torch.rand(B, device=device).view(-1, 1, 1, 1)

    xt = x0 * tt + (1 - tt) * x1
    vt = trainer.model(xt, tt.view(-1))

    return trainer.lossfun(vt, x1 - x0)

#======================================================================#
#
