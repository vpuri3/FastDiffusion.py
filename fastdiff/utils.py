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

# model: velocity v(x(t), t)
# x0: noise ~N(0, I)
# N: number of steps
# Output:
# x1: sample ~ data distribution

@torch.no_grad()
def sample_shortcut(model, x0, N):
    d = 0
    return

@torch.no_grad()
def sample_flow_matching(model, x0, N):
    xt = x0
    for t in range(N):
        t0 = t / N
        t1 = (t + 1) / N
        tt0 = torch.full((x0.size(0),), t0, device=x0.device)

        vt = model(xt, tt0)
        xt = xt + (t1 - t0) * vt

    return xt

def loss_shortcut(x1, trainer: mlutils.Trainer):
    x1_FM, x1_CS = x1.chunk(2)

    # FLOW MATCHING
    x0_FM, xt_FM, tt_FM, dd_FM = _sample_CM(x1_FM)
    vt_FM = trainer.model(xt_FM, tt_FM.view(-1), 0 * dd_FM.view(-1))
    loss_FM = trainer.lossfun(vt_FM, x1_FM - x0_FM)

    # CONSISTENCY
    x0_CS, xt_CS, tt_CS, dd_CS = _sample_CM(x1_FM)

    loss_CM = trainer.lossfun(0)

    return loss_FM + loss_CM

def loss_flow_matching(x1, trainer: mlutils.Trainer):
    x0, xt, tt, _ = _sample(x1)
    vt = trainer.model(xt, tt.view(-1))

    return trainer.lossfun(vt, x1 - x0)

def _sample(x1):
    B = x1.size(0)
    device = x1.device

    x0 = torch.randn_like(x1)
    tt = torch.rand(x1.size(0), device=device).view(-1, 1, 1, 1)
    dd = torch.zeros_like(tt)

    xt = x1 * tt + (1 - tt) * x0

    return x0, xt, tt, dd

#======================================================================#
#
