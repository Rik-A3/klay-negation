import math

import torch

CUTOFF = -math.log(2)


def log1mexp(x, eps):
    """
    Numerically accurate evaluation of log(1 - exp(x)) for x < 0.
    See [Maechler2012accurate]_ for details.
    https://github.com/pytorch/pytorch/issues/39242
    """
    mask = CUTOFF < x  # x < 0
    return torch.where(
        mask,
        (-x.expm1() + eps).log(),
        (-x.exp() + eps).log1p(),
    )


def negate_real(x, eps):
    return 1 - x


def unroll_ixs(ixs):
    deltas = torch.diff(ixs)
    ixs = torch.arange(len(deltas), dtype=torch.long, device=ixs.device)
    return ixs.repeat_interleave(repeats=deltas)
