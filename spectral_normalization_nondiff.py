# non-differentiable spectral normalization module
# weight tensors are normalized directly
import torch
from torch.optim.optimizer import Optimizer, required

from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations

    def _update_u_v(self):
        if not self._made_params():
            self._make_params()
        w = getattr(self.module, self.name)
        u = getattr(self.module, self.name + "_u")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u))
            u = l2normalize(torch.mv(w.view(height,-1).data, v))

        setattr(self.module, self.name + "_u", u)
        w.data = w.data / torch.dot(u, torch.mv(w.view(height,-1).data, v))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = l2normalize(w.data.new(height).normal_(0, 1))

        self.module.register_buffer(self.name + "_u", u)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)