import torch
from torch.optim.optimizer import Optimizer, required

from torch.autograd import Variable
import torch.nn.functional as F


class SpectralNormOptimizer(Optimizer):

	def __init__(self, params, lr=required):
		defaults = dict(lr=lr)

		super(SpectralNormOptimizer, self).__init__(params, defaults)

	def step(self, closure=None):
		loss = None
		if closure is not None:
			loss = closure()

		for group in self.param_groups:

			for p in group['params']:
				if p.grad is None:
					continue

				d_p = p.grad.data
				p.data.add_(-group['lr'], d_p)

		return loss


#approximates the spectral norm (maximum singular value) of W
#uses power iteration method
def spectral_norm(W, u=None, power_iterations=1):
	if u is None:
		u = Variable(W.data.new((W.data.shape[1]))).data.normal_(0,1)

	u_n = u
	for i in range(power_iterations):
		v_n = F.normalize(torch.mv(W.data, u_n), p=2, dim=0)
		u_n = F.normalize(torch.mv(torch.t(W.data), v_n), p=2, dim=0)

	prod = torch.mv(torch.t(W.data), v_n)
	singular_value = torch.dot(u_n, prod)
	return singular_value, u_n, v_n

from torch import nn

class SpectralNormLinear(nn.Linear):
	def __init__(self, in_size, out_size):
		self.u = None
		super(SpectralNormLinear, self).__init__(in_size, out_size)

	def W_bar(self):
		singular_value, u_n, _ = spectral_norm(self.weight, self.u)
		self.u = u_n
		return self.weight / singular_value

	def forward(self, x):
		return F.linear(x, self.W_bar(), self.bias)