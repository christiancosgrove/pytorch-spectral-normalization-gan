from torch import nn
import torch.nn.functional as F

from spectral_normalization import SpectralNormLinear

width = 28
h_dim = 100
z_dim = 100

class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.fc1 = nn.Linear(z_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, width * width)

    def forward(self, z):
        return nn.Tanh()(self.fc2(F.relu(self.fc1(z))).view(-1, width, width))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = SpectralNormLinear(width * width, h_dim)
        self.fc2 = SpectralNormLinear(h_dim, 1)


    def forward(self, x):
        shaped = x.view(-1, width * width)
        return nn.Sigmoid()(self.fc2(F.relu(self.fc1(shaped))))

