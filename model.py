from torch import nn
import torch.nn.functional as F

from spectral_normalization import SpectralNorm

width = 28
h_dim = 100
z_dim = 100

class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        self.model = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 2*h_dim),
            nn.ReLU(),
            nn.Linear(2*h_dim, width * width),
            nn.Tanh())

    def forward(self, z):
        return self.model(z).view(-1, width, width)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            SpectralNorm(nn.Linear(width * width, h_dim)),
            nn.LeakyReLU(),
            SpectralNorm(nn.Linear(h_dim, h_dim)),
            nn.LeakyReLU(),
            SpectralNorm(nn.Linear(h_dim, 1)),
            )

        # self.model = nn.Sequential(
        #     (nn.Linear(width * width, h_dim)),
        #     nn.LeakyReLU(),
        #     (nn.Linear(h_dim, h_dim)),
        #     nn.LeakyReLU(),
        #     (nn.Linear(h_dim, 1)),
        #     )


    def forward(self, x):
        shaped = x.view(-1, width * width)
        return self.model(shaped)

