from torch import nn
import torch.nn.functional as F

from spectral_normalization import SpectralNorm

width = 32
channels = 3
h_dim = 100
z_dim = 128
w_g = 4
leak = 0.1

class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels, out_channels, 3, stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, 3, 1, padding=0)
            )
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ConvTranspose2d(in_channels,out_channels, 3, stride, padding=0)
                )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.ReLU(),
            SpectralNorm(nn.Conv2d(in_channels, out_channels, 3, stride, padding=1)),
            nn.ReLU(),
            SpectralNorm(nn.Conv2d(out_channels, out_channels, 3, 1, padding=1))
            )
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Sequential(
                SpectralNorm(nn.Conv2d(in_channels,out_channels, 3, stride, padding=1))
            )

    def forward(self, x):
        return self.model(x) + self.bypass(x)



class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        self.dense = nn.ConvTranspose2d(z_dim, 1024, 4, stride=1)
        self.model = nn.Sequential(
            ResBlockGenerator(1024, 256, stride=2),
            ResBlockGenerator(256, 256, stride=2),
            ResBlockGenerator(256, 256, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, channels, 3, stride=1, padding=(1,1)),
            nn.Tanh())

    def forward(self, z):
        return self.model(self.dense(z.view(-1, self.z_dim, 1, 1)))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
                ResBlockDiscriminator(channels, 128, stride=2),
                ResBlockDiscriminator(128, 128, stride=2),
                ResBlockDiscriminator(128, 128),
                ResBlockDiscriminator(128, 128),
                nn.AvgPool2d(8),
                nn.ReLU(),
                SpectralNorm(nn.Conv2d(128, 1, 1))
            )

    def forward(self, x):
        m = self.model(x)
        return self.model(x).view(-1,1)