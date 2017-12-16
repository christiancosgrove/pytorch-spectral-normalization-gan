import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from model import Discriminator, Generator
from spectral_normalization import SpectralNormOptimizer

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--momentum', type=float, default=0.5)
args = parser.parse_args()

# loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data/', train=True, download=True,
#         transform=transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,),(0.5,))])),
#         batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data/', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
        batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)

Z_dim = 128

discriminator = Discriminator().cuda()
generator = Generator(Z_dim).cuda()

optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr, betas=(0.5,0.999))
optim_gen  = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5,0.999))

# optim_disc = optim.SGD(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr)
# optim_gen  = optim.SGD(generator.parameters(), lr=args.lr)

def train(epoch):
    discriminator.train()
    for batch_idx, (data, target) in enumerate(loader):
        if data.size()[0] != args.batch_size:
            continue
        data, target = Variable(data.cuda()), Variable(target.cuda())
        z = Variable(torch.randn(args.batch_size, Z_dim).cuda())
        disc_loss = nn.BCEWithLogitsLoss()(discriminator(data), Variable(torch.ones(args.batch_size, 1).cuda())) + \
            nn.BCEWithLogitsLoss()(discriminator(generator(z)), Variable(torch.zeros(args.batch_size, 1).cuda()))
        disc_loss.backward()
        optim_disc.step()

        optim_disc.zero_grad()
        optim_gen.zero_grad()
        gen_loss = nn.BCEWithLogitsLoss()(discriminator(generator(z)), Variable(torch.ones(args.batch_size, 1).cuda()))
        gen_loss.backward()
        optim_gen.step()

        if batch_idx % 100 == 0:
            print('disc loss', disc_loss.data[0], 'gen loss', gen_loss.data[0])

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

fixed_z = Variable(torch.randn(args.batch_size, Z_dim).cuda())
def evaluate(epoch):

    samples = generator(fixed_z).cpu().data.numpy()[:64]


    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.transpose((1,2,0)) * 0.5 + 0.5)

    if not os.path.exists('out/'):
        os.makedirs('out/')

    plt.savefig('out/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
    plt.close(fig)

for epoch in range(2000):
    train(epoch)
    evaluate(epoch)
