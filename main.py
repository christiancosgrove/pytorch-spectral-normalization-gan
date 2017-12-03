import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from model import Discriminator, Generator

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--momentum', type=float, default=0.5)
args = parser.parse_args()

loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data/', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor()])),
        batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)

Z_dim = 100

discriminator = Discriminator().cuda()
generator = Generator(Z_dim).cuda()

optim_disc = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.momentum,0.999))
optim_gen  = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.momentum,0.999))

def train(epoch):
    discriminator.train()
    for batch_idx, (data, target) in enumerate(loader):
        if data.size()[0] != args.batch_size:
            continue
        data, target = Variable(data.cuda()), Variable(target.cuda())
        z = Variable(torch.randn(args.batch_size, Z_dim).cuda())
        optim_disc.zero_grad()
        optim_gen.zero_grad()
        output = discriminator(data)

        disc_loss = nn.BCEWithLogitsLoss()(discriminator(data), Variable(torch.ones(args.batch_size, 1).cuda())) + \
            nn.BCEWithLogitsLoss()(discriminator(generator(z)), Variable(torch.zeros(args.batch_size, 1).cuda()))
        disc_loss.backward(retain_graph=True)
        optim_disc.step()

        optim_disc.zero_grad()
        optim_gen.zero_grad()
        gen_loss = nn.BCEWithLogitsLoss()(discriminator(generator(z)), Variable(torch.ones(args.batch_size, 1).cuda()))
        gen_loss.backward()
        optim_gen.step()

        if batch_idx % 10 == 0:
            print('disc loss', disc_loss.data[0], 'gen loss', gen_loss.data[0])

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

def evaluate(epoch):
    z = Variable(torch.randn(args.batch_size, Z_dim).cuda())

    samples = generator(z).cpu().data.numpy()[:16]


    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    if not os.path.exists('out/'):
        os.makedirs('out/')

    plt.savefig('out/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')

for epoch in range(100):
    train(epoch)
    evaluate(epoch)
