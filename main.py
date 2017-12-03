import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from model import Discriminator, Generator

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.5)
args = parser.parse_args()

loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data/', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)

discriminator = Discriminator().cuda()

optim_disc = optim.SGD(discriminator.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch):
    discriminator.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = Variable(data.cuda()), Variable(target.cuda())
        optim_disc.zero_grad()
        output = discriminator(data)

        loss = F.nll_loss(output, target)
        loss.backward()
        optim_disc.step()
        if batch_idx % 10 == 0:
            print('loss', loss.data[0])


for epoch in range(100):
    train(epoch)
