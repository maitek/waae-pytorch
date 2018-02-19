import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

cuda = True
cnt = 0
lr = 1e-4
out_dir = "out_aae3"
batch_size = 256

nc = 3 # number of channels
nz = 64 # size of latent vector
ngf = 64 # decoder (generator) filter factor
ndf = 64 # encoder filter factor
h_dim = 128 # discriminator hidden size
lam = 1 # regulization coefficient


transform = transforms.Compose([
        transforms.CenterCrop(160),
        transforms.Scale(64),
        transforms.ToTensor(),
         #transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

dataset = datasets.ImageFolder('data/', transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=20, drop_last=True)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.main = [
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        ]
        for idx, module in enumerate(self.main):
            self.add_module(str(idx), module)

    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.main = [
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, nz, 4, 1, 0, bias=False),
        ]

        for idx, module in enumerate(self.main):
            self.add_module(str(idx), module)

    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = [
            nn.Linear(nz, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        ]
        for idx, module in enumerate(self.main):
            self.add_module(str(idx), module)
    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        return x

Q = Encoder()
P = Decoder()
D = Discriminator()


if os.path.exists("Q_latest.pth"):
    Q = torch.load("Q_latest.pth")
if os.path.exists("P_latest.pth"):
    P =torch.load("P_latest.pth")
if os.path.exists("D_latest.pth"):
    D = torch.load("D_latest.pth")

if cuda:
    Q = Q.cuda()
    P = P.cuda()
    D = D.cuda()

def reset_grad():
    Q.zero_grad()
    P.zero_grad()
    D.zero_grad()

Q_solver = optim.Adam(Q.parameters(), lr=lr)
P_solver = optim.Adam(P.parameters(), lr=lr)
D_solver = optim.Adam(D.parameters(), lr=lr*0.1)

for it in range(100000):

    for batch_idx, batch_item in enumerate(data_loader):
        #X = sample_X(mb_size)
        """ Reconstruction phase """
        X = Variable(batch_item[0])
        if cuda:
            X = X.cuda()

        z_sample = Q(X)

        X_sample = P(z_sample)
        recon_loss = F.mse_loss(X_sample, X)

        recon_loss.backward()
        P_solver.step()
        Q_solver.step()
        reset_grad()

        """ Regularization phase """
        # Discriminator
        for _ in range(5):
            z_real = Variable(torch.randn(batch_size, nz))
            if cuda:
                z_real = z_real.cuda()

            z_fake = Q(X).view(batch_size,-1)

            D_real = D(z_real)
            D_fake = D(z_fake)

            #D_loss = -torch.mean(torch.log(D_real) + torch.log(1 - D_fake))
            D_loss = -(torch.mean(D_real) - torch.mean(D_fake))

            D_loss.backward()
            D_solver.step()

            # Weight clipping
            for p in D.parameters():
                p.data.clamp_(-0.01, 0.01)

            reset_grad()

        # Generator
        z_fake = Q(X).view(batch_size,-1)
        D_fake = D(z_fake)

        #G_loss = -torch.mean(torch.log(D_fake))
        G_loss = -torch.mean(D_fake)

        G_loss.backward()
        Q_solver.step()
        reset_grad()

        if batch_idx % 10 == 0:
            print('Iter-{}; D_loss: {:.4}; G_loss: {:.4}; recon_loss: {:.4}'
                  .format(batch_idx, D_loss.data[0], G_loss.data[0], recon_loss.data[0]))
            torch.save(Q,"Q_latest.pth")
            torch.save(P,"P_latest.pth")
            torch.save(D,"D_latest.pth")

        # Print and plot every now and then
        if batch_idx % 100 == 0:

            z_real = z_real.unsqueeze(2).unsqueeze(3) # add 2 dimensions
            if cnt % 2 == 0:
                samples = P(z_real) # Generated
            else:
                samples = X_sample # Reconstruction
            #samples = X_sample
            if cuda:
                samples = samples.cpu()
            samples = samples.data.numpy()[:16]

            fig = plt.figure(figsize=(4, 4))
            gs = gridspec.GridSpec(4, 4)
            gs.update(wspace=0.05, hspace=0.05)

            for i, sample in enumerate(samples):
                ax = plt.subplot(gs[i])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                sample = np.swapaxes(sample,0,2)
                plt.imshow(sample)


            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            plt.savefig('{}/{}.png'
                        .format(out_dir,str(cnt).zfill(3)), bbox_inches='tight')
            cnt += 1
            plt.close(fig)
