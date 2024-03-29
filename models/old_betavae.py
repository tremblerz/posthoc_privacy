import torch
import torch.nn as nn
from torch.nn import init

from torch.autograd import Variable


def loss_function(recon_x, x, mu, logvar, beta):
    batch_size = x.shape[0]
    recons_loss = torch.nn.functional.mse_loss(recon_x, x, size_average=False).div(batch_size)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    # kld_loss = klds.sum(1).mean(0, True)
    loss = recons_loss# + beta * kld_loss
    return loss.mean(), recons_loss, None


class VAE(nn.Module):
    def __init__(self, hparams):
        super(VAE, self).__init__()

        self.hparams = hparams
        self.have_cuda = False
        nc = hparams["nc"]
        nz = hparams["nz"]
        ndf = hparams["ndf"]
        ngf = hparams["ngf"]
        self.nz = nz

        self.encoder = nn.Sequential(
            # input is (nc) x 128 x 128
            # nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            # nn.Conv2d(nc, ndf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 2),
            # nn.LeakyReLU(0.2, inplace=True),
            #################
            # 32x32
            nn.Conv2d(nc, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            ###############
            # 16x16
            nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            ###############
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 256, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(256, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 2, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(True),
            #####################
            # nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf),
            # nn.LeakyReLU(True),
            # state size. (ngf*2) x 16 x 16
            #nn.ConvTranspose2d(ngf * 2, nc, 3, 2, 2, bias=False),
            # nn.BatchNorm2d(ngf),
            # nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            # nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            # nn.Tanh()
            nn.Conv2d(ngf * 2, nc, 3, 1, 1, bias=False),
            nn.Conv2d(nc, nc, 3, 1, 0, bias=False),
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )

        self.fc1 = nn.Linear(256, 128)
        self.fc21 = nn.Linear(128, nz)
        self.fc22 = nn.Linear(128, nz)

        self.fc3 = nn.Linear(nz, 128)
        self.fc4 = nn.Linear(128, 256)

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        conv = self.encoder(x)
        h1 = self.fc1(conv.view(-1, 256))
        # print("encode h1", h1.size())
        return self.fc21(h1), self.fc22(h1)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        deconv_input = self.fc4(h3)
        # print("deconv_input", deconv_input.size())
        deconv_input = deconv_input.view(-1, 256, 1, 1)
        return self.decoder(deconv_input)

    def reparametrize(self, mu, logvar):
        # std = torch.exp(0.5 * logvar)
        # eps = torch.randn_like(std)
        return mu #eps * std + mu

    def forward(self, x):
        # print("x", x.size())
        mu, logvar = self.encode(x)
        # print("mu, logvar", mu.size(), logvar.size())
        z = self.reparametrize(mu, logvar)
        # print("z", z.size())
        decoded = self.decode(z)
        return decoded, mu, logvar, z
