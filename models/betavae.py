from abc import ABC, abstractproperty
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F



class Vec2Vol(nn.Module):
    """ Converts a vector into a convolution volume
    by unsqueezing in the height and width dimension
    """
    def __init__(self, *args):
        super(Vec2Vol, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.unsqueeze(2).unsqueeze(3)


class Reshape(nn.Module):
    """ NN layer for reshaping
    """
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.reshape(self.shape)


def loss_function(recon_x, x, mu, sigma, beta):
    batch_size = x.shape[0]
    recon_x = torch.clamp(recon_x, 1e-8, 1 - 1e-8)
    marginal_likelihood = -F.binary_cross_entropy(recon_x, x, reduction='sum') / batch_size

    KL_divergence = 0.5 * torch.sum(
                                torch.pow(mu, 2) +
                                torch.pow(sigma, 2) -
                                torch.log(1e-8 + torch.pow(sigma, 2)) - 1
                               ).sum() / batch_size

    ELBO = marginal_likelihood - beta * KL_divergence

    loss = -ELBO

    return loss.mean(), marginal_likelihood, KL_divergence


class BaseVAE(nn.Module, ABC):
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()
        self.flatten = nn.Flatten()

    @property
    @abstractproperty
    def encoder(self):
        pass

    @property
    @abstractproperty
    def decoder(self):
        pass

    @property
    @abstractproperty
    def fc_upscale(self):
        pass

    @property
    @abstractproperty
    def fc_downscale(self):
        pass

    @property
    @abstractproperty
    def mu_net(self):
        pass

    @property
    @abstractproperty
    def sigma_net(self):
        pass

    def make_volume(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)
        return x.reshape(-1, 1, 28, 28)

    def encode(self, x):
        feature_vec = self.encoder(x)
        h_downscaled = self.fc_downscale(feature_vec)
        stddev = 1e-6 + F.softplus(self.sigma_net(h_downscaled[:, self.nz:]))
        mu = self.mu_net(h_downscaled[:, :self.nz])
        return mu, stddev

    def decode(self, z):
        h_upscaled = self.fc_upscale(z)
        return self.decoder(h_upscaled)

    def reparametrize(self, mu, sigma):
        z = mu + sigma * torch.randn_like(mu)
        return z

    def forward(self, x):
        mu, sigma = self.encode(x)
        z = self.reparametrize(mu, sigma)
        decoded = self.decode(z)
        return decoded, mu, sigma, z


class UTKVAE(BaseVAE):
    def __init__(self, hparams) -> None:
        super(UTKVAE, self).__init__()

        self.nz = hparams["nz"]

        self.encoder = nn.Sequential(
            # state size. 3 x 64 x 64
            nn.Conv2d(3, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            #################
            # 32x32
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            ###############
            # 16x16
            nn.Conv2d(256, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            ###############
            # state size. 256 x 8 x 8
            nn.Conv2d(256, 512, 3, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 512 x 4 x 4
            nn.Conv2d(512, 512, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten()
            # state size. 512
        )

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(256, 128, 4, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True),
            # state size. 128 x 4 x 4
            nn.ConvTranspose2d(128, 64, 3, 1, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),
            # state size. 64 x 6 x 6
            nn.ConvTranspose2d(64, 64, 3, 1, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),
            # state size. 64 x 8 x 8
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),
            # state size. 32 x 16 x 16
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True),
            # state size. 16 x 32 x 32
            nn.ConvTranspose2d(16, 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(True),
            # state size. 8 x 64 x 64
            nn.Conv2d(8, 3, 3, 1, 1, bias=False),
            nn.Sigmoid()
            # state size. 3 x 64 x 64
        )

        self._fc_downscale = nn.Sequential(
            # state size. 512
            nn.Linear(512, 128),
            nn.Linear(128, 2*self.nz)
        )

        self._fc_upscale = nn.Sequential(
            # state size. nz
            nn.Linear(self.nz, 128),
            nn.Linear(128, 256),
            Vec2Vol()
        )
        self._mu_net = nn.Linear(self.nz, self.nz)
        self._sigma_net = nn.Linear(self.nz, self.nz)

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder

    @property
    def fc_downscale(self):
        return self._fc_downscale

    @property
    def fc_upscale(self):
        return self._fc_upscale

    @property
    def mu_net(self):
        return self._mu_net
    
    @property
    def sigma_net(self):
        return self._sigma_net


class MnistVAE(BaseVAE):
    def __init__(self, hparams):
        super(MnistVAE, self).__init__()

        self.hparams = hparams
        nz = hparams["nz"]
        self.nz = nz
        self._encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.ELU(inplace=True),
            nn.Dropout(0.1),

            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Dropout(0.1),

            nn.Linear(512, nz*2)
        )
        self._decoder = nn.Sequential(
            nn.Linear(nz, 512),
            nn.Tanh(),
            nn.Dropout(0.01),

            nn.Linear(512, 512),
            nn.ELU(),
            nn.Dropout(0.01),

            nn.Linear(512, 784),
            nn.Sigmoid(),
            Reshape(-1, 1, 28, 28)
        )
        self._fc_downscale = nn.Sequential(
            nn.Identity()
        )
        self._fc_upscale = nn.Sequential(
            nn.Identity()
        )
        self._mu_net = nn.Linear(self.nz, self.nz)
        self._sigma_net = nn.Linear(self.nz, self.nz)

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder

    @property
    def fc_downscale(self):
        return self._fc_downscale

    @property
    def fc_upscale(self):
        return self._fc_upscale

    @property
    def mu_net(self):
        return self._mu_net
    
    @property
    def sigma_net(self):
        return self._sigma_net
