from abc import ABC, abstractproperty
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from torch.autograd import Variable


def loss_function(recon_x, x, mu, logvar, beta):
    batch_size = x.shape[0]
    recons_loss = torch.nn.functional.mse_loss(recon_x, x, size_average=False).div(batch_size)
    #BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    kld_loss = klds.sum(1).mean(0, True)
    loss = recons_loss + beta * kld_loss
    return loss.mean(), recons_loss, kld_loss


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
        feature_volume = self.encoder(self.flatten(x))
        feature_vector = feature_volume
        h_downscaled = self.fc_downscale(feature_vector)
        # print("encode h_downscaled", h_downscaled.sih_downscalede())
        return self.mu_net(h_downscaled[:, :self.nz]), self.sigma_net(h_downscaled[:, self.nz:])

    def decode(self, z):
        h_upscaled = self.fc_upscale(z)
        feature_volume = h_upscaled
        return self.make_volume(self.decoder(feature_volume))

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        # print("x", x.size())
        mu, logvar = self.encode(x)
        # print("mu, logvar", mu.size(), logvar.size())
        z = self.reparametrize(mu, logvar)
        # print("z", z.size())
        decoded = self.decode(z)
        return decoded, mu, logvar, z

        
class MnistVAE(BaseVAE):
    def __init__(self, hparams):
        super(MnistVAE, self).__init__()

        self.hparams = hparams
        self.have_cuda = False
        nz = hparams["nz"]
        self.nz = nz
        self._encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ELU(inplace=True),
            nn.Dropout(0.1),

            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Dropout(0.1),

            nn.Linear(512, nz*2)
            # # input is 1 x 28 x 28
            # nn.Conv2d(1, 16, 3, 2, 1, bias=False),
            # nn.LeakyReLU(0.1, inplace=True),
            # ##################
            # # state size. 16 x 14 x 14
            # nn.Conv2d(16, 32, 3, 2, 1, bias=False),
            # nn.BatchNorm2d(32),
            # nn.LeakyReLU(0.1, inplace=True),
            # ##################
            # # state size. 32 x 7 x 7
            # nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(32),
            # nn.LeakyReLU(0.1, inplace=True),
            # #################
            # # state size. 32 x 7 x 7
            # nn.Conv2d(32, 16, 3, 2, 0, bias=False),
            # nn.BatchNorm2d(16),
            # nn.LeakyReLU(0.1, inplace=True),
            # ###############
            # # state size. 16 x 3 x 3
        )
        self._decoder = nn.Sequential(
            nn.Linear(nz, 512),
            nn.Tanh(),
            nn.Dropout(0.99),

            nn.Linear(512, 512),
            nn.ELU(),
            nn.Dropout(0.99),

            nn.Linear(512, 784),
            nn.Sigmoid()
            # # # state size. 32 x 1 x 1
            # nn.ConvTranspose2d(32, 16, 2, 1, 0, bias=False),
            # # nn.BatchNorm2d(16),
            # nn.LeakyReLU(True),
            # # #####################
            # # # state size. 16 x 2 x 2
            # nn.ConvTranspose2d(16, 8, 3, 2, 0, bias=False),
            # # nn.BatchNorm2d(8),
            # nn.LeakyReLU(True),
            # # #####################
            # # # state size. 8 x 5 x 5
            # nn.ConvTranspose2d(8, 4, 3, 1, 0, bias=False),
            # # nn.BatchNorm2d(4),
            # nn.LeakyReLU(True),
            # # #####################
            # # # state size. 4 x 7 x 7
            # nn.ConvTranspose2d(4, 2, 4, 2, 1, bias=False),
            # # nn.BatchNorm2d(2),
            # nn.LeakyReLU(True),
            # # #####################
            # # # state size. 2 x 14 x 14
            # nn.ConvTranspose2d(2, 1, 4, 2, 1, bias=False),
            # # nn.BatchNorm2d(1),
            # nn.LeakyReLU(True),
            # # #####################
            # # # state size. 1 x 28 x 28
            # nn.Conv2d(1, 1, 3, 1, 1, bias=False),
            # nn.Sigmoid()
            # # #####################
            # # # state size. 1 x 28 x 28
        )
        self._fc_downscale = nn.Sequential(
            nn.Identity()
            # ###############
            # # state size. 576
            # nn.Linear(144, 32),
            # nn.ReLU(),
            # ###############
            # # state size. 32
            # nn.Linear(32, self.nz),
            # nn.ReLU()
            # ###############
            # # state size. self.nz
        )
        self._fc_upscale = nn.Sequential(
            nn.Identity()
            # ###############
            # # state size. self.nz
            # nn.Linear(self.nz, 32),
            # nn.ReLU()
            # ###############
            # # state size. 32
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


class SmallVAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(SmallVAE, self).__init__()
        
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
        
    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h)).view(-1, 1, 28, 28)
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var


class NewSmallVAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(NewSmallVAE, self).__init__()
        
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc3 = nn.Linear(h_dim2, h_dim2 // 2)
        self.fc4 = nn.Linear(h_dim2 // 2, h_dim2 // 4)
        self.fc5 = nn.Linear(h_dim2 // 4, h_dim2 // 8)
        self.fc31 = nn.Linear(h_dim2 // 8, z_dim)
        self.fc32 = nn.Linear(h_dim2 // 8, z_dim)
        # decoder part
        self.fc4_ = nn.Linear(z_dim, h_dim2 // 8)
        self.fc41 = nn.Linear(h_dim2 // 8, h_dim2 // 4)
        self.fc42 = nn.Linear(h_dim2 // 4, h_dim2 // 2)
        self.fc43 = nn.Linear(h_dim2 // 2, h_dim2)
        self.fc5_ = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
        
    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc4(h))
        h = F.relu(self.fc5(h))
        return self.fc31(h), self.fc32(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        h = F.relu(self.fc4_(z))
        h = F.relu(self.fc41(h))
        h = F.relu(self.fc42(h))
        h = F.relu(self.fc43(h))
        h = F.relu(self.fc5_(h))
        return F.sigmoid(self.fc6(h)).view(-1, 1, 28, 28)
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var



class BigVAE(nn.Module):
    def __init__(self, hparams):
        super(BigVAE, self).__init__()

        self.hparams = hparams
        self.have_cuda = False
        nc = hparams["nc"]
        nz = hparams["nz"]
        ndf = hparams["ndf"]
        ngf = hparams["ngf"]
        self.nz = nz

        self.encoder = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            #################
            # 32x32
            nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1, bias=False),
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
            nn.ConvTranspose2d(ngf * 2, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, nc, 3, 2, 2, bias=False),
            # nn.BatchNorm2d(ngf),
            # nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            # nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            # nn.Tanh()
            nn.Conv2d(nc, nc, 4, 1, 0, bias=False),
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
        # print("deconv_input", deconv_input.size())
        return self.decoder(deconv_input)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        # print("x", x.size())
        mu, logvar = self.encode(x)
        # print("mu, logvar", mu.size(), logvar.size())
        z = self.reparametrize(mu, logvar)
        # print("z", z.size())
        decoded = self.decode(z)
        return decoded, mu, logvar, z

