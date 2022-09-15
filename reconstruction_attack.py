import torch
import numpy as np
import torch.nn as nn
from adversarial_training import ARL
from models.decoder import Decoder
import lpips
import pytorch_msssim
from models.gen import AdversaryModelGen
from torch.distributions import Laplace



class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))


class MetricLoader():
    # Add more metrics
    def __init__(self, grayscale=False, device=None, data_range=1):
        self.l1_dist = nn.L1Loss()
        self.ce = nn.CrossEntropyLoss()
        self.grayscale = grayscale
        if grayscale:
            self._ssim = pytorch_msssim.ssim
        else:
            self._ssim = pytorch_msssim.SSIM(data_range=data_range)
        self.l2_dist = nn.MSELoss()
        self._psnr = PSNR()
        _lpips = lpips.LPIPS(net='vgg')
        if device is None:
            self._lpips = _lpips.cuda()
        else:
            self._lpips = _lpips.to(device)

    def acc(self, preds, y):
        return (preds.argmax(dim=1) == y).sum().item() / preds.shape[0]

    def l1(self, img1, img2):
        return self.l1_dist(img1, img2)

    def cross_entropy(self, preds, lbls):
        return self.ce(preds, lbls)
    
    def KLdivergence(self, preds, y):
        return nn.KLDivLoss(reduction = 'batchmean')(torch.log(preds), y)

    def ssim(self, img1, img2):
        if self.grayscale:
            return self._ssim(img1, img2, data_range=1)
        return self._ssim(img1, img2)

    def l2(self, img1, img2):
        return self.l2_dist(img1, img2)

    def psnr(self, img1, img2):
        return self._psnr(img1, img2)

    def lpips(self, img1, img2):
        score = self._lpips(img1, img2)
        return score.mean()


class Trainer():
    def __init__(self, arl_obj, eps, R, proposed) -> None:
        self.arl_obj = arl_obj
        self.setup_decoder()
        self.decoder_optim = torch.optim.Adam(self.decoder.parameters())
        if self.arl_obj.dset == "mnist" or self.arl_obj.dset == "fmnist":
            self.metrics = MetricLoader(grayscale=True)
        else:
            self.metrics = MetricLoader()
        self.eps = eps
        self.R = R
        self.proposed = proposed
        self.noise_sigma = self.proposed / (self.eps / self.R)
        self.min_loss = -torch.inf
        self.base_dir = "experiments/attacks/{}_eps_{}_R_{}".format(arl_obj.base_dir.split("/")[-1], eps, R)
        self.model_name =  self.base_dir + "/attacker.pt"

    def setup_decoder(self):
        if self.arl_obj.dset == "mnist" or self.arl_obj.dset == "fmnist":
            adv_model_params = {"channels": self.arl_obj.obf_out_size, "output_channels": 1,
                                "downsampling": 0, "offset": 4, "dset": self.arl_obj.dset}
        else:
            adv_model_params = {"channels": self.arl_obj.obf_out_size, "output_channels": 3,
                                "downsampling": 0, "offset": 4, "dset": self.arl_obj.dset}
        self.decoder = AdversaryModelGen(adv_model_params).cuda()

    def save_model(self):
        torch.save(self.decoder.state_dict(), self.model_name)

    def perturb(self, x):
        return x + Laplace(torch.zeros_like(x), self.noise_sigma).sample()

    def forward(self, data):
        with torch.no_grad():
            z, _ = arl_obj.vae.encode(data)#.view(-1, 784))
            z_tilde = arl_obj.obfuscator(z)
            z_hat = self.perturb(z_tilde)
        x_hat = self.decoder(z_hat)
        return x_hat

    def train(self):
        for ind, (data, labels, _) in enumerate(self.arl_obj.train_loader):
            data, labels = data.cuda(), labels.cuda()
            x_hat = self.forward(data)
            rec_loss = self.metrics.l1(data, x_hat)
            self.decoder_optim.zero_grad()
            rec_loss.backward()
            self.decoder_optim.step()
        print("epoch {}, loss {:.4f}".format(self.epoch, rec_loss.item()))

    def test(self):
        ssim, l1, l2, psnr, lpips, num_samples = 0, 0, 0, 0, 0, 0
        with torch.no_grad():
            for ind, (data, labels, _) in enumerate(self.arl_obj.test_loader):
                data, labels = data.cuda(), labels.cuda()
                x_hat = self.forward(data)
                ssim += self.metrics.ssim(data, x_hat); l1 += self.metrics.l1(data, x_hat)
                l2 += self.metrics.l2(data, x_hat); psnr += self.metrics.psnr(data, x_hat); lpips += self.metrics.lpips(data, x_hat)
                # num_samples += data.shape[0]
        ssim /= ind; l1 /= ind; l2 /= ind; psnr /= ind; lpips /= ind
        if ssim < self.min_loss:
            self.min_loss = ssim
            self.save_model()
        print("epoch {}, ssim {:4f}, l1 {:.4f}, l2 {:.4f}, psnr {:.4f}, lpips {:.4f}".format(self.epoch, ssim, l1, l2, psnr, lpips))

    def run(self):
        epochs = 20
        self.epoch = 0
        for _ in range(epochs):
            self.epoch += 1
            self.test()
            self.train()


if __name__ == '__main__':
    eps, radius, proposed = 1.0, 1.0, 0.34
    arl_config = {'alpha': 0.99, 'dset': 'mnist', 'obf_in': 8, 'obf_out': 8,
                  'lip_reg': False, 'lip_coeff': 0.01, 'noise_reg': True,
                  'sigma': 0.01, 'siamese_reg': True, 'margin': 25, 'lambda': 100.0}
    arl_obj = ARL(arl_config)
    arl_obj.load_state()
    arl_obj.vae.eval()
    arl_obj.obfuscator.eval()
    
    print(arl_config, eps, radius, proposed)
    trainer = Trainer(arl_obj, eps, radius, proposed)
    trainer.run()
