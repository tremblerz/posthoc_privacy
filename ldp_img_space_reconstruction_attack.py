from models.unet import UNet
import torch
import torch.nn as nn
import lpips
import pytorch_msssim
from torch.distributions import Laplace
from embedding import get_dataloader


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
    def __init__(self, config) -> None:
        self.dset = config["dset"]
        self.setup_data()
        self.setup_denoiser()
        self.denoiser_optim = torch.optim.Adam(self.denoiser.parameters())
        if self.dset in ["mnist", "fmnist"]:
            self.metrics = MetricLoader(grayscale=True)
            sens = 28 * 28
        else:
            self.metrics = MetricLoader()
            sens = 64 * 64 * 3
        eps = config["eps"]
        self.noise_sigma = sens / eps
        self.min_loss = -torch.inf
        self.base_dir = "experiments/attacks/{}_eps_{}".format(self.dset, eps)
        self.model_name =  self.base_dir + "/attacker.pt"

    def setup_data(self):
        self.train_loader, self.test_loader = get_dataloader(self.dset)

    def setup_denoiser(self):
        if self.dset in ["mnist", "fmnist"]:
            self.denoiser = UNet().cuda()
        else:
            self.denoiser = UNet(3, 3).cuda()

    def save_model(self):
        torch.save(self.denoiser.state_dict(), self.model_name)

    def gen_lap_noise(self, z):
        noise_dist = torch.distributions.Laplace(torch.zeros_like(z), self.noise_sigma)
        return noise_dist.sample()

    def forward(self, data):
        x_hat = self.denoiser(data)
        return x_hat

    def train(self):
        for data, labels in self.train_loader:
            data, labels = data.cuda(), labels.cuda()
            z = self.gen_lap_noise(data) + data
            x_hat = self.forward(z)
            rec_loss = self.metrics.l1(data, x_hat)
            self.denoiser_optim.zero_grad()
            rec_loss.backward()
            self.denoiser_optim.step()
        print("epoch {}, loss {:.4f}".format(self.epoch, rec_loss.item()))

    def test(self):
        ssim, l1, l2, psnr, lpips, num_samples = 0, 0, 0, 0, 0, 0
        with torch.no_grad():
            for ind, (data, labels) in enumerate(self.test_loader):
                data, labels = data.cuda(), labels.cuda()
                z = self.gen_lap_noise(data) + data
                x_hat = self.forward(z)
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
    eps = 2**8
    config = {'dset': 'mnist', 'eps': eps}
    print(config)
    trainer = Trainer(config)
    trainer.run()
