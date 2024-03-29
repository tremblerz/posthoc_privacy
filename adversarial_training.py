from collections import OrderedDict
from distutils.command.config import config
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch import optim
from torch.nn.modules.loss import _Loss
from models.fc import MnistPredModel, FMnistPredModel, UTKPredModel

from utils import check_and_create_path

from embedding import setup_vae
from embedding import get_dataloader

from models.gen import AdversaryModelGen
from lipmip.relu_nets import ReLUNet


class ContrastiveLoss(_Loss):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def get_mask(self, labels):
        labels = labels.unsqueeze(-1).to(dtype=torch.float64)
        class_diff = torch.cdist(labels, labels, p=1.0)
        return torch.clamp(class_diff, 0, 1)

    def get_pairwise(self, z):
        z = z.view(z.shape[0], -1)
        return torch.cdist(z, z, p=2.0)

    def forward(self, z, labels):
        mask = self.get_mask(labels).to(z.device)
        pairwise_dist = self.get_pairwise(z)
        loss = (1 - mask) * pairwise_dist +\
               mask * torch.maximum(torch.tensor(0.).to(z.device), self.margin - pairwise_dist)
        return loss.mean()


class ARL(object):
    def __init__(self, config) -> None:
        self.input_space = config.get('input_space') or False
        # Weighing term between privacy and accuracy, higher alpha has
        # higher weight towards privacy
        self.tag = config.get("tag") or None
        self.alpha = config["alpha"]
        self.obf_in_size = config["obf_in"]
        self.obf_out_size = config["obf_out"]

        self.noise_reg = config.get('noise_reg') or False
        self.siamese_reg = config.get('siamese_reg') or False

        # obf_layer_sizes = [self.obf_in_size, 10, 10, 6, self.obf_out_size]
        obf_layer_sizes = [self.obf_in_size, 10, self.obf_out_size]
        self.obfuscator = ReLUNet(obf_layer_sizes).cuda()

        self.dset = config["dset"]
        if self.dset == "mnist":
            self.pred_model = MnistPredModel(self.obf_out_size).cuda()
        elif self.dset == "fmnist":
            self.pred_model = FMnistPredModel(self.obf_out_size).cuda()
        elif self.dset == "utkface":
            self.pred_model = UTKPredModel(self.obf_out_size).cuda()

        if self.dset == "mnist" or self.dset == "fmnist":
            adv_model_params = {"channels": self.obf_out_size, "output_channels": 1,
                                "downsampling": 0, "offset": 4, "dset": self.dset}
        else:
            adv_model_params = {"channels": self.obf_out_size, "output_channels": 3,
                                "downsampling": 0, "offset": 4, "dset": self.dset}
        self.adv_model = AdversaryModelGen(adv_model_params).cuda()

        self.pred_loss_fn = torch.nn.CrossEntropyLoss()
        self.rec_loss_fn = torch.nn.MSELoss()

        self.obf_optimizer = optim.Adam(self.obfuscator.parameters(), lr=1e-4)
        self.pred_optimizer = optim.Adam(self.pred_model.parameters(), lr=1e-4)
        self.adv_optimizer = optim.Adam(self.adv_model.parameters())

        if self.noise_reg:
            self.sigma = config["sigma"]
        if self.siamese_reg:
            self._lambda = config["lambda"]
            self.margin = config["margin"]
            self.setup_siamese_reg()

        if not self.input_space:
            self.setup_vae()
        self.setup_data()

        self.min_final_loss = np.inf
        self.assign_paths(self.noise_reg, self.siamese_reg)

    def setup_siamese_reg(self):
        self.contrastive_loss = ContrastiveLoss(self.margin)

    def cleanse_state_dict(self, state_dict):
        """
        This is an mismatch of expected keys, etc. Issua comes up is saved via gpu, but running on cpu, etc.
        Ex. mismatch: keys not matching
        Expecting: {"model.0.weight", ...}
        Received: {"module.model.0.weight", ...}
        """
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
        return new_state_dict

    def setup_vae(self):
        self.vae, _ = setup_vae(self.dset)
        if self.dset == "mnist":
            self.vae.load_state_dict(torch.load("saved_models/mnist_beta15_vae.pt"))
        elif self.dset == "fmnist":
            self.vae.load_state_dict(torch.load("saved_models/fmnist_beta2_vae.pt"))
        elif self.dset == "utkface":
            wts = torch.load("saved_models/utkface_beta0_vae.pt", map_location=torch.device('cpu'))
            # wts = self.cleanse_state_dict(wts)
            self.vae.load_state_dict(wts)
        self.vae = self.vae.cuda()
        print("loaded vae")

    def setup_data(self):
        self.train_loader, self.test_loader = get_dataloader(self.dset)

    def assign_paths(self, noise_reg, siamese_reg):
        if self.dset == "mnist":
            self.base_dir = "./experiments/in_{}_out_{}_alpha_{}".format(self.obf_in_size,
                                                                        self.obf_out_size,
                                                                        self.alpha)
        elif self.dset == "fmnist":
            self.base_dir = "./experiments/fmnist_in_{}_out_{}_alpha_{}".format(self.obf_in_size,
                                                                        self.obf_out_size,
                                                                        self.alpha)
        elif self.dset == "utkface":
            if self.tag:
                self.base_dir = "./experiments/utkface_{}_in_{}_out_{}_alpha_{}".format(self.obf_in_size, self.tag,
                                                                            self.obf_out_size,
                                                                            self.alpha)
            else:
                self.base_dir = "./experiments/utkface_in_{}_out_{}_alpha_{}".format(self.obf_in_size,
                                                                            self.obf_out_size,
                                                                            self.alpha)
        else:
            print("unknown dataset {}".format(self.dset))
        if noise_reg:
            self.base_dir += "_noisereg_{}".format(self.sigma)
        if siamese_reg:
            self.base_dir += "_siamesereg_{}_{}".format(self.margin, self._lambda)
        self.obf_path = self.base_dir + "/obf.pt"
        self.adv_path = self.base_dir + "/adv.pt"
        self.pred_path = self.base_dir + "/pred.pt"

    def setup_path(self):
        # Should be only called when it is required to save updated models
        check_and_create_path(self.base_dir)
        self.imgs_dir = self.base_dir + "/imgs/"
        os.mkdir(self.imgs_dir)

    def save_state(self):
        torch.save(self.obfuscator.state_dict(), self.obf_path)
        torch.save(self.adv_model.state_dict(), self.adv_path)
        torch.save(self.pred_model.state_dict(), self.pred_path)

    def load_state(self):
        self.obfuscator.load_state_dict(torch.load(self.obf_path))
        self.adv_model.load_state_dict(torch.load(self.adv_path))
        self.pred_model.load_state_dict(torch.load(self.pred_path))
        print("sucessfully loaded models")

    def on_cpu(self):
        self.obfuscator.cpu()
        self.adv_model.cpu()
        self.pred_model.cpu()

    def gen_lap_noise(self, z):
        noise_dist = torch.distributions.Laplace(torch.zeros_like(z), self.sigma)
        return noise_dist.sample()

    def train(self, epoch):
        if not self.input_space:
            self.vae.eval()
        self.pred_model.train()
        train_loss = 0
        # Start training
        for batch_idx, (data, labels) in enumerate(self.train_loader):

            data, labels = data.cuda(), labels.cuda()
            if not self.input_space:
                # get sample embedding from the VAE
                with torch.no_grad():
                    if self.dset in ["mnist", "fmnist"]:
                        mu, sigma = self.vae.encode(data.view(-1, 784))
                    else:
                        mu, sigma = self.vae.encode(data)
                    z = self.vae.reparametrize(mu, sigma)
            else:
                z = torch.flatten(data, start_dim=1)

            # pass it through obfuscator
            z_tilde = self.obfuscator(z)

            # Train predictor model
            if self.noise_reg:
                z_server = z_tilde.detach() + self.gen_lap_noise(z_tilde)
                preds = self.pred_model(z_server)
            else:
                preds = self.pred_model(z_tilde.detach())
            pred_loss = self.pred_loss_fn(preds, labels)
            self.pred_optimizer.zero_grad()
            pred_loss.backward()
            self.pred_optimizer.step()
            # Train adversary model
            rec = self.adv_model(z_tilde.detach())
            rec_loss = self.rec_loss_fn(rec, data)
            self.adv_optimizer.zero_grad()
            rec_loss.backward()
            self.adv_optimizer.step()

            # Train obfuscator model by maximizing reconstruction loss
            # and minimizing prediction loss
            z_tilde = self.obfuscator(z)
            preds = self.pred_model(z_tilde)
            pred_loss = self.pred_loss_fn(preds, labels)
            rec = self.adv_model(z_tilde)
            rec_loss = self.rec_loss_fn(rec, data)
            total_loss = -self.alpha*rec_loss + (1-self.alpha)*pred_loss

            if self.siamese_reg:
                siamese_loss = self.contrastive_loss(z, labels)

            self.obf_optimizer.zero_grad()
            if self.siamese_reg:
                total_loss += self._lambda*siamese_loss
            total_loss.backward()
            self.obf_optimizer.step()
            
            train_loss += total_loss.item()
            
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, pred_loss {:.3f}, rec_loss {:.3f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), total_loss.item() / len(data), pred_loss.item(), rec_loss.item()))
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(self.train_loader.dataset)))

    def test(self):
        if not self.input_space:
           self.vae.eval()
        self.pred_model.eval()
        test_pred_loss= 0
        test_rec_loss= 0
        pred_correct = 0
        with torch.no_grad():
            for data, labels in self.test_loader:
                data, labels = data.cuda(), labels.cuda()
                if not self.input_space:
                    if self.dset in ["mnist", "fmnist"]:
                        mu, sigma = self.vae.encode(data.view(-1, 784))
                    else:
                        mu, sigma = self.vae.encode(data)
                    z = self.vae.reparametrize(mu, sigma)
                else:
                    z = torch.flatten(data, start_dim=1)

                # pass it through obfuscator
                z_tilde = self.obfuscator(z)
                # pass obfuscated z through pred_model
                preds = self.pred_model(z_tilde)
                pred_correct += (preds.argmax(dim=1) == labels).sum()

                # train obfuscator and pred_model
                pred_loss = self.pred_loss_fn(preds, labels)

                # pass obfuscated z to adv_model
                rec = self.adv_model(z_tilde)
                rec_loss = self.rec_loss_fn(rec, data)

                test_pred_loss += pred_loss.item()
                test_rec_loss += rec_loss.item()
            
        test_pred_loss /= len(self.test_loader.dataset)
        test_rec_loss /= len(self.test_loader.dataset)
        final_loss = self.alpha*test_rec_loss + (1-self.alpha)*test_pred_loss
        if final_loss < self.min_final_loss:
            if self.dset in ["mnist", "fmnist"]:
                rec_imgs = rec.view(-1, 1, 28, 28)
            elif self.dset == "utkface":
                rec_imgs = rec.view(-1, 3, 32, 32)
            save_image(rec_imgs,
                   '{}/epoch_{}.png'.format(self.base_dir, epoch))
            self.save_state()
            self.min_final_loss = final_loss
        pred_acc = pred_correct.item() / len(self.test_loader.dataset)
        print('====> Test pred loss: {:.4f}, rec loss {:.4f}, acc {:.5f}'.format(test_pred_loss, test_rec_loss, pred_acc))


if __name__ == '__main__':
    config = {"alpha": 0.99,
               "dset": "utkface", "obf_in": 512, "obf_out": 10,
               "lip_reg": False, "lip_coeff": 0.01,
               "noise_reg": False, "sigma": 0.1,
               "siamese_reg": False, "margin": 25, "lambda": 1.0,
               "tag": "gender", 'input_space': False
               }
    # config = {'alpha': 0.95, 'dset': 'mnist', 'obf_in': 784, 'obf_out': 5,
    #          'lip_reg': False, 'lip_coeff': 0.01, 'noise_reg': False,
    #          'sigma': 0.01, 'siamese_reg': False, 'margin': 25, 'lambda': 100.0, 'input_space': True}
    # config = {'dset': 'fmnist', 'alpha': 0.995, 'obf_in': 8, 'obf_out': 8, 'lip_reg': False, 'lip_coeff': 0.01, 'noise_reg': True, 'sigma': 0.01, 'siamese_reg': True, 'margin': 25., 'lambda': 50.0}
    arl = ARL(config)
    arl.setup_path()
    print("starting training", config)
    for epoch in range(1, 501):
        arl.train(epoch)
        arl.test()
