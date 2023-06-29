from collections import OrderedDict
import os
from trace import Trace
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torch import optim
from models.fc import MnistFullPredModel, MnistPredModel, FMnistPredModel, UTKPredModel

from utils import check_and_create_path

from embedding import setup_vae
from embedding import get_dataloader

from lipmip.relu_nets import ReLUNet


class ARL(object):
    def __init__(self, config) -> None:
        # Weighing term between privacy and accuracy, higher alpha has
        # higher weight towards privacy
        self.is_encoder = config.get("is_encoder")
        self.dset = config["dset"]
        self.pred_loss_fn = torch.nn.CrossEntropyLoss()
        self.eps = config["eps"]
        # since data is normalized between 0 and 1, the difference between max and min is a vector full of 1's 
        if self.is_encoder:
            sens = config["enc_out"]
        else:
            if self.dset in ["mnist", "fmnist"]:
                sens = 28 * 28
            else:
                sens = 64 * 64 * 3
        stability_param = 1e-10
        self.sigma = sens / self.eps + stability_param
        self.setup_data()
        if self.is_encoder:
            self.enc_in_size = config["enc_in"]
            self.enc_out_size = config["enc_out"]

            enc_layer_sizes = [self.enc_in_size, 8, 8, self.enc_out_size]
            self.encoder = ReLUNet(enc_layer_sizes).cuda()

            if self.dset == "mnist":
                self.pred_model = MnistPredModel(self.enc_out_size).cuda()
            elif self.dset == "fmnist":
                self.pred_model = FMnistPredModel(self.enc_out_size).cuda()
            elif self.dset == "utkface":
                self.pred_model = UTKPredModel(self.enc_out_size).cuda()

            params = [self.encoder.parameters() + self.pred_model.parameters()]
            self.setup_vae()
        else:
            model = resnet18()
            if self.dset in ["mnist", "fmnist"]:
                num_classes = 10
                model.fc = nn.Linear(512, num_classes)
                model = list(model.children())
                model.insert(-1, nn.Flatten())
                pre_layer = nn.Conv2d(1, 3, 1, 1)
                model_layers = nn.ModuleList([pre_layer] + model)
                model = nn.Sequential(*model_layers)
                # model = MnistFullPredModel().cuda()
            else:
                num_classes = 2
                model.fc = nn.Linear(512, num_classes)
            self.pred_model = model.cuda()
            params = self.pred_model.parameters()
        self.optimizer = optim.Adam(params)

        self.min_final_loss = np.inf
        self.assign_paths()

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
            self.vae.load_state_dict(torch.load("saved_models/mnist_beta3_vae.pt"))
        elif self.dset == "fmnist":
            self.vae.load_state_dict(torch.load("saved_models/fmnist_beta1_vae.pt"))
        elif self.dset == "utkface":
            wts = torch.load("/u/abhi24/Workspace/sanitizer/experiments/rebuttal/vae_beta0.5_age_seed15/saved_models/model_vae.pt", map_location=torch.device('cpu'))
            wts = self.cleanse_state_dict(wts)
            self.vae.load_state_dict(wts)
        self.vae = self.vae.cuda()
        print("loaded vae")

    def setup_data(self):
        self.train_loader, self.test_loader = get_dataloader(self.dset)

    def assign_paths(self):
        self.base_dir = "./experiments/ldp_{}_eps_{}".format(self.dset, self.eps)
        if self.is_encoder:
            self.base_dir += "_in_{}_out_{}".format(self.enc_in_size,
                                                   self.enc_out_size)
            self.enc_path = self.base_dir + "/enc.pt"

        self.pred_path = self.base_dir + "/pred.pt"

    def setup_path(self):
        # Should be only called when it is required to save updated models
        check_and_create_path(self.base_dir)

    def save_state(self):
        # torch.save(self.encoder.state_dict(), self.enc_path)
        torch.save(self.pred_model.state_dict(), self.pred_path)

    def load_state(self):
        if self.is_encoder:
            self.encoder.load_state_dict(torch.load(self.enc_path))
        self.pred_model.load_state_dict(torch.load(self.pred_path))
        print("sucessfully loaded models")

    def on_cpu(self):
        self.encoder.cpu()
        self.adv_model.cpu()
        self.pred_model.cpu()

    def gen_lap_noise(self, z):
        noise_dist = torch.distributions.Laplace(torch.zeros_like(z), self.sigma)
        return noise_dist.sample()

    def train(self, epoch):
        if self.is_encoder:
            self.vae.eval()
        self.pred_model.train()
        train_loss = 0
        # Start training
        for batch_idx, (data, labels) in enumerate(self.train_loader):
            data, labels = data.cuda(), labels.cuda()
            if self.is_encoder:
                # get sample embedding from the VAE
                with torch.no_grad():
                    if self.dset in ["mnist", "fmnist"]:
                        mu, sigma = self.vae.encode(data.view(-1, 784))
                    else:
                        mu, sigma = self.vae.encode(data)
                    z = self.vae.reparametrize(mu, sigma)    
                # pass it through encoder
                z = self.encoder(z)
            else:
                z = data#.view(-1, 784)
            
            z_server = z + self.gen_lap_noise(z)
            preds = self.pred_model(z_server)
            pred_loss = self.pred_loss_fn(preds, labels)
            self.optimizer.zero_grad()
            pred_loss.backward()
            self.optimizer.step()

            train_loss += pred_loss.item()
            
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)] pred_loss {:.3f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), pred_loss.item()))
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(self.train_loader.dataset)))

    def test(self):
        if self.is_encoder:
            self.vae.eval()
        self.pred_model.eval()
        test_pred_loss = 0
        pred_correct = 0
        with torch.no_grad():
            for data, labels in self.test_loader:
                data, labels = data.cuda(), labels.cuda()
                if self.is_encoder:
                    if self.dset in ["mnist", "fmnist"]:
                        mu, sigma = self.vae.encode(data.view(-1, 784))
                    else:
                        mu, sigma = self.vae.encode(data)
                    z = self.vae.reparametrize(mu, sigma)
                    z = self.encoder(z)
                else:
                    z = data#.view(-1, 784)
                z_server = z + self.gen_lap_noise(z)
                preds = self.pred_model(z_server)
                pred_correct += (preds.argmax(dim=1) == labels).sum()

                pred_loss = self.pred_loss_fn(preds, labels)

                test_pred_loss += pred_loss.item()
            
        test_pred_loss /= len(self.test_loader.dataset)
        if test_pred_loss < self.min_final_loss:
            self.save_state()
            self.min_final_loss = test_pred_loss
        pred_acc = pred_correct.item() / len(self.test_loader.dataset)
        print('====> Test pred loss: {:.4f}, acc {:.2f}'.format(test_pred_loss, pred_acc))


if __name__ == '__main__':
    train_expt = True
    config = {'encoder': False, 'dset': 'utkface', 'encoder_dim': 10, 'eps': 2**13}
    arl = ARL(config)
    if train_expt:
        arl.setup_path()
        print("starting training", config)
        for epoch in range(1, 51):
            arl.train(epoch)
            arl.test()
    else:
        arl.load_state()
        arl.test()    
