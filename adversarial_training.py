import pickle
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from utils import check_and_create_path

from embedding import setup_vae
from embedding import get_dataloader

from models.gen import AdversaryModelGen
from lipmip.relu_nets import ReLUNet

from lip_reg import net_Local_Lip, robust_loss


class ARL(object):
    def __init__(self, config) -> None:
        # Weighing term between privacy and accuracy, higher alpha has
        # higher weight towards privacy
        self.alpha = config["alpha"]
        self.obf_in_size = config["obf_in"]
        self.obf_out_size = config["obf_out"]
        lip_reg = config.get('lip_reg') or False
        noise_reg = config.get('noise_reg') or False
        obf_layer_sizes = [self.obf_in_size, 10, 10, 6, self.obf_out_size]
        self.obfuscator = ReLUNet(obf_layer_sizes).cuda()

        pred_layer_sizes = [self.obf_out_size, 10, 10]
        self.pred_model = ReLUNet(pred_layer_sizes).cuda()

        adv_model_params = {"channels": self.obf_out_size, "output_channels": 1,
                            "downsampling": 0, "offset": 4}
        self.adv_model = AdversaryModelGen(adv_model_params).cuda()

        self.pred_loss_fn = torch.nn.CrossEntropyLoss()
        self.rec_loss_fn = torch.nn.MSELoss()

        self.obf_optimizer = optim.Adam(self.obfuscator.parameters())
        self.pred_optimizer = optim.Adam(self.pred_model.parameters())
        self.adv_optimizer = optim.Adam(self.adv_model.parameters())

        self.dset = "mnist"

        self.setup_vae()
        self.setup_data()

        self.min_final_loss = np.inf
        self.assign_paths(lip_reg, noise_reg)

        if lip_reg:
            self.lip_reg_setup()

    def setup_vae(self):
        self.vae, _ = setup_vae()
        self.vae.load_state_dict(torch.load("saved_models/vae.pt"))
        print("loaded vae")

    def setup_data(self):
        self.train_loader, self.test_loader = get_dataloader(self.dset)

    def assign_paths(self, lip_reg, noise_reg):
        self.base_dir = "./experiments/in_{}_out_{}_alpha_{}".format(self.obf_in_size,
                                                                     self.obf_out_size,
                                                                     self.alpha)
        if lip_reg:
            self.base_dir += "_lipreg"
        if noise_reg:
            self.base_dir += "_noisereg"
        self.obf_path = self.base_dir + "/obf.pt"
        self.adv_path = self.base_dir + "/adv.pt"
        self.pred_path = self.base_dir + "/pred.pt"

    def setup_path(self):
        # Should be only called when it is required to save updated models
        check_and_create_path(self.base_dir)

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

    def train(self, epoch):
        self.vae.eval()
        train_loss = 0
        for batch_idx, (data, labels, _) in enumerate(self.train_loader):

            data, labels = data.cuda(), labels.cuda()
            # get sample embedding from the VAE
            with torch.no_grad():
                mu, log_var = self.vae.encoder(data.view(-1, 784))
                z = self.vae.sampling(mu, log_var)
            
            # pass it through obfuscator
            z_tilde = self.obfuscator(z)

            # Train predictor model
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
            total_loss = self.alpha*rec_loss + (1-self.alpha)*pred_loss
            self.obf_optimizer.zero_grad()
            total_loss.backward()
            self.obf_optimizer.step()
            
            train_loss += total_loss.item()
            
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, pred_loss {:.3f}, rec_loss {:.3f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), total_loss.item() / len(data), pred_loss.item(), rec_loss.item()))
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(self.train_loader.dataset)))

    def lip_reg_setup(self):
        # compute the feature size at each layer
        model = nn.Sequential(*(list(self.obfuscator.net)+list(self.pred_model.net)))
        input_size = []
        depth = len(model)
        x = torch.randn(1, 8).cuda()
        for i, layer in enumerate(model.children()):
            if i < depth-1:
                input_size.append(x.size()[1:])
                x = layer(x)

        # create u on cpu to store singular vector for every input at every layer
        self.u_train = []

        for i in range(len(input_size)):
            print(i)
            if not model[i].__class__.__name__=='Flatten' and not isinstance(model[i], nn.ReLU):
                self.u_train.append(torch.randn((len(self.train_loader.dataset), *(input_size[i])), pin_memory=True))
            else:
                self.u_train.append(None)

    def train_lip_reg(self, u_list, u_train):
        self.vae.eval()
        full_net = nn.Sequential(*(list(self.obfuscator.net)+list(self.pred_model.net)))
        net_local = net_Local_Lip(full_net)
        kappa, starting_kappa = 0.0, 1.0
        epsilon, starting_epsilon = 1.58, 0.
        schedule_length = 150
        sniter, opt_iter = 5, 10
        train_loss = 0
        for batch_idx, (data, labels, idx) in enumerate(self.train_loader):
            data, labels = data.cuda(), labels.cuda()
            # get sample embedding from the VAE
            with torch.no_grad():
                mu, log_var = self.vae.encoder(data.view(-1, 784))
                z = self.vae.sampling(mu, log_var)

            # pass it through obfuscator
            z_tilde = self.obfuscator(z)

            # Train predictor model
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

            # Lipschitz regularization
            epsilon1 = epsilon+batch_idx/len(self.train_loader)*(epsilon-starting_epsilon)/schedule_length
            kappa1 = kappa+batch_idx/len(self.train_loader)*(kappa-starting_kappa)/schedule_length
            # extract singular vector for each data point
            u_train_data = []
            for ll in range(len(u_train)):
                if u_train[ll] is not None:
                    u_train_data.append(u_train[ll][idx,:].cuda())
                else:
                    u_train_data.append(None)

            # robust loss
            local_loss, local_err, u_list, u_train_idx = robust_loss(net_local, epsilon1, z, labels, u_list, u_train_data,
                                                                     sniter, opt_iter, gloro=True)
            for ll in range(len(u_train)):
                if u_train_idx[ll] is not None:
                    u_train[ll][idx,:] = u_train_idx[ll].clone().detach().cpu()


            # Train obfuscator model by maximizing reconstruction loss
            # and minimizing prediction loss
            z_tilde = self.obfuscator(z)
            preds = self.pred_model(z_tilde)
            pred_loss = self.pred_loss_fn(preds, labels)
            rec = self.adv_model(z_tilde)
            rec_loss = self.rec_loss_fn(rec, data)
            total_loss = self.alpha*rec_loss + (1-self.alpha)*pred_loss
            self.obf_optimizer.zero_grad()
            (local_loss + total_loss).backward()
            self.obf_optimizer.step()

            train_loss += total_loss.item()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, pred_loss {:.3f}, rec_loss {:.3f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), total_loss.item() / len(data), pred_loss.item(), rec_loss.item()))
        return u_list, u_train

    def test(self):
        self.vae.eval()
        test_pred_loss= 0
        test_rec_loss= 0
        pred_correct = 0
        with torch.no_grad():
            for data, labels, _ in self.test_loader:
                data, labels = data.cuda(), labels.cuda()
                # get sample embedding from the VAE
                mu, log_var = self.vae.encoder(data.view(-1, 784))
                z = self.vae.sampling(mu, log_var)

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
            self.save_state()
            self.min_final_loss = final_loss
        pred_acc = pred_correct.item() / len(self.test_loader.dataset)
        print('====> Test pred loss: {:.4f}, rec loss {:.4f}, acc {:.2f}'.format(test_pred_loss, test_rec_loss, pred_acc))


if __name__ == '__main__':
    config = {"alpha": 0.901, "obf_in": 8, "obf_out": 5, "lip_reg": True, "noise_reg": False}
    arl = ARL(config)
    arl.setup_path()
    print("starting training")
    for epoch in range(1, 151):
        if config["lip_reg"]:
            if epoch == 1:
                u_train = arl.u_train
                u_list = None
            u_list, u_train = arl.train_lip_reg(u_list, u_train)
        else:
            arl.train(epoch)
        arl.test()
