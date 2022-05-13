import os
import torch
from torch import optim, save
from torchvision.utils import save_image
import torch.nn as nn

from models.betavae import UTKVAE, MnistVAE, loss_function as vae_loss_fn

from utils import get_dataloader

min_test_loss = 1000.

def init_weights(vae):
    def init_(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    for m in vae.modules():
        m.apply(init_)

def setup_vae(dset):
    print(dset)
    if dset == "mnist":
        vae = MnistVAE({"nz": 8})
    elif dset == "utkface":
        vae = UTKVAE({"nz": 10})
    else:
        print("unknown dataset", dset)
    init_weights(vae)
    vae = vae.cuda()
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    return vae, optimizer

def train(vae, train_loader, optimizer, beta, epoch):
    vae.train()
    train_loss = 0
    for batch_idx, (data, _, _) in enumerate(train_loader):
        data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, log_var, z = vae(data)
        loss, _, _ = vae_loss_fn(recon_batch, data, mu, log_var, beta)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

def test(vae, test_loader, beta, epoch, model_name, dset):
    vae.eval()
    test_loss= 0
    with torch.no_grad():
        for data, _, _ in test_loader:
            data = data.cuda()
            recon, mu, log_var, z = vae(data)
            
            # sum up batch loss
            vae_l, _, _ = vae_loss_fn(recon, data, mu, log_var, beta)
            test_loss += vae_l.item()
        #z = torch.randn(64, 8).cuda()
        sample = vae.decode(z).cuda()
        if dset == "mnist":
            save_image(sample.view(-1, 1, 28, 28),
                   './samples/training/{}/epoch_{}.png'.format(model_name, epoch))
        elif dset == "utkface":
            save_image(sample.view(-1, 3, 64, 64),
                   './samples/training/{}/epoch_{}.png'.format(model_name, epoch))
        
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss

def save_model(vae, model_name):
    torch.save(vae.state_dict(), "saved_models/{}.pt".format(model_name))

def train_vae():
    global min_test_loss
    dset, beta = "utkface", 5
    model_name = "{}_beta{}_vae".format(dset, beta)
    samples_dir = "./samples/training/{}".format(model_name)
    if not os.path.isdir(samples_dir):
        os.makedirs(samples_dir)
    train_loader, test_loader = get_dataloader(dset)
    vae, optimizer = setup_vae(dset)
    for epoch in range(1, 101):
        train(vae, train_loader, optimizer, beta, epoch)
        loss_ = test(vae, test_loader, beta, epoch, model_name, dset)
        if loss_ < min_test_loss:
            save_model(vae, model_name)
            min_test_loss = loss_

if __name__ == '__main__':
    train_vae()