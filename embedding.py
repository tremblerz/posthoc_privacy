import torch
from torch import optim, save
from torchvision import datasets, transforms

from models.betavae import loss_function as vae_loss_fn
from models.betavae import SmallVAE, BigVAE
from models.gen import AdversaryModelGen

min_test_loss = 1000.

def get_dataloader(dset, batch_size=200):

    if dset == 'mnist':
        # MNIST Dataset
        train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=False)
    else:
        print("dataset {} not implemented".format(dset))

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def setup_vae():
    # load and train BigVAE
    # nc = 1
    # nz = 8
    # ndf = 16
    # ngf = 16
    beta = 1
    # hparams_vae = {"nc": nc, "nz": nz, "ndf": ndf, "ngf": ngf}
    # vae = BigVAE(hparams_vae)
    # vae.cuda()

    # Load and train SmallVAE
    vae = SmallVAE(x_dim=784, h_dim1= 512, h_dim2=256, z_dim=8).cuda()
    optimizer = optim.Adam(vae.parameters())
    return vae, optimizer

def train(vae, train_loader, optimizer, beta, epoch):
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, log_var = vae(data)
        loss, _, _ = vae_loss_fn(recon_batch, data, mu, log_var, beta)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

def test(vae, test_loader, beta):
    vae.eval()
    test_loss= 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.cuda()
            recon, mu, log_var = vae(data)
            
            # sum up batch loss
            vae_l, _, _ = vae_loss_fn(recon, data, mu, log_var, beta)
            test_loss += vae_l.item()
        
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss

def save_model(vae, dset, beta):
    torch.save(vae.state_dict(), "saved_models/{}_beta{}_vae.pt".format(dset, beta))

def train_vae():
    global min_test_loss
    dset, beta = "mnist", 5
    train_loader, test_loader = get_dataloader(dset)
    vae, optimizer = setup_vae()
    for epoch in range(1, 51):
        train(vae, train_loader, optimizer, beta, epoch)
        loss_ = test(vae, test_loader, beta)
        if loss_ < min_test_loss:
            save_model(vae, dset, beta)
            min_test_loss = loss_

if __name__ == '__main__':
    train_vae()