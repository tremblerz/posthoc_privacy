from abc import abstractmethod
import numpy as np
import torch.utils.data as data
from glob import glob
from PIL import Image
import os
import pickle
import shutil
import logging
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch


class BaseDataset(data.Dataset):
    """docstring for BaseDataset"""

    def __init__(self, config):
        super(BaseDataset, self).__init__()
        self.format = config["format"]
        self.set_filepaths(config["path"])
        self.transforms = config["transforms"]

    def set_filepaths(self, path):
        filepaths = path + "/*.{}".format(self.format)
        self.filepaths = glob(filepaths)

    def load_image(self, filepath):
        img = Image.open(filepath)
        # img = np.array(img)
        return img

    @staticmethod
    def to_tensor(obj):
        return torch.tensor(obj)

    @abstractmethod
    def load_label(self):
        pass

    def __getitem__(self, index):
        filepath = self.filepaths[index]
        img = self.load_image(filepath)
        img = self.transforms(img)
        label = self.load_label(filepath)
        label = self.to_tensor(label)
        return img, label

    def __len__(self):
        return len(self.filepaths)


class UTKFace(BaseDataset):
    """docstring for UTKFace"""

    def __init__(self, config):
        super(UTKFace, self).__init__(config)
        self.attribute = config["attribute"]

    def load_label(self, filepath):
        labels = filepath.split("/")[-1].split("_")
        if self.attribute == "race":
            try:
                label = int(labels[2])
            except:
                print("corrupt label")
                label = np.random.randint(0, 4)
        elif self.attribute == "gender":
            label = int(labels[1])
        elif self.attribute == "age":
            # label = float(labels[0])
            if int(labels[0]) < 15:
                label = 0
            elif int(labels[0]) < 30:
                label = 1
            elif int(labels[0]) < 50:
                label = 2
            elif int(labels[0]) < 70:
                label = 3
            else:
                label = 4
            #label = float(label)
        return label


class MyDataset(Dataset):
    def __init__(self, trainset):
        self.set = trainset

    def __getitem__(self, index):
        data, target = self.set[index]
        return data, target, index

    def __len__(self):
        return len(self.set)

def get_split(train_split, dataset):

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(train_split * dataset_size))
    np.random.shuffle(indices)

    train_indices, test_indices = indices[:split], indices[split:]
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    return train_dataset, test_dataset


def get_dataloader(dset, batch_size=200):

    if dset == 'mnist':
        # MNIST Dataset
        train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=False)
    elif dset == 'fmnist':
        # Fashion MNIST
        train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transforms.ToTensor())
        test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transforms.ToTensor())
    elif dset == 'utkface':
        # UTKFace Dataset
        trainTransform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()])

        dataset = UTKFace({"path": "<add your path here>/UTKFace/",
                           "transforms": trainTransform,
                           "format": "jpg",
                           "attribute": "gender"})
        train_dataset, test_dataset = get_split(0.8, dataset)
    else:
        print("dataset {} not implemented".format(dset))

    train_dataset = MyDataset(train_dataset)
    test_dataset = MyDataset(test_dataset)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def check_and_create_path(path):
    if os.path.isdir(path):
        print("Experiment in {} already present".format(path))
        inp = input("Press e to exit, r to replace it: ")
        if inp == "e":
            exit()
        elif inp == "r":
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            print("Input not understood")
            exit()
    else:
        os.makedirs(path)

def save_object(obj, path):
    with open('{}/obj_dump.pkl'.format(path), 'wb') as fileobj:
        pickle.dump(obj, fileobj)


class Logger():
    def __init__(self, log_dir) -> None:
        log_format = "%(asctime)s::%(levelname)s::%(name)s::"\
                     "%(filename)s::%(lineno)d::%(message)s"
        logging.basicConfig(filename="{log_path}/log_console.log".format(
                                                     log_path=log_dir),
                            level='DEBUG', format=log_format)
        logging.getLogger().addHandler(logging.StreamHandler())
        self.log_dir = log_dir

    def init_tb(self):
        tb_path = self.log_dir + "/tensorboard"
        # if not os.path.exists(tb_path) or not os.path.isdir(tb_path):
        os.makedirs(tb_path)
        self.writer = SummaryWriter(tb_path)

    def log_console(self, msg):
        logging.info(msg)

    def log_tb(self, key, value, iteration):
        self.writer.add_scalar(key, value, iteration)
