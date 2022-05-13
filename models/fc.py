import torch.nn as nn


class MnistPredModel(nn.Sequential):
    def __init__(self, in_dims):
        super(MnistPredModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dims, 10),
            nn.BatchNorm1d(10),
            nn.LeakyReLU(True),
            nn.Dropout(0.2),
            nn.Linear(10, 10),
            nn.BatchNorm1d(10),
            nn.LeakyReLU(True),
            nn.Dropout(0.2),
            nn.Linear(10, 10),
            nn.BatchNorm1d(10),
            nn.LeakyReLU(True),
            nn.Dropout(0.1),
            nn.Linear(10, 10),
            nn.BatchNorm1d(10),
            nn.LeakyReLU(True),
            nn.Dropout(0.1),
            nn.Linear(10, 10),
        )