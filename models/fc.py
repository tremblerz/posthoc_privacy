import torch.nn as nn


class MnistPredModel(nn.Sequential):
    def __init__(self, in_dims):
        super(MnistPredModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dims, 50),
            #nn.BatchNorm1d(10),
            nn.LeakyReLU(True),
            #nn.Dropout(0.2),
            #nn.Linear(10, 10),
            #nn.BatchNorm1d(10),
            #nn.LeakyReLU(True),
            #nn.Dropout(0.2),
            #nn.Linear(10, 10),
            #nn.BatchNorm1d(10),
            #nn.LeakyReLU(True),
            #nn.Dropout(0.1),
            #nn.Linear(10, 10),
            #nn.BatchNorm1d(10),
            #nn.LeakyReLU(True),
            #nn.Dropout(0.1),
            nn.Linear(50, 10),
        )


class MnistFullPredModel(nn.Sequential):
    def __init__(self, in_dims=784):
        super(MnistFullPredModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dims, in_dims//2),
            nn.BatchNorm1d(in_dims//2),
            nn.LeakyReLU(True),
            nn.Dropout(0.2),
            nn.Linear(in_dims//2, in_dims//4),
            nn.BatchNorm1d(in_dims//4),
            nn.LeakyReLU(True),
            nn.Dropout(0.2),
            nn.Linear(in_dims//4, in_dims//8),
            nn.BatchNorm1d(in_dims//8),
            nn.LeakyReLU(True),
            nn.Dropout(0.1),
            nn.Linear(in_dims//8, 10),
            nn.BatchNorm1d(10),
            nn.LeakyReLU(True),
            nn.Dropout(0.1),
            nn.Linear(10, 10),
        )


class FMnistPredModel(nn.Sequential):
    def __init__(self, in_dims):
        super(FMnistPredModel, self).__init__()
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


class UTKPredModel(nn.Sequential):
    def __init__(self, in_dims):
        super(UTKPredModel, self).__init__()
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
            nn.Linear(10, 2),
            # nn.BatchNorm1d(10),
            # nn.LeakyReLU(True),
            # nn.Dropout(0.1),
            # nn.Linear(10, 2),
        )
