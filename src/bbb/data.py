import logging
from typing import Union, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import T_co
from torchvision import datasets, transforms
import pandas as pd

from bbb.config.constants import MUSHROOM_DATASET_PATH
from bbb.utils.pytorch_setup import DEVICE


logger = logging.getLogger(__name__)


def load_mnist(train: bool, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        datasets.MNIST('./data/mnist', train=train, download=True, transform=transforms.ToTensor()),
        batch_size=batch_size, 
        shuffle=shuffle,
        drop_last=True,  # This should be set to true, else it will disrupt average calculations
        pin_memory=True,
        num_workers=0
    )

def load_bandit() -> Tuple[torch.Tensor, torch.Tensor]:
    # reading the dataset
    df = pd.read_csv(MUSHROOM_DATASET_PATH)

    # randomizing the dataset
    df = df.sample(frac=1.0)

    # check the class distribution
    logger.debug(f'Class distribution of mushrooms: {df["edible"].value_counts()}')

    # splitting our df
    X = df.copy().drop('edible', axis=1)

    # edible -> 0, poisonous -> 1
    y = df.copy()['edible'].astype('category').cat.codes

    # One-hot
    X = pd.get_dummies(X, drop_first=True)

    def df_to_tensor(df):
        return torch.from_numpy(df.values).float().to(DEVICE)

    X = df_to_tensor(X.copy())
    y = df_to_tensor(y.copy()).unsqueeze(-1)

    return X,y

class RegressionDataset(Dataset):
    def __init__(
        self,
        size: int,
        l_lim: float,
        u_lim: float,
        seed: Union[int, None] = 0
    ) -> None:
        """Creating the regression dataset used in the paper.
        Values ashere to the following equation, where ɛ~𝒩(0,0.02).

        y = x + 0.3sin(2π(x+ɛ)) + 0.3sin(4π(x+ɛ)) + ɛ

        :param size: size of vector to generate
        :type size: int
        :param l_lim: lower x-range limit to generate data over
        :type l_lim: float
        :param u_lim: upper x-range limit to generate data over
        :type u_lim: float
        :param seed: random seed to be used, defaults to 0
        :type seed: Union[int, None], optional
        """
        super().__init__()

        self.size = size
        self.seed = seed

        if self.seed is not None:
            torch.manual_seed(self.seed)

        self.x = torch.unsqueeze(torch.linspace(l_lim, u_lim, self.size, requires_grad=False), dim=1)

        epsilon = torch.randn(self.x.size()) * 0.02
        self.y = self.x + 0.3*torch.sin(2*np.pi*(self.x + epsilon)) + 0.3*torch.sin(4*np.pi*(self.x + epsilon)) + epsilon

    def __len__(self):
        return self.size

    def __getitem__(self, index) -> T_co:
        return self.x[index], self.y[index]

def generate_regression_data(train: bool, size: int, batch_size: int, shuffle: bool, seed: int = None) -> DataLoader:
    if train:
        return DataLoader(
            RegressionDataset(size=size, l_lim=0.0, u_lim=0.5),
            batch_size=batch_size, 
            shuffle=shuffle,
            drop_last=True,  # This should be set to true, else it will disrupt average calculations
            pin_memory=True,
            num_workers=0
        )
    else:
        return DataLoader(
            RegressionDataset(size=size, l_lim=-0.2, u_lim=1.4),
            batch_size=batch_size, 
            shuffle=shuffle,
            drop_last=True,  # This should be set to true, else it will disrupt average calculations
            pin_memory=True,
            num_workers=0
        )
