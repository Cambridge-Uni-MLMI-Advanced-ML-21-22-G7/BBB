from typing import Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import T_co
from torchvision import datasets, transforms


def load_mnist(train: bool, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        datasets.MNIST('./data/mnist', train=train, download=True, transform=transforms.ToTensor()),
        batch_size=batch_size, 
        shuffle=shuffle,
        drop_last=True
    )

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
            num_workers=8
        )
    else:
        return DataLoader(
            RegressionDataset(size=size, l_lim=-0.2, u_lim=1.4),
            batch_size=batch_size, 
            shuffle=shuffle,
            drop_last=True,  # This should be set to true, else it will disrupt average calculations
            pin_memory=True,
            num_workers=8
        )
