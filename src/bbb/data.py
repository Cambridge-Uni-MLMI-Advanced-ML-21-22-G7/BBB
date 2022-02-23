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
        size: int = 100,
        seed: Union[int, None] = 0
    ) -> None:
        """Creating the regression dataset used in the paper.
        Values ashere to the following equation, where ɛ~𝒩(0,0.02).

        y = x + 0.3sin(2π(x+ɛ)) + 0.3sin(4π(x+ɛ)) + ɛ

        :param size: size of vector to generate, defaults to 100
        :type size: int, optional
        :param seed: random seed to be used, defaults to 0
        :type seed: Union[int, None], optional
        """
        super().__init__()

        self.size = size
        self.seed = seed

        if self.seed is not None:
            torch.manual_seed(self.seed)

        self.x = torch.unsqueeze(torch.linspace(-0.2, 1.2, self.size, requires_grad=False), dim=1)

        epsilon = torch.randn(self.x.size()) * 0.02
        self.y = self.x + 0.3*torch.sin(2*np.pi*(self.x + epsilon)) + 0.3*torch.sin(4*np.pi*(self.x + epsilon)) + epsilon

    def __len__(self):
        return self.size

    def __getitem__(self, index) -> T_co:
        return self.x[index], self.y[index]

def generate_regression_data(size: int, batch_size: int, shuffle: bool, seed: int = None) -> DataLoader:
    return DataLoader(
        RegressionDataset(size=size),
        batch_size=batch_size, 
        shuffle=shuffle,
        drop_last=True
    )
