
import logging
from typing import Union, Tuple
import numpy as np
import torch
from typing import Union, Tuple
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import T_co
from torchvision import datasets, transforms
import pandas as pd

from bbb.config.constants import MUSHROOM_DATASET_PATH, MUSHROOM_BUFFER_PATH, MUSHROOM_TRAIN_PATH
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

class MnistAdversarialDataset(Dataset):
    def __init__(self, size: int, epsilon: int = 0.1, randomise: bool = False) -> None:
        super().__init__()

        if epsilon > 0:
            x_data = np.load(f'../data/adversarial_mnist/x_grad_{epsilon}.npy')
        else:
            x_data = np.load('../data/adversarial_mnist/x_orig.npy')

        y_data = np.load('../data/adversarial_mnist/y_grad.npy')

        if randomise:
            idx = np.random.permutation(np.arange(size))
            x_data = x_data[idx,:,:,:]
            y_data = y_data[idx]

        self.x = x_data[:size,:,:,:]
        self.y = y_data[:size]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index) -> T_co:
        return self.x[index], self.y[index]

def load_adverserial_mnist(
    train: bool,
    size: int,
    batch_size: int,
    shuffle: bool,
    epsilon: int = 0.1,
    randomise: bool = False
) -> DataLoader:
    if train:
        raise RuntimeError('Cannot train using adversarial data!')
    else:
        return DataLoader(
            MnistAdversarialDataset(size=size, epsilon=epsilon, randomise=randomise),
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

def load_bandit_buffer() -> Tuple[torch.Tensor, torch.Tensor]:
    data = torch.load(MUSHROOM_BUFFER_PATH).to(DEVICE)
    X = data[:,:97]
    y = data[:,97].unsqueeze(-1)
    z = data[:,98].unsqueeze(-1)
    # x is context + action, y is reward, z is label
    return X,y,z

def load_bandit_train() -> Tuple[torch.Tensor, torch.Tensor]:
    data = torch.load(MUSHROOM_TRAIN_PATH).to(DEVICE)
    X = data[:,:95]
    y = data[:,95].unsqueeze(-1)
    # x is contecxt, y is label
    return X,y

class RegressionDataset(Dataset):
    def __init__(
        self,
        size: int,
        l_lim: float,
        u_lim: float,
        train: bool = False,
        seed: Union[int, None] = 0
    ) -> None:
        """Creating the regression dataset used in the paper.
        Values ashere to the following equation, where ??~????(0,0.02).

        y = x + 0.3sin(2??(x+??)) + 0.3sin(4??(x+??)) + ??

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

class RegressionModifiedDataset(Dataset):
    def __init__(
        self,
        size: int,
        l_lim: float,
        u_lim: float,
        train: bool,
        seed: Union[int, None] = 0,
    ) -> None:
        """Creating a modified version of the regression dataset used in the paper.
        Values ashere to the following equation, where ??~????(0,0.02).

        y = - 0.3sin(2??(x+??)) - 0.3sin(4??(x+??)) + ??

        Training data is split into two regions.

        :param size: size of vector to generate
        :type size: int
        :param l_lim: lower x-range limit to generate data over
        :type l_lim: float
        :param u_lim: upper x-range limit to generate data over
        :type u_lim: float
        :param train: whether this is training data
        :type train: bool
        :param seed: random seed to be used, defaults to 0
        :type seed: Union[int, None], optional
        """
        super().__init__()

        self.size = size
        self.seed = seed

        if self.seed is not None:
            torch.manual_seed(self.seed)

        self.x = torch.unsqueeze(torch.linspace(l_lim, u_lim, self.size, requires_grad=False), dim=1)

        if train:
            self.x[self.x>0.3] += 0.45

        epsilon = torch.randn(self.x.size()) * 0.02

        self.y =-0.3*torch.sin(2*np.pi*(self.x + epsilon)) - 0.3*torch.sin(4*np.pi*(self.x + epsilon)) + epsilon

    def __len__(self):
        return self.size

    def __getitem__(self, index) -> T_co:
        return self.x[index], self.y[index]

def generate_modified_regression_data(train: bool, size: int, batch_size: int, shuffle: bool, seed: int = None) -> DataLoader:
    if train:
        return DataLoader(
            RegressionModifiedDataset(size=size, train=train, l_lim=0.0, u_lim=0.5),
            batch_size=batch_size, 
            shuffle=shuffle,
            drop_last=True,  # This should be set to true, else it will disrupt average calculations
            pin_memory=True,
            num_workers=0
        )
    else:
        return DataLoader(
            RegressionModifiedDataset(size=size, train=train, l_lim=-0.5, u_lim=1.5),
            batch_size=batch_size, 
            shuffle=shuffle,
            drop_last=True,  # This should be set to true, else it will disrupt average calculations
            pin_memory=True,
            num_workers=0
        )
