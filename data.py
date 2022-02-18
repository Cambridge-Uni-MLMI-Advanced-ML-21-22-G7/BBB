from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def load_mnist(train, batch_size, shuffle):
    return DataLoader(
        datasets.MNIST('./mnist', train=train, download=True, transform=transforms.ToTensor()),
        batch_size=batch_size, 
        shuffle=shuffle,
        drop_last=True
    )