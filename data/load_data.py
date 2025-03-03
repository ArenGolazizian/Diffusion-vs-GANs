import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def load_fashion_mnist(batch_size, img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

    dataset = datasets.FashionMNIST(
        root="./data/FashionMNIST",
        download=True,
        train=True,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    return dataloader