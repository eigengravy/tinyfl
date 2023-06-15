import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Tuple

from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"


class FashionMNISTModel(nn.Module):
    def __init__(self) -> None:
        super(FashionMNISTModel, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x

    @classmethod
    def create_model(cls):
        model = cls().to(device)
        return model

    def train_model(self, epochs: int, trainloader: DataLoader):
        optimizer = torch.optim.Adam(self.parameters())
        for _ in range(epochs):
            self.train()
            for _, (X, y) in enumerate(trainloader):
                X, y = X.to(device), y.to(device)
                pred = self(X)
                loss = nn.CrossEntropyLoss()(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def test_model(self, testloader: DataLoader) -> Tuple[float, float]:
        size = len(testloader.dataset)
        batches = len(testloader)
        self.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in testloader:
                X, y = X.to(device), y.to(device)
                pred = self(X)
                test_loss += nn.CrossEntropyLoss()(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= batches
        correct /= size
        return 100 * correct, test_loss

    @staticmethod
    def create_datasets():
        return (
            datasets.FashionMNIST(
                root="data",
                train=True,
                download=True,
                transform=transforms.ToTensor(),
            ),
            datasets.FashionMNIST(
                root="data",
                train=False,
                download=True,
                transform=transforms.ToTensor(),
            ),
        )
