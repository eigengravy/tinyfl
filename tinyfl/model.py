import torch
from torch import nn
from torch.utils.data import Dataset, Subset, DataLoader
import copy
from collections import defaultdict
from random import shuffle
from typing import List, Tuple

device = "cuda" if torch.cuda.is_available() else "cpu"


class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
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


def create_model() -> Model:
    model = Model().to(device)
    return model


def train_model(model: Model, epochs: int, trainloader: DataLoader):
    optimizer = torch.optim.Adam(model.parameters())
    for _ in range(epochs):
        model.train()
        for _, (X, y) in enumerate(trainloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = nn.CrossEntropyLoss()(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def test_model(model: Model, testloader: DataLoader) -> Tuple[float, float]:
    size = len(testloader.dataset)
    batches = len(testloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in testloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += nn.CrossEntropyLoss()(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= batches
    correct /= size
    return 100 * correct, test_loss


def fedavg_models(weights):
    avg = copy.deepcopy(weights[0])
    for i in range(1, len(weights)):
        for key in avg:
            avg[key] += weights[i][key]
        avg[key] = torch.div(avg[key], len(weights))
    return avg


strategies = {
    "fedavg": fedavg_models,
}


def stratified_split_dataset(dataset: Dataset, num_parties: int) -> List[List[int]]:
    def partition_list(l, n):
        indices = list(range(len(l)))
        shuffle(indices)
        index_partitions = [sorted(indices[i::n]) for i in range(n)]
        return [[l[i] for i in index_partition] for index_partition in index_partitions]

    labels = dataset.targets.tolist()
    indices_per_label = defaultdict(list)
    for idx, label in enumerate(labels):
        indices_per_label[label].append(idx)

    indices_split = [[] for _ in range(num_parties)]

    for label, indices in indices_per_label.items():
        partitioned_indices = partition_list(indices, num_parties)
        shuffle(partitioned_indices)
        for i, subset in enumerate(partitioned_indices):
            indices_split[i].extend(subset)

    return indices_split


def subset_from_indices(dataset: Dataset, indices: List[int]) -> Subset:
    return Subset(dataset=dataset, indices=indices)
