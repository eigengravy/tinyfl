import torch
from torch import nn
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import datasets, transforms
import copy
from collections import defaultdict
from random import shuffle
from typing import List, Tuple

from tinyfl.models.new_fashion_mnist import FashionMNISTModel
from tinyfl.models.plant_disease_model import (
    PlantDiseaseModel,
)


models = {
    "fashion-mnist": FashionMNISTModel,
    "plant_disease": PlantDiseaseModel,
}
#
# my_datasets = {
#     "fashion-mnist": (
#         datasets.FashionMNIST(
#             root="data",
#             train=True,
#             download=True,
#             transform=transforms.ToTensor(),
#         ),
#         datasets.FashionMNIST(
#             root="data",
#             train=False,
#             download=True,
#             transform=transforms.ToTensor(),
#         ),
#     ),
#     "plant_disease": PlantDiseaseModel.create_datasets(),
# }
#
# my_loaders = {
#     "fashion-mnist": (
#         DataLoader(my_datasets["fashion-mnist"][0], batch_size=64),
#         DataLoader(my_datasets["fashion-mnist"][1], batch_size=64),
#     ),
#     "plant_disease": PlantDiseaseModel.create_dataloaders(),
# }


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


def simple_split_dataset(dataset: Dataset, num_parties: int) -> List[List[int]]:
    indices = list(range(len(dataset)))
    shuffle(indices)
    index_partitions = [sorted(indices[i::num_parties]) for i in range(num_parties)]
    return index_partitions


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
