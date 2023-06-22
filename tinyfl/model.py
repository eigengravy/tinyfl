import torch
from torch.utils.data import Dataset, Subset
import copy
import numpy as np
from collections import defaultdict
from random import shuffle
from typing import List
from tinyfl.models.fashion_mnist_model import FashionMNISTModel
from tinyfl.models.plant_disease_model import PlantDiseaseModel


models = {
    "fashion_mnist": FashionMNISTModel,
    "plant_disease": PlantDiseaseModel,
}


def fedavg_models(weights):
    avg = copy.deepcopy(weights[0])
    for i in range(1, len(weights)):
        for key in avg:
            avg[key] += weights[i][key]
        avg[key] = torch.div(avg[key], len(weights))
    return avg


strategies = {
    "fedavg": fedavg_models,
    "fedprox": fedavg_models,
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


splits = {"simple": simple_split_dataset, "stratified": stratified_split_dataset}


def subset_from_indices(dataset: Dataset, indices: List[int]) -> Subset:
    return Subset(dataset=dataset, indices=indices)


# TODO: update to new models api
def _compute_accuracy(weight, testloader):
    model = create_model()
    model.load_state_dict(weight)
    return test_model(model, testloader)[0]


def accuracy_scorer(weights, testloader):
    return [_compute_accuracy(weight, testloader) for weight in weights]


def marginal_gain_scorer(weights, prev_scores, testloader):
    assert len(weights) == len(prev_scores)
    return [
        max(a - b, 0)
        for a, b in zip(
            [_compute_accuracy(weight, testloader) for weight in weights],
            prev_scores,
        )
    ]


def multikrum_scorer(weights):
    R = len(weights)
    f = R // 3 - 1
    closest_updates = R - f - 2

    keys = weights[0].keys()

    return [
        sum(
            sorted(
                [
                    sum(
                        [
                            np.linalg.norm(
                                weights[i][key].cpu() - weights[j][key].cpu()
                            )
                            for key in keys
                        ]
                    )
                    for j in range(R)
                    if j != i
                ]
            )[:closest_updates]
        )
        for i in range(R)
    ]


scorers = {
    "accuracy": accuracy_scorer,
    "marginal_gain": marginal_gain_scorer,
    "multi_krum": multikrum_scorer,
}
