from tinyfl.model import create_model, test_model
import numpy as np
from torch.utils.data import DataLoader
from typing import Any,List,Mapping


def _compute_accuracy(weight:Mapping[str, Any], testloader: DataLoader)-> float:
    """
    Computes accuracy of model.

    Compares output of model with current set of weights to calculate percentage of correct answers.

    Args:
        weight: Weights of the model stored in a dictionary
        testloader: The loaded dataset

    Returns:
        A float value of the accuracy of the model (% of correct answers)
    """
    model = create_model()
    model.load_state_dict(weight)
    return test_model(model, testloader)[0]


def accuracy_scorer(weights: List[Mapping[str, Any]], testloader: DataLoader)-> List(float):
    """Computes accuracy of models.

    Args:
        weights: A list of weights of each model which are stored in dictionaries
        testloader: The loaded dataset

    Returns:
        A list with float values of the accuracies of the models (% of correct answers)
    """
    return [_compute_accuracy(weight, testloader) for weight in weights]


def marginal_gain_scorer(weights: List[Mapping[str, Any]], prev_scores: List[float], testloader: DataLoader)-> List[float]:
    """Calculates marginal gain in accuracy of model

    Calculates increase in accuracy of model after pulling wieghts

    Args:
        weights: A list of weights of each model which are stored in dictionaries
        prev_scores: List storing accuracies of models prior to most recent updation of weights
    
    Returns:
        List of floats which represent the marginal increases in accuracies(if any) of each party
    """
    assert len(weights) == len(prev_scores)
    return [
        max(a - b, 0)
        for a, b in zip(
            [_compute_accuracy(weight, testloader) for weight in weights],
            prev_scores,
        )
    ]


def multikrum_scorer(weights: List[Mapping[str, Any]]):
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
