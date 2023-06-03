from tinyfl.model import create_model, test_model
import numpy as np


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
