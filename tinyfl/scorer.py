from tinyfl.model import create_model, test_model
import numpy as np

# Computes the accuracy of the givien weight, takes args weight
# Returns the accuracy of the model
def _compute_accuracy(weight):
    model = create_model()
    model.load_state_dict(weight)
    return test_model(model)[0]

# Computes the accuracy of a list of weights, takes args weights: list of weights
# Returns a list of accuracy scores
def accuracy_scorer(weights):
    return [_compute_accuracy(weight) for weight in weights]

# Computes the marginal gain scores for a list of weights
# Takes args weights and previous scores
# Returns a list of marignal gains in scores
def marginal_gain_scorer(weights, prev_scores):
    assert len(weights) == len(prev_scores)
    return [
        max(a - b, 0)
        for a, b in zip(
            [_compute_accuracy(weight) for weight in weights],
            prev_scores,
        )
    ]

# Computes the multikrum scores for a list of weights, takes args weights
# Returns a list of multikrum scores
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
