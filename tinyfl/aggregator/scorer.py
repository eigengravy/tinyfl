from tinyfl.model import create_model, test_model


def accuracy_scorer(weights):
    model = create_model()
    return [test_model(model.load_state_dict(weight))[0] for weight in weights]


def marginal_gain_scorer(weights, prev_scores):
    assert len(weights) == len(prev_scores)
    model = create_model()
    return [
        max(a - b, 0)
        for a, b in zip(
            [test_model(model.load_state_dict(weight))[0] for weight in weights],
            prev_scores,
        )
    ]


def multikrum_scorer(weights):
    pass
