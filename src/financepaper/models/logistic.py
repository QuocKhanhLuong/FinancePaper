from sklearn.linear_model import LogisticRegression


def make_logistic(seed: int = 42, C: float = 1.0, **kwargs) -> LogisticRegression:
    params = dict(C=C, solver="lbfgs", penalty="l2", class_weight=None,
                  max_iter=3000, random_state=seed)
    params.update(kwargs)
    return LogisticRegression(**params)
