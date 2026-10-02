from xgboost import XGBClassifier


def make_xgboost(seed: int = 42, **kwargs) -> XGBClassifier:
    params = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                  n_jobs=1, random_state=seed, max_depth=3, learning_rate=0.05,
                  n_estimators=200, min_child_weight=5, subsample=1.0,
                  colsample_bytree=1.0, reg_lambda=1.0)
    params.update(kwargs)
    return XGBClassifier(**params)
