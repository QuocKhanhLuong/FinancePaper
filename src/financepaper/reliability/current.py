"""A deliberately narrow inference boundary: no restored quantities or labels."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from financepaper.data.schema import FEATURE_NAMES

TEMPORAL = np.arange(5, 23)
MONTHS = [(10,16,22),(9,15,21),(8,14,20),(7,13,19),(6,12,18),(5,11,17)]


@dataclass(frozen=True)
class CurrentEvidence:
    probability: np.ndarray
    hidden: np.ndarray
    attribution: np.ndarray
    completion_probabilities: np.ndarray
    completion_attributions: np.ndarray | None = None

    def __post_init__(self):
        n = len(self.probability)
        if self.hidden.shape != (n,23) or self.attribution.shape != (n,23):
            raise ValueError("original-field attribution and mask must have 23 columns")
        if self.hidden.dtype != bool or self.completion_probabilities.shape[0] != n:
            raise ValueError("invalid current evidence dimensions")
        for v in (self.probability,self.attribution,self.completion_probabilities):
            if not np.isfinite(v).all():
                raise ValueError("nonfinite current evidence")
        if np.any((self.probability<0)|(self.probability>1)):
            raise ValueError("invalid probability")


def current_features(evidence: CurrentEvidence) -> pd.DataFrame:
    """Feature names are explicit allowlists; arbitrary dataframes are not input."""
    e = evidence
    p = np.clip(e.probability,1e-9,1-1e-9)
    observed = np.where(e.hidden,0,e.attribution)
    ordered = np.sort(np.where(e.hidden,-np.inf,e.attribution),axis=1)[:,::-1]
    values = dict(probability=p,entropy=-(p*np.log(p)+(1-p)*np.log1p(-p)),
        prediction_variance=np.var(e.completion_probabilities,axis=1),
        missing_fraction=e.hidden.mean(1),missing_temporal=e.hidden[:,5:].mean(1),
        missing_static=e.hidden[:,:5].mean(1),
        positive_count=((e.attribution>.01)&~e.hidden).sum(1),
        attribution_abs_sum=np.abs(observed).sum(1),
        positive_sum=np.maximum(observed,0).sum(1),
        negative_sum=np.minimum(observed,0).sum(1),
        third_reason=np.nan_to_num(ordered[:,2],neginf=0),
        rank_gap=np.nan_to_num(ordered[:,2]-ordered[:,3],nan=0,posinf=0,neginf=0))
    for j,f in enumerate(FEATURE_NAMES):
        values[f"missing_{f}"] = e.hidden[:,j].astype(float)
    for j,ids in enumerate(MONTHS):
        values[f"missing_month_{j}"] = e.hidden[:,ids].mean(1)
    if e.completion_attributions is not None:
        phi = e.completion_attributions
        if phi.shape[:1]+phi.shape[2:] != (len(p),23):
            raise ValueError("expected [customer, completion, original field]")
        values["attribution_variance"] = np.mean(np.where(e.hidden,0,np.var(phi,axis=1)),axis=1)
        eligible = (~e.hidden)&(e.attribution>.01)
        reason_order = np.argsort(-np.where(eligible,e.attribution,-np.inf),axis=1,kind="stable")[:,:3]
        events,signs,ranks = [],[],[]
        for d in range(phi.shape[1]):
            score = np.take_along_axis(phi[:,d],reason_order,axis=1)
            rank = np.sum((phi[:,d,:,None] > score[:,None,:]+1e-6)&(~e.hidden[:,:,None]),axis=1)
            signs.append(np.mean(score<=1e-6,axis=1))
            ranks.append(np.mean(rank>=3,axis=1))
            events.append(np.any((score<=1e-6)|(rank>=3),axis=1))
        values["mc_revision"] = np.mean(events,axis=0)
        values["sign_instability"] = np.mean(signs,axis=0)
        values["rank_instability"] = np.mean(ranks,axis=0)
    return pd.DataFrame(values)


PREDICTION_FEATURES = ["probability","entropy","prediction_variance","missing_fraction"]
ATTRIBUTION_FEATURES = ["positive_count","attribution_abs_sum","positive_sum","negative_sum","third_reason","rank_gap"]
MASK_FEATURES = ["missing_temporal","missing_static"]+[f"missing_{f}" for f in FEATURE_NAMES]+[f"missing_month_{j}" for j in range(6)]
RECIPES = {"prediction":PREDICTION_FEATURES,
           "explanation":PREDICTION_FEATURES+ATTRIBUTION_FEATURES,
           "mask_explanation":PREDICTION_FEATURES+ATTRIBUTION_FEATURES+MASK_FEATURES}
