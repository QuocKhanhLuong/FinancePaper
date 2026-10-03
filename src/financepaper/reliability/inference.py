"""Serving output deliberately cannot contain verification-only information."""
import numpy as np
from scipy.special import expit,logit
from financepaper.data.schema import FEATURE_NAMES
from financepaper.explanations.reasons import extract_reasons
from financepaper.reliability.current import CurrentEvidence,current_features,MONTHS


def selector_probability(selector, features):
    score=selector["model"].predict_proba(features[selector["columns"]])[:,1]
    calibrated=selector["calibrator"].transform(logit(np.clip(score,1e-7,1-1e-7)))
    return score,calibrated


def diagnostic_output(evidence: CurrentEvidence, logits, selector, policy, *, debug=False):
    features=current_features(evidence)
    scores,probabilities=selector_probability(selector,features)
    reasons=[extract_reasons(a,h,FEATURE_NAMES,3,.01) for a,h in zip(evidence.attribution,evidence.hidden)]
    release=policy.release(probabilities,np.array([len(r)==3 for r in reasons]))
    result=[]
    for i,names in enumerate(reasons):
        ids=[FEATURE_NAMES.index(n) for n in names]
        row=dict(prediction=dict(logit=float(logits[i]),prob_raw=float(expit(logits[i])),prob_calibrated=float(evidence.probability[i])),
            missingness=dict(fraction_total=float(evidence.hidden[i].mean()),fraction_temporal=float(evidence.hidden[i,5:].mean()),
                fraction_static=float(evidence.hidden[i,:5].mean()),features_missing=[f for f,h in zip(FEATURE_NAMES,evidence.hidden[i]) if h],
                months_missing=[int(evidence.hidden[i,j].sum()) for j in MONTHS]),
            uncertainty=dict(predictive_entropy=float(features.entropy.iloc[i]),prediction_variance=float(features.prediction_variance.iloc[i]),
                ensemble_or_mc_std=float(np.sqrt(features.prediction_variance.iloc[i])),kind="donor_completion_sensitivity"),
            explanation=dict(top_features=list(names),top_scores=evidence.attribution[i,ids].tolist(),
                signs=np.sign(evidence.attribution[i,ids]).astype(int).tolist(),rank=list(range(1,len(ids)+1))),
            reliability=dict(revision_score=float(scores[i]),revision_probability=float(probabilities[i]),release_decision=bool(release[i])))
        if debug: row["debug"]=dict(current_features=features.iloc[i].to_dict(),current_attribution=evidence.attribution[i].tolist())
        result.append(row)
    return result
