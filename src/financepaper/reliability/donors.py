"""Training-only approximate conditional joint donor completions, not a posterior."""
import numpy as np
import pandas as pd
from financepaper.data.schema import FEATURE_NAMES, CATEGORICAL_FEATURES


class ConditionalDonors:
    def __init__(self, reference: pd.DataFrame, *, neighbours=32, max_reference=2048, seed=42):
        if tuple(reference.columns) != FEATURE_NAMES or reference.isna().any().any():
            raise ValueError("reference must be complete canonical training features only")
        rng=np.random.default_rng(seed)
        pos=np.sort(rng.choice(len(reference),min(len(reference),max_reference),replace=False))
        self.reference=reference.iloc[pos].copy()
        self.neighbours=min(neighbours,len(pos))
        q=reference.quantile([.25,.75]).to_numpy()
        self.scale=np.maximum(q[1]-q[0],1.)
        self.categorical=np.array([f in CATEGORICAL_FEATURES for f in FEATURE_NAMES])

    def complete(self, partial: pd.DataFrame, *, draws=8, seed=42):
        if tuple(partial.columns) != FEATURE_NAMES or draws<1:
            raise ValueError("invalid completion request")
        values=partial.to_numpy(float)
        observed=np.isfinite(values)
        ref=self.reference.to_numpy(float)
        rng=np.random.default_rng(seed)
        result=np.repeat(values[:,None,:],draws,axis=1)
        for i,x in enumerate(values):
            distance=np.minimum(np.abs(ref-np.nan_to_num(x))/self.scale,5.)
            distance[:,self.categorical]=(ref[:,self.categorical]!=x[self.categorical])
            distance=np.where(observed[i],distance,0).sum(1)
            nearest=np.argsort(distance,kind="stable")[:self.neighbours]
            donors=ref[rng.choice(nearest,draws,replace=True)]
            result[i]=np.where(observed[i],x,donors)
        return [pd.DataFrame(result[:,d],index=partial.index,columns=FEATURE_NAMES) for d in range(draws)]
