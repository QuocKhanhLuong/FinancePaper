"""Sharp box-supported first-moment tail bounds for the C5 SHAP reduction."""
from fractions import Fraction as F
from itertools import groupby

from financepaper.evaluation.completion_rank import rank_cells


def parameters(m,p,moments):
    cells=rank_cells(m,F(p))
    b=tuple(map(F,moments))
    if len(b)!=len(cells) or any(not -1<=v<=1 for v in b):
        raise ValueError("Need m-1 reference means in [-1,1]")
    mean=sum(alpha*v for (alpha,_),v in zip(cells,b,strict=True))
    offset=sum((alpha-beta)*v for (alpha,beta),v in zip(cells,b,strict=True))
    a=tuple(beta for _,beta in cells)
    mu=tuple((1-v)/2 for v in b)
    upper=offset+sum(a)
    return dict(b=b,a=a,mu=mu,mean=mean,offset=offset,upper=upper,tau=upper/2)


def box_tail_bound(weights,means,threshold):
    """Exact worst-case P(sum a_i X_i >= threshold), 0<=X_i<=1.

    Only coordinate means are fixed; arbitrary dependence is allowed.
    Sorting and a linear sweep after sorting avoid numerical root finding.
    """
    a,mu,tau=tuple(map(F,weights)),tuple(map(F,means)),F(threshold)
    if not a or len(a)!=len(mu) or min(a)<=0 or any(not 0<=v<=1 for v in mu):
        raise ValueError("Positive weights and unit-interval means required")
    expectation=sum(w*v for w,v in zip(a,mu,strict=True))
    if tau<=expectation:
        return F(1)
    if tau>sum(a):
        return F(0)
    remaining,cap=sum(a),F(0)
    for value,group in groupby(sorted(zip(mu,a)),key=lambda pair:pair[0]):
        if cap+value*(remaining-tau)<0:
            return cap/(tau-remaining)
        mass=sum(w for _,w in group)
        cap+=value*mass
        remaining-=mass
    assert remaining==0
    return cap/tau


def reference_moment_bound(m,p,moments):
    info=parameters(m,p,moments)
    return box_tail_bound(info["a"],info["mu"],info["tau"])


def range_markov_bound(m,p,moments):
    info=parameters(m,p,moments)
    if info["mean"]<=0:
        return F(1)
    return 1-info["mean"]/info["upper"]


def moment_witness(m,p,moments):
    info=parameters(m,p,moments)
    z=box_tail_bound(info["a"],info["mu"],info["tau"])
    if z in (0,1):
        law=(F(1),)
        contrasts=(info["b"],)
    else:
        bad=tuple(min(F(1),v/z) for v in info["mu"])
        good=tuple((v-z*x)/(1-z) for v,x in zip(info["mu"],bad,strict=True))
        law=(z,1-z)
        contrasts=tuple(tuple(1-2*x for x in point) for point in (bad,good))
    gaps=tuple(info["offset"]+sum(a*d for a,d in zip(info["a"],point,strict=True)) for point in contrasts)
    return dict(law=law,contrasts=contrasts,gaps=gaps,risk=z,mean=info["mean"])


def witness_table(m,witness,constant=F(1,2)):
    constant=F(constant)
    if not 0<=constant<=1:
        raise ValueError("Prediction must be bounded")
    table=[]
    for point in witness["contrasts"]:
        for code in range(2**m):
            k=(code//4).bit_count()
            if code%4 in (0,3):
                table.append(constant)
            else:
                d=point[k]
                table.append((1+d)/2 if code%4==1 else (1-d)/2)
    return tuple(table)
