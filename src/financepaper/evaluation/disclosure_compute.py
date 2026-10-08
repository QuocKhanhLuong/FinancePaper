"""Finite synthetic cost/coverage controls for one grouped hidden SHAP player."""
from dataclasses import dataclass
from fractions import Fraction as F
import math

import numpy as np
from scipy.stats import beta


@dataclass(frozen=True)
class Polynomial:
    k: int
    masks: tuple[int, ...]
    coefficients: tuple[int, ...]

    def __post_init__(self):
        if not 1 <= self.k <= 63 or len(self.masks) != len(self.coefficients) or not self.masks:
            raise ValueError("Invalid finite polynomial dimensions")
        if len(set(self.masks)) != len(self.masks) or any(m <= 0 or m >= 1 << self.k for m in self.masks):
            raise ValueError("Supports must be distinct and nonempty")
        if any(c <= 0 for c in self.coefficients):
            raise ValueError("This prespecified family uses positive integer coefficients")

    @property
    def weight(self):
        return sum(self.coefficients)


def make_polynomial(k, family, seed):
    rng = np.random.default_rng(seed)
    if family == "linear":
        masks = [1 << j for j in range(k)]
    elif family == "shared_gate":
        masks = [1] + [1 | (1 << j) for j in range(1, k)]
    elif family == "sparse_polynomial":
        target = min(2*k, 64, sum(math.comb(k, d) for d in range(1, min(3,k)+1)))
        supports = set()
        while len(supports) < target:
            degree = int(rng.integers(1, min(3,k)+1))
            supports.add(sum(1 << int(j) for j in rng.choice(k, degree, replace=False)))
        masks = sorted(supports)
    else:
        raise ValueError("Unknown frozen model family")
    return Polynomial(k, tuple(masks), tuple(map(int,rng.integers(1,10,len(masks)))))


def numerator(poly, codes):
    codes = np.asarray(codes, dtype=np.uint64)
    answer = np.zeros(codes.shape, dtype=np.int64)
    for mask, coefficient in zip(poly.masks, poly.coefficients, strict=True):
        parity = (np.bitwise_count(codes & np.uint64(mask)) & 1).astype(np.int64)
        answer += coefficient*(1-2*parity)
    return answer


def exact_means(poly, qs):
    degrees = {}
    for mask, coefficient in zip(poly.masks,poly.coefficients,strict=True):
        degrees[mask.bit_count()] = degrees.get(mask.bit_count(),0)+coefficient
    return [sum(F(c)*(2*F(q)-1)**d for d,c in degrees.items())/poly.weight for q in qs]


def exact_second_moments(poly, qs):
    degrees = {}
    for mask,c in zip(poly.masks,poly.coefficients,strict=True):
        for other,v in zip(poly.masks,poly.coefficients,strict=True):
            degree = (mask ^ other).bit_count()
            degrees[degree] = degrees.get(degree,0)+c*v
    reference = [sum(F(c)*(2*F(q)-1)**d for d,c in degrees.items())/poly.weight**2 for q in qs]
    uniform = F(degrees.get(0,0),poly.weight**2)
    return reference, uniform


def event_threshold(poly, reference_mean):
    boundary = -poly.weight*F(reference_mean)
    return boundary.numerator // boundary.denominator


def reference_aware_bound(mean, reference_mean, epsilon):
    t,b,e = map(F,(mean,reference_mean,epsilon))
    if not 0 <= t <= 1 or not -1 <= b <= 1 or not 0 <= e <= 1:
        raise ValueError("Invalid mean or TV budget")
    if b == -1:
        if t != 0:
            raise ValueError("Infeasible positive mean when b=-1")
        return F(1)
    if not -1 <= 2*t-b <= 1:
        raise ValueError("Completion contrast mean outside [-1,1]")
    return max(F(0),min(F(1),1-t,1-2*t/(1+b),e+(1-b)/(1+b)))


def cantelli_bound(mean, variance):
    t,v = map(F,(mean,variance))
    if v < 0:
        raise ValueError("Negative variance")
    return F(1) if t <= 0 else v/(v+t*t)


def enumerate_laws(poly, qs, means, chunk=65536):
    """Integer event classification; floating probability aggregation only."""
    thresholds = [event_threshold(poly,b) for b in means]
    counts = np.zeros(poly.k+1)
    sums = np.zeros(poly.k+1)
    squares = np.zeros(poly.k+1)
    bad = np.zeros((len(qs),poly.k+1))
    for start in range(0,1 << poly.k,chunk):
        codes = np.arange(start,min(start+chunk,1 << poly.k),dtype=np.uint64)
        neg = np.bitwise_count(codes).astype(np.int64)
        values = numerator(poly,codes)
        counts += np.bincount(neg,minlength=poly.k+1)
        sums += np.bincount(neg,weights=values,minlength=poly.k+1)
        squares += np.bincount(neg,weights=values*values,minlength=poly.k+1)
        for i,threshold in enumerate(thresholds):
            bad[i] += np.bincount(neg[values<=threshold],minlength=poly.k+1)
    rows=[]
    for i,q in enumerate(qs):
        q=float(q)
        probability=np.array([q**(poly.k-j)*(1-q)**j for j in range(poly.k+1)])
        rows.append(dict(risk_r=float(bad[i]@probability),risk_u=float(bad[i].sum()/2**poly.k),
            mean_r=float(sums@probability/poly.weight),second_r=float(squares@probability/poly.weight**2),
            probability_sum=float(counts@probability)))
    return dict(laws=rows,mean_u=float(sums.sum()/2**poly.k/poly.weight),
        second_u=float(squares.sum()/2**poly.k/poly.weight**2),states=2**poly.k)


def branch_interval(poly, q, reference_mean, node_budget=5000):
    """Sound partial polynomial ranges; unresolved probability is retained."""
    if node_budget < 1:
        raise ValueError("Positive node budget required")
    threshold=event_threshold(poly,reference_mean)
    influence=[sum(c for m,c in zip(poly.masks,poly.coefficients,strict=True) if m & (1<<j)) for j in range(poly.k)]
    order=sorted(range(poly.k),key=lambda j:(-influence[j],j))
    assigned=[0]
    for j in order:
        assigned.append(assigned[-1] | (1<<j))
    stack=[(0,0,1.,1.)]
    bad_r=bad_u=0.
    nodes=0
    q=float(q)
    while stack and nodes<node_budget:
        depth,negative,mass_r,mass_u=stack.pop()
        nodes+=1
        fixed=radius=0
        for mask,c in zip(poly.masks,poly.coefficients,strict=True):
            if mask & ~assigned[depth]:
                radius+=c
            else:
                fixed+=c*(1-2*((mask & negative).bit_count()%2))
        if fixed+radius<=threshold:
            bad_r+=mass_r
            bad_u+=mass_u
        elif fixed-radius>threshold:
            continue
        else:
            assert depth<poly.k
            bit=1 << order[depth]
            stack.append((depth+1,negative|bit,mass_r*(1-q),mass_u/2))
            stack.append((depth+1,negative,mass_r*q,mass_u/2))
    unresolved_r=math.fsum(node[2] for node in stack)
    unresolved_u=math.fsum(node[3] for node in stack)
    return dict(lower_r=bad_r,upper_r=bad_r+unresolved_r,lower_u=bad_u,upper_u=bad_u+unresolved_u,
        nodes=nodes,unresolved_r=unresolved_r,unresolved_u=unresolved_u,complete=not stack)


def binomial_upper(failures, draws, delta):
    if not 0 <= failures <= draws or draws<1 or not 0<delta<1:
        raise ValueError("Invalid binomial interval input")
    return 1. if failures==draws else float(beta.ppf(1-delta,failures+1,draws-failures))


def monte_carlo(poly,q,epsilon,reference_mean,draws,delta,seed):
    rng=np.random.default_rng(seed)
    uniform=rng.random(draws)<float(epsilon)
    negative=rng.random((draws,poly.k))<np.where(uniform[:,None],.5,1-float(q))
    powers=np.left_shift(np.uint64(1),np.arange(poly.k,dtype=np.uint64))
    codes=np.sum(negative.astype(np.uint64)*powers,axis=1,dtype=np.uint64)
    failures=int(np.count_nonzero(numerator(poly,codes)<=event_threshold(poly,reference_mean)))
    return dict(failures=failures,draws=draws,upper=binomial_upper(failures,draws,delta),delta=delta)
