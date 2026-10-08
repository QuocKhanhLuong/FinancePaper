from fractions import Fraction as F
import numpy as np
import pytest

from financepaper.evaluation.completion_moments import FiniteMarginal,coalition_oracle
from financepaper.evaluation.disclosure_compute import (
    make_polynomial,numerator,exact_means,exact_second_moments,event_threshold,
    reference_aware_bound,cantelli_bound,enumerate_laws,branch_interval,binomial_upper,
)
from financepaper.evaluation.law_robust_disclosure import law_risk_upper


@pytest.mark.parametrize("family",["linear","sparse_polynomial","shared_gate"])
def test_grouped_hidden_identity_moments_and_tail(family):
    poly=make_polynomial(4,family,83)
    q=F(3,4)
    b=exact_means(poly,[q])[0]
    seconds,u=exact_second_moments(poly,[q])
    enumeration=enumerate_laws(poly,[q],[b],chunk=3)
    assert enumeration["laws"][0]["mean_r"]==pytest.approx(float(b),abs=1e-14)
    assert enumeration["laws"][0]["second_r"]==pytest.approx(float(seconds[0]),abs=1e-14)
    assert enumeration["second_u"]==pytest.approx(float(u),abs=1e-14)
    assert enumeration["mean_u"]==pytest.approx(0.,abs=1e-14)
    marginal=FiniteMarginal((0.,1.),(.5,.5))
    probs=tuple(float(q**(4-h.bit_count())*(1-q)**h.bit_count()) for h in range(16))
    hidden=FiniteMarginal(tuple(map(float,range(16))),probs)
    def predict(batch):
        d=numerator(poly,batch[:,2].astype(np.uint64))/poly.weight
        return .5+.5*d*(batch[:,0]-batch[:,1])
    for h in range(16):
        phi=coalition_oracle(predict,[marginal,marginal,hidden],[1.,1.,float(h)],[0,1])
        assert (phi[0]-phi[1])/.5==pytest.approx((float(numerator(poly,[h])[0]/poly.weight)+float(b))/2,abs=1e-14)


@pytest.mark.parametrize("nodes",[1,5,5000])
@pytest.mark.parametrize("family",["linear","sparse_polynomial","shared_gate"])
def test_branch_retains_unresolved_probability(nodes,family):
    poly=make_polynomial(8,family,811)
    q=F(9,10)
    b=exact_means(poly,[q])[0]
    truth=enumerate_laws(poly,[q],[b])["laws"][0]
    branch=branch_interval(poly,q,b,nodes)
    for law in ("r","u"):
        assert branch[f"lower_{law}"]<=truth[f"risk_{law}"]+1e-12
        assert truth[f"risk_{law}"]<=branch[f"upper_{law}"]+1e-12
    assert branch["nodes"]<=nodes
    if nodes==5000:
        assert branch["complete"]


def test_reference_information_dominates_c6_exact_grid():
    for it in range(11):
        t=F(it,10)
        for ie in range(11):
            e=F(ie,10)
            for ib in range(-9,11):
                b=F(ib,10)
                if -1<=2*t-b<=1:
                    assert reference_aware_bound(t,b,e)<=law_risk_upper(t,e)
    assert reference_aware_bound(F(1,2),F(1,2),F(1,10))==F(1,3)
    assert law_risk_upper(F(1,2),F(1,10))==F(11,30)


def test_integer_tie_and_degenerate_variance():
    poly=make_polynomial(2,"linear",7)
    assert event_threshold(poly,F(0))==0
    assert cantelli_bound(F(0),F(0))==1
    assert cantelli_bound(F(1,2),F(0))==0
    assert reference_aware_bound(0,-1,0)==1


def test_one_sided_exact_binomial_bound():
    assert binomial_upper(0,256,.001)==pytest.approx(1-.001**(1/256))
    assert binomial_upper(256,256,.001)==1.
    assert binomial_upper(4,256,.001)>binomial_upper(0,256,.001)
