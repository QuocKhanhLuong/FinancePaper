from fractions import Fraction as F
from itertools import product
from math import comb

import pytest

from financepaper.evaluation.reference_moments import reference_moment_bound
from financepaper.evaluation.uncertain_reference_moments import (
    moment_box, interval_risk_bound, hoeffding_radius, c5_certifies,
)


@pytest.mark.parametrize("m", [2,3,5])
@pytest.mark.parametrize("p", [F(1,10),F(1,2),F(9,10)])
def test_box_bound_dominates_every_corner(m,p):
    center=tuple(F(k,m-1) for k in range(m-1))
    lower,upper=moment_box(center,F(1,5))
    worst=interval_risk_bound(m,p,lower,upper)
    for corner in product(*zip(lower,upper)):
        assert reference_moment_bound(m,p,corner)<=worst


def test_boundary_gain_and_uncertainty_monotonicity():
    previous=F(0)
    for radius in (0,F(1,1000),F(1,100),F(1,10),1,2):
        lower,upper=moment_box([0,1],radius)
        risk=interval_risk_bound(3,F(1,2),lower,upper)
        assert risk>=previous
        previous=risk
    assert previous==1


def test_confidence_radius_penalties_and_exact_release_boundary():
    assert hoeffding_radius(2048,1,F(1,20))<hoeffding_radius(2048,15,F(1,20))
    assert hoeffding_radius(32768,15,F(1,20))<hoeffding_radius(2048,15,F(1,20))
    assert not c5_certifies(3,F(1,2),-1,F(1,20))
    assert c5_certifies(3,F(1,2),1,F(1,20))


def test_hoeffding_one_sided_coverage_by_exact_binomial_enumeration():
    n=32
    delta=F(1,20)
    radius=hoeffding_radius(n,1,delta)
    for probability in (F(1,10),F(1,2),F(9,10)):
        failure=sum(F(comb(n,k))*probability**k*(1-probability)**(n-k)
            for k in range(n+1) if 2*F(k,n)-1-radius>2*probability-1)
        assert failure<=delta


@pytest.mark.parametrize("operation",[
    lambda:moment_box([2],0),lambda:moment_box([0],-1),
    lambda:interval_risk_bound(3,F(1,2),[0],[1]),
    lambda:interval_risk_bound(2,F(1,2),[1],[0]),
    lambda:hoeffding_radius(0,1,F(1,20)),
    lambda:hoeffding_radius(10,0,F(1,20)),
    lambda:hoeffding_radius(10,1,0),lambda:c5_certifies(2,F(1,2),0,0),
])
def test_invalid_inputs(operation):
    with pytest.raises(ValueError):
        operation()
