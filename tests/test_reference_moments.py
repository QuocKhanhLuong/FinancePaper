from fractions import Fraction as F
import numpy as np
import pytest

from financepaper.evaluation.bounded_completion import finite_definition_matrix
from financepaper.evaluation.completion_rank import rank_frontier,rank_risk_upper
from financepaper.evaluation.reference_moments import (
    box_tail_bound,parameters,reference_moment_bound,range_markov_bound,moment_witness,witness_table,
)


@pytest.mark.parametrize("m",[2,3,5,8])
@pytest.mark.parametrize("p",[F(1,10),F(1,2),F(9,10)])
def test_random_moments_witness_and_information_order(m,p):
    rng=np.random.default_rng(128)
    for _ in range(5):
        b=tuple(F(int(x),4) for x in rng.integers(-4,5,m-1))
        info=parameters(m,p,b)
        result=moment_witness(m,p,b)
        assert all(-1<=d<=1 for point in result["contrasts"] for d in point)
        for k,value in enumerate(b):
            assert sum(q*point[k] for q,point in zip(result["law"],result["contrasts"],strict=True))==value
        assert sum(q for q,g in zip(result["law"],result["gaps"],strict=True) if g<=0)==result["risk"]
        assert sum(q*g for q,g in zip(result["law"],result["gaps"],strict=True))==info["mean"]
        assert result["risk"]<=range_markov_bound(m,p,b)
        if info["mean"]>=0:
            assert result["risk"]<=rank_risk_upper(m,p,info["mean"])


@pytest.mark.parametrize("m",[2,3,5])
def test_attainer_matches_independent_shap_definition(m):
    p=F(1,2)
    z=F(1,4)
    loss=rank_frontier(m,p,z)["loss"]
    b=tuple(1-2*z*l for l in loss)
    witness=moment_witness(m,p,b)
    assert witness["risk"]==z
    table=witness_table(m,witness)
    assert all(table[(h+1)*2**m-1]==F(1,2) for h in range(len(witness["law"])))
    for h,gap in enumerate(witness["gaps"]):
        op=finite_definition_matrix(m,p,witness["law"],h)
        actual=sum((a-b)*v for a,b,v in zip(op[0],op[1],table,strict=True))/(1-p)
        assert actual==gap


def test_boundary_information_can_force_zero_risk():
    assert reference_moment_bound(3,F(1,2),[0,1])==0
    assert range_markov_bound(3,F(1,2),[0,1])==F(1,10)
    assert rank_risk_upper(3,F(1,2),F(3,4))>0
    assert box_tail_bound([1,1],[F(1,4),F(3,4)],2)==F(1,4)
    assert box_tail_bound([1,1],[0,F(1,2)],F(3,2))==0
    assert box_tail_bound([1],[F(1,2)],F(1,2))==1
    assert box_tail_bound([1],[F(1,2)],2)==0


@pytest.mark.parametrize("operation",[
    lambda:box_tail_bound([0],[F(1,2)],1),
    lambda:box_tail_bound([1],[2],1),
    lambda:reference_moment_bound(3,F(1,2),[0]),
    lambda:reference_moment_bound(3,F(1,2),[0,2]),
])
def test_invalid_moment_information(operation):
    with pytest.raises(ValueError):
        operation()
