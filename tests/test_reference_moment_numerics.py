"""Failure-driven regression tests; added after the first audit's LP mismatch."""
from fractions import Fraction as F
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from run_reference_moment_audit_v2 import generic_moment_lp
from financepaper.evaluation.reference_moments import parameters,reference_moment_bound


@pytest.mark.parametrize("m",[16,32])
@pytest.mark.parametrize("profile",["zero","constant","ascending","alternating"])
def test_ipm_large_reference_moment_profiles(m,profile):
    n=m-1
    b={"zero":[F(0)]*n,"constant":[F(1,2)]*n,
       "ascending":[F(k,n-1) for k in range(n)],
       "alternating":[F(k%2) for k in range(n)]}[profile]
    info=parameters(m,F(9,10),b)
    assert abs(generic_moment_lp(info)-float(reference_moment_bound(m,F(9,10),b)))<1e-9
