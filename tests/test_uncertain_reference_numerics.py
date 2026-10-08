from fractions import Fraction as F
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from run_uncertain_reference_moments_v2 import uncertain_lp_residual
from financepaper.evaluation.reference_moments import parameters


def test_degenerate_fixed_upper_endpoint_lp_terminates():
    for z,expected in ((F(0),0.),(F(1,2),.25)):
        assert abs(uncertain_lp_residual(4,F(1,10),(F(1),)*3,(F(1),)*3,z)-expected)<1e-9


def test_independent_interval_lp_matches_rational_lower_corner():
    lower=(F(-1,100),F(99,100))
    upper=(F(1,100),F(1))
    info=parameters(3,F(1,2),lower)
    for z in (F(1,100),F(1,4),F(3,4)):
        exact=z*info["tau"]-sum(a*min(z,mu) for a,mu in zip(info["a"],info["mu"],strict=True))
        assert abs(uncertain_lp_residual(3,F(1,2),lower,upper,z)-float(exact))<1e-9
