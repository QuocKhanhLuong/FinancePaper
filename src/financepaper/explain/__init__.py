"""Attribution utilities for the temporal pilot.

The temporal attribution code operates on the model's raw risk logit.  It is
kept separate from the legacy tree/LR SHAP implementation so that the two
attribution conventions cannot be confused accidentally.
"""

from .integrated_gradients import aggregate_original, integrated_gradients

__all__ = ["aggregate_original", "integrated_gradients"]
