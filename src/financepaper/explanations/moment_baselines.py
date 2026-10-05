"""Known orthogonal-basis moment control, NOT an official FourierSHAP port.

Restricted to four equiprobable independent atoms per hidden coordinate. The
compiled point-attribution representation is shared with the candidate. The
mean/Gram identity is standard Parseval, not a claimed contribution.
"""
from itertools import product
import time
import numpy as np
from .conditional_moments import MomentBudgetExceeded


def uniform_four_atom_moments(compiler, x, observed, atoms, *, max_frequencies=200000,
                              timeout_seconds=60.):
    start = time.perf_counter()
    x, observed = np.asarray(x, float), np.asarray(observed)
    if x.shape != (compiler.dimension,) or observed.shape != x.shape or observed.dtype != bool:
        raise ValueError('Matching input and explicit boolean observation mask required')
    partial = np.full(compiler.dimension, np.nan)
    partial[observed] = x[observed].astype(compiler.input_dtype)
    atoms = np.asarray(atoms, dtype=compiler.input_dtype).astype(float)
    if atoms.shape != (4,) or not np.isfinite(atoms).all() or not np.isfinite(partial[observed]).all():
        raise ValueError('Four finite equiprobable atoms and finite observed values required')
    # Rows correspond to atom indices; columns are Walsh frequencies.
    h = np.array([[1, 1, 1, 1], [1, -1, 1, -1],
                  [1, 1, -1, -1], [1, -1, -1, 1]], dtype=float)
    spectrum, factors, expanded_terms = {}, {}, 0
    for rectangle, coefficient in zip(compiler.rectangles, compiler.coefficients):
        if time.perf_counter()-start > timeout_seconds:
            raise MomentBudgetExceeded('Fourier control wall-clock budget exceeded')
        if any(observed[j] and not lo <= partial[j] < hi for j, lo, hi in rectangle):
            continue
        hidden = [(j, lo, hi) for j, lo, hi in rectangle if not observed[j]]
        axes = []
        for key in hidden:
            j, lo, hi = key
            if key not in factors:
                expansion = h.T @ ((atoms >= lo) & (atoms < hi)) / 4
                factors[key] = [(frequency, value) for frequency, value in enumerate(expansion) if value != 0]
            axes.append(factors[key])
        for choices in product(*axes):
            frequency_key = tuple((hidden[i][0], frequency) for i, (frequency, _) in enumerate(choices) if frequency)
            scale = np.prod([v for _, v in choices]) if choices else 1.
            spectrum.setdefault(frequency_key, np.zeros(compiler.dimension))[:] += scale*coefficient
            expanded_terms += 1
            if len(spectrum) > max_frequencies:
                raise MomentBudgetExceeded('Fourier control frequency budget exceeded')
    mean = spectrum.pop((), np.zeros(compiler.dimension))
    nonconstant = np.array([v for v in spectrum.values() if np.any(v != 0)]).reshape(-1, compiler.dimension)
    covariance = nonconstant.T @ nonconstant
    return {'mean': mean, 'covariance': covariance, 'frequencies': len(nonconstant),
            'expanded_terms': expanded_terms, 'coefficient_bytes': nonconstant.nbytes,
            'elapsed_seconds': time.perf_counter()-start}
