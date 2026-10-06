"""Generic expression controls for a bounded structural audit, not a new solver.

All simplifications are clamping, linear projection and identical-term merging.
Term-support graphs are NOT the primal graph of a thresholded rank event.
"""
from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass
class Expression:
    rectangles: tuple
    coefficients: np.ndarray

    def __post_init__(self):
        self.coefficients = np.asarray(self.coefficients, dtype=float)
        if (self.coefficients.ndim != 2 or len(self.rectangles) != len(self.coefficients)
                or not np.isfinite(self.coefficients).all()):
            raise ValueError("Finite coefficient matrix aligned to rectangles required")

    def project(self, matrix):
        matrix = np.asarray(matrix, float)
        if matrix.ndim != 2 or matrix.shape[0] != self.coefficients.shape[1] or not np.isfinite(matrix).all():
            raise ValueError("Projection must match output dimension")
        c = self.coefficients @ matrix
        keep = np.any(c != 0, axis=1)
        return Expression(tuple(r for r, yes in zip(self.rectangles, keep) if yes), c[keep])

    def indicators(self, points):
        points = np.asarray(points)
        if points.ndim != 2 or not np.isfinite(points).all():
            raise ValueError("Complete finite points required")
        out = np.ones((len(points), len(self.rectangles)), bool)
        for r, rectangle in enumerate(self.rectangles):
            for j, lo, hi in rectangle:
                out[:, r] &= (points[:, j] >= lo) & (points[:, j] < hi)
        return out

    def values(self, points):
        return self.indicators(points) @ self.coefficients

    def specialize(self, values, fixed):
        values, fixed = np.asarray(values), np.asarray(fixed)
        if (values.ndim != 1 or fixed.shape != values.shape or fixed.dtype != bool
                or not np.isfinite(values[fixed]).all()):
            raise ValueError("Explicit boolean fixed mask and finite fixed values required")
        merged, surviving = {}, 0
        for rectangle, c in zip(self.rectangles, self.coefficients):
            if any(fixed[j] and not lo <= values[j] < hi for j, lo, hi in rectangle):
                continue
            surviving += 1
            key = tuple((j, lo, hi) for j, lo, hi in rectangle if not fixed[j])
            if key not in merged:
                merged[key] = c.copy()
            else:
                merged[key] += c
        keep = [r for r, c in merged.items() if np.any(c != 0)]
        result = Expression(tuple(keep), np.array([merged[r] for r in keep]).reshape(-1, self.coefficients.shape[1]))
        return result, dict(input_terms=len(self.rectangles), surviving_terms=surviving,
                            merged_terms=len(merged), exact_zero_terms=len(merged)-len(keep))


def completion_contract(origins, names, groups, hidden, encoded, *, expected_metadata):
    """Determine legal fixed completion inputs without reading hidden truth.

    Encoded donors must come from the frozen preprocessor. This verifies shared
    fixed coordinates and complete metadata; it does not invent one-hot samples.
    """
    hidden, encoded = np.asarray(hidden), np.asarray(encoded)
    fields = sum(groups.values(), [])
    if (hidden.shape != (len(names),) or hidden.dtype != bool or encoded.ndim != 2
            or len(encoded) == 0 or encoded.shape[1] != len(origins)
            or not np.isfinite(encoded).all() or sorted(fields) != sorted(names)
            or len(set(fields)) != len(names)):
        raise ValueError("Invalid schema, hidden mask or encoded completions")
    fixed, mapping, original = [], np.zeros((len(origins), len(groups))), []
    metadata_names = {name for name in origins if name.startswith(("__observed_", "__delta_"))}
    if set(expected_metadata) != metadata_names or not all(np.isfinite(v) for v in expected_metadata.values()):
        raise ValueError("Explicit complete-state metadata must cover every metadata origin")
    group_of = {f: g for g, members in enumerate(groups.values()) for f in members}
    for j, name in enumerate(origins):
        if name in names:
            field = names.index(name)
            fixed.append(not hidden[field])
            original.append(field)
            if not hidden[field]:
                mapping[j, group_of[name]] = 1
        elif name.startswith("__observed_") or name.startswith("__delta_"):
            prefix = "__observed_" if name.startswith("__observed_") else "__delta_"
            if name[len(prefix):] not in names:
                raise ValueError("Unknown metadata origin")
            expected = encoded.dtype.type(expected_metadata[name])
            if not np.all(encoded[:, j] == expected):
                raise ValueError(f"Completion metadata differs from complete-state encoding: {name}")
            fixed.append(True)
            original.append(-1)
        else:
            raise ValueError(f"Unrecognized encoded origin: {name}")
    fixed = np.array(fixed, bool)
    if not np.all(encoded[:, fixed] == encoded[0, fixed]):
        raise ValueError("Completion changed fixed observed coordinates")
    available = np.array([not hidden[[names.index(f) for f in members]].all()
                          for members in groups.values()])
    # Unknown coordinates in this partial vector stay erased, even though legal
    # finite completion atoms are separately available for evaluation.
    partial = np.where(fixed, encoded[0], np.nan)
    return partial, fixed, mapping, available, np.array(original)


def contrast_projection(current, available, *, k=2, magnitude=.01):
    current, available = np.asarray(current, float), np.asarray(available)
    if (current.ndim != 1 or available.shape != current.shape or available.dtype != bool
            or not np.isfinite(current).all() or not isinstance(k, int) or not 1 <= k <= len(current)):
        raise ValueError("Invalid current candidate state")
    valid = (current > magnitude) & available
    if valid.sum() < k:
        return None
    candidates = np.argsort(-np.where(valid, current, -np.inf), kind="stable")[:k]
    identity = np.eye(len(current))
    columns, slices = [identity[:, g] for g in candidates], []
    for g in candidates:
        start = len(columns)
        columns.extend(identity[:, h]-identity[:, g] for h in np.flatnonzero(available) if h != g)
        slices.append(slice(start, len(columns)))
    return np.column_stack(columns), candidates, slices


def channel_events(values, rank_slices, *, k=2):
    values = np.asarray(values, float)
    if values.ndim != 2 or not np.isfinite(values).all() or len(rank_slices) != k:
        raise ValueError("Finite sign/contrast channels required")
    out = (values[:, :k] <= 1e-6).any(1)
    for s in rank_slices:
        out |= (values[:, s] > 1e-6).sum(1) >= k
    return out


def graph_summary(supports):
    """Deterministic min-fill heuristic; does not infer event independence."""
    adjacency = {j: set() for support in supports for j in support}
    for support in supports:
        for a, b in combinations(sorted(set(support)), 2):
            adjacency[a].add(b)
            adjacency[b].add(a)
    remaining, components = set(adjacency), []
    while remaining:
        todo, component = [min(remaining)], set()
        while todo:
            j = todo.pop()
            if j in component:
                continue
            component.add(j)
            todo.extend(adjacency[j]-component)
        components.append(len(component))
        remaining -= component
    graph = {j: set(v) for j, v in adjacency.items()}
    width = 0
    while graph:
        def key(j):
            missing = sum(b not in graph[a] for a, b in combinations(sorted(graph[j]), 2))
            return missing, len(graph[j]), j
        j = min(graph, key=key)
        neighbours = graph[j]
        width = max(width, len(neighbours))
        for a, b in combinations(neighbours, 2):
            graph[a].add(b)
            graph[b].add(a)
        for a in neighbours:
            graph[a].remove(j)
        del graph[j]
    return dict(variables=len(adjacency), components=len(components),
                largest_component=max(components, default=0), min_fill_width_upper=width)


def structure(expression, original_indices):
    supports = [set(int(original_indices[j]) for j, _, _ in r) for r in expression.rectangles]
    if any(-1 in s for s in supports):
        raise ValueError("Metadata must be specialized before structural analysis")
    c = expression.coefficients
    channels = []
    for col in range(c.shape[1]):
        active = c[:, col] != 0
        channel_supports = [s for s, yes in zip(supports, active) if yes]
        channels.append(dict(terms=int(active.sum()), **graph_summary(channel_supports)))
    return dict(terms=len(c), nonzeros=int(np.count_nonzero(c)), channels=c.shape[1],
                coefficient_bytes=c.nbytes, constant_terms=sum(not s for s in supports),
                zero_channels=int(np.all(c == 0, axis=0).sum()),
                constant_channels=sum(row["variables"] == 0 for row in channels),
                unique_channels=len({c[:, j].tobytes() for j in range(c.shape[1])}),
                **graph_summary(supports), scalar_channels=channels)
