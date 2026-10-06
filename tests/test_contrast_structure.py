import numpy as np
import pytest

from financepaper.explanations.contrast_structure import (
    Expression, completion_contract, contrast_projection, channel_events, graph_summary, structure,
)
from financepaper.explanations.rank_event_audit import toy_compiler
from financepaper.reliability.validation import revision_arrays


def test_specialization_merges_and_cancels_identical_rectangles():
    expression = Expression((((0, 0., 2.), (1, 0., 2.)), ((1, 0., 2.),), ((0, 3., 4.),)),
                            np.array([[1., 2.], [-1., -2.], [9., 9.]]))
    result, counts = expression.specialize([1., np.nan], np.array([True, False]))
    assert len(result.rectangles) == 0
    assert counts == dict(input_terms=3, surviving_terms=2, merged_terms=1, exact_zero_terms=1)
    np.testing.assert_array_equal(result.values([[1., 1.]]), [[0., 0.]])


@pytest.mark.parametrize("seed", [101, 102, 103])
def test_projection_specialization_commute_and_match_full_frozen_event(seed):
    compiler = toy_compiler()
    expression = Expression(compiler.rectangles, compiler.coefficients)
    grouping = np.eye(4)[:, :3]
    points = np.column_stack([np.ones((5, 3)), [-1., -.5, 0., 1., 2.]])
    before = compiler.values(points[[0]]) @ grouping
    projection, candidates, slices = contrast_projection(before[0], np.ones(3, bool))
    raw = expression.project(grouping)
    groups, _ = raw.specialize([1., 1., 1., np.nan], np.array([1, 1, 1, 0], bool))
    a = groups.project(projection)
    b, _ = raw.project(projection).specialize([1., 1., 1., 100000.], np.array([1, 1, 1, 0], bool))
    order = np.random.default_rng(seed).permutation(5)
    np.testing.assert_allclose(a.values(points[order]), b.values(points[order]), atol=1e-12)
    frozen = revision_arrays(before, compiler.values(points)[None] @ grouping, np.zeros((1, 3), bool), k=2)
    np.testing.assert_array_equal(channel_events(a.values(points), slices), frozen["event"][0])
    np.testing.assert_array_equal(candidates, frozen["reasons"][0])


def test_complete_metadata_fixed_but_hidden_values_remain_variable():
    origins = ("a", "a", "b", "__observed_a", "__delta_a")
    encoded = np.array([[1, 0, 3, 1, .2], [0, 1, 3, 1, .2]], np.float32)
    partial, fixed, mapping, available, original = completion_contract(
        origins, ("a", "b"), {"g": ["a", "b"]}, np.array([True, False]), encoded,
        expected_metadata={"__observed_a": 1., "__delta_a": .2})
    np.testing.assert_array_equal(fixed, [False, False, True, True, True])
    assert np.isnan(partial[:2]).all()
    np.testing.assert_array_equal(mapping[:, 0], [0, 0, 1, 0, 0])
    np.testing.assert_array_equal(original, [0, 0, 1, -1, -1])
    assert available.item()
    # Equal hidden values in a finite sample do not justify treating them as observed.
    duplicate = np.repeat(encoded[[0]], 2, axis=0)
    assert not completion_contract(origins, ("a", "b"), {"g": ["a", "b"]},
                                   np.array([True, False]), duplicate,
                                   expected_metadata={"__observed_a": 1., "__delta_a": .2})[1][0]


@pytest.mark.parametrize("corruption", ["mask", "delta", "observed", "unknown"])
def test_completion_encoding_guards(corruption):
    origins = ["a", "b", "__observed_a", "__delta_a"]
    points = np.array([[1, 2, 1, .2], [0, 2, 1, .2]], np.float32)
    if corruption == "mask": points[0, 2] = 0
    if corruption == "delta": points[0, 3] = .4
    if corruption == "observed": points[0, 1] = 5
    if corruption == "unknown": origins[-1] = "__unknown_a"
    with pytest.raises(ValueError):
        completion_contract(origins, ("a", "b"), {"g1": ["a"], "g2": ["b"]},
                            np.array([True, False]), points,
                            expected_metadata={"__observed_a": 1., "__delta_a": .2})


def test_complete_delta_first_month_is_zero_and_later_months_one_fifth():
    from financepaper.data.temporal import TEMPORAL_FIELDS, elapsed_delta
    names = tuple(f for fields in TEMPORAL_FIELDS for f in fields)
    origins = names + tuple(f"__observed_{f}" for f in names) + tuple(f"__delta_{f}" for f in names)
    delta = elapsed_delta(np.ones((6, 3), bool))/5.
    np.testing.assert_array_equal(delta[0], np.zeros(3))
    np.testing.assert_array_equal(delta[1:], np.full((5, 3), .2, np.float32))
    metadata = {f"__observed_{f}": 1. for f in names}
    metadata.update({f"__delta_{f}": v for f, v in zip(names, delta.ravel())})
    points = np.tile(np.r_[np.zeros(18), np.ones(18), delta.ravel()], (2, 1)).astype(np.float32)
    result = completion_contract(origins, names, {"all": list(names)}, np.ones(18, bool),
                                 points, expected_metadata=metadata)
    assert not result[1][:18].any() and result[1][18:].all()
    corrupt = points.copy()
    corrupt[:, 36:39] = .2
    with pytest.raises(ValueError, match="complete-state encoding"):
        completion_contract(origins, names, {"all": list(names)}, np.ones(18, bool),
                            corrupt, expected_metadata=metadata)
    with pytest.raises(ValueError, match="cover every"):
        completion_contract(origins, names, {"all": list(names)}, np.ones(18, bool),
                            points, expected_metadata={})


def test_tied_outsider_is_not_automatically_revision():
    projection, _, slices = contrast_projection([2., 1.5, 0.], np.ones(3, bool))
    after = np.array([[1., 1., 2.], [0., 1., 2.]])
    np.testing.assert_array_equal(channel_events(after @ projection, slices), [False, True])
    assert contrast_projection([0., .001, .002], np.ones(3, bool)) is None


@pytest.mark.parametrize("available", [np.ones(3, bool), np.ones(8, bool),
                                     np.array([True, False, True, False, True, False, False, False])])
def test_sign_plus_all_contrasts_cannot_remove_nonzero_union_terms(available):
    # Post-run algebraic diagnostic, not a revised headroom criterion: the
    # anchor sign plus all differences reconstruct every available group.
    n = len(available)
    projection, candidates, _ = contrast_projection(np.arange(1., n+1.), available)
    c = np.random.default_rng(20261006).integers(-3, 4, size=(20, n)).astype(float)
    c[:, ~available] = 0
    c[0] = 0
    q = c @ projection
    np.testing.assert_array_equal(np.any(q != 0, axis=1), np.any(c != 0, axis=1))
    assert np.linalg.matrix_rank(projection[available]) == int(available.sum())
    anchor = candidates[0]
    np.testing.assert_array_equal(q[:, 0], c[:, anchor])
    for offset, h in enumerate(h for h in np.flatnonzero(available) if h != anchor):
        np.testing.assert_array_equal(q[:, 2+offset]+q[:, 0], c[:, h])


def test_term_graph_width_is_only_a_structural_diagnostic():
    assert graph_summary([{0, 1}, {1, 2}])["min_fill_width_upper"] == 1
    assert graph_summary([{0, 1, 2}])["min_fill_width_upper"] == 2
    assert graph_summary([])["largest_component"] == 0
    info = graph_summary([{0}, {1}])
    assert info["components"] == 2 and info["min_fill_width_upper"] == 0
    # Even disconnected additive supports do not factor the threshold event.
    atoms = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    event = atoms.sum(1) > .5
    assert not np.array_equal(event, (atoms[:, 0] > .5) & (atoms[:, 1] > .5))


def test_tiny_nonzero_terms_and_constant_channels_are_preserved():
    e = Expression(((), ((0, 0, 1),)), np.array([[2., 2., 0.], [1e-30, 1e-30, 0.]]))
    assert e.project(np.eye(3)).coefficients[1, 0] == 1e-30
    info = structure(e, [5])
    assert info["zero_channels"] == 1 and info["unique_channels"] == 2
    assert info["constant_channels"] == 1
    with pytest.raises(ValueError): structure(e, [-1])


def test_fixed_value_validation_ignores_only_unknown_placeholders():
    e = Expression((((0, 0, 2),),), np.ones((1, 1)))
    with pytest.raises(ValueError): e.specialize([np.nan], np.array([True]))
    with pytest.raises(ValueError): e.specialize([1], np.array([1]))
    result, _ = e.specialize([np.nan], np.array([False]))
    assert result.values([[1.]]).item() == 1
