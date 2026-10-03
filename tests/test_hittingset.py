"""Exactness of the hitting set solver, checked against brute force."""

from __future__ import annotations

import itertools
import random

import pytest

from effectkey.hittingset import greedy_hitting_set, min_hitting_set


def brute_force(sets, universe):
    for size in range(len(universe) + 1):
        for combo in itertools.combinations(sorted(universe), size):
            if all(set(s) & set(combo) for s in sets):
                return size
    raise AssertionError("no hitting set exists over the universe")


def random_instance(rng):
    universe = [f"f{i}" for i in range(rng.randint(3, 9))]
    sets = [
        set(rng.sample(universe, rng.randint(1, min(4, len(universe)))))
        for _ in range(rng.randint(2, 12))
    ]
    return sets, universe


def test_empty_instance_has_empty_solution():
    result = min_hitting_set([])
    assert result.elements == ()
    assert result.optimal


def test_singletons_are_forced():
    result = min_hitting_set([{"a"}, {"b"}, {"a", "c"}])
    assert set(result.elements) == {"a", "b"}
    assert result.optimal


def test_matches_brute_force_on_random_instances():
    rng = random.Random(1234)
    for _ in range(300):
        sets, universe = random_instance(rng)
        result = min_hitting_set(sets)
        assert result.size == brute_force(sets, universe)
        assert result.optimal


def test_solution_is_always_valid_and_bounded():
    rng = random.Random(99)
    for _ in range(300):
        sets, _ = random_instance(rng)
        exact = min_hitting_set(sets)
        greedy = greedy_hitting_set(sets)
        assert all(set(s) & set(exact.elements) for s in sets)
        assert all(set(s) & set(greedy.elements) for s in sets)
        assert exact.size <= greedy.size
        assert exact.lower_bound <= exact.size


def test_forbidden_fields_are_never_selected():
    result = min_hitting_set([{"a", "b"}, {"b", "c"}], allowed={"a", "c"})
    assert "b" not in result.elements
    assert set(result.elements) == {"a", "c"}


def test_unhittable_constraint_is_reported_not_swallowed():
    result = min_hitting_set([{"p", "q"}], allowed={"z"})
    assert not result.feasible
    assert result.unhittable == (("p", "q"),)


def test_weights_break_ties_toward_stable_fields():
    sets = [{"a", "b"}]
    assert min_hitting_set(sets, weights={"a": 0.9, "b": 0.1}).elements == ("b",)
    assert min_hitting_set(sets, weights={"a": 0.1, "b": 0.9}).elements == ("a",)


def test_node_budget_still_returns_a_valid_cover():
    rng = random.Random(7)
    sets = [set(rng.sample([f"f{i}" for i in range(14)], 5)) for _ in range(40)]
    result = min_hitting_set(sets, node_budget=5)
    assert all(set(s) & set(result.elements) for s in sets)
    assert not result.optimal
