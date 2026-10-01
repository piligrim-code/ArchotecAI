from types import SimpleNamespace

import pytest

from capabilities.infection_spread_model import adapter


@pytest.mark.parametrize('counts', [[], [0], [10], [0, 0], [4, 8]])
def test_no_contacts(counts):
    assert adapter.infection_spread_model(counts) == {'contacts': {}, 'total_contacts': 0}


def test_original_two_location_regression():
    assert adapter.infection_spread_model([5, 5]) == {'contacts': {(0, 1): 1}, 'total_contacts': 1}


def test_pairs_are_unique_and_total_is_deterministic():
    result = adapter.infection_spread_model([10, 20, 5])
    assert result == {'contacts': {(0, 1): 2, (0, 2): 1, (1, 2): 1}, 'total_contacts': 4}
    assert result == adapter.infection_spread_model([10, 20, 5])


@pytest.mark.parametrize('counts', [[-1], [True], [1.5], ['5'], [float('nan')], [10**9 + 1], [5] * 129])
def test_invalid_inputs(counts):
    with pytest.raises(ValueError):
        adapter.infection_spread_model(counts)


def test_missing_solver_is_explicit(monkeypatch):
    monkeypatch.setattr(adapter.pywraplp.Solver, 'CreateSolver', lambda _: None)
    with pytest.raises(RuntimeError, match='unavailable'):
        adapter.infection_spread_model([5, 5])


def test_failed_solve_is_not_reported_as_success(monkeypatch):
    objective = SimpleNamespace(SetCoefficient=lambda *args: None, SetMaximization=lambda: None)
    solver = SimpleNamespace(SetTimeLimit=lambda _: None, BoolVar=lambda _: object(),
                             Objective=lambda: objective, Solve=lambda: -1)
    monkeypatch.setattr(adapter.pywraplp.Solver, 'CreateSolver', lambda _: solver)
    with pytest.raises(RuntimeError, match='optimal'):
        adapter.infection_spread_model([5, 5])
