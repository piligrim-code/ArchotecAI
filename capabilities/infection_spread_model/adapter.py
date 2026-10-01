from numbers import Integral

from ortools.linear_solver import pywraplp

def infection_spread_model(infections_at_locations):
    """Return a toy contact estimate, not an epidemiological prediction.

    The heuristic weight for an unordered pair is min(counts) // 5.
    The original unconstrained positive-weight objective selects every such
    pair. Retaining the solver here demonstrates the public adapter contract;
    it does not imply a calibrated disease model or meaningful optimization.
    """
    counts = list(infections_at_locations)
    if len(counts) > 128:
        raise ValueError('The demo supports at most 128 locations')
    if any(isinstance(n, bool) or not isinstance(n, Integral) or not 0 <= n <= 10**9 for n in counts):
        raise ValueError('Counts must be integers between 0 and 1000000000')
    weights = {
        (i, j): min(int(counts[i]), int(counts[j])) // 5
        for i in range(len(counts)) for j in range(i + 1, len(counts))
        if min(counts[i], counts[j]) >= 5
    }
    if not weights:
        return {'contacts': {}, 'total_contacts': 0}
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        raise RuntimeError('SCIP solver is unavailable')
    solver.SetTimeLimit(2000)
    variables = {pair: solver.BoolVar(f'contact_{pair[0]}_{pair[1]}') for pair in weights}
    objective = solver.Objective()
    for pair, weight in weights.items():
        objective.SetCoefficient(variables[pair], weight)
    objective.SetMaximization()
    if solver.Solve() != pywraplp.Solver.OPTIMAL:
        raise RuntimeError('The demo requires an optimal solver result')
    contacts = {pair: weight for pair, weight in weights.items() if variables[pair].solution_value() > 0.5}
    return {'contacts': contacts, 'total_contacts': sum(contacts.values())}
