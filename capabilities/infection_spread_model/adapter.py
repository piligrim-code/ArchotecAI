from ortools.linear_solver import pywraplp
import numpy as np

def infection_spread_model(infections_at_locations):
    # Define variables
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        raise Exception('Failed to create the solver.')

    num_locations = len(infections_at_locations)
    contacts_between_locations = np.zeros((num_locations, num_locations), dtype=int)

    for i in range(num_locations):
        for j in range(i + 1, num_locations):
            # Assuming a simple model where contact between locations is proportional to infections
            contacts_between_locations[i][j] = max(0, min(infections_at_locations[i], infections_at_locations[j]) // 5)
            contacts_between_locations[j][i] = contacts_between_locations[i][j]

    contacts_flattened = contacts_between_locations.flatten().tolist()
    
    # Decision variables: whether to activate contact model between locations
    binary_contacts = {}
    for i in range(num_locations * (num_locations - 1) // 2):
        binary_contacts[f"contact_{i}"] = solver.BoolVar(f"contact_{i}")

    # Objective function: maximize total contacts while considering constraints
    objective = solver.Objective()
    
    for i in range(len(contacts_flattened)):
        if contacts_flattened[i] > 0:
            objective.SetCoefficient(binary_contacts[f"contact_{i}"], contacts_flattened[i])

    # Constraint: ensure the model does not double-count any contacts
    constraints = [solver.Constraint(-solver.infinity(), solver.infinity()) for _ in range(num_locations)]
    
    for i in range(num_locations * (num_locations - 1) // 2):
        row, col = divmod(i, num_locations - 1)
        if contacts_between_locations[row][col] > 0:
            constraints[row].SetCoefficient(binary_contacts[f"contact_{i}"], -1)

    # Solve the model
    solver.EnableOutput()
    solver.Solve()

    result = {}
    for i in range(len(contacts_flattened)):
        contact_var_name = f"contact_{i}"
        if binary_contacts[contact_var_name].solution_value() == 1:
            row, col = divmod(i, num_locations - 1)
            result[(row, col)] = contacts_between_locations[row][col]

    return {"contacts": result, "total_contacts": sum(contacts for _, contacts in result.items())}