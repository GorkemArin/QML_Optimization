import gurobipy as gp
from gurobipy import GRB, Model, quicksum

def CheckGurobiLicense():
    try:
        env = gp.Env(empty=True)  # Create an empty environment
        env.start()  # Try to start the environment (checks for a valid license)
        print("+ Gurobi license is active.")
    except gp.GurobiError as e:
        print(f"! Gurobi license error. Please activate one.")
        raise e

def solve_gurobi(qubo_model: dict) -> dict:
    """
    Solve a QUBO given as a dictionary using Gurobi.

    QUBO format:
        {(i, j): coefficient}

    Returns:
        {variable_name: 0 or 1}
    """
    CheckGurobiLicense()

    model = Model("QUBO")
    model.setParam("OutputFlag", 0)

    # Collect variable names
    vars_set = set()
    for i, j in qubo_model:
        vars_set.add(i)
        vars_set.add(j)

    # Create binary variables
    x = model.addVars(vars_set, vtype=GRB.BINARY, name="x")

    # Build quadratic objective
    obj = gp.QuadExpr()
    handled = set()

    for (i, j), coeff in qubo_model.items():
        key = tuple(sorted((i, j)))
        if key in handled:
            continue
        handled.add(key)

        if i == j:
            obj.add(coeff * x[i])
        else:
            obj.add(coeff * x[i] * x[j])

    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    # Extract solution
    solution = {v: int(round(x[v].X)) for v in vars_set}
    return solution