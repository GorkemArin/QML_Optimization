from pyomo.environ import *

# Create a model
model = ConcreteModel()

# Define binary variables
model.x = Var([1, 2], within=Binary)

# Define objective function in COO format
Q_data = [
    (1, 1, 4),  # x1^2 coefficient
    (2, 2, 5),  # x2^2 coefficient
    (1, 2, 3)   # x1*x2 coefficient
]
c_data = [
    (1, 1),  # x1 coefficient
    (2, 2)   # x2 coefficient
]

# Objective function
def objective_rule(model):
    return sum(c * model.x[i] for i, c in c_data) + sum(q * model.x[i] * model.x[j] for i, j, q in Q_data)

model.obj = Objective(rule=objective_rule, sense=minimize)

# Constraint
model.constr = Constraint(expr=model.x[1] + model.x[2] <= 1)

model.pprint()

# Solve with a solver supporting binary quadratic models
solver = SolverFactory("gurobi")  # Use an appropriate solver
solver.solve(model)

# Print results
print(f"x1 = {model.x[1].value}, x2 = {model.x[2].value}")
