'''
Open-source solvers like GLPK and CBC are free and sufficient for most basic optimization needs.
They are excellent choices for smaller-scale projects and educational purposes. 

However, commercial solvers such as CPLEX and Gurobi typically offer superior performance,
especially for larger, more complex problems. These solvers have advanced features,
including enhanced quadratic and nonlinear programming support, and are optimized for
large-scale industrial applications.
'''

import pyomo
import pyomo.environ as pyo
from pyomo.environ import *
import numpy as np

def train_function_3(x):
    return [int(xi[3] - xi[1] >= 1) for xi in x]

def train_function_2(x):
    return [int(xi[3]) for xi in x]

def train_function_1(x):
    return (np.sum(x, axis=1) >= 3).astype(int)

def generate_data(N):
    inputs = np.random.randint(0, 2, size=(N, numOfInputs))
    outputs = train_function_3(inputs)
    return inputs, outputs

def feed_forward(input, weights, bias):
    return (np.dot(input, weights) + 1 * bias > 0).astype(int)

def test_model(input, weights, expected_outputs):
    test_out = feed_forward(input, weights[:-1], weights[-1])
    error_count = np.sum(test_out != expected_outputs)
    print('outputs:\t', np.array(expected_outputs))
    print('test_out:\t', test_out)
    print('error_count: ', error_count)

def test_model_all(weights_all, N):
    inputs, outputs = generate_data(N)
    test_out = feed_forward(inputs, weights_all[:-1], weights_all[-1])
    error_count = np.sum(test_out != outputs)
    print('outputs:\t', np.array(outputs))
    print('test_out:\t', test_out)
    print('error_count: ', error_count)

print('pyomo version:', pyomo.__version__)

class Perceptron(Block):
    
    def __init__(self,
                 numOfInputs: int,
                 inputVarType: str = 'Binary',
                 activationFunction: str = 'Binary Step'):
        super()
        self.numOfInputs = numOfInputs
        self.inputVarType = inputVarType
        self.activationFunction = activationFunction

    def Formulate(self,
                  inputs,
                  outputs):
        pass

    def Pprint():
        pass

percepTest = Perceptron(5)

# Params
numOfInputs = 5
Train_N = 100
Test_N = 200

#weights = [f'w{i}' for i in range(numOfInputs + 1)]
inputs, outputs = generate_data(Train_N)
weights_num = 2 * np.random.rand(numOfInputs + 1) - 1 #Random from -1 to +1

# print('inputs: ', inputs)
# print('outputs:\t', outputs)

# test_out = feed_forward(inputs, weights_num[:-1], weights_num[-1])
# print('test_out:\t', test_out)
# print('error_count: ', test_model(inputs, weights_num, outputs))

# Create a model
model = pyo.ConcreteModel()
M = numOfInputs + 1 #Max possible value
epsilon = 0.5

model.Iw = Set(initialize = list(range(numOfInputs + 1)))
model.w = Var(model.Iw, domain=Reals, bounds=(-1, 1))

model.Iy = RangeSet(0, Train_N-1)
model.y = Var(model.Iy, domain=Binary)

model.Ie = RangeSet(0, Train_N-1)
model.e = Var(model.Iy, domain=Binary)
model.constraintList = pyo.ConstraintList()

for n, (input, output) in enumerate(zip(inputs, outputs)):
    input = np.concatenate([input, [1]])

    z_expr = sum(model.w[i] * input[i] for i in model.Iw)
    # print("Added: ", z_expr <= M * model.y[n])
    # print("Added: ", z_expr >= epsilon - M * (1 - model.y[n]))

    model.constraintList.add(z_expr <= M * model.y[n])
    model.constraintList.add(z_expr >= epsilon - M * (1 - model.y[n]))

    model.constraintList.add(model.e[n] >= output - model.y[n])
    model.constraintList.add(model.e[n] >= model.y[n] - output)

def objective(model):
    return sum(model.e[i] for i in model.Ie) 
model.obj = pyo.Objective(rule=objective, sense=pyo.minimize)

solver = pyo.SolverFactory('gurobi')
# Solve the problem
result = solver.solve(model)

# model.pprint()
# model.display()

print('Status:', result.solver.status)
print('Termination Condition:', result.solver.termination_condition)

trained_weights = [pyo.value(model.w[wi]) for wi in model.Iw]
print('Optimal w:', trained_weights)
# print('Optimal e:', [pyo.value(model.e[ei]) for ei in model.Ie])
# print('Optimal y:', [pyo.value(model.y[yi]) for yi in model.Iy])
print('Optimal Objective:', pyo.value(model.obj))

print('\n## Test ##')
#test_model(inputs, trained_weights, outputs)
test_model_all(trained_weights, Test_N)

# model.x = pyo.Var(within=pyo.NonNegativeReals)
# model.y = pyo.Var(within=pyo.NonNegativeReals)
# # Define objective
# model.obj = pyo.Objective(expr=model.x + model.y, sense=pyo.minimize)
# # Define constraints
# model.con1 = pyo.Constraint(expr=model.x + 2 * model.y >= 4)
# model.con2 = pyo.Constraint(expr=model.x - model.y <= 1)
# # Select solver
# solver = pyo.SolverFactory('gurobi')
# # Solve the problem
# result = solver.solve(model)
# # Display results
# print('Status:', result.solver.status)
# print('Termination Condition:', result.solver.termination_condition)
# print('Optimal x:', pyo.value(model.x))
# print('Optimal y:', pyo.value(model.y))
# print('Optimal Objective:', pyo.value(model.obj))