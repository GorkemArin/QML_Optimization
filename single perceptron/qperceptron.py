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
from pyomo.core.base.block import _BlockData
from pyomo.core.base.block import Block
import numpy as np

def train_function_3(x):
    return [int(xi[3] - xi[1] >= 1) for xi in x]

def train_function_2(x):
    return [int(xi[3]) for xi in x]

def train_function_1(x):
    return (np.sum(x, axis=1) >= 3).astype(int)

def generate_data(N, funcId=1):
    inputs = np.random.randint(0, 2, size=(N, numOfInputs))

    if(funcId == 1):
        outputs = train_function_1(inputs)
    elif(funcId == 2):
        outputs = train_function_2(inputs)
    elif(funcId == 3):
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

def test_model_all(weights_all, N, funcId):
    inputs, outputs = generate_data(N, funcId)
    test_out = feed_forward(inputs, weights_all[:-1], weights_all[-1])
    error_count = np.sum(test_out != outputs)
    print('outputs:\t', np.array(outputs))
    print('test_out:\t', test_out)
    print('error_count: ', error_count)

class Perceptron():
    def __init__(self,
                 numOfInputs: int,
                 inputVarType: str = 'Binary',
                 activationFunction: str = 'Binary Step'):
        self.numOfInputs = numOfInputs
        self.inputVarType = inputVarType
        self.activationFunction = activationFunction
        self.block = pyo.Block(concrete=True)

    def objective(self, model):
        return sum(model.e[i] for i in model.Ie) 

    def Construct(self,
                  inputs,
                  outputs,
                  epsilon=0.5):
        
        block = self.block
        Train_N = len(inputs)
        M = numOfInputs + 1 #Max possible value

        block.Iw = Set(initialize = list(range(numOfInputs + 1)))
        block.w = Var(block.Iw, domain=Reals, bounds=(-1, 1))

        block.Iy = RangeSet(0, Train_N-1)
        block.y = Var(block.Iy, domain=Binary)

        block.Ie = RangeSet(0, Train_N-1)
        block.e = Var(block.Iy, domain=Binary)
        block.constraintList = pyo.ConstraintList()

        for n, (input, output) in enumerate(zip(inputs, outputs)):
            input = np.concatenate([input, [1]])
            z_expr = sum(block.w[i] * input[i] for i in block.Iw)

            block.constraintList.add(z_expr <= M * block.y[n])
            block.constraintList.add(z_expr >= epsilon - M * (1 - block.y[n]))

            block.constraintList.add(block.e[n] >= output - block.y[n])
            block.constraintList.add(block.e[n] >= block.y[n] - output)

        block.obj = pyo.Objective(rule=self.objective, sense=pyo.minimize)

    def GetBlock(self) -> pyo.Block:
        return self.block

    def Pprint(self):
        pass

print('pyomo version:', pyomo.__version__)

# Params
numOfInputs = 5
Train_N = 100
Test_N = 200

percepBlock1 = Perceptron(numOfInputs)
percepBlock2 = Perceptron(numOfInputs)
model = pyo.ConcreteModel()

inputs1, outputs1 = generate_data(Train_N, funcId=3)
inputs2, outputs2 = generate_data(Train_N, funcId=1)

percepBlock1.Construct(inputs1, outputs1)
model.b1 = percepBlock1.GetBlock()

percepBlock2.Construct(inputs2, outputs2)
model.b2 = percepBlock2.GetBlock()

model.b1.obj.deactivate()
model.b2.obj.deactivate()
model.obj = Objective(expr=model.b1.obj.expr + model.b2.obj.expr)

solver = pyo.SolverFactory('gurobi')

# Solve the problem
result = solver.solve(model)

# model.pprint()
# model.display()

print('Status:', result.solver.status)
print('Termination Condition:', result.solver.termination_condition)

trained_weights_1 = [pyo.value(model.b1.w[wi]) for wi in model.b1.Iw]
print('Optimal w1:', trained_weights_1)
trained_weights_2 = [pyo.value(model.b2.w[wi]) for wi in model.b2.Iw]
print('Optimal w2:', trained_weights_2)
# print('Optimal e:', [pyo.value(model.e[ei]) for ei in model.Ie])
# print('Optimal y:', [pyo.value(model.y[yi]) for yi in model.Iy])
print('Optimal Objective:', pyo.value(model.b1.obj))

print('\n## Test ##')
#test_model(inputs, trained_weights, outputs)
test_model_all(trained_weights_1, Test_N, funcId=3)
test_model_all(trained_weights_2, Test_N, funcId=1)




##### REMAINING CODE #####

# print('inputs: ', inputs)
# print('outputs:\t', outputs)

# test_out = feed_forward(inputs, weights_num[:-1], weights_num[-1])
# print('test_out:\t', test_out)
# print('error_count: ', test_model(inputs, weights_num, outputs))

# Create a model
# # # model = pyo.ConcreteModel()
# # # M = numOfInputs + 1 #Max possible value
# # # epsilon = 0.5

# # # model.Iw = Set(initialize = list(range(numOfInputs + 1)))
# # # model.w = Var(model.Iw, domain=Reals, bounds=(-1, 1))

# # # model.Iy = RangeSet(0, Train_N-1)
# # # model.y = Var(model.Iy, domain=Binary)

# # # model.Ie = RangeSet(0, Train_N-1)
# # # model.e = Var(model.Iy, domain=Binary)
# # # model.constraintList = pyo.ConstraintList()

# # # for n, (input, output) in enumerate(zip(inputs, outputs)):
# # #     input = np.concatenate([input, [1]])

# # #     z_expr = sum(model.w[i] * input[i] for i in model.Iw)
# # #     # print("Added: ", z_expr <= M * model.y[n])
# # #     # print("Added: ", z_expr >= epsilon - M * (1 - model.y[n]))

# # #     model.constraintList.add(z_expr <= M * model.y[n])
# # #     model.constraintList.add(z_expr >= epsilon - M * (1 - model.y[n]))

# # #     model.constraintList.add(model.e[n] >= output - model.y[n])
# # #     model.constraintList.add(model.e[n] >= model.y[n] - output)

# # # def objective(model):
# # #     return sum(model.e[i] for i in model.Ie) 
# # # model.obj = pyo.Objective(rule=objective, sense=pyo.minimize)

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