'''
Open-source solvers like GLPK and CBC are free and sufficient for most basic optimization needs.
They are excellent choices for smaller-scale projects and educational purposes. 

However, commercial solvers such as CPLEX and Gurobi typically offer superior performance,
especially for larger, more complex problems. These solvers have advanced features,
including enhanced quadratic and nonlinear programming support, and are optimized for
large-scale industrial applications.
'''

import pyomo.environ as pyo
from pyomo.environ import *
import numpy as np
import math

class Perceptron():
    def __init__(self,
                 numOfInputs: int,
                 weightLimits = (-1, 1),
                 inputVarType: str = 'Binary',
                 activationFunction: str = 'Binary Step'):
        self.numOfInputs = numOfInputs
        self.weightLimits = weightLimits
        self.inputVarType = inputVarType
        self.activationFunction = activationFunction
        self.block = pyo.Block(concrete=True)
        self.w_expr = []

    def objective(self, model):
        return sum(model.e[i] for i in model.Ie) 
    
    def __GetCountAndOffsetOfBinaries(self, sensitivity):
        bottomLimit, topLimit = self.weightLimits
        assert topLimit > bottomLimit, 'top limit cannot be smaller than or equal to bottom limit'
        assert sensitivity > 0, 'sensitivity must be a positive number'
        count = math.ceil(math.log2((topLimit - bottomLimit) / sensitivity + 1))
        return count, bottomLimit #bottom lim. is offset
    
    def Construct(self,
                  inputs,
                  outputs,
                  epsilon=0.5,
                  binSensitivity=1/8):
        
        check_sens = math.log2(binSensitivity)
        assert int(check_sens) == check_sens, \
            "binSensitivity must be a degree of 2. (neg. or positive) e.g. 1/2, 1/8"

        numerical_in =  type(inputs) != Var
        numerical_out = type(outputs) != Var
        #TO-DO: Branch the flow of the construct accordingly.

        block = self.block
        Train_N = len(inputs)
        M = self.numOfInputs + 1 # Max possible value or weight array length
        binN, binOffset = self.__GetCountAndOffsetOfBinaries(binSensitivity)

        block.Iw = Set(initialize = list(range(M)))
        block.Iwbin = Set(initialize = list(range(binN)))
        block.w = Var(block.Iw, block.Iwbin, domain=Binary, bounds=(-1, 1))

        block.Iy = RangeSet(0, Train_N-1)
        block.y = Var(block.Iy, domain=Binary)

        block.Ie = RangeSet(0, Train_N-1)
        block.e = Var(block.Iy, domain=Binary)
        block.constraintList = pyo.ConstraintList()
        
        self.w_expr = []
        for iw in block.Iw:
            self.w_expr.append(sum(block.w[iw, ibin] * (2**ibin) * binSensitivity for ibin in block.Iwbin) + binOffset)
            block.constraintList.add(self.w_expr[-1] <= self.weightLimits[1])
            block.constraintList.add(self.weightLimits[0] <= self.w_expr[-1])

        # for i, wexp in enumerate(self.w_expr):
        #     print(f'w_expr{i}: \n', wexp) 

        for n, (input, output) in enumerate(zip(inputs, outputs)):
            input = np.concatenate([input, [1]])
            
            z_expr = sum(self.w_expr[i] * input[i] for i in block.Iw)

            block.constraintList.add(z_expr <= M * block.y[n])
            block.constraintList.add(z_expr >= epsilon - M * (1 - block.y[n]))

            block.constraintList.add(block.e[n] >= output - block.y[n])
            block.constraintList.add(block.e[n] >= block.y[n] - output)

        block.obj = pyo.Objective(rule=self.objective, sense=pyo.minimize)

    def GetTrainedWeights(self):
        return [pyo.value(w) for w in self.w_expr]

    def GetBlock(self) -> pyo.Block:
        return self.block

    def Pprint(self):
        pass

