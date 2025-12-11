import datetime
import numpy as np
import torch.nn as nn

from QUBO_Conversion import train_optimizer_QUBO
from Solution_Wrapper import wrap_solution

# Solvers
from Gurobi_Solver import solve_gurobi

def get_time_diff(start: datetime, end: datetime) -> float:
    diff = end - start
    return diff.total_seconds()

class ModelBasedSolver:
    def __init__(self):
        self.modeling_time = 0.0
        self.solving_time = 0.0
        self.total_training_time = 0.0

        self.status = 0
        self.error_message = 'none'

        self.decoded_model = None

    def pprint_times(self):
        print(f'Modeling Time: {self.modeling_time:.3f} s')
        print(f'Solving Time: {self.solving_time:.3f} s')
        print(f'Total Training Time: {self.total_training_time:.3f} s')
    
    def train(self, solver: str, nn_model: nn.Module, X, Y, bitdepth:int = 3) -> int:
        if len(X) != len(Y):
            self.status = 1
            self.error_message = 'X and Y lengths don\'t match'
            return self.status

        time_train_start = datetime.datetime.now()

        qubo, loss_model = train_optimizer_QUBO(nn_model, X, Y, bitdepth)
        time_modeling_end = datetime.datetime.now()

        if solver == 'gurobi':
            solution = solve_gurobi(qubo)
        else:
            self.status = 1
            self.error_message = f'Invalid solver: {solver}'
            return self.status
        time_solving_end = datetime.datetime.now()
        
        wrap_solution(nn_model, solution, bitdepth)
        time_train_end = datetime.datetime.now()

        self.decoded_model = loss_model.decode_sample(solution, vartype='BINARY')
        self.modeling_time = get_time_diff(time_train_start, time_modeling_end)
        self.solving_time = get_time_diff(time_modeling_end, time_solving_end)
        self.total_training_time = get_time_diff(time_train_start, time_train_end)
        self.status = 0

        return self.status
            

    

