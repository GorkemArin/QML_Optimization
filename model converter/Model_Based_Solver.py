import datetime
import numpy as np
import torch.nn as nn
import torch.optim as optim

from QUBO_Conversion import train_optimizer_QUBO
from QCP_Conversion import train_optimizer_QCP
from Solution_Wrapper import wrap_solution

# Solvers
from Gurobi_Solver import solve_gurobi_model
from Kipu_Solver import solve_kipu ## works with python 3.13

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

    def pprint_times(self, title_on=True):
        if title_on:
            print('\n== Timings ==')
        print(f'Modeling Time: {self.modeling_time:.3f} s')
        print(f'Solving Time: {self.solving_time:.3f} s')
        print(f'Total Training Time: {self.total_training_time:.3f} s')
    
    def train(self, solver: str, nn_model: nn.Module, X, Y, bitdepth:int = 3) -> int:
        if len(X) != len(Y):
            self.status = 1
            self.error_message = 'X and Y lengths don\'t match'
            return self.status

        time_train_start = datetime.datetime.now()
        
        if solver == 'gurobi':
            gurobi_model = train_optimizer_QCP(nn_model, X, Y, bitdepth)
        else:
            qubo, loss_model = train_optimizer_QUBO(nn_model, X, Y, bitdepth)
        
        time_modeling_end = datetime.datetime.now()

        if solver == 'gurobi':
            solution = solve_gurobi_model(gurobi_model)
        elif solver.startswith('kipu'):
            solution = solve_kipu(qubo, solver)
        else:
            self.status = 1
            self.error_message = f'Invalid solver: {solver}'
            return self.status
        
        time_solving_end = datetime.datetime.now()
        
        wrap_solution(nn_model, solution, bitdepth)

        time_train_end = datetime.datetime.now()

        if solver != 'gurobi':
            self.decoded_model = loss_model.decode_sample(solution, vartype='BINARY')

        self.modeling_time = get_time_diff(time_train_start, time_modeling_end)
        self.solving_time = get_time_diff(time_modeling_end, time_solving_end)
        self.total_training_time = get_time_diff(time_train_start, time_train_end)
        self.status = 0

        return self.status


class ClassicalSolver:
    def train_gd(model, X, y):
        time_train_start = datetime.datetime.now()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        # ----- 4. Training Loop -----
        epochs = 250
        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()           # reset gradients
            y_pred = model.forward(X)              # forward pass

            loss = criterion(y_pred, y)    # compute loss

            loss.backward()                 # backpropagation
            optimizer.step()                # update weights

            losses.append(loss.item())
            if epoch < 5 or (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
                # for name, param in model.named_parameters():
                #     if param.requires_grad:
                #         print(f"{name} → mean: {param.data.mean():.4f}, grad mean: {param.grad.mean():.4f}")

        time_train_end = datetime.datetime.now()
        total_training_time = get_time_diff(time_train_start, time_train_end)
        print(f'Total Training Time: {total_training_time:.3f} s')

        return losses    
    


    

