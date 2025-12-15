import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import torch.nn.functional as F
import datetime

from Model_Based_Solver import ModelBasedSolver
from Model_Based_Solver import ClassicalSolver
import Model_Evaluator

def print_time_diff(start: datetime, end: datetime):
    diff = end - start
    print(f'Training time: {diff.total_seconds():.3f} seconds')

# ----- 1. Define the Neural Network -----
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_size, output_size)
        )

    def forward(self, x):
        return self.model(x)
    

# ---- 2. Set Variables -----

### Data
train_data_size = 100
test_data_size = 1000

### Network
input_size = 2
output_size = 2
model = SimpleNN(input_size, output_size)

### Solver
solver = 'kipu-miray'
bitdepth = 3

# ----- 3. Create Datasets -----

X = torch.randn(train_data_size, 2)
Y = ((X[:, 0] - 0.5*X[:, 1]) <= 0.5).long()

X_test = torch.randn(test_data_size, 2)
Y_test = ((X_test[:, 0] - 0.5*X_test[:, 1]) <= 0.5).long()

# ----- 4. Train the Model -----

if solver == 'adam':
    before_train = datetime.datetime.now()
    ClassicalSolver.train_gd(model, X, Y)
    after_train = datetime.datetime.now()
    print_time_diff(before_train, after_train)
else:
    model_based_solver = ModelBasedSolver()
    if model_based_solver.train(solver, model, X, Y, bitdepth) == 1:
        raise Exception(model_based_solver.error_message)
    model_based_solver.pprint_times()

# ----- 5. Evaluate the Model -----

print('\n== Evaluation ==')
Model_Evaluator.print_weights(model)
Model_Evaluator.evaluate_model(model, X, Y, X_test, Y_test, plotData2D=True)
Model_Evaluator.save_model(model, 'my_model.pt')