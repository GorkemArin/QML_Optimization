import torch.nn as nn

def wrap_solution(neural_network: nn.Module, solution: dict):
    # neural_network
    linears = [m for m in neural_network.modules() if isinstance(m, nn.Linear)]
    print(linears)
    pass