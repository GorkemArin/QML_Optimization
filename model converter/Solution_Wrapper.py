import torch
import torch.nn as nn
import numpy as np

def wrap_solution(neural_network: nn.Module, solution: dict, bit_depth: int):
    # neural_network
    linears = [m for m in neural_network.modules() if isinstance(m, nn.Linear)]
    print(linears)

    for layer_index, linear_layer in enumerate(linears):
        out_features = linear_layer.out_features
        in_features = linear_layer.in_features

        # e.g. for bit_depth n
        # n = 1 -> Δ = 2    | -1, 1
        # n = 2 -> Δ = 1    | -1, 0, 1, 2
        # n = 3 -> Δ = 0.33 | -1, -0.66, -0.33, 0, 0.33, 0.66, 1, 1.33
        # n = ...

        # scaling step
        if(bit_depth == 1):
            Δ = 2
        else:
            Δ = 1 / (2**(bit_depth-1) - 1)
        offset = -1 # center around 0, starting at -1

        # Create a dict for weight expressions
        weights_matrix = np.zeros((out_features, in_features), dtype=float)

        for i in range(out_features):
            for j in range(in_features):
                weight_value = offset
                for k in range(bit_depth):
                    bit = f"w{layer_index}_{i}_{j}_bit{k}"
                    if(bit in solution and solution[bit] == 1.0):
                        weight_value += (2**k) * Δ
                weights_matrix[i, j] = weight_value 

        with torch.no_grad():
            linear_layer.weight.copy_(torch.tensor(weights_matrix))

        if(linear_layer.bias is not None):
            bias_vector = np.zeros(out_features, dtype=float)

            for i in range(out_features):
                weight_value = offset
                for k in range(bit_depth):
                    bit = f"b{layer_index}_{i}_bit{k}"
                    if(bit in solution and solution[bit] == 1.0):
                        weight_value += (2**k) * Δ
                bias_vector[i] = weight_value 

            with torch.no_grad():
                linear_layer.bias.copy_(torch.tensor(bias_vector))
