import torch.nn.functional as F
import torch.nn as nn
from torch.fx import symbolic_trace
from pyqubo import Binary, Constraint
import numpy as np

from Gurobi_Solver import solve_gurobi
from Solution_Wrapper import wrap_solution

def layer_to_expression(layer: nn.modules.linear.Linear, layer_index: int, bit_depth: int):
    out_features = layer.out_features
    in_features = layer.in_features

    # scaling step
    if(bit_depth == 1):
        Δ = 2
    else:
        Δ = 1 / (2**(bit_depth-1) -1)
    offset = -1 # center around 0

    # Create a dict for weight expressions
    weights = {}

    for i in range(out_features):
        for j in range(in_features):
            bits = [Binary(f"w{layer_index}_{i}_{j}_bit{k}") for k in range(bit_depth)]
            expr = sum((2**k) * bits[k] for k in range(bit_depth)) * Δ + offset
            weights[(i, j)] = expr

    W_expr = np.array([[weights[(i, j)] for j in range(in_features)]
                   for i in range(out_features)])
    
    if(layer.bias is not None):
        biases = []
        for i in range(out_features):
            bits = [Binary(f"b{layer_index}_{i}_bit{k}") for k in range(bit_depth)]
            expr = sum((2**k) * bits[k] for k in range(bit_depth)) * Δ + offset
            biases.append(expr)

        biases = np.array(biases).reshape(out_features, 1)
        W_expr = np.concatenate((W_expr, biases), axis=1)
        
    return W_expr.T, (layer.bias is not None)

def hidden_layer(units_count: int, bit_depth: int, unique_index: int):
    # scaling step
    if(bit_depth == 1):
        Δ = 2
    else:
        Δ = 1 / (2**(bit_depth-1) -1)
    offset = -1 # center around 0

    hidden = []
    for i in range(units_count):
        bits = [Binary(f"h{unique_index}_{i}_bit{k}") for k in range(bit_depth)]
        expr = sum((2**k) * bits[k] for k in range(bit_depth)) * Δ + offset
        hidden.append(expr)
    
    return np.array(hidden)

def get_polynomial_of_activation_func(func: str):
    if func == 'relu':
        return lambda x: x/2 + 3*(x**2)/4
        # return lambda x: x/2 + 3*(x**2)/4 - (x**4) / 4 # original approx.
    elif func == 'sigmoid':
        return lambda x: 1/2 + 3*x/4 - (x**3)/4
    elif func == 'tanh':
        return lambda x: x - (x**3)/3
    
def get_equality_constraint(exp_A, exp_B):
    # diff = exp_A - exp_B
    # return [Constraint(equality, '') for equality in diff]

    return sum((a - b)**2 for (a, b) in zip(exp_A, exp_B))

def train_optimizer_QUBO(nn_model: nn.Module, X, Y, bitdepth = 3):
    traced = symbolic_trace(nn_model)
    print(traced.graph)

    executed_layers = []
    for node in traced.graph.nodes:
        if node.op == 'call_module':      # only modules, skip functions/constants
            layer = dict(traced.named_modules())[node.target]
            executed_layers.append(layer)
        elif node.op == 'call_function':
        # functional operation
            executed_layers.append(node.target.__name__)  # function object, e.g., F.relu
        elif node.op == 'call_method':
        # functional operation
            executed_layers.append(node.target)  # function object, e.g., F.relu

    print('executed layers:', executed_layers)

    bias_included = False
    expressions_list = []
    for i, layer in enumerate(executed_layers):
        layer_name = str(layer).lower()
        layer_type = type(layer)
        print(f'{i+1}. {layer_name} / {layer_type}')

        if(layer_name.startswith('linear')):
            exp, b_incld = layer_to_expression(layer, i, bitdepth)
            expressions_list.append(exp)
            if(i == 0):
                bias_included = b_incld
        elif(layer_name.startswith('relu')):
            expressions_list.append(get_polynomial_of_activation_func('relu'))
        elif(layer_name.startswith('sigmoid')):
            expressions_list.append(get_polynomial_of_activation_func('sigmoid'))
        elif(layer_name.startswith('tanh')):
            expressions_list.append(get_polynomial_of_activation_func('tanh'))

    #Y = torch.from_numpy(np.eye(Y.max() + 1)[Y]) #one-hot coding

    Y = F.one_hot(Y, num_classes=2)

    train_count = len(X)
    train_indx = 1
    losses = []
    hidden_unique_indx = 0  # to follow a unique index
    for x, y_target in zip(X,Y):
        cur_layer = x
        for i, exp in enumerate(expressions_list):
            if(isinstance(exp, np.ndarray)): # linear network
                if(cur_layer.shape[0] == exp.shape[0] - 1):
                    cur_layer = np.concatenate((cur_layer, [1])) # add one for bias
                output = np.dot(cur_layer, exp)
            elif(callable(exp) and exp.__name__ == "<lambda>"): # activation function
                output = exp(cur_layer)
            else: # invalid
                raise NameError(f'Invalid expression: {exp}')
            
            # if last expression is reached, no need for a hidden layer.
            # Equalize it to output.
            if (i == len(expressions_list) - 1): # last expression
                losses.append(get_equality_constraint(output, np.array(y_target)))
                continue

            # if medium layer, create a new hidden layer and equalize it.
            hidden = hidden_layer(output.shape[0], bitdepth, hidden_unique_indx)
            hidden_unique_indx += 1
            losses.append(get_equality_constraint(output, hidden))
            cur_layer = hidden
                    
        print(f'train ff calculated: {train_indx}/{train_count}.')
        train_indx += 1

    # Sum all loss functions into a single HUBO
    total_loss = sum(losses)
    print('total loss', total_loss)
    print('Type of single loss: ', type(losses[0]))
    # print('Total loss calculated: ', total_loss)

    print('Compiling...')
    # Compile to PyQUBO model
    loss_model = total_loss.compile()

    qubo, offset = loss_model.to_qubo()
    return qubo, loss_model

    ##### Rest is residual
    

    #print('model: ', type(loss_model))
    # Convert to QUBO for a solver (higher-order terms will be reduced internally)

    # print('qubo, offset: ', type(qubo), type(offset))
    # print('offset:', offset)
    # print('qubo model:', qubo)
    
    # print('Total Loss')
    # # print(total_loss)

    # print('QUBO:')
    # print(qubo)

    # solution = solve_gurobi(qubo)
    # wrap_solution(nn_model, solution, bitdepth)

    # dec = loss_model.decode_sample(solution, vartype='BINARY')

    # print('Broken Constraints')
    # # print(dec.constraints())
    # print(dec.constraints(only_broken=True))

    #print(solution)

    # layer0 = model[0]  # first Linear layer
    # print(layer0.weight)

    # with torch.no_grad():
    #     model.fc1.weight[:] = torch.tensor([[0.1, 0.2],
    #                                         [0.3, 0.4],
    #                                         [0.5, 0.6],
    #                                         [0.7, 0.8]])
    #     model.fc1.bias[:] = torch.tensor([0.1, 0.2, 0.3, 0.4])

    # with torch.no_grad():
    #     for name, param in model.named_parameters():
    #         param.fill_(0.5)