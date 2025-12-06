import torch.nn as nn
from torch.fx import symbolic_trace
from pyqubo import Binary
import numpy as np

def layer_to_expression(layer: nn.modules.linear.Linear, layer_index: int, bit_depth: int):
    in_features = layer.in_features
    out_features = layer.out_features

    Δ = 0.1  # scaling step
    offset = (2 ** (bit_depth - 1)) * Δ  # center around 0

    # Create a dict for weight expressions
    weights = {}

    for i in range(out_features):
        for j in range(in_features):
            bits = [Binary(f"w{layer_index}_{i}_{j}_bit{k}") for k in range(bit_depth)]
            expr = sum((2**k) * bits[k] for k in range(bit_depth)) * Δ - offset
            weights[(i, j)] = expr

    W_expr = np.array([[weights[(i, j)] for j in range(in_features)]
                   for i in range(out_features)])
    
    if(layer.bias is not None):
        biases = []
        for i in range(out_features):
            bits = [Binary(f"b{layer_index}_{i}_bit{k}") for k in range(bit_depth)]
            expr = sum((2**k) * bits[k] for k in range(bit_depth)) * Δ - offset
            biases.append(expr)

        biases = np.array(biases).reshape(out_features, 1)
        W_expr = np.concatenate((W_expr, biases), axis=1)
        
    return W_expr.T, (layer.bias is not None)

def get_polynomial_of_activation_func(func: str):
    if func == 'relu':
        return lambda x: x/2 + 3*(x**2)/4 - (x**4) / 4
    elif func == 'sigmoid':
        return lambda x: 1/2 + 3*x/4 - (x**3)/4
    elif func == 'tanh':
        return lambda x: x - (x**3)/3
    
def train_optimizer_HUBO(model: nn.Module, X, y, bitdepth = 8):

    traced = symbolic_trace(model)
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

    train_count = len(X)
    train_indx = 1
    losses = []
    for x, y_t in zip(X,y):
        y_pred = np.array(x)
        for i, exp in enumerate(expressions_list):
            if(isinstance(exp, np.ndarray)):
                if(y_pred.shape[0] == exp.shape[0] - 1):
                    y_pred = np.concatenate((y_pred, [1])) # add one for bias
                y_pred = np.dot(y_pred, exp)
            elif(callable(exp) and exp.__name__ == "<lambda>"):
                y_pred = exp(y_pred)
            else:
                raise NameError(f'Invalid expression: {exp}')

        y_target = np.array(y_t)
        if(np.isscalar(y_target) or y_target.shape == ()):
            temp = np.zeros(y_pred.shape)
            temp[y_target] = 1
            y_target = temp

        mse_loss = np.sum((y_pred - y_target)**2)
        losses.append(mse_loss)
        
        print(f'train ff calculated: {train_indx}/{train_count}.')
        train_indx += 1

    # Sum all loss functions into a single HUBO
    total_loss = sum(losses)
    print('Type of single loss: ', type(losses[0]))
    # print('Total loss calculated: ', total_loss)

    print('Compiling...')
    # Compile to PyQUBO model
    model = total_loss.compile()
    print('model: ', type(model))
    # Convert to QUBO for a solver (higher-order terms will be reduced internally)
    qubo, offset = model.to_qubo()
    print('qubo, offset: ', type(qubo), type(offset))

    exit()


    

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

    return [-1]