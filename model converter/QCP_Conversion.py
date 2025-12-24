import torch.nn.functional as F
import torch.nn as nn
from torch.fx import symbolic_trace

from gurobipy import GRB, Model, Var, quicksum
import gurobipy as gp
import numpy as np

model = Model()
sensitivity = 0

def gp_dot(a: np.ndarray, b: np.ndarray):
    """
    Compute dot product(s) of NumPy arrays containing Gurobi Vars / expressions.

    Supported cases:
    - 1D · 1D  -> scalar LinExpr / QuadExpr
    - 2D · 1D  -> 1D array of expressions
    - 1D · 2D  -> 1D array of expressions
    - 2D · 2D  -> 2D array of expressions (matrix product)

    Notes:
    - Uses gp.quicksum (NumPy vectorization is NOT supported with Gurobi objects)
    - Resulting expressions are:
        * LinExpr if all products are (constant × var)
        * QuadExpr if any (var × var) appears
    """

    a = np.asarray(a, dtype=object)
    b = np.asarray(b, dtype=object)

    if a.ndim == 1 and b.ndim == 1:
        assert a.shape[0] == b.shape[0]
        return quicksum(a[i] * b[i] for i in range(a.shape[0]))

    if a.ndim == 2 and b.ndim == 1:
        assert a.shape[1] == b.shape[0]
        return np.array([
            quicksum(a[i, j] * b[j] for j in range(a.shape[1]))
            for i in range(a.shape[0])
        ], dtype=object)

    if a.ndim == 1 and b.ndim == 2:
        assert a.shape[0] == b.shape[0]
        return np.array([
            quicksum(a[i] * b[i, j] for i in range(a.shape[0]))
            for j in range(b.shape[1])
        ], dtype=object)

    if a.ndim == 2 and b.ndim == 2:
        assert a.shape[1] == b.shape[0]
        return np.array([
            [
                quicksum(a[i, k] * b[k, j] for k in range(a.shape[1]))
                for j in range(b.shape[1])
            ]
            for i in range(a.shape[0])
        ], dtype=object)

    raise ValueError("Only 1D or 2D NumPy arrays are supported.")

def layer_to_expression(layer: nn.modules.linear.Linear, layer_index: int, bit_depth: int):
    global model

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
            bits = [model.addVar(vtype=GRB.BINARY, name=f"w{layer_index}_{i}_{j}_bit{k}") for k in range(bit_depth)]
            expr = quicksum((2**k) * bits[k] for k in range(bit_depth)) * Δ + offset
            weights[(i, j)] = expr

    W_expr = np.array([[weights[(i, j)] for j in range(in_features)]
                   for i in range(out_features)])
    
    if(layer.bias is not None):
        biases = []
        for i in range(out_features):
            bits = [model.addVar(vtype=GRB.BINARY, name=f"b{layer_index}_{i}_bit{k}") for k in range(bit_depth)]
            expr = quicksum((2**k) * bits[k] for k in range(bit_depth)) * Δ + offset
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
        bits = [model.addVar(vtype=GRB.BINARY, name=f"h{unique_index}_{i}_bit{k}") for k in range(bit_depth)]
        expr = quicksum((2**k) * bits[k] for k in range(bit_depth)) * Δ + offset
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
    
def set_equality_constraint(exp_A, exp_B):
    global sensitivity
    for (a, b) in zip(exp_A, exp_B):
        model.addConstr(a <= b + sensitivity - 1e-4)
        model.addConstr(a >= b - sensitivity + 1e-4)

    # diff = exp_A - exp_B
    # return [Constraint(equality, '') for equality in diff]
    #return quicksum((a - b)**2 for (a, b) in zip(exp_A, exp_B))

slack_unique_id = 0
def get_slack_variable(bitdepth: int, max_value: float, count:int):
    global slack_unique_id
    Δ = max_value / ((2**bitdepth)-1)
    slack_vars = []
    for i in range(count):
        bits = [model.addVar(vtype=GRB.BINARY, name=f's{slack_unique_id}_bit{k}') for k in range(bitdepth)]
        slack = quicksum((2**k) * bits[k] for k in range(bitdepth)) * Δ
        slack_vars.append(slack)
        slack_unique_id += 1
    return np.array(slack_vars)

def is_array_like(x):
    return isinstance(x, (list, tuple, np.ndarray))

# A <= B --> A - B + s = 0 --> +(A - B + s)^2
def set_LEQ_inequality_constraint(exp_A, exp_B):
    global sensitivity
    for (a, b) in zip(exp_A, exp_B):
        if \
        type(a) != gp.LinExpr and type(a) != gp.QuadExpr and \
        type(b) != gp.LinExpr and type(b) != gp.QuadExpr:
            continue
        model.addConstr(a <= b)
    
    # if(is_array_like(exp_A)):
    #     count = len(exp_A)
    # elif(is_array_like(exp_B)):
    #     count = len(exp_B)
    # else:
    #     count = 1
    # slack = get_slack_variable(bitdepth, max_value=7, count=count)
    # return quicksum((a - b + s)**2 for (a, b, s) in zip(exp_A, exp_B, slack))

# A => B
def set_GEQ_inequality_constraint(exp_A, exp_B):
    return set_LEQ_inequality_constraint(exp_B, exp_A)


# y = ReLU(x) = max(0,x)

# y >= x
# y >= 0
# y <= x + M(1-aux)
# y <= M*aux
# aux in {0,1}
aux_unique_id = 0
def set_ReLU_constraints(x, y):
    global aux_unique_id
    M = 10
    
    aux = []
    for i in range(len(y)):
        aux.append(model.addVar(vtype=GRB.BINARY, name=f'aux{aux_unique_id}') * M)
        aux_unique_id += 1
    aux = np.array(aux)
    
    set_GEQ_inequality_constraint(y, x)
    set_GEQ_inequality_constraint(y, np.zeros(y.shape))
    set_LEQ_inequality_constraint(y, x + M * (1-aux))
    set_LEQ_inequality_constraint(y, aux)

def get_objective_loss(y_pred, y_expect):
    y_diff = y_pred - y_expect
    mul = gp_dot(y_diff, y_diff)

    if is_array_like(mul):
        return quicksum(mul)
    else:
        return mul

def get_expressions_list(nn_model: nn.Module, bitdepth):
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

    print('Executed layers:', executed_layers)

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
            expressions_list.append('relu')
            #expressions_list.append(get_polynomial_of_activation_func('relu'))
        elif(layer_name.startswith('sigmoid')):
            expressions_list.append(get_polynomial_of_activation_func('sigmoid'))
        elif(layer_name.startswith('tanh')):
            expressions_list.append(get_polynomial_of_activation_func('tanh'))
    
    return expressions_list

def set_sensitivity(bitdepth):
    global sensitivity

    if(bitdepth == 1):
        sensitivity = 2
    else:
        sensitivity = 1 / (2**(bitdepth-1) -1)
    

def train_optimizer_QCP(nn_model: nn.Module, X, Y, bitdepth = 3):
    global model
    model = Model()

    expressions_list = get_expressions_list(nn_model, bitdepth)

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
                output = gp_dot(cur_layer, exp)
            elif(isinstance(exp, str) and exp == 'relu'):
                if (i == len(expressions_list) - 1): # last expression
                    hidden = np.array(y_target)
                else:
                    hidden = hidden_layer(cur_layer.shape[0], bitdepth, hidden_unique_indx)
                    hidden_unique_indx += 1
                set_ReLU_constraints(cur_layer, hidden)
                cur_layer = hidden
                continue
            elif(callable(exp) and exp.__name__ == "<lambda>"): # activation function
                output = exp(cur_layer)
            else: # invalid
                raise NameError(f'Invalid expression: {exp}')
            
            # if last expression is reached, no need for a hidden layer.
            # Equalize it to output.
            if (i == len(expressions_list) - 1): # last expression
                losses.append(get_objective_loss(output, np.array(y_target)))
                continue

            # if medium layer, create a new hidden layer and equalize it.
            hidden = hidden_layer(output.shape[0], bitdepth, hidden_unique_indx)
            hidden_unique_indx += 1
            set_equality_constraint(output, hidden)
            cur_layer = hidden
                    
        print(f'Modeling in progress: {train_indx}/{train_count}.')
        train_indx += 1

    model.setObjective(quicksum(losses))
    model.write("model.lp")
    return model

    # Sum all loss functions into a single HUBO
    # total_loss = quicksum(losses)
    # # print('total loss', total_loss)
    # # print('Type of single loss: ', type(losses[0]))
    # # print('Total loss calculated: ', total_loss)

    # print('Compiling the model...')
    # # Compile to PyQUBO model
    # loss_model = total_loss.compile()

    # qubo, offset = loss_model.to_qubo()
    # print(qubo)

    # return qubo, loss_model

    # ##### Rest is residual


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