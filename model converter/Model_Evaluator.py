import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def plot_double(x1, y1, training_cond, x2, y2, output_cond):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

    # ----- First plot -----
    colors1 = ['blue' if training_cond[i] else 'red' for i in range(len(x1))]
    axes[0].scatter(x1, y1, c=colors1, alpha=0.7, edgecolors='k')
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=1)
    axes[0].axvline(0, color='gray', linestyle='--', linewidth=1)
    axes[0].set_title('Training Data')
    axes[0].set_xlabel('X-axis')
    axes[0].set_ylabel('Y-axis')

    # ----- Second plot -----
    colors2 = ['blue' if output_cond[i] else 'red' for i in range(len(x2))]
    axes[1].scatter(x2, y2, c=colors2, alpha=0.7, edgecolors='k')
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=1)
    axes[1].axvline(0, color='gray', linestyle='--', linewidth=1)
    axes[1].set_title('Test Data')
    axes[1].set_xlabel('X-axis')
    axes[1].set_ylabel('Y-axis')

    plt.tight_layout()
    plt.show(block=True)

def print_weights(nn_model: nn.Module):
    print('Weights after training:')
    linears = [m for m in nn_model.modules() if isinstance(m, nn.Linear)]
    for i in range(1):
        weights_2d = linears[i].weight.detach().cpu().numpy()
        print(f'layer {i}:\n', weights_2d)

def evaluate_model(nn_model: nn.Module, train_X, train_Y, test_X, test_Y, plotData2D: bool):
    # ----- 5. Evaluation -----
    # with torch.no_grad():
    #print(f'Final Loss: ', losses[-1])

    preds = torch.argmax(nn_model.forward(train_X), dim=1)
    accuracy = (preds == train_Y).float().mean()
    print(f'Training Accuracy: {accuracy*100:.2f}%')
        
    test_out = torch.argmax(nn_model.forward(test_X), dim=1)
    test_accuracy = (test_out == test_Y).float().mean()
    print(f'Test Accuracy: {test_accuracy*100:.2f}%')

    if plotData2D:
        plot_double(train_X[:, 0], train_X[:, 1], train_Y==1, test_X[:, 0], test_X[:, 1], test_out==1)

def save_model(nn_model:nn.Module, path:str):
    torch.save(nn_model.state_dict(), path)