import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

def plot(x, y, conditional):
    
    # ----- 2. Define a condition -----
    # Example: points where x + y > 0 will be blue, otherwise red
    colors = ['blue' if conditional[i] else 'red' for i in range(len(x))]

    # ----- 3. Plot -----
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, c=colors, alpha=0.7, edgecolors='k')
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.axvline(0, color='gray', linestyle='--', linewidth=1)
    plt.title('Conditional Coloring Example')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

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
    axes[1].set_title('Final Result')
    axes[1].set_xlabel('X-axis')
    axes[1].set_ylabel('Y-axis')

    plt.tight_layout()
    plt.show()

# ----- 1. Define the Neural Network -----
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_size)
        )

    def forward(self, x):
        return self.model(x)


# ----- 2. Create Dataset -----
# Dummy dataset: 1000 samples, 10 features, 2 output classes
X = torch.randn(1000, 2)

# Label: 1 if inside circle, 0 otherwise
radius = 1
# y = ((X[:, 0]**2 + X[:, 1]**2) <= radius**2).float()
y = (X[:, 0] > 0.5).long()

# print(X, y)
# plot(X[:, 0], X[:, 1], y==1)


# ----- 3. Initialize Model, Loss, Optimizer -----
input_size = 2
hidden_sizes = [4, 8]
output_size = 2

model = SimpleNN(input_size, hidden_sizes, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ----- 4. Training Loop -----
epochs = 200
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

# ----- 5. Evaluation -----
with torch.no_grad():
    preds = torch.argmax(model.forward(X), dim=1)  # shape [1000]
    accuracy = (preds == y).float().mean()
    print(f'Final Accuracy: {accuracy:.2f}')
    print(f'Final Loss: ', losses[-1])

final_out = torch.argmax(model.forward(X), dim=1)
plot_double(X[:, 0], X[:, 1], y==1, X[:, 0], X[:, 1], final_out==1)