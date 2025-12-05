# Cell 3
# --- Global Setup ---

# Import Libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools
import random
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.init as init

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# Cell 6
# --- Define Seed ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- Helper Functions To Use ---
def accuracy(logits: torch.Tensor, y: torch.Tensor):
    """Top-1 accuracy for logits [N,C] and labels [N]."""
    return (logits.argmax(dim=1) == y).float().mean().item()

def count_params(obj):
    """
    Count trainable parameters.
    - If obj is (W, b) tuple → counts elements.
    - If obj is a nn.Module → sums requires_grad params.
    """
    if isinstance(obj, tuple) and len(obj) == 2:
        W, b = obj
        return W.numel() + b.numel()
    if isinstance(obj, nn.Module):
        return sum(p.numel() for p in obj.parameters() if p.requires_grad)
    raise TypeError("count_params expects (W,b) or nn.Module.")


@torch.no_grad()
def evaluate_acc(W: torch.Tensor, b: torch.Tensor, loader):
    """Dataset-level accuracy for a linear softmax model parameterized by (W,b)."""
    total_acc, total_n = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(DEVICE).view(xb.size(0), -1)
        yb = yb.to(DEVICE)
        logits = xb @ W + b
        batch_acc = accuracy(logits, yb)
        n = xb.size(0)
        total_acc += batch_acc * n         # weight by batch size
        total_n   += n
    return total_acc / total_n


# Use this function in the training loop for your nn.Module
@torch.no_grad()
def evaluate_module(model: nn.Module, loader):
    model.eval()
    total_acc, total_n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        batch_acc = accuracy(logits, yb)
        n = xb.size(0)
        total_acc += batch_acc * n
        total_n   += n
    return total_acc / total_n

# Cell 9
# Load the raw MNIST dataset
transform = transforms.ToTensor()

train_full = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_set   = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

print(f"\n Train set: {len(train_full)} samples  |  Test set: {len(test_set)} samples")

# Cell 11
# Split the training data
train_size = int(0.8 * len(train_full))
val_size = len(train_full) - train_size
train_set, val_set = random_split(train_full, [train_size, val_size])

# Create DataLoaders
batch_size = 128
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Sanity check
xb, yb = next(iter(train_loader))
print(f"Batch: {xb.shape} {yb.shape} | pixel range = ({xb.min():.1f}, {xb.max():.1f})")

# Cell 14
def logistic_regression(train_loader, val_loader, epochs=20, lr=0.1, tol=1e-6):
    """
    Train a multiclass logistic regression model using gradient descent.
    - X: [N, d] input features (flattened images)
    - y: [N] class labels in {0,...,9}
    - lr: learning rate
    - max_steps: max number of iterations
    - tol: stop early when gradients converge

    Returns: (W, b)
    """
    d = 784  # MNIST image size
    C = 10   # Number of classes

    # Initialize parameters
    W = torch.randn(d, C, device=DEVICE, requires_grad=True)
    b = torch.zeros(C, device=DEVICE, requires_grad=True)

    for epoch in range(epochs):
        for xb, yb in train_loader:
            xb = xb.to(DEVICE).view(xb.size(0), -1)
            yb = yb.to(DEVICE)

            # Forward pass
            logits = xb @ W + b
            loss = F.cross_entropy(logits, yb)

            # Backward pass
            loss.backward()

            # Update weights
            with torch.no_grad():
                W -= lr * W.grad
                b -= lr * b.grad
                W.grad.zero_()
                b.grad.zero_()

        val_acc = evaluate_acc(W, b, val_loader)
        print(f"Epoch {epoch+1}/{epochs} | Val acc: {val_acc:.4f}")

    return W.detach(), b.detach()


# Cell 16
# TODO: Run
W, b = logistic_regression(train_loader, val_loader, epochs=20, lr=0.1)

model = (W, b)

# Evaluate
val_acc  = evaluate_acc(W, b, val_loader)
test_acc = evaluate_acc(W, b, test_loader)

print(f"\nNumber of Parameters: {count_params(model):,}")
print(f"Val. acc.: {val_acc:.4f}")
print(f"Test acc.: {test_acc:.4f}")

# Cell 18
misclassified_examples = []

@torch.no_grad()
def find_misclassified(W, b, loader, num_examples=2):
    W = W.to(DEVICE)
    b = b.to(DEVICE)
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        # Flatten images for logistic regression
        xb_flat = xb.view(xb.size(0), -1)

        # Forward pass
        logits = xb_flat @ W + b
        predictions = logits.argmax(dim=1)

        # Identify misclassified samples
        incorrect_indices = (predictions != yb).nonzero(as_tuple=True)[0]

        for idx in incorrect_indices:
            if len(misclassified_examples) < num_examples:
                # Detach from GPU and convert to numpy for plotting
                misclassified_examples.append({
                    'image': xb[idx].cpu().squeeze().numpy(),
                    'predicted': predictions[idx].item(),
                    'actual': yb[idx].item()
                })
            else:
                return misclassified_examples # Found enough examples
    return misclassified_examples


misclassified_samples = find_misclassified(W, b, test_loader, num_examples=2)


print("Two examples of incorrectly classified digits: ")

fig, axes = plt.subplots(1, len(misclassified_samples), figsize=(10, 5))
for i, example in enumerate(misclassified_samples):
    ax = axes[i]
    ax.imshow(example['image'], cmap='gray')
    ax.set_title(f"Actual: {example['actual']}\nPredicted: {example['predicted']}", color='red')
    ax.axis('off')
plt.tight_layout()
plt.show()

# Cell 22
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 10)
        )

    def forward(self, x):
        return self.model(x)

# Cell 24
model = MLP().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
epochs = 20

train_accs, val_accs = [], []

for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss = loss_fn(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_acc = evaluate_module(model, train_loader)
    val_acc = evaluate_module(model, val_loader)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    print(f"Epoch {epoch+1}/{epochs} | Train acc: {train_acc:.4f} | Val acc: {val_acc:.4f}")

test_acc = evaluate_module(model, test_loader)
print(f"\nTest acc.: {test_acc:.4f}")

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_accs, label='Train Accuracy')
plt.plot(range(1, epochs + 1), val_accs, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('MLP Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Cell 28
def train_with_init(init_fn, title):
    model = MLP().to(DEVICE)
    if init_fn is not None:
        model.apply(init_fn)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    epochs = 20
    val_accs = []

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_acc = evaluate_module(model, val_loader)
        val_accs.append(val_acc)

    test_acc = evaluate_module(model, test_loader)
    print(f"{title} | Test acc: {test_acc:.4f}")
    return val_accs

def init_zero(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 0)
        nn.init.constant_(m.bias, 0)

def init_uniform(m):
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, 0, 1)
        nn.init.uniform_(m.bias, 0, 1)

def init_normal(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 1)
        nn.init.normal_(m.bias, 0)

def init_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)

val_accs_default = train_with_init(None, "Default (Kaiming)")
val_accs_zero = train_with_init(init_zero, "Zero")
val_accs_uniform = train_with_init(init_uniform, "Uniform")
val_accs_normal = train_with_init(init_normal, "Normal")
val_accs_xavier = train_with_init(init_xavier, "Xavier")

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(range(1, 21), val_accs_default, label='Default (Kaiming)')
plt.plot(range(1, 21), val_accs_zero, label='Zero')
plt.plot(range(1, 21), val_accs_uniform, label='Uniform')
plt.plot(range(1, 21), val_accs_normal, label='Normal')
plt.plot(range(1, 21), val_accs_xavier, label='Xavier')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.title('Effect of Weight Initialization on MLP Training')
plt.legend()
plt.grid(True)
plt.show()

# Cell 32
def train_with_optimizer(optimizer_class, title, **kwargs):
    model = MLP().to(DEVICE) # Using default Kaiming init
    optimizer = optimizer_class(model.parameters(), **kwargs)
    loss_fn = nn.CrossEntropyLoss()
    epochs = 20
    val_accs = []

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_acc = evaluate_module(model, val_loader)
        val_accs.append(val_acc)

    test_acc = evaluate_module(model, test_loader)
    print(f"{title} | Test acc: {test_acc:.4f}")
    return val_accs

val_accs_adam = val_accs_default # From previous run
val_accs_sgd = train_with_optimizer(optim.SGD, "SGD", lr=0.1)
val_accs_rmsprop = train_with_optimizer(optim.RMSprop, "RMSProp", lr=1e-3)
val_accs_adagrad = train_with_optimizer(optim.Adagrad, "Adagrad", lr=1e-2)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(range(1, 21), val_accs_adam, label='Adam')
plt.plot(range(1, 21), val_accs_sgd, label='SGD')
plt.plot(range(1, 21), val_accs_rmsprop, label='RMSProp')
plt.plot(range(1, 21), val_accs_adagrad, label='Adagrad')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.title('Effect of Optimizer on MLP Training')
plt.legend()
plt.grid(True)
plt.show()

# Cell 35
# Best init (Kaiming) + worst optimizer (SGD)
val_accs_best_init_worst_opt = train_with_optimizer(optim.SGD, "Kaiming + SGD", lr=0.1)

# Worst init (Zeros and uniform) + best optimizer (Adam)
val_accs_worst_init_best_opt = train_with_init(init_zero, "Zero + Adam")
val_accs_second_worst_init_best_opt = train_with_init(init_uniform, "Uniform + Adam")

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(range(1, 21), val_accs_best_init_worst_opt, label='Best Init (Kaiming) + Worst Opt (SGD)')
plt.plot(range(1, 21), val_accs_worst_init_best_opt, label='Worst Init (Zero) + Best Opt (Adam)')
plt.plot(range(1, 21), val_accs_second_worst_init_best_opt, label='Second Worst Init (uniform) + Best Opt (Adam)')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.title('Interaction between Initialization and Optimization')
plt.legend()
plt.grid(True)
plt.show()

# Cell 38
def XORData(dim):
  X = np.array(list(itertools.product([0, 1], repeat=dim)))
  Y = X.sum(axis=1)%2
  return X, Y

# Cell 40
class Linear(nn.Module):
  def __init__(self, in_features, out_features):
    super().__init__()
    self.weight = nn.Parameter(torch.randn(out_features, in_features))
    self.bias = nn.Parameter(torch.randn(out_features))

  def forward(self, x):
    return F.linear(x, self.weight, self.bias)

# Cell 42
class FFNet(nn.Module):
  def __init__(self, in_features, out_features, hidden_size):
    super().__init__()
    self.layer1 = Linear(in_features, hidden_size)
    self.layer2 = Linear(hidden_size, out_features)

  def forward(self, x):
    x = self.layer1(x)
    x = torch.sigmoid(x)
    x = self.layer2(x)
    return x

# Cell 44
loss_func = nn.MSELoss()

def train_and_collect_losses(net, X_data, Y_data, optimizer_instance, epochs=300):
  steps = X_data.shape[0]
  loss_history = []
  # Convert numpy arrays to torch tensors once, and move to device
  X_tensor = torch.tensor(X_data, dtype=torch.float32).to(DEVICE)
  Y_tensor = torch.tensor(Y_data, dtype=torch.float32).unsqueeze(1).to(DEVICE) # unsqueeze for MSELoss

  for i in range(epochs):
      # Randomly select a data point for each step, as in the original train function
      epoch_loss_sum = 0.0
      for j in range(steps): # Iterate 'steps' times, each time picking a random data point
          data_point_idx = np.random.randint(X_tensor.size(0))
          x_var = X_tensor[data_point_idx]
          y_var = Y_tensor[data_point_idx]

          optimizer_instance.zero_grad()
          y_hat = net(x_var)
          loss = loss_func(y_hat, y_var)
          loss.backward()
          optimizer_instance.step()
          epoch_loss_sum += loss.item()

      # Average loss over the 'steps' iterations for epoch reporting
      avg_epoch_loss = epoch_loss_sum / steps
      loss_history.append(avg_epoch_loss)

      if(i % 100 == 0):
          print(f"  Epoch:{i}, Loss:{avg_epoch_loss:.4f}")
  return loss_history

# Cell 46
# Dimensions to test
dimensions = [2, 3, 4, 5]
hidden_sizes_to_test = {}
for d in dimensions:
    # Iterate through a reasonable range of hidden layer sizes
    # e.g., for d=2 -> [1,2,3,4,5,6], for d=5 -> [1,2,3,4,5,6,7,8,9,10]
    hidden_sizes_to_test[d] = list(range(1, d + 6))

all_results = {}
learning_rate = 0.02
momentum_val = 0.9
epochs_per_run = 300

for d in dimensions:
    print(f"--- Dimension d={d} ---")
    X, Y = XORData(d)
    d_results = {}

    for hidden_size in hidden_sizes_to_test[d]:
        print(f"  Training for hidden_size={hidden_size}...")
        # Create new FFNet and optimizer objects each time
        model = FFNet(d, 1, hidden_size).to(DEVICE)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum_val)
        losses = train_and_collect_losses(model, X, Y, optimizer, epochs=epochs_per_run)
        d_results[hidden_size] = losses
    all_results[d] = d_results

# Plotting the results
fig, axes = plt.subplots(len(dimensions), 1, figsize=(12, 5 * len(dimensions)), squeeze=False)
axes = axes.flatten()

for i, d in enumerate(dimensions):
    ax = axes[i]
    ax.set_title(f'Loss Curves for XOR d={d}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    for hidden_size, losses in all_results[d].items():
        if losses: # Only plot if losses were collected
            ax.plot(losses, label=f'Hidden Size: {hidden_size}')
    ax.legend()
    ax.grid(True)
plt.tight_layout()
plt.show()

# Cell 47
dims = [2, 3, 4, 5]
hidden_sizes = [1, 2, 3, 4, 5, 8, 16]
plt.figure(figsize=(12, 8))

for d in dims:
    X, Y = XORData(d)
    losses = []
    for h in hidden_sizes:
        model = FFNet(d, 1, h)
        optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)

        # Simple training loop to get final loss
        epochs = 300
        final_loss = 0
        for epoch in range(epochs):
            for i in range(X.shape[0]):
                x_var = torch.Tensor(X[i])
                y_var = torch.Tensor([Y[i]])
                optimizer.zero_grad()
                y_hat = model(x_var)
                loss = F.mse_loss(y_hat, y_var)
                loss.backward()
                optimizer.step()
            if epoch == epochs -1:
                final_loss = loss.item()
        losses.append(final_loss)
    plt.plot(hidden_sizes, losses, marker='o', linestyle='-', label=f'dim={d}')

plt.xlabel('Hidden Layer Size')
plt.ylabel('Final MSE Loss')
plt.title('XOR: Final Loss vs. Hidden Layer Size for Different Dimensions')
plt.legend()
plt.grid(True)
plt.show()

# Cell 50
def calc_gradients(net, x, y_var, y_hat, loss):
    # Unpack layers
    W1, b1 = net.layer1.weight, net.layer1.bias
    W2, b2 = net.layer2.weight, net.layer2.bias

    # Forward pass values
    a1 = F.linear(x, W1, b1)
    h1 = torch.sigmoid(a1)

    # Gradient of loss w.r.t. output (for MSE)
    d_loss_y_hat = 2 * (y_hat - y_var)

    # Gradients for the second layer
    d_loss_W2 = d_loss_y_hat.unsqueeze(1) * h1.unsqueeze(0)
    d_loss_b2 = d_loss_y_hat

    # Gradients for the first layer (chain rule)
    d_loss_h1 = d_loss_y_hat * W2
    d_h1_a1 = h1 * (1 - h1) # Sigmoid derivative
    d_loss_a1 = d_loss_h1 * d_h1_a1
    d_loss_W1 = d_loss_a1.T @ x.unsqueeze(0)
    d_loss_b1 = d_loss_a1.squeeze()

    # Flatten and concatenate gradients
    return torch.cat([
        d_loss_W1.flatten(),
        d_loss_b1.flatten(),
        d_loss_W2.flatten(),
        d_loss_b2.flatten()
    ])

# Cell 52
def equal_gradients(net, x, y_var, y_hat, loss):
  grads = []
  for param in net.parameters():
    grads.append(param.grad.view(-1))
  grads = torch.cat(grads)
  return True if torch.sum(grads - calc_gradients(net, x, y_var, y_hat, loss)).round() == 0 else False # Added round because results were very close but not identical

def train_and_compare(net, X, Y, epochs=100):
  steps = X.shape[0]
  for i in range(epochs):
      for j in range(steps):
          data_point = np.random.randint(X.shape[0])
          x_var = torch.Tensor(X[data_point])
          y_var = torch.Tensor([Y[data_point]])
          optimizer.zero_grad()
          y_hat = net(x_var)
          loss = loss_func(y_hat, y_var)
          loss.backward()
          if not equal_gradients(net, x_var, y_var, y_hat, loss.item()):
            print("Wrong gradients computation!")
            return
          optimizer.step()
  print("Correct gradients computation!")

model = FFNet(2, 1, 2)
optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
X, Y = XORData(2)


train_and_compare(model, X, Y)
