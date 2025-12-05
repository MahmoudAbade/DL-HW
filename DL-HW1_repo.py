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
# TODO: Implement preprocessing

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

    # TODO: Implement logistic_regression



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

# Cell 21
class MLP(nn.Module):
    def __init__(self):
        # TODO: implement

    def forward(self, x):
        return self.model(x)

# Cell 23
# TODO: Implement

# Cell 27
# TODO: Implement

# Cell 31
# TODO: Implement

# Cell 34
# Implement

# Cell 37
def XORData(dim):
  X = np.array(list(itertools.product([0, 1], repeat=dim)))
  Y = X.sum(axis=1)%2
  return X, Y

# Cell 39
class Linear(nn.Module):
  def __init__(self, in_features, out_features):
    # Implement here

  def forward(self, x):
    # Implement here

# Cell 41
class FFNet(nn.Module):
  def __init__(self, in_features, out_features, hidden_size):
    # Implement here

  def forward(self, x):
    # Implement here

# Cell 43
loss_func = nn.MSELoss()

def train(net, X, Y, epochs=300):
  steps = X.shape[0]
  for i in range(epochs):
      for j in range(steps):
          data_point = np.random.randint((X.shape[0]))
          x_var = torch.Tensor(X[data_point])
          y_var = torch.Tensor([Y[data_point]])
          optimizer.zero_grad()
          y_hat = net(x_var)
          loss = loss_func(y_hat, y_var)
          loss.backward()
          optimizer.step()
      if(i % 100 == 0):
          print("Epoch:{}, Loss:{}".format(i, loss.detach()))

# Cell 45
# Implement here

# Cell 48
def calc_gradients(net, x, y_hat, loss):
  # Implement here

# Cell 50
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
