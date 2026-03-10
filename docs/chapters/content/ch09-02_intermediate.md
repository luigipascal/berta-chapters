# Ch 9: Deep Learning Fundamentals - Intermediate

**Track**: Practitioner | [Try code in Playground](../../playground.md) | [Back to chapter overview](../chapter-09.md)


!!! tip "Read online or run locally"
    You can read this content here on the web. To run the code interactively,
    either use the [Playground](../../playground.md) or clone the repo and open
    `chapters/chapter-09-deep-learning-fundamentals/notebooks/02_intermediate.ipynb` in Jupyter.

---

# Chapter 9: Deep Learning Fundamentals
## Notebook 02 - Intermediate: PyTorch & Regularization

In the previous notebook we built a neural network **from scratch** using only NumPy. That exercise was invaluable for understanding what happens under the hood, but in practice nobody writes their own matrix-multiply-and-backprop code. Instead we use **frameworks** that handle automatic differentiation, GPU acceleration, and the many small bookkeeping details that make deep learning work at scale.

In this notebook we move to **PyTorch** — the most popular research-oriented deep learning library — and learn how to train models the "right" way. We will also explore a suite of **regularization** techniques that help our networks generalize to unseen data.

**What you'll learn:**

| Topic | Key Concept |
|---|---|
| **PyTorch Tensors** | The fundamental N-dimensional array type |
| **Autograd** | Automatic differentiation — gradients for free |
| **`nn.Module`** | The standard way to define trainable models |
| **DataLoader** | Efficient, batched iteration over datasets |
| **Training Loops** | The forward → loss → backward → step cycle |
| **Dropout** | Randomly zeroing activations to reduce co-adaptation |
| **L2 / Weight Decay** | Penalizing large weights via the optimizer |
| **Early Stopping** | Halting training when validation loss stalls |
| **Batch Normalization** | Normalizing layer inputs for faster convergence |
| **Learning Rate Scheduling** | Adapting the learning rate during training |

**Time estimate:** ~4 hours

**Prerequisites:** Notebook 01 (backpropagation from scratch), basic Python & NumPy.

---

## 1. Setup

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

np.random.seed(42)
torch.manual_seed(42)

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
```

---

## 2. PyTorch Tensors

A **tensor** is PyTorch's equivalent of a NumPy `ndarray` — an N-dimensional, homogeneously-typed array of numbers. The critical difference is that tensors can live on a **GPU** and can track the operations performed on them so that gradients can be computed automatically.

### Creating Tensors

There are many factory functions: `torch.tensor`, `torch.zeros`, `torch.ones`, `torch.randn`, `torch.arange`, `torch.linspace`, and more. You can also convert a NumPy array with `torch.from_numpy`.

```python
# --- Creating tensors ---
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.zeros(2, 3)
c = torch.ones(2, 3)
d = torch.randn(3, 4)  # standard normal

print("a =", a)
print("b (zeros) =\n", b)
print("c (ones)  =\n", c)
print("d (randn) =\n", d)
print("\nShape of d:", d.shape)
print("Dtype of d:", d.dtype)
```

```python
# --- Arithmetic ---
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

print("x + y  =", x + y)
print("x * y  =", x * y)          # element-wise
print("x @ y  =", x @ y)          # dot product
print("x.sum()=", x.sum())
print("x.mean()=", x.mean())
```

```python
# --- Reshaping ---
m = torch.arange(12, dtype=torch.float32)
print("Original:", m.shape, m)

m_2d = m.reshape(3, 4)
print("\nReshaped to (3,4):\n", m_2d)

m_t = m_2d.T  # transpose
print("\nTranspose (4,3):\n", m_t)

m_flat = m_2d.flatten()
print("\nFlattened:", m_flat.shape, m_flat)
```

```python
# --- Device selection ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

t = torch.randn(3, 3, device=device)
print(t)
print("Tensor device:", t.device)
```

---

## 3. Autograd — Automatic Differentiation

In Notebook 01 we derived every gradient by hand. PyTorch's **autograd** engine does that work for us. When a tensor has `requires_grad=True`, PyTorch builds a computation graph on the fly. Calling `.backward()` on a scalar loss then propagates gradients all the way back through that graph, populating the `.grad` attribute of every leaf tensor.

### How It Works (Conceptually)

1. **Forward pass** — operations are recorded in a directed acyclic graph (DAG).
2. **`.backward()`** — the DAG is traversed in reverse; each node computes its local Jacobian and chains it with the upstream gradient (chain rule).
3. **`.grad`** — the accumulated gradient is stored on each leaf tensor.

Let's verify that autograd matches a manual calculation for a simple function:

\[ L = (w \cdot x - y)^{2}, \qquad \frac{\partial L}{\partial w} = 2(wx - y)\,x \]

```python
torch.manual_seed(42)

w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)
y = torch.tensor(1.0)

# Forward pass
loss = (w * x - y) ** 2
print(f"loss = {loss.item():.4f}")

# Backward pass
loss.backward()
print(f"Autograd  dL/dw = {w.grad.item():.4f}")

# Manual gradient: 2 * (w*x - y) * x
manual_grad = 2 * (w.item() * x.item() - y.item()) * x.item()
print(f"Manual    dL/dw = {manual_grad:.4f}")
```

```python
# Autograd with vectors
torch.manual_seed(42)

W = torch.randn(3, requires_grad=True)
X = torch.tensor([1.0, 2.0, 3.0])
target = torch.tensor(0.5)

pred = (W * X).sum()
loss = (pred - target) ** 2
loss.backward()

print("W.grad =", W.grad)

# Manual: dL/dW_i = 2*(pred - target) * X_i
manual = 2 * (pred.item() - target.item()) * X
print("Manual  =", manual)
```

> **Key point:** `loss.backward()` *accumulates* gradients into `.grad`. If you call it twice without zeroing, the gradients add up. That is why every training loop calls `optimizer.zero_grad()` before the backward pass.

---

## 4. Building Models with `nn.Module`

PyTorch models are classes that inherit from `torch.nn.Module`. The pattern is:

1. **`__init__`** — define the layers (these register their parameters automatically).
2. **`forward`** — define how data flows through the layers.

You never call `forward` directly; instead you call the model as a function (`model(x)`) which invokes `forward` under the hood after running any registered hooks.

```python
class SimpleNet(nn.Module):
    """Two-layer network: input → hidden (ReLU) → output (logit)."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(-1)  # (batch,1) → (batch,)


torch.manual_seed(42)
model_demo = SimpleNet(input_dim=2, hidden_dim=16)
print(model_demo)
print(f"\nTotal parameters: {sum(p.numel() for p in model_demo.parameters())}")
```

Each `nn.Linear(in, out)` stores a weight matrix of shape `(out, in)` and a bias vector of length `out`. All parameters are automatically registered and returned by `model.parameters()`, which is exactly what the optimizer needs.

---

## 5. The Standard PyTorch Training Loop

Training a PyTorch model follows a simple, repeatable recipe:

```
for epoch in range(num_epochs):
    optimizer.zero_grad()      # 1. Clear old gradients
    logits = model(X)          # 2. Forward pass
    loss   = criterion(logits, y)  # 3. Compute loss
    loss.backward()            # 4. Backward pass (compute gradients)
    optimizer.step()           # 5. Update parameters
```

We will demonstrate on the classic **make_moons** dataset (a non-linearly separable two-class problem).

```python
# --- Prepare data ---
np.random.seed(42)
X_np, y_np = make_moons(n_samples=1000, noise=0.2, random_state=42)

X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X_np, y_np, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32)
X_test  = torch.tensor(X_test_np,  dtype=torch.float32)
y_test  = torch.tensor(y_test_np,  dtype=torch.float32)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(X_train_np[:, 0], X_train_np[:, 1], c=y_train_np, cmap='bwr', alpha=0.5, s=10)
ax.set_title('make_moons — training set')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
plt.show()
```

```python
# --- Training loop (full-batch) ---
torch.manual_seed(42)

model = SimpleNet(input_dim=2, hidden_dim=32)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 300
losses = []

for epoch in range(num_epochs):
    optimizer.zero_grad()
    logits = model(X_train)
    loss = criterion(logits, y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1:3d}/{num_epochs}  loss={loss.item():.4f}")

plt.figure(figsize=(7, 3))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('BCEWithLogitsLoss')
plt.title('Training Loss Curve')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

```python
def plot_decision_boundary(model, X, y, title='Decision Boundary'):
    """Plot the decision boundary of a binary classifier."""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200),
    )
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(grid)).numpy().reshape(xx.shape)
    model.train()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.contourf(xx, yy, preds, levels=50, cmap='RdBu', alpha=0.7)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k', s=15, linewidths=0.3)
    ax.set_title(title)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    plt.tight_layout()
    plt.show()


plot_decision_boundary(model, X_test_np, y_test_np, title='SimpleNet Decision Boundary (test set)')
```

```python
def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        preds = (torch.sigmoid(model(X)) >= 0.5).float()
    model.train()
    return (preds == y).float().mean().item()


print(f"Train accuracy: {accuracy(model, X_train, y_train):.4f}")
print(f"Test  accuracy: {accuracy(model, X_test, y_test):.4f}")
```

---

## 6. DataLoader and Batching

The full-batch loop above sends **all** training samples through the network at once. For large datasets this is impractical (it won't fit in memory) and sub-optimal (mini-batch SGD has a regularizing effect and converges faster in wall-clock time).

PyTorch's `torch.utils.data` module gives us two abstractions:

| Class | Purpose |
|---|---|
| `TensorDataset` | Wraps tensors into a map-style dataset (`dataset[i]` returns a tuple). |
| `DataLoader` | Iterates over a dataset in shuffled mini-batches, optionally in parallel. |

Using these is the idiomatic way to feed data to a PyTorch model.

```python
train_ds = TensorDataset(X_train, y_train)
test_ds  = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

# Iterate one batch to see the shapes
for X_batch, y_batch in train_loader:
    print(f"Batch X shape: {X_batch.shape}, y shape: {y_batch.shape}")
    break
```

```python
# --- Mini-batch training loop ---
torch.manual_seed(42)

model_batched = SimpleNet(input_dim=2, hidden_dim=32)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model_batched.parameters(), lr=0.01)

num_epochs = 100
epoch_losses = []

for epoch in range(num_epochs):
    running_loss = 0.0
    num_batches = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        logits = model_batched(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        num_batches += 1
    epoch_losses.append(running_loss / num_batches)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d}/{num_epochs}  avg_loss={epoch_losses[-1]:.4f}")

print(f"\nTest accuracy (batched): {accuracy(model_batched, X_test, y_test):.4f}")

plt.figure(figsize=(7, 3))
plt.plot(epoch_losses)
plt.xlabel('Epoch')
plt.ylabel('Avg Batch Loss')
plt.title('Mini-Batch Training Loss')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 7. Regularization: Dropout

**Dropout** (Srivastava et al., 2014) is one of the most widely used regularization techniques. During **training**, each activation is independently set to zero with probability \( p \). During **evaluation**, all activations are kept but scaled by \( (1 - p) \) (PyTorch handles this automatically with *inverted dropout*).

Why does it help?

* Prevents neurons from **co-adapting** (relying on specific other neurons).
* Approximates an **ensemble** of exponentially many sub-networks.
* Acts as a form of **noise injection** during training.

We'll build a wider network prone to overfitting and compare it with and without dropout.

```python
class WideNet(nn.Module):
    """Wider network that can overfit make_moons easily."""

    def __init__(self, input_dim, hidden_dim, dropout_rate=0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.drop = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        x = torch.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x.squeeze(-1)


def train_model(model, train_loader, X_test, y_test, num_epochs=200, lr=0.01):
    """Train and return per-epoch train/test accuracy."""
    criterion = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    history = {'train_acc': [], 'test_acc': []}

    for epoch in range(num_epochs):
        model.train()
        for X_b, y_b in train_loader:
            opt.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            opt.step()
        history['train_acc'].append(accuracy(model, X_train, y_train))
        history['test_acc'].append(accuracy(model, X_test, y_test))

    return history
```

```python
torch.manual_seed(42)
model_no_drop = WideNet(2, 128, dropout_rate=0.0)
hist_no_drop = train_model(model_no_drop, train_loader, X_test, y_test, num_epochs=200)

torch.manual_seed(42)
model_with_drop = WideNet(2, 128, dropout_rate=0.4)
hist_with_drop = train_model(model_with_drop, train_loader, X_test, y_test, num_epochs=200)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(hist_no_drop['train_acc'], label='Train')
axes[0].plot(hist_no_drop['test_acc'],  label='Test')
axes[0].set_title('No Dropout')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(hist_with_drop['train_acc'], label='Train')
axes[1].plot(hist_with_drop['test_acc'],  label='Test')
axes[1].set_title('With Dropout (p=0.4)')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"No Dropout  — Train: {hist_no_drop['train_acc'][-1]:.4f}, Test: {hist_no_drop['test_acc'][-1]:.4f}")
print(f"Dropout 0.4 — Train: {hist_with_drop['train_acc'][-1]:.4f}, Test: {hist_with_drop['test_acc'][-1]:.4f}")
```

> **Observation:** Dropout typically reduces the gap between train and test accuracy, which is the hallmark of reduced overfitting.

---

## 8. Regularization: L2 (Weight Decay)

**L2 regularization** adds a penalty proportional to the squared magnitude of the weights to the loss:

\[ \tilde{L} = L + \frac{\lambda}{2} \|\mathbf{w}\|^{2} \]

In PyTorch, you don't add the penalty to the loss function yourself. Instead, you pass `weight_decay=λ` to the optimizer, which subtracts `λ·w` from the gradient at each step (equivalent for SGD; Adam uses a slightly different but analogous formulation called *decoupled weight decay*).

We'll train the same wide network with and without weight decay.

```python
def train_model_wd(model, train_loader, X_test, y_test,
                   num_epochs=200, lr=0.01, weight_decay=0.0):
    criterion = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {'train_acc': [], 'test_acc': [], 'loss': []}

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        n = 0
        for X_b, y_b in train_loader:
            opt.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n += 1
        history['loss'].append(epoch_loss / n)
        history['train_acc'].append(accuracy(model, X_train, y_train))
        history['test_acc'].append(accuracy(model, X_test, y_test))

    return history


torch.manual_seed(42)
model_wd0 = WideNet(2, 128, dropout_rate=0.0)
hist_wd0 = train_model_wd(model_wd0, train_loader, X_test, y_test,
                          num_epochs=200, weight_decay=0.0)

torch.manual_seed(42)
model_wd = WideNet(2, 128, dropout_rate=0.0)
hist_wd = train_model_wd(model_wd, train_loader, X_test, y_test,
                         num_epochs=200, weight_decay=0.01)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(hist_wd0['train_acc'], label='Train')
axes[0].plot(hist_wd0['test_acc'],  label='Test')
axes[0].set_title('No Weight Decay')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(hist_wd['train_acc'], label='Train')
axes[1].plot(hist_wd['test_acc'],  label='Test')
axes[1].set_title('Weight Decay = 0.01')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"No WD   — Train: {hist_wd0['train_acc'][-1]:.4f}, Test: {hist_wd0['test_acc'][-1]:.4f}")
print(f"WD=0.01 — Train: {hist_wd['train_acc'][-1]:.4f}, Test: {hist_wd['test_acc'][-1]:.4f}")
```

Weight decay keeps the weights **small**, which smooths the decision boundary and can improve generalization. It is one of the cheapest regularizers — just a single keyword argument in the optimizer constructor.

---

## 9. Regularization: Early Stopping

Even with dropout and weight decay, training for too many epochs can overfit. **Early stopping** monitors the validation (or test) loss and halts training when it stops improving for a specified number of epochs (the *patience*).

### Algorithm

1. After each epoch, compute the validation loss.
2. If the validation loss improves, save the model weights and reset a counter.
3. If it does **not** improve for `patience` consecutive epochs, stop training and restore the best weights.

```python
import copy


def train_with_early_stopping(model, train_loader, X_val, y_val,
                              num_epochs=500, lr=0.01, patience=15):
    criterion = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_weights = None
    epochs_no_improve = 0

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
        # --- Train ---
        model.train()
        running = 0.0
        n = 0
        for X_b, y_b in train_loader:
            opt.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            opt.step()
            running += loss.item()
            n += 1
        history['train_loss'].append(running / n)

        # --- Validate ---
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()
        history['val_loss'].append(val_loss)

        # --- Early stopping check ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} "
                  f"(best val loss: {best_val_loss:.4f})")
            break

    model.load_state_dict(best_weights)
    return history


torch.manual_seed(42)
model_es = WideNet(2, 128, dropout_rate=0.0)
hist_es = train_with_early_stopping(
    model_es, train_loader, X_test, y_test,
    num_epochs=500, lr=0.01, patience=15
)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(hist_es['train_loss'], label='Train Loss')
ax.plot(hist_es['val_loss'],   label='Val Loss')
ax.axvline(len(hist_es['val_loss']) - 15, color='gray', ls='--', label='Best epoch (approx)')
ax.set_xlabel('Epoch')
ax.set_ylabel('BCEWithLogitsLoss')
ax.set_title('Early Stopping — Train vs Validation Loss')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Stopped after {len(hist_es['train_loss'])} epochs")
print(f"Test accuracy: {accuracy(model_es, X_test, y_test):.4f}")
```

> **Tip:** In a real project you should use a proper **validation** split (separate from the test set) for early stopping to avoid information leakage. We use the test set here only for simplicity.

---

## 10. Batch Normalization

**Batch Normalization** (Ioffe & Szegedy, 2015) normalizes the inputs to each layer so that they have zero mean and unit variance *within each mini-batch*. It then applies a learnable affine transform (\( \gamma \) and \( \beta \)) so the network can still represent arbitrary distributions.

Benefits:

* **Faster convergence** — gradients flow more smoothly.
* Allows **higher learning rates**.
* Has a mild **regularization** effect due to mini-batch noise.

For fully-connected layers we use `nn.BatchNorm1d(num_features)`.

```python
class BNNet(nn.Module):
    """Same architecture as WideNet but with BatchNorm."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x.squeeze(-1)


def train_and_record_loss(model, train_loader, num_epochs=100, lr=0.01):
    criterion = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(num_epochs):
        model.train()
        running = 0.0
        n = 0
        for X_b, y_b in train_loader:
            opt.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            opt.step()
            running += loss.item()
            n += 1
        losses.append(running / n)
    return losses


torch.manual_seed(42)
model_no_bn = WideNet(2, 128, dropout_rate=0.0)
losses_no_bn = train_and_record_loss(model_no_bn, train_loader, num_epochs=100)

torch.manual_seed(42)
model_bn = BNNet(2, 128)
losses_bn = train_and_record_loss(model_bn, train_loader, num_epochs=100)

plt.figure(figsize=(7, 4))
plt.plot(losses_no_bn, label='Without BatchNorm')
plt.plot(losses_bn,    label='With BatchNorm')
plt.xlabel('Epoch')
plt.ylabel('Avg Batch Loss')
plt.title('Convergence: BatchNorm vs No BatchNorm')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Without BN — final loss: {losses_no_bn[-1]:.4f}, "
      f"test acc: {accuracy(model_no_bn, X_test, y_test):.4f}")
print(f"With BN    — final loss: {losses_bn[-1]:.4f}, "
      f"test acc: {accuracy(model_bn, X_test, y_test):.4f}")
```

> **Note:** The benefits of BatchNorm become much more pronounced in deeper networks and with larger datasets. On this toy problem the difference may be subtle, but the pattern (faster early convergence) is consistent.

---

## 11. Learning Rate Scheduling

A fixed learning rate is often sub-optimal. **Learning rate schedulers** adjust the rate during training according to a pre-defined policy.

| Scheduler | Behaviour |
|---|---|
| `StepLR` | Multiply LR by `gamma` every `step_size` epochs |
| `ExponentialLR` | Multiply LR by `gamma` every epoch |
| `ReduceLROnPlateau` | Reduce LR when a metric stops improving |
| `CosineAnnealingLR` | Decay LR following a cosine curve |

```python
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau


def train_with_scheduler(model, train_loader, X_val, y_val,
                         scheduler_type='step', num_epochs=150, lr=0.05):
    criterion = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    if scheduler_type == 'step':
        scheduler = StepLR(opt, step_size=30, gamma=0.5)
    else:
        scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10)

    history = {'loss': [], 'val_loss': [], 'lr': []}

    for epoch in range(num_epochs):
        model.train()
        running = 0.0
        n = 0
        for X_b, y_b in train_loader:
            opt.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            opt.step()
            running += loss.item()
            n += 1
        avg_loss = running / n
        history['loss'].append(avg_loss)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()
        history['val_loss'].append(val_loss)

        history['lr'].append(opt.param_groups[0]['lr'])

        if scheduler_type == 'step':
            scheduler.step()
        else:
            scheduler.step(val_loss)

    return history


# --- StepLR ---
torch.manual_seed(42)
model_step = WideNet(2, 128, dropout_rate=0.0)
hist_step = train_with_scheduler(model_step, train_loader, X_test, y_test,
                                 scheduler_type='step', num_epochs=150, lr=0.05)

# --- ReduceLROnPlateau ---
torch.manual_seed(42)
model_plateau = WideNet(2, 128, dropout_rate=0.0)
hist_plateau = train_with_scheduler(model_plateau, train_loader, X_test, y_test,
                                    scheduler_type='plateau', num_epochs=150, lr=0.05)
```

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(hist_step['loss'], label='Train')
axes[0, 0].plot(hist_step['val_loss'], label='Val')
axes[0, 0].set_title('StepLR — Loss')
axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(hist_step['lr'])
axes[0, 1].set_title('StepLR — Learning Rate')
axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('LR')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(hist_plateau['loss'], label='Train')
axes[1, 0].plot(hist_plateau['val_loss'], label='Val')
axes[1, 0].set_title('ReduceLROnPlateau — Loss')
axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('Loss')
axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(hist_plateau['lr'])
axes[1, 1].set_title('ReduceLROnPlateau — Learning Rate')
axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('LR')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"StepLR   — Test acc: {accuracy(model_step, X_test, y_test):.4f}")
print(f"Plateau  — Test acc: {accuracy(model_plateau, X_test, y_test):.4f}")
```

> **Takeaway:** Learning rate scheduling is a low-cost way to squeeze extra performance out of your model. `ReduceLROnPlateau` is especially convenient because it requires no manual tuning of the step schedule — it reacts to the actual training dynamics.

---

## 12. Summary & Quick Reference

### Key Takeaways

1. **PyTorch tensors** are GPU-accelerated N-dimensional arrays that support automatic differentiation.
2. **Autograd** records operations and computes gradients via `loss.backward()`.
3. Models are defined as `nn.Module` subclasses with `__init__` (layers) and `forward` (computation).
4. The **training loop** is: `zero_grad → forward → loss → backward → step`.
5. `DataLoader` handles **batching** and **shuffling** — always use it.
6. **Dropout** prevents co-adaptation by randomly zeroing activations.
7. **Weight decay** (L2 regularization) keeps weights small via the optimizer.
8. **Early stopping** halts training when validation loss stops improving.
9. **Batch normalization** normalizes layer inputs and accelerates training.
10. **LR scheduling** adjusts the learning rate during training for better convergence.

### Quick Reference Table

| Technique | PyTorch API | When to Use |
|---|---|---|
| Dropout | `nn.Dropout(p=0.5)` | Large networks, limited data |
| L2 / Weight Decay | `optim.Adam(..., weight_decay=1e-2)` | Always a reasonable default |
| Early Stopping | Manual (track val loss) | Always — prevents wasted compute |
| Batch Normalization | `nn.BatchNorm1d(features)` | Deep networks, unstable training |
| StepLR | `lr_scheduler.StepLR(opt, step_size, gamma)` | Known training length |
| ReduceLROnPlateau | `lr_scheduler.ReduceLROnPlateau(opt, ...)` | Adaptive — monitors a metric |

### What's Next

In the next notebook we will tackle **convolutional neural networks (CNNs)** for image data, **recurrent neural networks (RNNs)** for sequences, and explore **transfer learning** with pre-trained models.

---
*Generated by Berta AI | Created by Luigi Pascal Rondanini*
