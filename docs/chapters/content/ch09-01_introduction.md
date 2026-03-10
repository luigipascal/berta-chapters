# Ch 9: Deep Learning Fundamentals - Introduction

**Track**: Practitioner | [Try code in Playground](../../playground.md) | [Back to chapter overview](../chapter-09.md)


!!! tip "Read online or run locally"
    You can read this content here on the web. To run the code interactively,
    either use the [Playground](../../playground.md) or clone the repo and open
    `chapters/chapter-09-deep-learning-fundamentals/notebooks/01_introduction.ipynb` in Jupyter.

---

# Chapter 9: Deep Learning Fundamentals
## Notebook 01 - Introduction: Neural Networks from Scratch

Deep learning is the engine behind modern AI — from image recognition and natural language understanding to game-playing agents and protein-structure prediction. At its core, deep learning is about stacking simple, differentiable building blocks (neurons) into networks that can learn arbitrarily complex functions from data.

In this notebook we strip away every framework and build a neural network **entirely from scratch** using only NumPy. By the end you will have a visceral understanding of every gradient that flows through a network.

**What you'll learn:**

| Topic | Section |
|---|---|
| The perceptron — a single artificial neuron | §2 |
| Activation functions (sigmoid, tanh, ReLU) | §3 |
| Multi-layer network architecture & notation | §4 |
| The forward pass | §5 |
| Loss functions (MSE, binary cross-entropy) | §6 |
| Backpropagation from scratch | §7 |
| Training on a spiral dataset | §8 |
| Gradient descent variants (batch, SGD, mini-batch) | §9 |

**Time estimate:** ~4 hours

---

## 1. Setup

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

np.random.seed(42)

plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
```

---

## 2. What is a Neural Network?

### The biological analogy

A biological neuron receives electrical signals through its **dendrites**, integrates them in the **cell body**, and fires an output signal along its **axon** when the combined input exceeds a threshold.

An *artificial* neuron mimics this in three steps:

1. **Weighted sum** — each input \( x_i \) is multiplied by a learnable weight \( w_i \) and the results are summed together with a bias \( b \):
   \[ z = \sum_{i=1}^{n} w_i x_i + b = \mathbf{w}^\top \mathbf{x} + b \]

2. **Activation function** — a non-linear function \( \sigma \) is applied to \( z \) to produce the output:
   \[ a = \sigma(z) \]

3. **Output** — the value \( a \) is passed downstream to other neurons or used directly as the network's prediction.

A single neuron with a sigmoid activation is historically called a **perceptron** (though the original perceptron used a step function). Let's build one and teach it the AND gate.

```python
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(a):
    """Derivative given the *activated* value a = sigmoid(z)."""
    return a * (1.0 - a)
```

```python
# --- AND gate dataset ---
X_and = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]])
y_and = np.array([[0], [0], [0], [1]])

np.random.seed(42)
w = np.random.randn(2, 1) * 0.5
b = np.zeros((1, 1))
lr = 1.0

losses = []
for epoch in range(1000):
    z = X_and @ w + b
    a = sigmoid(z)
    loss = np.mean((a - y_and) ** 2)
    losses.append(loss)

    # Gradient of MSE through sigmoid
    dz = 2 * (a - y_and) * sigmoid_derivative(a) / len(y_and)
    dw = X_and.T @ dz
    db = np.sum(dz, axis=0, keepdims=True)

    w -= lr * dw
    b -= lr * db

print("Learned weights:", w.flatten())
print("Learned bias:   ", b.flatten())
print("\nPredictions (rounded):")
preds = sigmoid(X_and @ w + b)
for xi, pi in zip(X_and, preds):
    print(f"  {xi} -> {pi[0]:.4f}  (label {int(pi[0] > 0.5)})")

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Perceptron learning the AND gate")
plt.show()
```

The loss decreases smoothly and the perceptron quickly learns to output values close to 0 for all inputs except \( (1, 1) \). A single neuron can solve any *linearly separable* problem — but not XOR. We'll revisit that soon.

---

## 3. Activation Functions

Without a non-linear activation, stacking layers of neurons would collapse into a single linear transformation (a composition of linear maps is linear). Activation functions inject the non-linearity that lets networks approximate complex functions.

### Sigmoid

\[ \sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \sigma'(z) = \sigma(z)(1 - \sigma(z)) \]

* Outputs in \( (0, 1) \) — useful for probabilities.
* Saturates for large \( |z| \) → vanishing gradients.

### Tanh

\[ \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}, \qquad \tanh'(z) = 1 - \tanh^2(z) \]

* Outputs in \( (-1, 1) \) — zero-centered, which helps gradient flow.
* Still saturates at extremes.

### ReLU (Rectified Linear Unit)

\[ \text{ReLU}(z) = \max(0, z), \qquad \text{ReLU}'(z) = \begin{cases} 1 & z > 0 \\ 0 & z \leq 0 \end{cases} \]

* Cheap to compute, no saturation for \( z > 0 \).
* "Dead ReLU" problem: neurons with \( z \leq 0 \) stop learning.
* Most popular default for hidden layers in modern networks.

```python
z = np.linspace(-5, 5, 300)

# Activations
sig = 1.0 / (1.0 + np.exp(-z))
th  = np.tanh(z)
rel = np.maximum(0, z)

# Derivatives
sig_d = sig * (1 - sig)
th_d  = 1 - th ** 2
rel_d = (z > 0).astype(float)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Activations ---
axes[0].plot(z, sig, label='Sigmoid', linewidth=2)
axes[0].plot(z, th,  label='Tanh',    linewidth=2)
axes[0].plot(z, rel, label='ReLU',    linewidth=2)
axes[0].axhline(0, color='grey', linewidth=0.5)
axes[0].axvline(0, color='grey', linewidth=0.5)
axes[0].set_title('Activation Functions')
axes[0].set_xlabel('z')
axes[0].set_ylabel('σ(z)')
axes[0].legend()
axes[0].set_ylim(-1.5, 5)

# --- Derivatives ---
axes[1].plot(z, sig_d, label="Sigmoid'", linewidth=2)
axes[1].plot(z, th_d,  label="Tanh'",    linewidth=2)
axes[1].plot(z, rel_d, label="ReLU'",    linewidth=2)
axes[1].axhline(0, color='grey', linewidth=0.5)
axes[1].axvline(0, color='grey', linewidth=0.5)
axes[1].set_title('Derivatives')
axes[1].set_xlabel('z')
axes[1].set_ylabel("σ'(z)")
axes[1].legend()
axes[1].set_ylim(-0.1, 1.5)

plt.tight_layout()
plt.show()
```

Notice how the sigmoid and tanh derivatives become very small when the input magnitude is large. During backpropagation these small derivatives get **multiplied** across many layers, causing gradients to shrink exponentially — the infamous *vanishing gradient* problem. ReLU's constant gradient of 1 for positive inputs is one reason it became the default choice.

---

## 4. Forward Pass

A neural network organises neurons into **layers**:

| Layer | Role |
|---|---|
| **Input layer** (layer 0) | Passes raw features into the network. No parameters. |
| **Hidden layers** (layers 1 … \( L-1 \)) | Transform representations using weights and activations. |
| **Output layer** (layer \( L \)) | Produces the final prediction. |

### Notation

For layer \( l \):

* \( n^{[l]} \) — number of neurons.
* \( \mathbf{W}^{[l]} \in \mathbb{R}^{n^{[l]} \times n^{[l-1]}} \) — weight matrix.
* \( \mathbf{b}^{[l]} \in \mathbb{R}^{n^{[l]} \times 1} \) — bias vector.
* \( \mathbf{z}^{[l]} = \mathbf{W}^{[l]}\mathbf{a}^{[l-1]} + \mathbf{b}^{[l]} \) — pre-activation.
* \( \mathbf{a}^{[l]} = g^{[l]}(\mathbf{z}^{[l]}) \) — activation (output of layer \( l \)).
* \( \mathbf{a}^{[0]} = \mathbf{X} \) — the input data.

The **forward pass** is the process of computing the output of the network given an input. Starting from \( \mathbf{a}^{[0]} = \mathbf{X} \), we iterate:

\[ \mathbf{z}^{[l]} = \mathbf{W}^{[l]}\,\mathbf{a}^{[l-1]} + \mathbf{b}^{[l]} \]
\[ \mathbf{a}^{[l]} = g(\mathbf{z}^{[l]}) \]

until we reach the output layer. Let's implement this for a 2-layer network and apply it to the XOR problem.

```python
def forward_pass(X, W1, b1, W2, b2):
    """
    2-layer network: input -> hidden (sigmoid) -> output (sigmoid).

    Parameters
    ----------
    X  : (m, n_in)  input matrix (m samples)
    W1 : (n_in, n_h)
    b1 : (1, n_h)
    W2 : (n_h, n_out)
    b2 : (1, n_out)

    Returns
    -------
    a2 : (m, n_out) output activations
    cache : dict with intermediate values for backprop
    """
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)
    cache = {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
    return a2, cache
```

```python
# --- XOR dataset ---
X_xor = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])

# Single neuron attempt (no hidden layer)
np.random.seed(42)
w_single = np.random.randn(2, 1) * 0.5
b_single = np.zeros((1, 1))

for _ in range(5000):
    a = sigmoid(X_xor @ w_single + b_single)
    dz = 2 * (a - y_xor) * sigmoid_derivative(a) / 4
    w_single -= 1.0 * (X_xor.T @ dz)
    b_single -= 1.0 * np.sum(dz, axis=0, keepdims=True)

print("Single neuron on XOR (should FAIL):")
preds_single = sigmoid(X_xor @ w_single + b_single)
for xi, pi in zip(X_xor, preds_single):
    print(f"  {xi} -> {pi[0]:.4f}")

# 2-layer network attempt
np.random.seed(42)
W1 = np.random.randn(2, 4) * 0.5
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 0.5
b2 = np.zeros((1, 1))

for _ in range(5000):
    a2, cache = forward_pass(X_xor, W1, b1, W2, b2)
    m = len(y_xor)
    dz2 = 2 * (a2 - y_xor) * sigmoid_derivative(a2) / m
    dW2 = cache['a1'].T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)
    dz1 = (dz2 @ W2.T) * sigmoid_derivative(cache['a1'])
    dW1 = X_xor.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)
    W1 -= 2.0 * dW1
    b1 -= 2.0 * db1
    W2 -= 2.0 * dW2
    b2 -= 2.0 * db2

print("\n2-layer network on XOR (should SUCCEED):")
a2_final, _ = forward_pass(X_xor, W1, b1, W2, b2)
for xi, pi in zip(X_xor, a2_final):
    print(f"  {xi} -> {pi[0]:.4f}")
```

The single neuron produces ~0.5 for every input — it cannot separate XOR. The 2-layer network with a hidden layer of 4 neurons solves it cleanly. **Hidden layers create new representations that are linearly separable.**

---

## 5. Loss Functions

A loss function measures how far the network's predictions are from the true labels. The optimizer will adjust weights to **minimize** this loss.

### Mean Squared Error (MSE)

\[ \mathcal{L}_{\text{MSE}} = \frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i - y_i)^2 \]

Simple and intuitive; commonly used for regression.

### Binary Cross-Entropy (BCE)

\[ \mathcal{L}_{\text{BCE}} = -\frac{1}{m}\sum_{i=1}^{m}\bigl[y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i)\bigr] \]

The standard loss for binary classification. It penalises confident wrong predictions much more heavily than MSE does.

```python
def mse_loss(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)

def bce_loss(y_true, y_pred, eps=1e-12):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
```

```python
# Loss landscape for a simple 1-parameter model: y_hat = sigmoid(w * x)
# True label y=1, input x=1  =>  y_hat = sigmoid(w)
w_range = np.linspace(-6, 6, 300)
y_hat = sigmoid(w_range)

mse_vals = (y_hat - 1.0) ** 2
bce_vals = -(np.log(np.clip(y_hat, 1e-12, None)))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(w_range, mse_vals, linewidth=2, color='steelblue')
axes[0].set_title('MSE Loss Landscape (y=1, x=1)')
axes[0].set_xlabel('Weight w')
axes[0].set_ylabel('Loss')

axes[1].plot(w_range, bce_vals, linewidth=2, color='coral')
axes[1].set_title('BCE Loss Landscape (y=1, x=1)')
axes[1].set_xlabel('Weight w')
axes[1].set_ylabel('Loss')

plt.tight_layout()
plt.show()
```

Notice the difference: MSE flattens out for very negative \( w \) (where \( \hat{y} \approx 0 \)), making gradients small. BCE grows unboundedly, providing a strong gradient signal even when the prediction is far off. This is why **BCE is preferred for classification**.

---

## 6. Backpropagation from Scratch

Backpropagation is just the **chain rule** applied systematically across the computational graph of the network.

### The chain rule refresher

If \( \mathcal{L} \) depends on \( \mathbf{a}^{[L]} \) which depends on \( \mathbf{z}^{[L]} \) which depends on \( \mathbf{W}^{[L]} \), then:

\[
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[L]}} = 
\frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[L]}}
\cdot \frac{\partial \mathbf{a}^{[L]}}{\partial \mathbf{z}^{[L]}}
\cdot \frac{\partial \mathbf{z}^{[L]}}{\partial \mathbf{W}^{[L]}}
\]

### Gradient flow (backward pass)

Starting from the loss, we compute:

1. \( \delta^{[L]} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[L]}} \) (output error signal).
2. \( \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = (\mathbf{a}^{[l-1]})^\top \, \delta^{[l]} \).
3. \( \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}} = \sum_{\text{samples}} \delta^{[l]} \).
4. Propagate: \( \delta^{[l-1]} = \delta^{[l]}\,(\mathbf{W}^{[l]})^\top \odot g'(\mathbf{z}^{[l-1]}) \).

We repeat steps 2–4 backwards through all layers.

Below we wrap everything into a reusable `NeuralNetworkScratch` class.

```python
class NeuralNetworkScratch:
    """Fully-connected neural network trained with backpropagation.

    Uses sigmoid activations throughout and MSE loss.
    """

    def __init__(self, layer_sizes):
        """
        Parameters
        ----------
        layer_sizes : list[int]
            e.g. [2, 4, 1] means 2 inputs, 4 hidden neurons, 1 output.
        """
        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes) - 1
        self.params = {}
        self._init_weights()

    def _init_weights(self):
        np.random.seed(42)
        for l in range(1, self.L + 1):
            n_in = self.layer_sizes[l - 1]
            n_out = self.layer_sizes[l]
            self.params[f'W{l}'] = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
            self.params[f'b{l}'] = np.zeros((1, n_out))

    def forward(self, X):
        """Compute forward pass, storing activations for backprop."""
        self.cache = {'a0': X}
        a = X
        for l in range(1, self.L + 1):
            z = a @ self.params[f'W{l}'] + self.params[f'b{l}']
            a = sigmoid(z)
            self.cache[f'z{l}'] = z
            self.cache[f'a{l}'] = a
        return a

    def backward(self, X, y):
        """Compute gradients of MSE loss via backpropagation."""
        m = X.shape[0]
        grads = {}
        a_out = self.cache[f'a{self.L}']

        delta = 2 * (a_out - y) * sigmoid_derivative(a_out) / m

        for l in range(self.L, 0, -1):
            a_prev = self.cache[f'a{l-1}']
            grads[f'dW{l}'] = a_prev.T @ delta
            grads[f'db{l}'] = np.sum(delta, axis=0, keepdims=True)
            if l > 1:
                delta = (delta @ self.params[f'W{l}'].T) * sigmoid_derivative(self.cache[f'a{l-1}'])

        self.grads = grads
        return grads

    def _update_params(self, lr):
        for l in range(1, self.L + 1):
            self.params[f'W{l}'] -= lr * self.grads[f'dW{l}']
            self.params[f'b{l}'] -= lr * self.grads[f'db{l}']

    def train(self, X, y, epochs=1000, lr=1.0, verbose=True):
        """Train with full-batch gradient descent."""
        self.losses = []
        for epoch in range(epochs):
            y_hat = self.forward(X)
            loss = mse_loss(y, y_hat)
            self.losses.append(loss)
            self.backward(X, y)
            self._update_params(lr)
            if verbose and (epoch % (epochs // 5) == 0 or epoch == epochs - 1):
                print(f"  epoch {epoch:>5d}  loss = {loss:.6f}")
        return self.losses

    def predict(self, X):
        return self.forward(X)
```

```python
print("Training on XOR with NeuralNetworkScratch([2, 4, 1]):\n")
nn = NeuralNetworkScratch([2, 4, 1])
losses = nn.train(X_xor, y_xor, epochs=5000, lr=2.0)

print("\nFinal predictions:")
preds = nn.predict(X_xor)
for xi, pi, yi in zip(X_xor, preds, y_xor):
    print(f"  {xi} -> {pi[0]:.4f}  (target {yi[0]})")

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Loss curve — XOR with backpropagation')
plt.show()
```

The network converges to near-zero loss and correctly separates XOR — entirely from scratch, using nothing but NumPy and the chain rule.

---

## 7. Training on a Spiral Dataset

XOR has only 4 points. Let's test our scratch network on something harder: a **two-class spiral** dataset where the two classes are interleaved. This is a classic non-linearly-separable problem.

```python
def make_spirals(n_points=200, noise=0.5):
    """Generate a 2-class spiral dataset."""
    np.random.seed(42)
    n = n_points // 2
    theta = np.linspace(0, 2 * np.pi, n) + np.random.randn(n) * 0.2
    r = np.linspace(0.5, 3, n)

    x1 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    x2 = np.column_stack([r * np.cos(theta + np.pi), r * np.sin(theta + np.pi)])

    X = np.vstack([x1, x2]) + np.random.randn(n_points, 2) * noise * 0.3
    y = np.hstack([np.zeros(n), np.ones(n)]).reshape(-1, 1)
    return X, y

X_spiral, y_spiral = make_spirals(300, noise=0.4)

plt.scatter(X_spiral[:, 0], X_spiral[:, 1], c=y_spiral.ravel(),
            cmap='bwr', edgecolors='k', s=25, alpha=0.8)
plt.title('Spiral Dataset')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.show()
```

```python
def plot_decision_boundary(model, X, y, title='Decision Boundary'):
    h = 0.1
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                          np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict(grid).reshape(xx.shape)

    plt.contourf(xx, yy, probs, levels=50, cmap='RdBu', alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr',
                edgecolors='k', s=20, alpha=0.9)
    plt.title(title)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
```

```python
# --- Normalize features for better training ---
mu = X_spiral.mean(axis=0)
std = X_spiral.std(axis=0)
X_norm = (X_spiral - mu) / std

# --- 1 hidden layer: [2, 16, 1] ---
print("=== 1 hidden layer [2, 16, 1] ===")
nn_1h = NeuralNetworkScratch([2, 16, 1])
losses_1h = nn_1h.train(X_norm, y_spiral, epochs=5000, lr=1.0, verbose=True)

# --- 2 hidden layers: [2, 16, 16, 1] ---
print("\n=== 2 hidden layers [2, 16, 16, 1] ===")
nn_2h = NeuralNetworkScratch([2, 16, 16, 1])
losses_2h = nn_2h.train(X_norm, y_spiral, epochs=5000, lr=1.0, verbose=True)
```

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

plt.sca(axes[0])
plot_decision_boundary(nn_1h, X_norm, y_spiral,
                       title='1 Hidden Layer [2,16,1]')

plt.sca(axes[1])
plot_decision_boundary(nn_2h, X_norm, y_spiral,
                       title='2 Hidden Layers [2,16,16,1]')

axes[2].plot(losses_1h, label='1 hidden layer', alpha=0.8)
axes[2].plot(losses_2h, label='2 hidden layers', alpha=0.8)
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('MSE Loss')
axes[2].set_title('Training Loss Comparison')
axes[2].legend()

plt.tight_layout()
plt.show()
```

The 2-hidden-layer network typically finds a more complex decision boundary and reaches a lower loss. Deeper networks can represent more intricate functions, but they also require more careful training (initialization, learning rate, more data).

---

## 8. Gradient Descent Variants

So far we have used **batch gradient descent** — we compute the gradient over the *entire* training set before each parameter update. This is simple but has limitations:

| Variant | Batch size | Pro | Con |
|---|---|---|---|
| **Batch GD** | \( m \) (all data) | Stable gradient estimate | Slow for large datasets |
| **Stochastic GD (SGD)** | 1 | Fast updates, can escape local minima | Very noisy gradient |
| **Mini-batch SGD** | \( B \) (e.g. 32) | Good trade-off: speed + stability | Requires batch-size tuning |

Mini-batch SGD is the de-facto standard in deep learning. Let's implement it.

```python
class NeuralNetworkMiniBatch(NeuralNetworkScratch):
    """Extends NeuralNetworkScratch with mini-batch training."""

    def train_minibatch(self, X, y, epochs=500, lr=0.5, batch_size=32):
        m = X.shape[0]
        self.losses = []
        for epoch in range(epochs):
            indices = np.random.permutation(m)
            X_shuf = X[indices]
            y_shuf = y[indices]
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, m, batch_size):
                end = min(start + batch_size, m)
                Xb = X_shuf[start:end]
                yb = y_shuf[start:end]
                y_hat = self.forward(Xb)
                epoch_loss += mse_loss(yb, y_hat)
                n_batches += 1
                self.backward(Xb, yb)
                self._update_params(lr)
            self.losses.append(epoch_loss / n_batches)
        return self.losses
```

```python
np.random.seed(42)

# Batch GD
nn_batch = NeuralNetworkScratch([2, 16, 1])
nn_batch._init_weights()
loss_batch = nn_batch.train(X_norm, y_spiral, epochs=500, lr=1.0, verbose=False)

# SGD (batch_size=1)
nn_sgd = NeuralNetworkMiniBatch([2, 16, 1])
nn_sgd._init_weights()
loss_sgd = nn_sgd.train_minibatch(X_norm, y_spiral, epochs=500, lr=0.5, batch_size=1)

# Mini-batch (batch_size=32)
nn_mb = NeuralNetworkMiniBatch([2, 16, 1])
nn_mb._init_weights()
loss_mb = nn_mb.train_minibatch(X_norm, y_spiral, epochs=500, lr=0.5, batch_size=32)

plt.figure(figsize=(10, 5))
plt.plot(loss_batch, label='Batch GD',       alpha=0.8, linewidth=2)
plt.plot(loss_sgd,   label='SGD (bs=1)',      alpha=0.5, linewidth=1)
plt.plot(loss_mb,    label='Mini-batch (bs=32)', alpha=0.8, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Convergence: Batch GD vs SGD vs Mini-batch SGD')
plt.legend()
plt.show()
```

**Observations:**

* **Batch GD** — smooth convergence, one gradient per epoch.
* **SGD** — noisy loss curve but can converge faster in wall-clock time on large datasets because it updates after every sample.
* **Mini-batch** — strikes the best balance: leverages vectorised matrix operations on the batch while still getting multiple updates per epoch.

---

## 9. Summary

In this notebook we built a neural network **completely from scratch** and explored every component:

| Concept | Key idea |
|---|---|
| **Perceptron** | A single neuron computes a weighted sum + activation. Solves linearly separable problems (AND) but not XOR. |
| **Activation functions** | Sigmoid, tanh, ReLU introduce non-linearity. ReLU is the modern default for hidden layers. |
| **Multi-layer architecture** | Hidden layers create new representations that are linearly separable. Notation: \( W^{[l]}, b^{[l]}, a^{[l]} \). |
| **Forward pass** | Layer-by-layer computation: \( z \to a \) through the network. |
| **Loss functions** | MSE for regression, BCE for classification. BCE provides stronger gradients for wrong predictions. |
| **Backpropagation** | The chain rule applied backwards through the network to compute gradients of every parameter. |
| **Gradient descent** | Batch, SGD, and mini-batch. Mini-batch SGD is the standard in practice. |

Armed with this foundation, you're ready to move to **Notebook 02** where we introduce modern frameworks (PyTorch) and tackle regularization techniques.

---
*Generated by Berta AI | Created by Luigi Pascal Rondanini*
