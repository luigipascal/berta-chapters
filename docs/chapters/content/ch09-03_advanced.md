# Ch 9: Deep Learning Fundamentals - Advanced

**Track**: Practitioner | [Try code in Playground](../../playground.md) | [Back to chapter overview](../chapter-09.md)


!!! tip "Read online or run locally"
    You can read this content here on the web. To run the code interactively,
    either use the [Playground](../../playground.md) or clone the repo and open
    `chapters/chapter-09-deep-learning-fundamentals/notebooks/03_advanced.ipynb` in Jupyter.

---

# Chapter 9: Deep Learning Fundamentals
## Notebook 03 - Advanced: CNNs, RNNs & Image Classification Capstone

Welcome to the advanced notebook for Chapter 9. Having built a solid foundation with feedforward networks and training mechanics, we now turn to **specialized architectures** that revolutionised deep learning: **Convolutional Neural Networks (CNNs)** for spatial data and **Recurrent Neural Networks (RNNs)** for sequential data. We finish with a full **image-classification capstone** that ties everything together.

**What you'll learn:**

| Topic | Key Ideas |
|---|---|
| Convolutions & Pooling | Learnable filters, stride, padding, feature maps |
| CNN for Images | Conv → ReLU → Pool → Flatten → FC pipeline |
| RNNs & LSTMs | Hidden states, vanishing gradients, gating mechanisms |
| Transfer Learning | Pre-trained models, freezing, fine-tuning |
| Capstone Project | End-to-end image classification with evaluation |

**Time estimate:** ~4 hours

---

## 1. Setup

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

np.random.seed(42)
torch.manual_seed(42)
```

---

## 2. Convolutional Neural Networks: Theory

### Why Convolutions?

Fully-connected layers treat every input pixel independently and ignore **spatial structure**. Images have strong local correlations – nearby pixels form edges, textures, and shapes. A **convolution** exploits this by sliding a small **learnable filter** (kernel) across the image, computing element-wise products and summing them at every position. The same filter is reused everywhere, which keeps the parameter count low and grants **translation invariance**: a feature detected in one region is detected anywhere.

### Core Concepts

| Term | Definition |
|---|---|
| **Kernel / Filter** | A small weight matrix (e.g. 3×3) that slides over the input. Each filter learns to detect one kind of feature (edge, corner, blob…). |
| **Stride** | The step size of the sliding window. Stride 1 moves one pixel at a time; stride 2 skips every other position, halving the spatial dimensions. |
| **Padding** | Zeros (or other values) added around the input border so the output retains the same spatial size as the input (*same* padding) or shrinks (*valid* padding). |
| **Feature Map** | The 2-D output of one filter applied to the input. A convolutional layer with *k* filters produces *k* feature maps stacked into a 3-D tensor. |

### Pooling

After a convolution + activation, a **pooling layer** down-samples each feature map to reduce spatial dimensions and introduce a small amount of translation tolerance.

- **Max Pooling** – takes the maximum value in each pooling window (e.g. 2×2). This preserves the strongest activation.
- **Average Pooling** – takes the mean. Smoother but may lose sharp features.

### Canonical CNN Architecture Pattern

```
Input Image
  → [Conv → ReLU → Pool]  × N   (feature extraction)
  → Flatten
  → [FC → ReLU]  × M            (classification head)
  → Output (softmax / logits)
```

Early layers capture low-level features (edges, colours); deeper layers combine them into high-level concepts (eyes, wheels, letters).

---

## 3. Convolution Operation from Scratch

Let's implement a naive 2-D convolution in pure NumPy and apply an edge-detection kernel to a synthetic image.

```python
np.random.seed(42)


def conv2d(image, kernel, stride=1, padding=0):
    """Naive 2-D convolution with nested loops."""
    if padding > 0:
        image = np.pad(image, padding, mode='constant', constant_values=0)
    ih, iw = image.shape
    kh, kw = kernel.shape
    oh = (ih - kh) // stride + 1
    ow = (iw - kw) // stride + 1
    output = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            region = image[i * stride:i * stride + kh,
                           j * stride:j * stride + kw]
            output[i, j] = np.sum(region * kernel)
    return output


# --- Synthetic 8x8 image with a vertical stripe ---
image = np.zeros((8, 8))
image[:, 3:5] = 1.0  # bright vertical bar

# --- Edge-detection kernel (vertical Sobel-like) ---
kernel = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=float)

output = conv2d(image, kernel, stride=1, padding=0)

# --- Visualise ---
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Input (8×8)')
axes[1].imshow(kernel, cmap='RdBu', vmin=-2, vmax=2)
axes[1].set_title('Kernel (3×3 edge detector)')
for r in range(kernel.shape[0]):
    for c in range(kernel.shape[1]):
        axes[1].text(c, r, f'{kernel[r, c]:.0f}',
                     ha='center', va='center', fontsize=12, fontweight='bold')
axes[2].imshow(output, cmap='RdBu')
axes[2].set_title(f'Output ({output.shape[0]}×{output.shape[1]})')
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()

print(f'Input shape:  {image.shape}')
print(f'Kernel shape: {kernel.shape}')
print(f'Output shape: {output.shape}')
```

The output highlights the **left and right edges** of the vertical bar, exactly where the intensity changes – this is what an edge-detection filter does.

---

## 4. Building a CNN with PyTorch

We construct a small CNN for classifying 32×32 grayscale images into 10 classes. To avoid downloading real datasets we generate **synthetic image data** with `torch.randn`.

```python
torch.manual_seed(42)
np.random.seed(42)

# ---- Synthetic dataset: 1-channel 32x32 images, 10 classes ----
NUM_TRAIN = 2000
NUM_TEST = 400
NUM_CLASSES = 10
IMG_SIZE = 32


def make_synthetic_images(n, num_classes=10):
    """Each class gets a distinct random mean so the CNN can learn to separate them."""
    class_means = torch.linspace(-1.5, 1.5, num_classes)
    labels = torch.randint(0, num_classes, (n,))
    images = torch.randn(n, 1, IMG_SIZE, IMG_SIZE) * 0.3
    for idx in range(n):
        images[idx] += class_means[labels[idx]]
    return images, labels


X_train, y_train = make_synthetic_images(NUM_TRAIN, NUM_CLASSES)
X_test, y_test = make_synthetic_images(NUM_TEST, NUM_CLASSES)

train_loader = DataLoader(TensorDataset(X_train, y_train),
                          batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test),
                         batch_size=64, shuffle=False)

print(f'Training images : {X_train.shape}')
print(f'Test images     : {X_test.shape}')
```

```python
class SimpleCNN(nn.Module):
    """Two conv blocks followed by a fully-connected classifier."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # -> 16 x 32 x 32
            nn.ReLU(),
            nn.MaxPool2d(2),                                        # -> 16 x 16 x 16
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # -> 32 x 16 x 16
            nn.ReLU(),
            nn.MaxPool2d(2),                                        # -> 32 x  8 x  8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = SimpleCNN(NUM_CLASSES)
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f'\nTotal parameters: {total_params:,}')
```

```python
# ---- Training ----
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 15
history = {'train_loss': [], 'train_acc': [], 'test_acc': []}

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    epoch_loss = running_loss / total
    epoch_acc = correct / total

    # -- Evaluate on test set --
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            preds = model(imgs).argmax(1)
            test_correct += (preds == labels).sum().item()
            test_total += imgs.size(0)
    test_acc = test_correct / test_total

    history['train_loss'].append(epoch_loss)
    history['train_acc'].append(epoch_acc)
    history['test_acc'].append(test_acc)

    if epoch % 3 == 0 or epoch == 1:
        print(f'Epoch {epoch:>2d}/{EPOCHS}  '
              f'Loss: {epoch_loss:.4f}  '
              f'Train Acc: {epoch_acc:.3f}  '
              f'Test Acc: {test_acc:.3f}')

print('\nTraining complete.')
```

---

## 5. Visualising Learned CNN Filters

The first convolutional layer's filters operate directly on the input image, so we can plot each filter as a small grayscale image to see what low-level patterns the network learned to detect.

```python
filters = model.features[0].weight.data.clone()  # shape: (16, 1, 3, 3)
filters = filters.squeeze(1)  # -> (16, 3, 3)

fig, axes = plt.subplots(2, 8, figsize=(14, 4))
for idx, ax in enumerate(axes.flat):
    ax.imshow(filters[idx], cmap='RdBu',
              vmin=filters.min(), vmax=filters.max())
    ax.set_title(f'F{idx}', fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle('Learned conv1 Filters (3×3)', fontsize=13)
plt.tight_layout()
plt.show()
```

---

## 6. Recurrent Neural Networks: Theory

### Sequential Data

Many real-world signals unfold over time or in ordered sequences: stock prices, sentences, audio waveforms, sensor readings. A feedforward network sees all inputs simultaneously with no notion of **order**. A **Recurrent Neural Network (RNN)** processes one element at a time and maintains a **hidden state** that acts as the network's memory of what it has seen so far.

### RNN Update Rule

At each timestep \( t \):

\[ h_t = \tanh(W_{xh}\, x_t + W_{hh}\, h_{t-1} + b_h) \]
\[ y_t = W_{hy}\, h_t + b_y \]

- \( x_t \) – input at time \( t \)
- \( h_t \) – hidden state at time \( t \) (carries context from previous steps)
- \( y_t \) – output at time \( t \)

### The Vanishing Gradient Problem

During back-propagation through time (BPTT), gradients are multiplied by \( W_{hh} \) at every step. When \( \|W_{hh}\| < 1 \), gradients **shrink exponentially**, making it nearly impossible for the network to learn dependencies spanning many timesteps.

### LSTM to the Rescue

The **Long Short-Term Memory (LSTM)** cell introduces a **cell state** \( C_t \) and three **gates** that control information flow:

| Gate | Purpose |
|---|---|
| **Forget gate** \( f_t \) | Decides what to discard from the cell state |
| **Input gate** \( i_t \) | Decides what new information to store |
| **Output gate** \( o_t \) | Decides what part of the cell state to expose as the hidden state |

Because the cell state flows through the network with mostly **additive** updates (rather than multiplicative), gradients can propagate over hundreds of timesteps without vanishing.

---

## 7. RNN from Scratch (NumPy)

```python
np.random.seed(42)


class SimpleRNNCell:
    """Minimal RNN cell: h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b)."""

    def __init__(self, input_size, hidden_size):
        scale = 0.1
        self.W_xh = np.random.randn(hidden_size, input_size) * scale
        self.W_hh = np.random.randn(hidden_size, hidden_size) * scale
        self.b_h = np.zeros(hidden_size)
        self.hidden_size = hidden_size

    def forward_sequence(self, xs):
        """Process a sequence of inputs and return all hidden states."""
        T = len(xs)
        h = np.zeros(self.hidden_size)
        states = [h.copy()]
        for t in range(T):
            h = np.tanh(self.W_xh @ xs[t] + self.W_hh @ h + self.b_h)
            states.append(h.copy())
        return np.array(states)  # shape: (T+1, hidden_size)


# --- Run the cell on a short sine-wave sequence ---
seq_len = 20
input_size = 1
hidden_size = 8

t_vals = np.linspace(0, 4 * np.pi, seq_len)
inputs = np.sin(t_vals).reshape(seq_len, input_size)

rnn_cell = SimpleRNNCell(input_size, hidden_size)
hidden_states = rnn_cell.forward_sequence(inputs)

# --- Plot input and hidden state evolution ---
fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
axes[0].plot(t_vals, inputs[:, 0], 'o-', color='steelblue')
axes[0].set_ylabel('Input x(t)')
axes[0].set_title('RNN from Scratch – Hidden State Evolution')

for unit in range(hidden_size):
    axes[1].plot(range(seq_len + 1), hidden_states[:, unit],
                 label=f'h{unit}', alpha=0.8)
axes[1].set_xlabel('Timestep')
axes[1].set_ylabel('Hidden units')
axes[1].legend(loc='upper right', fontsize=7, ncol=4)
plt.tight_layout()
plt.show()

print(f'Sequence length : {seq_len}')
print(f'Hidden size     : {hidden_size}')
print(f'States shape    : {hidden_states.shape}')
```

Each hidden unit develops its own response curve as the sine wave is fed in step-by-step. With training (back-propagation through time) these dynamics would be tuned so that the hidden state encodes useful information for a downstream task.

---

## 8. LSTM with PyTorch – Sine Wave Prediction

We train an LSTM to predict the **next value** of a noisy sine wave given a short look-back window. This is a classic sequence-to-one regression task.

```python
torch.manual_seed(42)
np.random.seed(42)

# ---- Generate noisy sine data ----
N_POINTS = 1000
LOOK_BACK = 20

t_series = np.linspace(0, 20 * np.pi, N_POINTS)
sine_data = np.sin(t_series) + np.random.randn(N_POINTS) * 0.05


def create_sequences(data, look_back):
    xs, ys = [], []
    for i in range(len(data) - look_back):
        xs.append(data[i:i + look_back])
        ys.append(data[i + look_back])
    return (torch.tensor(np.array(xs), dtype=torch.float32).unsqueeze(-1),
            torch.tensor(np.array(ys), dtype=torch.float32))


X_seq, y_seq = create_sequences(sine_data, LOOK_BACK)
split = int(0.8 * len(X_seq))
X_tr, y_tr = X_seq[:split], y_seq[:split]
X_te, y_te = X_seq[split:], y_seq[split:]

train_dl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)
test_dl = DataLoader(TensorDataset(X_te, y_te), batch_size=64)

print(f'Sequences shape : {X_seq.shape}  (samples, timesteps, features)')
print(f'Train / Test    : {len(X_tr)} / {len(X_te)}')
```

```python
class SineLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


lstm_model = SineLSTM()
criterion_lstm = nn.MSELoss()
optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=1e-3)

LSTM_EPOCHS = 30
lstm_losses = []

for epoch in range(1, LSTM_EPOCHS + 1):
    lstm_model.train()
    batch_losses = []
    for xb, yb in train_dl:
        optimizer_lstm.zero_grad()
        pred = lstm_model(xb)
        loss = criterion_lstm(pred, yb)
        loss.backward()
        optimizer_lstm.step()
        batch_losses.append(loss.item())
    epoch_loss = np.mean(batch_losses)
    lstm_losses.append(epoch_loss)
    if epoch % 5 == 0 or epoch == 1:
        print(f'Epoch {epoch:>2d}/{LSTM_EPOCHS}  MSE: {epoch_loss:.6f}')

print('\nLSTM training complete.')
```

```python
# ---- Predict and plot ----
lstm_model.eval()
with torch.no_grad():
    y_pred_test = lstm_model(X_te).numpy()

fig, axes = plt.subplots(2, 1, figsize=(14, 6))

axes[0].plot(lstm_losses, color='crimson')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('LSTM Training Loss')

show = 150
axes[1].plot(y_te.numpy()[:show], label='Actual', color='steelblue')
axes[1].plot(y_pred_test[:show], '--', label='Predicted', color='darkorange')
axes[1].set_xlabel('Time step')
axes[1].set_ylabel('Value')
axes[1].set_title('Sine Wave – Predicted vs Actual (test set)')
axes[1].legend()

plt.tight_layout()
plt.show()

test_mse = criterion_lstm(torch.tensor(y_pred_test), y_te).item()
print(f'Test MSE: {test_mse:.6f}')
```

---

## 9. Transfer Learning Concepts

### The Idea

Training a deep CNN from scratch requires **millions** of labelled images and days of GPU time. **Transfer learning** sidesteps this by starting from a model that was already trained on a large, diverse dataset (typically ImageNet – 1.2 M images, 1000 classes). The early layers of such a model have learned general features (edges, textures, colour blobs) that are useful for virtually any visual task.

### Workflow

1. **Load a pre-trained backbone** (e.g. ResNet-18, VGG-16).
2. **Freeze the backbone** – set `requires_grad = False` for all backbone parameters so they are not updated during training.
3. **Replace the final classification head** with a new one matching your number of classes.
4. **Train only the new head** on your (possibly small) dataset.
5. *(Optional)* **Fine-tune** – unfreeze some of the later backbone layers and train end-to-end at a very low learning rate.

### Why It Works

- Low-level features are **task-agnostic**: edges and textures look the same whether you're classifying dogs, X-rays, or satellite imagery.
- By freezing early layers you prevent catastrophic forgetting of those general representations.
- You only need to learn the task-specific mapping in the final layer(s), which requires far fewer labelled examples.

### Code Pattern (no download required)

```python
import torchvision.models as models

# --- Transfer-learning pattern (illustrative – no weights downloaded) ---
#
# Step 1: Load pre-trained backbone
# backbone = models.resnet18(pretrained=True)
#
# Step 2: Freeze all layers
# for param in backbone.parameters():
#     param.requires_grad = False
#
# Step 3: Replace final FC layer for your task
# num_features = backbone.fc.in_features
# backbone.fc = nn.Linear(num_features, MY_NUM_CLASSES)
#
# Step 4: Only the new fc layer is trainable
# optimizer = optim.Adam(backbone.fc.parameters(), lr=1e-3)
#
# Step 5 (optional): Fine-tune later layers
# for param in backbone.layer4.parameters():
#     param.requires_grad = True

# Demonstrate the architecture (using default init, no download)
demo_resnet = models.resnet18(pretrained=False)
print('ResNet-18 final layer (original):')
print(f'  {demo_resnet.fc}')

# Replace for a 5-class task
num_ftrs = demo_resnet.fc.in_features
demo_resnet.fc = nn.Linear(num_ftrs, 5)
print('\nResNet-18 final layer (replaced for 5 classes):')
print(f'  {demo_resnet.fc}')

trainable = sum(p.numel() for p in demo_resnet.fc.parameters())
total = sum(p.numel() for p in demo_resnet.parameters())
print(f'\nTrainable (head only): {trainable:,} / {total:,} '
      f'({100 * trainable / total:.1f}%)')
```

---

## 10. Capstone: Image Classification Pipeline

We bring everything together: synthetic data generation, CNN construction, a full training loop with `DataLoader`, evaluation with a confusion matrix, training curves, and sample predictions.

### Data Generation

```python
torch.manual_seed(42)
np.random.seed(42)

# ---- Synthetic 3-channel (RGB) images, 3 classes ----
CAP_CLASSES = 3
CAP_IMG_SIZE = 32
CAP_TRAIN = 1500
CAP_TEST = 300
CLASS_NAMES = ['Circle-ish', 'Cross-ish', 'Stripe-ish']


def generate_capstone_data(n, img_size=32, num_classes=3):
    """Create simple synthetic RGB patterns that differ per class."""
    images = torch.zeros(n, 3, img_size, img_size)
    labels = torch.randint(0, num_classes, (n,))
    for idx in range(n):
        c = labels[idx].item()
        base = torch.randn(3, img_size, img_size) * 0.15
        if c == 0:  # circle pattern – bright centre
            yy, xx = torch.meshgrid(torch.arange(img_size),
                                    torch.arange(img_size), indexing='ij')
            dist = ((xx - img_size // 2) ** 2 +
                    (yy - img_size // 2) ** 2).float().sqrt()
            mask = (dist < img_size // 3).float()
            base[0] += mask * 1.2
        elif c == 1:  # cross pattern
            base[1, img_size // 2 - 2:img_size // 2 + 2, :] += 1.2
            base[1, :, img_size // 2 - 2:img_size // 2 + 2] += 1.2
        else:  # horizontal stripes
            for row in range(0, img_size, 4):
                base[2, row:row + 2, :] += 1.2
        images[idx] = base
    return images, labels


X_cap_train, y_cap_train = generate_capstone_data(CAP_TRAIN)
X_cap_test, y_cap_test = generate_capstone_data(CAP_TEST)

cap_train_loader = DataLoader(TensorDataset(X_cap_train, y_cap_train),
                              batch_size=64, shuffle=True)
cap_test_loader = DataLoader(TensorDataset(X_cap_test, y_cap_test),
                             batch_size=64)

# Show a few samples per class
fig, axes = plt.subplots(3, 5, figsize=(12, 7))
for cls_idx in range(CAP_CLASSES):
    idxs = (y_cap_train == cls_idx).nonzero(as_tuple=True)[0][:5]
    for col, i in enumerate(idxs):
        img = X_cap_train[i].permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        axes[cls_idx, col].imshow(img)
        axes[cls_idx, col].set_xticks([])
        axes[cls_idx, col].set_yticks([])
    axes[cls_idx, 0].set_ylabel(CLASS_NAMES[cls_idx], fontsize=11)
fig.suptitle('Sample Training Images (3 classes)', fontsize=14)
plt.tight_layout()
plt.show()

print(f'Training set : {X_cap_train.shape}')
print(f'Test set     : {X_cap_test.shape}')
```

### Model

```python
class CapstoneCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),   # -> 16 x 32 x 32
            nn.ReLU(),
            nn.MaxPool2d(2),                   # -> 16 x 16 x 16
            nn.Conv2d(16, 32, 3, padding=1),   # -> 32 x 16 x 16
            nn.ReLU(),
            nn.MaxPool2d(2),                   # -> 32 x  8 x  8
            nn.Conv2d(32, 64, 3, padding=1),   # -> 64 x  8 x  8
            nn.ReLU(),
            nn.MaxPool2d(2),                   # -> 64 x  4 x  4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


cap_model = CapstoneCNN(CAP_CLASSES)
print(cap_model)
print(f'\nParameters: {sum(p.numel() for p in cap_model.parameters()):,}')
```

### Training

```python
# ---- Training loop ----
cap_criterion = nn.CrossEntropyLoss()
cap_optimizer = optim.Adam(cap_model.parameters(), lr=1e-3)
CAP_EPOCHS = 20

cap_history = {'loss': [], 'train_acc': [], 'test_acc': []}

for epoch in range(1, CAP_EPOCHS + 1):
    cap_model.train()
    running_loss, correct, total = 0.0, 0, 0
    for xb, yb in cap_train_loader:
        cap_optimizer.zero_grad()
        logits = cap_model(xb)
        loss = cap_criterion(logits, yb)
        loss.backward()
        cap_optimizer.step()
        running_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)

    cap_history['loss'].append(running_loss / total)
    cap_history['train_acc'].append(correct / total)

    # Evaluate
    cap_model.eval()
    tc, tt = 0, 0
    with torch.no_grad():
        for xb, yb in cap_test_loader:
            tc += (cap_model(xb).argmax(1) == yb).sum().item()
            tt += yb.size(0)
    cap_history['test_acc'].append(tc / tt)

    if epoch % 4 == 0 or epoch == 1:
        print(f'Epoch {epoch:>2d}/{CAP_EPOCHS}  '
              f'Loss: {cap_history["loss"][-1]:.4f}  '
              f'Train Acc: {cap_history["train_acc"][-1]:.3f}  '
              f'Test Acc: {cap_history["test_acc"][-1]:.3f}')

print('\nCapstone training complete.')
```

### Training Curves

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(cap_history['loss'], color='crimson')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Cross-Entropy Loss')
axes[0].set_title('Training Loss')

axes[1].plot(cap_history['train_acc'], label='Train', color='steelblue')
axes[1].plot(cap_history['test_acc'], label='Test', color='darkorange')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy per Epoch')
axes[1].legend()

plt.tight_layout()
plt.show()
```

### Confusion Matrix

```python
# ---- Confusion matrix ----
cap_model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in cap_test_loader:
        all_preds.append(cap_model(xb).argmax(1))
        all_labels.append(yb)
all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

# Build confusion matrix manually (no sklearn dependency)
cm = np.zeros((CAP_CLASSES, CAP_CLASSES), dtype=int)
for true, pred in zip(all_labels, all_preds):
    cm[true, pred] += 1

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(CAP_CLASSES))
ax.set_yticks(range(CAP_CLASSES))
ax.set_xticklabels(CLASS_NAMES, fontsize=10)
ax.set_yticklabels(CLASS_NAMES, fontsize=10)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix (Test Set)')
for i in range(CAP_CLASSES):
    for j in range(CAP_CLASSES):
        color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                fontsize=14, fontweight='bold', color=color)
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()

accuracy = np.trace(cm) / cm.sum()
print(f'Overall test accuracy: {accuracy:.3f}')
for c in range(CAP_CLASSES):
    prec = cm[c, c] / cm[:, c].sum() if cm[:, c].sum() > 0 else 0
    rec = cm[c, c] / cm[c, :].sum() if cm[c, :].sum() > 0 else 0
    print(f'  {CLASS_NAMES[c]:>12s}  –  Precision: {prec:.3f}  Recall: {rec:.3f}')
```

### Sample Predictions

```python
# ---- Classify and display sample test images ----
cap_model.eval()
np.random.seed(42)
sample_indices = np.random.choice(len(X_cap_test), 12, replace=False)

fig, axes = plt.subplots(2, 6, figsize=(16, 5))
for ax, idx in zip(axes.flat, sample_indices):
    img_tensor = X_cap_test[idx].unsqueeze(0)
    with torch.no_grad():
        pred_class = cap_model(img_tensor).argmax(1).item()
    true_class = y_cap_test[idx].item()

    img = X_cap_test[idx].permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    ax.imshow(img)
    color = 'green' if pred_class == true_class else 'red'
    ax.set_title(f'P:{CLASS_NAMES[pred_class]}\nT:{CLASS_NAMES[true_class]}',
                 fontsize=8, color=color)
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle('Sample Predictions (green = correct, red = wrong)', fontsize=13)
plt.tight_layout()
plt.show()
```

---

## 11. Summary

### Key Takeaways

- **CNNs** use learnable convolutional filters and pooling to efficiently extract spatial features from images, dramatically reducing parameters compared to fully-connected approaches.
- **RNNs** process sequential data by maintaining a hidden state, but suffer from vanishing gradients over long sequences.
- **LSTMs** solve the vanishing gradient problem with gated memory cells, enabling learning of long-range dependencies.
- **Transfer learning** leverages pre-trained models to achieve strong performance on new tasks with limited labelled data.
- A complete **image-classification pipeline** combines data loading, model construction, training loops, and evaluation metrics.

### Architecture Comparison

| Property | Feedforward (MLP) | CNN | RNN / LSTM |
|---|---|---|---|
| **Input type** | Fixed-size vector | Grid / image | Variable-length sequence |
| **Parameter sharing** | None | Filters shared spatially | Weights shared across timesteps |
| **Key strength** | General-purpose | Spatial feature extraction | Temporal / sequential modelling |
| **Typical use** | Tabular data, small tasks | Image classification, object detection | Time series, NLP, speech |
| **Parameter efficiency** | Low (dense connections) | High (small kernels reused) | Moderate (shared across time) |
| **Handles variable length** | No | No (fixed spatial size) | Yes |

### What's Next

In **Chapter 10: Natural Language Processing (NLP)** we will apply these sequence modelling ideas to text data – tokenisation, word embeddings, and building language models with RNNs and Transformers.

---
*Generated by Berta AI | Created by Luigi Pascal Rondanini*
