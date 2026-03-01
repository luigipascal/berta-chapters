# Python Type System Reference

```mermaid
graph TD
    A[Python Objects] --> B[Numeric]
    A --> C[Sequences]
    A --> D[Mappings]
    A --> E[Sets]
    A --> F[Boolean]
    A --> G[None]

    B --> B1[int]
    B --> B2[float]
    B --> B3[complex]

    C --> C1[str - immutable]
    C --> C2[list - mutable]
    C --> C3[tuple - immutable]

    D --> D1[dict]

    E --> E1[set - mutable]
    E --> E2[frozenset - immutable]

    style A fill:#e1f5ff
    style B fill:#fff3e0
    style C fill:#f3e5f5
    style D fill:#e8f5e9
    style E fill:#fce4ec
    style F fill:#f5f5f5
    style G fill:#f5f5f5
```

## AI-Relevant Type Usage

| Type | AI Use Case | Example |
|------|-------------|---------|
| `int` | Epochs, batch sizes, indices | `epochs = 100` |
| `float` | Weights, losses, learning rates | `lr = 0.001` |
| `str` | Model names, file paths, prompts | `model = "gpt-4"` |
| `list` | Datasets, features, predictions | `preds = [0.9, 0.1]` |
| `dict` | Configs, metrics, JSON data | `config = {"lr": 0.01}` |
| `tuple` | Shapes, coordinates, return values | `shape = (224, 224, 3)` |
| `set` | Vocabulary, unique labels | `vocab = {"cat", "dog"}` |
| `bool` | Flags, conditions | `use_gpu = True` |
| `None` | Missing values, uninitialized | `best_model = None` |
