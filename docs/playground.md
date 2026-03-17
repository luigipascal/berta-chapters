---
hide:
  - toc
---

# :computer: Python Playground

Practice Python directly in your browser. No installation required. Errors are explained in plain English.

<div class="retro-banner">
:zap: Powered by <a href="https://pyodide.org">Pyodide</a> &mdash; real Python running in your browser via WebAssembly
</div>

<div class="playground-container">
  <div class="playground-header">
    BERTA PLAYGROUND &mdash; Python 3.11
    <select class="exercise-select" id="exerciseSelect" onchange="loadExercise()" style="float:right">
      <option value="">-- Load an exercise --</option>
      <optgroup label="Chapter 1: Python Fundamentals">
        <option value="ch1_hello">Hello World</option>
        <option value="ch1_variables">Variables &amp; Types</option>
        <option value="ch1_lists">Lists &amp; Loops</option>
        <option value="ch1_functions">Functions</option>
        <option value="ch1_dict">Dictionaries</option>
      </optgroup>
      <optgroup label="Chapter 2: Data Structures">
        <option value="ch2_search">Binary Search</option>
        <option value="ch2_stack">Stack</option>
        <option value="ch2_sorting">Sorting</option>
      </optgroup>
      <optgroup label="Chapter 3: Linear Algebra">
        <option value="ch3_vectors">Vector Operations</option>
        <option value="ch3_dotproduct">Dot Product</option>
      </optgroup>
      <optgroup label="Chapter 4: Probability">
        <option value="ch4_coin">Coin Flip Simulation</option>
        <option value="ch4_bayes">Bayes Theorem</option>
      </optgroup>
      <optgroup label="Chapter 6: Machine Learning">
        <option value="ch6_split">Train/Test Split</option>
        <option value="ch6_accuracy">Accuracy Score</option>
      </optgroup>
      <optgroup label="Chapter 7: Supervised Learning">
        <option value="ch7_linreg">Linear Regression</option>
        <option value="ch7_sigmoid">Sigmoid &amp; Logistic</option>
        <option value="ch7_gini">Gini Impurity</option>
      </optgroup>
      <optgroup label="Chapter 8: Unsupervised Learning">
        <option value="ch8_kmeans">K-Means Clustering</option>
        <option value="ch8_pca">PCA from Scratch</option>
        <option value="ch8_silhouette">Silhouette Score</option>
      </optgroup>
      <optgroup label="Chapter 9: Deep Learning">
        <option value="ch9_neuron">Single Neuron</option>
        <option value="ch9_activations">Activation Functions</option>
        <option value="ch9_backprop">Backpropagation</option>
      </optgroup>
      <optgroup label="Chapter 10: Natural Language Processing">
        <option value="ch10_tokenize">Simple Tokenization</option>
        <option value="ch10_bow">Bag of Words</option>
        <option value="ch10_cosine">Cosine Similarity</option>
      </optgroup>
    </select>
  </div>
  <div id="editorWrapper" style="position:relative;overflow:hidden;">
  <textarea class="playground-editor" id="codeEditor" spellcheck="false"># Welcome to the Berta Playground!
# Write Python code here and click [Run] to execute.
#
# Try: Load an exercise from the dropdown above,
# or write your own code below.

print("Hello from Berta Chapters!")
print("2 + 2 =", 2 + 2)

# Try making an error to see friendly explanations:
# print(undefined_variable)
</textarea>
  </div>
  <div class="playground-toolbar">
    <button class="run-btn" onclick="runCode()" id="runBtn" disabled>&#9654; Run</button>
    <button class="clear-btn" onclick="clearOutput()">Clear Output</button>
    <button class="clear-btn" onclick="resetEditor()">Reset</button>
    <span id="statusText" style="float:right; font-size:12px; color:#808080; padding-top:4px;">Loading Python...</span>
  </div>
  <div class="playground-output" id="output">Initializing Python environment... (this takes a few seconds on first load)</div>
  <div class="playground-status" id="statusBar">Pyodide: loading...</div>
</div>

---

## :bulb: How It Works

- **Write code** in the editor above (or load a pre-built exercise)
- **Click Run** to execute your Python code in the browser
- **Errors are explained** in plain English below the traceback
- **No data leaves your browser** &mdash; everything runs locally via WebAssembly

!!! info "Supported Libraries"
    The playground runs standard Python 3.11. You have access to built-in modules
    like `math`, `random`, `json`, `collections`, `itertools`, and more.
    NumPy and other heavy libraries are not loaded to keep startup fast.

!!! tip "Keyboard Shortcut"
    Press **Ctrl+Enter** (or **Cmd+Enter** on Mac) to run your code.

---

## :books: Exercise Guide

| Exercise | Chapter | What You'll Practice |
|----------|---------|---------------------|
| Hello World | Ch 1 | Print statements, f-strings |
| Variables & Types | Ch 1 | Type checking, conversion |
| Lists & Loops | Ch 1 | Iteration, comprehensions |
| Functions | Ch 1 | Defining and calling functions |
| Dictionaries | Ch 1 | Key-value operations |
| Binary Search | Ch 2 | Algorithm implementation |
| Stack | Ch 2 | LIFO data structure |
| Sorting | Ch 2 | Implementing merge sort |
| Vector Operations | Ch 3 | Math with lists |
| Dot Product | Ch 3 | Linear algebra basics |
| Coin Flip Simulation | Ch 4 | Probability and randomness |
| Bayes Theorem | Ch 4 | Conditional probability |
| Train/Test Split | Ch 6 | ML data preparation |
| Accuracy Score | Ch 6 | Model evaluation |
| Linear Regression | Ch 7 | Fitting a line from scratch |
| Sigmoid & Logistic | Ch 7 | Classification with probabilities |
| Gini Impurity | Ch 7 | Decision tree splitting |
| K-Means Clustering | Ch 8 | Grouping unlabeled data |
| PCA from Scratch | Ch 8 | Dimensionality reduction |
| Silhouette Score | Ch 8 | Cluster quality evaluation |
| Single Neuron | Ch 9 | Perceptron learning |
| Activation Functions | Ch 9 | Sigmoid, tanh, ReLU |
| Backpropagation | Ch 9 | Gradient computation |
| Simple Tokenization | Ch 10 | Splitting text into words |
| Bag of Words | Ch 10 | Word counts for documents |
| Cosine Similarity | Ch 10 | Comparing text vectors |

---

**Created by [Luigi Pascal Rondanini](https://rondanini.net) | Generated by [Berta AI](https://berta.one)**

<!-- Load Pyodide FIRST -->
<script src="https://cdn.jsdelivr.net/pyodide/v0.27.5/full/pyodide.js"></script>

<script>
// ============================================================
// Berta Playground - Python in the browser via Pyodide
// ============================================================

var EXERCISES = {};
EXERCISES["ch1_hello"] = "# Exercise: Hello World\n# Print your name and a greeting.\n\nname = \"Learner\"\nprint(\"Hello, \" + name + \"! Welcome to Berta Chapters.\")\n\n# YOUR TURN: Change the name variable to your own name\n# and add a second print statement.\n";
EXERCISES["ch1_variables"] = "# Exercise: Variables & Types\n\nage = 25\nheight = 1.75\nname = \"Berta\"\nis_ai = True\nskills = [\"Python\", \"ML\", \"NLP\"]\n\nprint(\"Name: \" + name + \" (type: \" + type(name).__name__ + \")\")\nprint(\"Age: \" + str(age) + \" (type: \" + type(age).__name__ + \")\")\nprint(\"Height: \" + str(height) + \" (type: \" + type(height).__name__ + \")\")\nprint(\"Is AI: \" + str(is_ai) + \" (type: \" + type(is_ai).__name__ + \")\")\nprint(\"Skills: \" + str(skills) + \" (type: \" + type(skills).__name__ + \")\")\n\n# YOUR TURN: Add a variable called 'learning_rate'\n# set it to 0.001, and print its type.\n";
EXERCISES["ch1_lists"] = "# Exercise: Lists & Loops\n\nscores = [85, 92, 78, 95, 88, 73, 91]\naverage = sum(scores) / len(scores)\nprint(\"Scores: \" + str(scores))\nprint(\"Average: \" + str(round(average, 1)))\n\nabove_avg = [s for s in scores if s > average]\nprint(\"Above average: \" + str(above_avg))\n\n# YOUR TURN: Create a list of the first 10 squares\n# (1, 4, 9, 16, ...) using a list comprehension.\n";
EXERCISES["ch1_functions"] = "# Exercise: Functions\n\ndef greet(name, greeting=\"Hello\"):\n    return greeting + \", \" + name + \"!\"\n\nprint(greet(\"Berta\"))\nprint(greet(\"Learner\", greeting=\"Welcome\"))\n\n# YOUR TURN: Write a function called 'calculate_bmi'\n# that takes weight (kg) and height (m) as parameters\n# and returns the BMI (weight / height**2).\n# Test it with weight=70, height=1.75\n";
EXERCISES["ch1_dict"] = "# Exercise: Dictionaries\n\nmodel_config = {\n    \"name\": \"transformer-v1\",\n    \"layers\": 12,\n    \"hidden_size\": 768,\n    \"learning_rate\": 0.0003,\n}\n\nprint(\"Model Configuration:\")\nfor key, value in model_config.items():\n    print(\"  \" + key + \": \" + str(value))\n\ndropout = model_config.get(\"dropout\", 0.1)\nprint(\"\\nDropout: \" + str(dropout) + \" (default used)\")\n\n# YOUR TURN: Add 'batch_size': 32 to the config\n# and print the updated config.\n";
EXERCISES["ch2_search"] = "# Exercise: Binary Search\n\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    steps = 0\n    while left <= right:\n        steps += 1\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid, steps\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1, steps\n\ndata = list(range(0, 100, 3))\nprint(\"Searching in \" + str(len(data)) + \" items\")\n\ntarget = 42\nidx, steps = binary_search(data, target)\nprint(\"Found \" + str(target) + \" at index \" + str(idx) + \" in \" + str(steps) + \" steps\")\n\n# YOUR TURN: Search for a value that does NOT exist.\n";
EXERCISES["ch2_stack"] = "# Exercise: Stack - Balanced Parentheses\n\nclass Stack:\n    def __init__(self):\n        self._items = []\n    def push(self, item):\n        self._items.append(item)\n    def pop(self):\n        return self._items.pop()\n    def is_empty(self):\n        return len(self._items) == 0\n\ndef is_balanced(expression):\n    stack = Stack()\n    pairs = {')': '(', ']': '[', '}': '{'}\n    for char in expression:\n        if char in '([{':\n            stack.push(char)\n        elif char in ')]}':\n            if stack.is_empty() or stack.pop() != pairs[char]:\n                return False\n    return stack.is_empty()\n\ntests = [\"(())\", \"([{}])\", \"(()\", \"({[)]}\", \"\"]\nfor expr in tests:\n    result = \"balanced\" if is_balanced(expr) else \"NOT balanced\"\n    print(\"  '\" + expr + \"' -> \" + result)\n";
EXERCISES["ch2_sorting"] = "# Exercise: Merge Sort\n\ndef merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n\ndef merge(left, right):\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            result.append(left[i])\n            i += 1\n        else:\n            result.append(right[j])\n            j += 1\n    result.extend(left[i:])\n    result.extend(right[j:])\n    return result\n\nimport random\nrandom.seed(42)\ndata = [random.randint(1, 100) for _ in range(12)]\nprint(\"Original: \" + str(data))\nprint(\"Sorted:   \" + str(merge_sort(data)))\n";
EXERCISES["ch3_vectors"] = "# Exercise: Vector Operations\n\ndef vector_add(a, b):\n    return [x + y for x, y in zip(a, b)]\n\ndef scalar_multiply(scalar, v):\n    return [scalar * x for x in v]\n\ndef vector_norm(v):\n    return sum(x**2 for x in v) ** 0.5\n\nv1 = [3, 4]\nv2 = [1, -2]\n\nprint(\"v1 =\", v1)\nprint(\"v2 =\", v2)\nprint(\"v1 + v2 =\", vector_add(v1, v2))\nprint(\"2 * v1 =\", scalar_multiply(2, v1))\nprint(\"|v1| =\", round(vector_norm(v1), 2))\nprint(\"|v2| =\", round(vector_norm(v2), 2))\n";
EXERCISES["ch3_dotproduct"] = "# Exercise: Dot Product & Cosine Similarity\nimport math\n\ndef dot_product(a, b):\n    return sum(x * y for x, y in zip(a, b))\n\ndef cosine_similarity(a, b):\n    dot = dot_product(a, b)\n    norm_a = math.sqrt(sum(x**2 for x in a))\n    norm_b = math.sqrt(sum(x**2 for x in b))\n    if norm_a == 0 or norm_b == 0:\n        return 0.0\n    return dot / (norm_a * norm_b)\n\nprint(\"Similar:\", round(cosine_similarity([1,2,3], [2,4,6]), 4))\nprint(\"Orthogonal:\", round(cosine_similarity([1,0], [0,1]), 4))\nprint(\"Opposite:\", round(cosine_similarity([1,1], [-1,-1]), 4))\n";
EXERCISES["ch4_coin"] = "# Exercise: Coin Flip Simulation\nimport random\nrandom.seed(42)\n\ndef simulate_flips(n):\n    heads = sum(1 for _ in range(n) if random.random() < 0.5)\n    return heads / n\n\nfor n in [10, 100, 1000, 10000, 100000]:\n    p = simulate_flips(n)\n    bar = '#' * int(p * 40)\n    print(\"n=\" + str(n).rjust(7) + \": P(H)=\" + str(round(p, 4)) + \" |\" + bar + \"|\")\n\nprint(\"\\nAs n grows, P(heads) approaches 0.5!\")\n";
EXERCISES["ch4_bayes"] = "# Exercise: Bayes' Theorem\n\ndef bayes(prior, sensitivity, false_positive_rate):\n    p_pos = sensitivity * prior + false_positive_rate * (1 - prior)\n    return (sensitivity * prior) / p_pos\n\nprior = 0.01\nsensitivity = 0.95\nfpr = 0.05\n\nresult = bayes(prior, sensitivity, fpr)\nprint(\"Prior probability of disease: \" + str(prior * 100) + \"%\")\nprint(\"Test sensitivity: \" + str(sensitivity * 100) + \"%\")\nprint(\"False positive rate: \" + str(fpr * 100) + \"%\")\nprint(\"\")\nprint(\"If you test positive, probability you have it: \" + str(round(result * 100, 1)) + \"%\")\nprint(\"\")\nprint(\"Surprised? This is why Bayes' theorem matters!\")\n";
EXERCISES["ch6_split"] = "# Exercise: Train/Test Split from Scratch\nimport random\n\ndef train_test_split(data, test_ratio=0.2, seed=42):\n    random.seed(seed)\n    shuffled = data[:]\n    random.shuffle(shuffled)\n    split = int(len(shuffled) * (1 - test_ratio))\n    return shuffled[:split], shuffled[split:]\n\ndataset = list(range(100))\ntrain, test = train_test_split(dataset, test_ratio=0.2)\n\nprint(\"Dataset: \" + str(len(dataset)) + \" items\")\nprint(\"Train: \" + str(len(train)) + \" items (\" + str(len(train)) + \"%)\")\nprint(\"Test: \" + str(len(test)) + \" items (\" + str(len(test)) + \"%)\")\nprint(\"No overlap: \" + str(len(set(train) & set(test)) == 0))\nprint(\"All accounted: \" + str(len(set(train) | set(test)) == len(dataset)))\n";
EXERCISES["ch6_accuracy"] = "# Exercise: Accuracy Score & Confusion Matrix\n\ndef accuracy(y_true, y_pred):\n    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)\n    return correct / len(y_true)\n\ndef confusion_matrix(y_true, y_pred):\n    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)\n    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)\n    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)\n    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)\n    return {\"TP\": tp, \"TN\": tn, \"FP\": fp, \"FN\": fn}\n\ny_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]\ny_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 1]\n\nacc = accuracy(y_true, y_pred)\ncm = confusion_matrix(y_true, y_pred)\n\nprint(\"True labels:  \" + str(y_true))\nprint(\"Predictions:  \" + str(y_pred))\nprint(\"\")\nprint(\"Accuracy: \" + str(round(acc * 100)) + \"%\")\nprint(\"\")\nprint(\"Confusion Matrix:\")\nprint(\"  TP=\" + str(cm['TP']) + \" FP=\" + str(cm['FP']))\nprint(\"  FN=\" + str(cm['FN']) + \" TN=\" + str(cm['TN']))\n\nprecision = cm['TP'] / max(cm['TP'] + cm['FP'], 1)\nrecall = cm['TP'] / max(cm['TP'] + cm['FN'], 1)\nprint(\"\")\nprint(\"Precision: \" + str(round(precision * 100)) + \"%\")\nprint(\"Recall: \" + str(round(recall * 100)) + \"%\")\n";
EXERCISES["ch7_linreg"] = "# Exercise: Linear Regression from Scratch\n# Fit y = wx + b using gradient descent (pure Python, no NumPy)\nimport random\nrandom.seed(42)\n\n# Generate data: y = 2.5x + 1.0 + noise\nX = [random.uniform(0, 10) for _ in range(50)]\ny = [2.5 * x + 1.0 + random.gauss(0, 1.5) for x in X]\n\n# Gradient descent\nw, b, lr = 0.0, 0.0, 0.01\nfor epoch in range(200):\n    dw, db = 0.0, 0.0\n    for xi, yi in zip(X, y):\n        pred = w * xi + b\n        error = pred - yi\n        dw += error * xi\n        db += error\n    w -= lr * (2 / len(X)) * dw\n    b -= lr * (2 / len(X)) * db\n\nprint('Learned: y = {:.2f}x + {:.2f}'.format(w, b))\nprint('True:    y = 2.50x + 1.00')\n\n# R-squared\ny_mean = sum(y) / len(y)\nss_res = sum((yi - (w * xi + b))**2 for xi, yi in zip(X, y))\nss_tot = sum((yi - y_mean)**2 for yi in y)\nprint('R² = {:.4f}'.format(1 - ss_res / ss_tot))\n\n# YOUR TURN: Change the learning rate or number of epochs.\n# What happens with lr=0.1? With only 10 epochs?\n";
EXERCISES["ch7_sigmoid"] = "# Exercise: Sigmoid & Logistic Regression\nimport math\nimport random\nrandom.seed(42)\n\ndef sigmoid(z):\n    return 1 / (1 + math.exp(-max(-500, min(500, z))))\n\n# Show sigmoid values\nprint('Sigmoid function:')\nfor z in [-5, -2, -1, 0, 1, 2, 5]:\n    s = sigmoid(z)\n    bar = '#' * int(s * 30)\n    print('  z={:+.0f}  sig={:.4f}  |{:<30}|'.format(z, s, bar))\n\n# Mini logistic regression on 2D data\n# Class 0: low values, Class 1: high values\nX = [(random.gauss(2, 1), random.gauss(2, 1)) for _ in range(20)] + \\\n    [(random.gauss(5, 1), random.gauss(5, 1)) for _ in range(20)]\ny = [0] * 20 + [1] * 20\n\nw1, w2, b, lr = 0.0, 0.0, 0.0, 0.1\nfor epoch in range(200):\n    for (x1, x2), yi in zip(X, y):\n        pred = sigmoid(w1 * x1 + w2 * x2 + b)\n        err = pred - yi\n        w1 -= lr * err * x1\n        w2 -= lr * err * x2\n        b -= lr * err\n\npreds = [1 if sigmoid(w1*x1 + w2*x2 + b) >= 0.5 else 0 for x1, x2 in X]\nacc = sum(p == t for p, t in zip(preds, y)) / len(y)\nprint('\\nLogistic regression accuracy: {:.0%}'.format(acc))\nprint('Weights: w1={:.2f}, w2={:.2f}, b={:.2f}'.format(w1, w2, b))\n";
EXERCISES["ch7_gini"] = "# Exercise: Gini Impurity & Decision Tree Splitting\n\ndef gini(labels):\n    \"\"\"Gini impurity: 1 - sum(p_i^2)\"\"\"\n    n = len(labels)\n    if n == 0:\n        return 0.0\n    counts = {}\n    for l in labels:\n        counts[l] = counts.get(l, 0) + 1\n    return 1 - sum((c / n) ** 2 for c in counts.values())\n\ndef find_best_split(X, y, feature_idx):\n    \"\"\"Find the threshold that minimizes weighted Gini.\"\"\"\n    values = sorted(set(x[feature_idx] for x in X))\n    best_gini, best_thr = float('inf'), None\n    for i in range(len(values) - 1):\n        thr = (values[i] + values[i+1]) / 2\n        left_y = [yi for xi, yi in zip(X, y) if xi[feature_idx] <= thr]\n        right_y = [yi for xi, yi in zip(X, y) if xi[feature_idx] > thr]\n        n = len(y)\n        weighted = len(left_y)/n * gini(left_y) + len(right_y)/n * gini(right_y)\n        if weighted < best_gini:\n            best_gini, best_thr = weighted, thr\n    return best_thr, best_gini\n\n# Example: should we approve a loan?\n# Features: [income, credit_score]\nX = [[30, 600], [50, 700], [40, 650], [80, 750],\n     [20, 550], [60, 720], [35, 580], [90, 800]]\ny = [0, 1, 0, 1, 0, 1, 0, 1]  # 0=deny, 1=approve\n\nprint('Parent Gini:', round(gini(y), 4))\nprint()\nfor i, name in enumerate(['Income', 'Credit Score']):\n    thr, g = find_best_split(X, y, i)\n    print('{}: best split at {:.0f}, Gini={:.4f}'.format(name, thr, g))\n    left = [yi for xi, yi in zip(X, y) if xi[i] <= thr]\n    right = [yi for xi, yi in zip(X, y) if xi[i] > thr]\n    print('  Left:  {} -> Gini {:.3f}'.format(left, gini(left)))\n    print('  Right: {} -> Gini {:.3f}'.format(right, gini(right)))\n\n# YOUR TURN: Which feature gives a better split? Why?\n";
EXERCISES["ch8_kmeans"] = "# Exercise: K-Means Clustering from Scratch\nimport random\nimport math\nrandom.seed(42)\n\ndef distance(a, b):\n    return math.sqrt(sum((ai - bi)**2 for ai, bi in zip(a, b)))\n\ndef kmeans(data, k=3, max_iters=20):\n    # Random initialization\n    centroids = random.sample(data, k)\n    for iteration in range(max_iters):\n        # Assign each point to nearest centroid\n        clusters = [[] for _ in range(k)]\n        for point in data:\n            dists = [distance(point, c) for c in centroids]\n            cluster_id = dists.index(min(dists))\n            clusters[cluster_id].append(point)\n        # Recompute centroids\n        new_centroids = []\n        for cluster in clusters:\n            if cluster:\n                mean = [sum(p[d] for p in cluster) / len(cluster)\n                        for d in range(len(cluster[0]))]\n                new_centroids.append(mean)\n            else:\n                new_centroids.append(random.choice(data))\n        if new_centroids == centroids:\n            break\n        centroids = new_centroids\n    return centroids, clusters\n\n# Generate 3 clusters of 2D points\ndata = []\nfor cx, cy in [(2, 2), (8, 3), (5, 8)]:\n    data += [(cx + random.gauss(0, 0.8), cy + random.gauss(0, 0.8))\n             for _ in range(20)]\nrandom.shuffle(data)\n\ncentroids, clusters = kmeans(data, k=3)\n\nprint('K-Means found 3 clusters:')\nfor i, (c, cl) in enumerate(zip(centroids, clusters)):\n    print('  Cluster {}: center=({:.1f}, {:.1f}), {} points'.format(\n        i+1, c[0], c[1], len(cl)))\n\n# Inertia (within-cluster sum of squared distances)\ninertia = sum(distance(p, centroids[i])**2\n              for i, cl in enumerate(clusters) for p in cl)\nprint('\\nInertia: {:.1f}'.format(inertia))\n\n# YOUR TURN: Try k=2 or k=4. Does inertia go up or down?\n";
EXERCISES["ch8_pca"] = "# Exercise: PCA from Scratch (pure Python)\nimport math\nimport random\nrandom.seed(42)\n\n# Generate correlated 2D data\nn = 100\ndata = []\nfor _ in range(n):\n    x = random.gauss(0, 3)\n    y = 0.7 * x + random.gauss(0, 1)  # correlated!\n    data.append([x, y])\n\n# Step 1: Center the data\nmean_x = sum(p[0] for p in data) / n\nmean_y = sum(p[1] for p in data) / n\ncentered = [[p[0] - mean_x, p[1] - mean_y] for p in data]\nprint('Mean: ({:.2f}, {:.2f})'.format(mean_x, mean_y))\n\n# Step 2: Covariance matrix\ncov_xx = sum(p[0]**2 for p in centered) / (n - 1)\ncov_yy = sum(p[1]**2 for p in centered) / (n - 1)\ncov_xy = sum(p[0]*p[1] for p in centered) / (n - 1)\nprint('\\nCovariance matrix:')\nprint('  [{:.2f}  {:.2f}]'.format(cov_xx, cov_xy))\nprint('  [{:.2f}  {:.2f}]'.format(cov_xy, cov_yy))\n\n# Step 3: Eigenvalues (quadratic formula for 2x2)\ntrace = cov_xx + cov_yy\ndet = cov_xx * cov_yy - cov_xy**2\ndisc = math.sqrt(max(trace**2 / 4 - det, 0))\nlam1 = trace / 2 + disc\nlam2 = trace / 2 - disc\ntotal = lam1 + lam2\n\nprint('\\nEigenvalues: {:.2f}, {:.2f}'.format(lam1, lam2))\nprint('Variance explained: {:.1%}, {:.1%}'.format(lam1/total, lam2/total))\nprint('\\nPC1 captures {:.1%} of the variance!'.format(lam1/total))\nprint('This means we could reduce 2D to 1D and keep most info.')\n\n# YOUR TURN: Change the correlation (0.7) to 0.0.\n# What happens to the variance explained?\n";
EXERCISES["ch8_silhouette"] = "# Exercise: Silhouette Score from Scratch\nimport math\nimport random\nrandom.seed(42)\n\ndef distance(a, b):\n    return math.sqrt(sum((ai-bi)**2 for ai, bi in zip(a, b)))\n\ndef silhouette_score(points, labels):\n    \"\"\"Compute mean silhouette score for a clustering.\"\"\"\n    n = len(points)\n    scores = []\n    unique_labels = list(set(labels))\n    for i in range(n):\n        # a(i) = mean dist to own cluster\n        same = [j for j in range(n) if labels[j] == labels[i] and j != i]\n        if not same:\n            scores.append(0.0)\n            continue\n        a_i = sum(distance(points[i], points[j]) for j in same) / len(same)\n        # b(i) = min mean dist to other clusters\n        b_i = float('inf')\n        for k in unique_labels:\n            if k == labels[i]:\n                continue\n            other = [j for j in range(n) if labels[j] == k]\n            if other:\n                mean_d = sum(distance(points[i], points[j]) for j in other) / len(other)\n                b_i = min(b_i, mean_d)\n        s_i = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0\n        scores.append(s_i)\n    return sum(scores) / len(scores)\n\n# Well-separated clusters\npoints = [(random.gauss(0, 0.5), random.gauss(0, 0.5)) for _ in range(15)] + \\\n         [(random.gauss(5, 0.5), random.gauss(5, 0.5)) for _ in range(15)] + \\\n         [(random.gauss(10, 0.5), random.gauss(0, 0.5)) for _ in range(15)]\nlabels_good = [0]*15 + [1]*15 + [2]*15\n\n# Bad clustering (random labels)\nlabels_bad = [random.randint(0, 2) for _ in range(45)]\n\nprint('Silhouette score (good clustering): {:.3f}'.format(\n    silhouette_score(points, labels_good)))\nprint('Silhouette score (random labels):   {:.3f}'.format(\n    silhouette_score(points, labels_bad)))\nprint()\nprint('Score ranges from -1 to 1:')\nprint('  +1 = perfectly separated clusters')\nprint('   0 = overlapping clusters')\nprint('  -1 = points in wrong clusters')\n";
EXERCISES["ch9_neuron"] = "# Exercise: Single Neuron (Perceptron) Learning\nimport math\nimport random\nrandom.seed(42)\n\ndef sigmoid(z):\n    return 1 / (1 + math.exp(-max(-500, min(500, z))))\n\n# A single neuron learning the AND gate\n# AND: (0,0)->0, (0,1)->0, (1,0)->0, (1,1)->1\nX = [[0, 0], [0, 1], [1, 0], [1, 1]]\ny = [0, 0, 0, 1]\n\nw1, w2, b = random.gauss(0, 0.5), random.gauss(0, 0.5), random.gauss(0, 0.5)\nlr = 1.0\n\nprint('=== Training a single neuron on AND gate ===')\nprint('Before training:')\nfor xi, yi in zip(X, y):\n    out = sigmoid(w1 * xi[0] + w2 * xi[1] + b)\n    print('  {} -> {:.3f} (target: {})'.format(xi, out, yi))\n\nfor epoch in range(1000):\n    for xi, yi in zip(X, y):\n        out = sigmoid(w1 * xi[0] + w2 * xi[1] + b)\n        err = out - yi\n        w1 -= lr * err * xi[0]\n        w2 -= lr * err * xi[1]\n        b -= lr * err\n\nprint('\\nAfter 1000 epochs:')\nfor xi, yi in zip(X, y):\n    out = sigmoid(w1 * xi[0] + w2 * xi[1] + b)\n    pred = 1 if out >= 0.5 else 0\n    status = 'correct' if pred == yi else 'WRONG'\n    print('  {} -> {:.3f} (pred: {}, target: {}) {}'.format(xi, out, pred, yi, status))\n\nprint('\\nWeights: w1={:.2f}, w2={:.2f}, b={:.2f}'.format(w1, w2, b))\n\n# YOUR TURN: Change the target to OR gate: y = [0, 1, 1, 1]\n# Then try XOR: y = [0, 1, 1, 0]. Can a single neuron learn XOR?\n";
EXERCISES["ch9_activations"] = "# Exercise: Activation Functions\nimport math\n\ndef sigmoid(x):\n    return 1 / (1 + math.exp(-max(-500, min(500, x))))\n\ndef tanh(x):\n    return math.tanh(x)\n\ndef relu(x):\n    return max(0, x)\n\ndef sigmoid_deriv(x):\n    s = sigmoid(x)\n    return s * (1 - s)\n\ndef tanh_deriv(x):\n    return 1 - math.tanh(x)**2\n\ndef relu_deriv(x):\n    return 1.0 if x > 0 else 0.0\n\nprint('Activation Functions Comparison')\nprint('=' * 55)\nprint('{:>5}  {:>8}  {:>8}  {:>8}'.format('x', 'sigmoid', 'tanh', 'relu'))\nprint('-' * 55)\nfor x in [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]:\n    print('{:>5.1f}  {:>8.4f}  {:>8.4f}  {:>8.4f}'.format(\n        x, sigmoid(x), tanh(x), relu(x)))\n\nprint('\\nDerivatives at x=0:')\nprint('  sigmoid: {:.4f}  (max gradient is 0.25!)'.format(sigmoid_deriv(0)))\nprint('  tanh:    {:.4f}  (stronger gradient)'.format(tanh_deriv(0)))\nprint('  relu:    {:.4f}  (constant gradient for x>0)'.format(relu_deriv(0.1)))\n\nprint('\\nKey takeaways:')\nprint('  sigmoid: output (0,1), vanishing gradients, good for output layer')\nprint('  tanh:    output (-1,1), still vanishes, better than sigmoid')\nprint('  relu:    output [0,inf), no vanishing gradient, most popular')\n\n# YOUR TURN: What is sigmoid_deriv(-5)? Why is this a problem?\n";
EXERCISES["ch9_backprop"] = "# Exercise: Backpropagation - Computing Gradients\nimport math\nimport random\nrandom.seed(42)\n\ndef sigmoid(z):\n    return 1 / (1 + math.exp(-max(-500, min(500, z))))\n\n# 2-layer network: [2 inputs] -> [2 hidden] -> [1 output]\n# Learn XOR: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0\nX = [[0,0], [0,1], [1,0], [1,1]]\ny = [0, 1, 1, 0]\n\n# Initialize weights randomly\nW1 = [[random.gauss(0, 1) for _ in range(2)] for _ in range(2)]  # 2x2\nb1 = [0.0, 0.0]\nW2 = [random.gauss(0, 1) for _ in range(2)]  # 2->1\nb2 = 0.0\nlr = 2.0\n\ndef forward(x):\n    h = [sigmoid(W1[j][0]*x[0] + W1[j][1]*x[1] + b1[j]) for j in range(2)]\n    out = sigmoid(W2[0]*h[0] + W2[1]*h[1] + b2)\n    return h, out\n\ndef train_step(x, target):\n    global W1, b1, W2, b2\n    h, out = forward(x)\n    # Output layer gradient\n    d_out = (out - target) * out * (1 - out)\n    # Hidden layer gradients (chain rule!)\n    d_h = [d_out * W2[j] * h[j] * (1 - h[j]) for j in range(2)]\n    # Update weights\n    for j in range(2):\n        W2[j] -= lr * d_out * h[j]\n    b2 -= lr * d_out\n    for j in range(2):\n        for i in range(2):\n            W1[j][i] -= lr * d_h[j] * x[i]\n        b1[j] -= lr * d_h[j]\n\nprint('Training a 2-layer network on XOR...')\nfor epoch in range(5000):\n    for xi, yi in zip(X, y):\n        train_step(xi, yi)\n\nprint('\\nResults after 5000 epochs:')\ncorrect = 0\nfor xi, yi in zip(X, y):\n    h, out = forward(xi)\n    pred = 1 if out >= 0.5 else 0\n    correct += (pred == yi)\n    print('  {} -> {:.3f} (pred={}, target={})'.format(xi, out, pred, yi))\nprint('\\nAccuracy: {}/{}'.format(correct, len(y)))\nprint('A single neuron CANNOT learn XOR, but 2 layers can!')\n";

EXERCISES["ch10_tokenize"] = "# Exercise: Simple Tokenization (NLP)\nimport re\n\ndef simple_tokenize(text):\n    \"\"\"Split text into words: lowercase, keep only letters.\"\"\"\n    text = text.lower().strip()\n    words = re.findall(r\"[a-z]+\", text)\n    return words\n\nsentence = \"Natural language processing is amazing!\"\ntokens = simple_tokenize(sentence)\nprint(\"Sentence:\", sentence)\nprint(\"Tokens:\", tokens)\nprint(\"Number of tokens:\", len(tokens))\n\n# Another example\ns2 = \"Hello, World. How are you?\"\nprint(\"\\nSentence:\", s2)\nprint(\"Tokens:\", simple_tokenize(s2))\n\n# YOUR TURN: Add a line that tokenizes your own sentence.\n";
EXERCISES["ch10_bow"] = "# Exercise: Bag of Words (word counts)\nfrom collections import Counter\nimport re\n\ndef simple_tokenize(text):\n    return re.findall(r\"[a-z]+\", text.lower())\n\ndef bag_of_words(text):\n    tokens = simple_tokenize(text)\n    return Counter(tokens)\n\ndoc1 = \"machine learning is fun\"\ndoc2 = \"machine learning and deep learning\"\n\nbow1 = bag_of_words(doc1)\nbow2 = bag_of_words(doc2)\n\nprint(\"Doc 1:\", doc1)\nprint(\"BoW 1:\", dict(bow1))\nprint()\nprint(\"Doc 2:\", doc2)\nprint(\"BoW 2:\", dict(bow2))\n\n# All unique words (vocabulary)\nvocab = sorted(set(bow1) | set(bow2))\nprint(\"\\nVocabulary:\", vocab)\n\n# YOUR TURN: Add a third document and print its BoW.\n";
EXERCISES["ch10_cosine"] = "# Exercise: Cosine Similarity (text vectors)\nimport math\n\ndef cosine_similarity(vec_a, vec_b):\n    \"\"\"Cosine similarity between two vectors (lists of numbers).\"\"\"\n    dot = sum(a * b for a, b in zip(vec_a, vec_b))\n    norm_a = math.sqrt(sum(x*x for x in vec_a))\n    norm_b = math.sqrt(sum(x*x for x in vec_b))\n    if norm_a == 0 or norm_b == 0:\n        return 0.0\n    return dot / (norm_a * norm_b)\n\n# Two short documents as word count vectors (same vocabulary order)\n# Vocabulary: [\"cat\", \"dog\", \"run\"]\nvec1 = [2, 1, 0]  # doc1: 2 cat, 1 dog, 0 run\nvec2 = [1, 2, 1]  # doc2: 1 cat, 2 dog, 1 run\nvec3 = [0, 0, 3]  # doc3: only \"run\"\n\nsim_12 = cosine_similarity(vec1, vec2)\nsim_13 = cosine_similarity(vec1, vec3)\nsim_23 = cosine_similarity(vec2, vec3)\n\nprint(\"Vec1 (cat, dog, run):\", vec1)\nprint(\"Vec2:\", vec2)\nprint(\"Vec3:\", vec3)\nprint()\nprint(\"Similarity 1-2: {:.3f}\".format(sim_12))\nprint(\"Similarity 1-3: {:.3f}\".format(sim_13))\nprint(\"Similarity 2-3: {:.3f}\".format(sim_23))\nprint(\"\\nDocuments with more overlapping words have higher cosine similarity.\")\n\n# YOUR TURN: Change vec3 to [3, 0, 0]. Which pair is most similar?\n";

// Friendly error explanations
var ERROR_EXPLANATIONS = {
  "NameError": function(msg) {
    var match = msg.match(/name '(.+)' is not defined/);
    if (match) return "You tried to use a variable called '" + match[1] + "', but it hasn't been created yet. Did you forget to define it? Check for typos. Python is case-sensitive.";
    return "You're using a name that Python doesn't recognize. Make sure all variables are defined before you use them.";
  },
  "TypeError": function(msg) {
    if (msg.indexOf("unsupported operand") >= 0) return "You're trying to do math with incompatible types. For example, you can't add a string and a number. Use int() or str() to convert.";
    if (msg.indexOf("not callable") >= 0) return "You're trying to call something that isn't a function. Did you accidentally put parentheses () on a variable?";
    if (msg.indexOf("argument") >= 0) return "Wrong number or type of arguments passed to a function. Check what the function expects.";
    return "Python got confused about data types. Make sure you're using the right types for each operation.";
  },
  "SyntaxError": function(msg) {
    if (msg.indexOf("EOL") >= 0) return "You started a string but didn't close it. Make sure every opening quote has a matching closing quote.";
    if (msg.indexOf("unexpected EOF") >= 0) return "Your code ended unexpectedly. You might be missing a closing parenthesis, bracket, or the body of an if/for/function.";
    if (msg.indexOf("invalid syntax") >= 0) return "Python can't understand this line. Common causes: missing colons after if/for/def, using = instead of == for comparison, or mismatched brackets.";
    return "There's a grammar mistake in your code. Check for missing colons, brackets, or quotes.";
  },
  "IndentationError": function() {
    return "Python uses indentation (spaces) to define code blocks. Make sure your code inside if/for/def/class is indented consistently. Use 4 spaces (not tabs).";
  },
  "IndexError": function() {
    return "You tried to access a position in a list that doesn't exist. Remember: Python lists start at index 0, and the last item is at index len(list)-1.";
  },
  "KeyError": function(msg) {
    var match = msg.match(/'(.+)'/);
    if (match) return "The key '" + match[1] + "' doesn't exist in this dictionary. Use .get('" + match[1] + "', default_value) for safe access.";
    return "You tried to access a dictionary key that doesn't exist. Use .get() for safe access.";
  },
  "ValueError": function(msg) {
    if (msg.indexOf("convert") >= 0) return "Python can't convert this value. For example, int('hello') fails because 'hello' isn't a number.";
    return "The value you provided isn't valid for this operation. Check your data format.";
  },
  "ZeroDivisionError": function() {
    return "You tried to divide by zero, which is undefined. Add a check: if denominator != 0 before dividing.";
  },
  "AttributeError": function(msg) {
    var match = msg.match(/'(.+)' object has no attribute '(.+)'/);
    if (match) return "The " + match[1] + " type doesn't have a method called '" + match[2] + "'. Check spelling, or use type(variable) to verify the type.";
    return "You tried to use a method that doesn't exist on this object. Check the type and available methods.";
  },
  "ImportError": function() {
    return "This module isn't available in the browser playground. Built-in modules (math, random, json, collections) work fine.";
  },
  "ModuleNotFoundError": function() {
    return "This module isn't available in the browser. You can use built-in modules like math, random, json, collections, itertools.";
  },
  "RecursionError": function() {
    return "Infinite recursion! Your function calls itself forever. Make sure there's a base case that stops the recursion.";
  },
  "FileNotFoundError": function() {
    return "File operations aren't available in the browser playground. This runs in your browser without a file system.";
  }
};

function getErrorExplanation(errorType, errorMsg) {
  var handler = ERROR_EXPLANATIONS[errorType];
  if (handler) {
    return handler(errorMsg);
  }
  return "Something went wrong. Read the error message above -- it usually tells you the line number and what happened.";
}

function escapeHtml(text) {
  var div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// Pyodide initialization
var pyodide = null;

async function initPyodide() {
  try {
    pyodide = await loadPyodide();
    document.getElementById("statusBar").textContent = "Pyodide: ready";
    document.getElementById("statusText").textContent = "Ready";
    document.getElementById("statusText").style.color = "#008080";
    document.getElementById("output").textContent = "Python environment ready. Write code and click Run!";
    document.getElementById("runBtn").disabled = false;
  } catch (e) {
    document.getElementById("statusBar").textContent = "Pyodide: failed to load";
    document.getElementById("output").textContent = "Failed to load Python environment: " + e.message;
  }
}

async function runCode() {
  if (!pyodide) {
    document.getElementById("output").textContent = "Python is still loading. Please wait...";
    return;
  }

  var code = document.getElementById("codeEditor").value;
  var outputEl = document.getElementById("output");
  var statusText = document.getElementById("statusText");

  statusText.textContent = "Running...";
  statusText.style.color = "#996600";
  outputEl.textContent = "";

  clearErrorHighlight();

  pyodide.globals.set("__user_code__", code);

  var wrapper = [
    "import sys, io, traceback",
    "sys.stdout = io.StringIO()",
    "sys.stderr = io.StringIO()",
    "__berta_error__ = ''",
    "__berta_error_type__ = ''",
    "__berta_error_line__ = 0",
    "try:",
    "    exec(compile(__user_code__, '<playground>', 'exec'))",
    "except BaseException as __e__:",
    "    __berta_error_type__ = type(__e__).__name__",
    "    __berta_error__ = traceback.format_exc()",
    "    import sys as _s",
    "    _tb = _s.exc_info()[2]",
    "    while _tb and _tb.tb_next:",
    "        _tb = _tb.tb_next",
    "    __berta_error_line__ = _tb.tb_lineno if _tb else (getattr(__e__, 'lineno', 0) or 0)",
    "    if __berta_error_type__ == 'SyntaxError' and hasattr(__e__, 'lineno') and __e__.lineno:",
    "        __berta_error_line__ = __e__.lineno",
  ].join("\n");

  try {
    pyodide.runPython(wrapper);
  } catch (e) {
    // This should almost never happen now, but just in case
    outputEl.innerHTML = "<span class=\"error\">" + escapeHtml(String(e)) + "</span>";
    statusText.textContent = "Error";
    statusText.style.color = "#cc0000";
    return;
  }

  var stdout = pyodide.runPython("sys.stdout.getvalue()");
  var stderr = pyodide.runPython("sys.stderr.getvalue()");
  var errorTraceback = pyodide.runPython("__berta_error__");
  var errorType = pyodide.runPython("__berta_error_type__");
  var errorLine = pyodide.runPython("__berta_error_line__");

  if (errorTraceback) {
    var explanation = getErrorExplanation(errorType, errorTraceback);

    var tbLines = errorTraceback.trim().split("\n");
    var lastLine = tbLines[tbLines.length - 1];

    var display = "";
    if (tbLines.length > 2) {
      display = "<span style=\"color:#aa6666\">" + escapeHtml(errorTraceback) + "</span>" +
        "\n<span style=\"color:#ff4444;font-weight:bold\">" + escapeHtml(lastLine) + "</span>";
    } else {
      display = "<span class=\"error\">" + escapeHtml(errorTraceback) + "</span>";
    }

    var preOutput = "";
    if (stdout) {
      preOutput = escapeHtml(stdout) + "\n\n";
    }

    var lineInfo = "";
    if (errorLine > 0) {
      lineInfo = "<div style=\"color:#ffcc00;margin-top:6px;\">Line " + errorLine + " in your code.</div>";
      highlightErrorLine(errorLine);
    }

    outputEl.innerHTML = preOutput + display + lineInfo +
      "<div class=\"error-explain\">EXPLANATION: " + escapeHtml(explanation) + "</div>";
    statusText.textContent = errorType + " (line " + errorLine + ")";
    statusText.style.color = "#cc0000";
  } else {
    var result = "";
    if (stdout) result += stdout;
    if (stderr) result += stderr;
    if (!result) result = "(No output. Add print() statements to see results.)";

    outputEl.innerHTML = escapeHtml(result);
    statusText.textContent = "Done";
    statusText.style.color = "#006600";
  }
}

function clearOutput() {
  document.getElementById("output").textContent = "Output cleared.";
}

function resetEditor() {
  document.getElementById("codeEditor").value = "# Welcome to the Berta Playground!\n\nprint(\"Hello from Berta Chapters!\")\nprint(\"2 + 2 =\", 2 + 2)\n";
  clearOutput();
}

function loadExercise() {
  var select = document.getElementById("exerciseSelect");
  var key = select.value;
  if (key && EXERCISES[key]) {
    document.getElementById("codeEditor").value = EXERCISES[key];
    clearOutput();
  }
}

// Keyboard shortcut: Ctrl+Enter to run
document.addEventListener("keydown", function(e) {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    e.preventDefault();
    runCode();
  }
});

// Tab key in editor inserts spaces; typing clears error highlight
document.addEventListener("DOMContentLoaded", function() {
  var editor = document.getElementById("codeEditor");
  if (editor) {
    editor.addEventListener("keydown", function(e) {
      if (e.key === "Tab") {
        e.preventDefault();
        var start = this.selectionStart;
        var end = this.selectionEnd;
        this.value = this.value.substring(0, start) + "    " + this.value.substring(end);
        this.selectionStart = this.selectionEnd = start + 4;
      }
    });
    editor.addEventListener("input", clearErrorHighlight);
  }
});

// Error line highlighting in the editor
function highlightErrorLine(lineNum) {
  var editor = document.getElementById("codeEditor");
  var lines = editor.value.split("\n");
  if (lineNum < 1 || lineNum > lines.length) return;

  // Replace textarea with a div overlay approach:
  // We'll set the textarea background and use a highlight div
  var highlightDiv = document.getElementById("editorHighlight");
  if (!highlightDiv) {
    highlightDiv = document.createElement("div");
    highlightDiv.id = "editorHighlight";
    highlightDiv.style.cssText = "position:absolute;left:0;right:0;height:1.3em;background:rgba(204,0,0,0.25);border-left:3px solid #cc0000;pointer-events:none;z-index:1;";
    var wrapper = document.getElementById("editorWrapper");
    if (wrapper) wrapper.appendChild(highlightDiv);
  }

  // Calculate position: each line is about 1.3em tall with 14px font
  var lineHeight = 18; // approximate px per line at 14px font
  var scrollTop = editor.scrollTop;
  var topOffset = (lineNum - 1) * lineHeight - scrollTop;
  highlightDiv.style.top = topOffset + "px";
  highlightDiv.style.display = "block";

  // Also select the error line in the textarea
  var start = 0;
  for (var i = 0; i < lineNum - 1; i++) {
    start += lines[i].length + 1;
  }
  var end = start + lines[lineNum - 1].length;
  editor.focus();
  editor.setSelectionRange(start, end);

  // Scroll the editor to show the error line
  var editorHeight = editor.clientHeight;
  var targetScroll = (lineNum - 1) * lineHeight - editorHeight / 3;
  if (targetScroll > 0) editor.scrollTop = targetScroll;
}

function clearErrorHighlight() {
  var highlightDiv = document.getElementById("editorHighlight");
  if (highlightDiv) highlightDiv.style.display = "none";
}

// Initialize after Pyodide script has loaded
initPyodide();
</script>
