# Complete AI/ML Engineer Learning Guide
### From Zero to Production — Everything You Need to Know

---

## Table of Contents

1. [Math Foundations](#1-math-foundations)
2. [Programming & Tools](#2-programming--tools)
3. [Machine Learning Fundamentals](#3-machine-learning-fundamentals)
4. [Deep Learning](#4-deep-learning)
5. [Natural Language Processing (NLP)](#5-natural-language-processing-nlp)
6. [Transformers & Attention](#6-transformers--attention)
7. [Large Language Models (LLMs)](#7-large-language-models-llms)
8. [Prompt Engineering](#8-prompt-engineering)
9. [RAG (Retrieval-Augmented Generation)](#9-rag-retrieval-augmented-generation)
10. [Vector Databases](#10-vector-databases)
11. [LangChain & LangGraph](#11-langchain--langgraph)
12. [MCP (Model Context Protocol)](#12-mcp-model-context-protocol)
13. [Fine-Tuning & PEFT](#13-fine-tuning--peft)
14. [MLOps & Deployment](#14-mlops--deployment)
15. [Production Best Practices](#15-production-best-practices)
16. [Capstone Projects](#16-capstone-projects)

---

# 1. Math Foundations

## 1.1 Linear Algebra

Linear algebra is the backbone of ML. Every data point, weight, and transformation lives in vector/matrix space.

### Vectors

A vector is an ordered list of numbers. Think of it as a point in space.

```
v = [3, 4]        # 2D vector
v = [1, 2, 3]     # 3D vector

Magnitude (length) of v = sqrt(3² + 4²) = sqrt(9+16) = 5
```

**Key Operations:**

```
Vector Addition:        Scalar Multiplication:
[1,2] + [3,4] = [4,6]  3 × [1,2] = [3,6]

Dot Product (measures similarity):
[1,2,3] · [4,5,6] = 1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32

If dot product = 0  → vectors are perpendicular (orthogonal)
If dot product > 0  → vectors point in similar direction
If dot product < 0  → vectors point in opposite directions
```

### Matrices

A matrix is a 2D grid of numbers. In ML, data is stored as matrices.

```
Data Matrix (4 samples, 3 features each):

        Feature1  Feature2  Feature3
Row 1  [  1.2,     3.4,     0.5  ]   ← sample 1
Row 2  [  2.1,     1.8,     0.9  ]   ← sample 2
Row 3  [  0.7,     2.9,     1.1  ]   ← sample 3
Row 4  [  3.5,     0.2,     0.3  ]   ← sample 4

Shape: (4, 3)  →  4 rows, 3 columns
```

**Matrix Multiplication (core of neural networks):**

```
A (2×3)  ×  B (3×2)  =  C (2×2)

[1, 2, 3]   [7, 8]     [1×7+2×9+3×11,  1×8+2×10+3×12]   [58,  64]
[4, 5, 6] × [9, 10]  = [4×7+5×9+6×11,  4×8+5×10+6×12] = [139, 154]
            [11,12]

Inner dimensions must match: (2×3) × (3×2) → (2×2) ✓
```

**Why this matters for Neural Networks:**

```
Input Data (batch of 3 samples, 4 features)     Weights (4 neurons, 3 outputs)
X = [[0.2, 0.5, 0.1, 0.8],                       W = [[0.1, 0.3, 0.5],
     [0.9, 0.1, 0.7, 0.3],                            [0.2, 0.4, 0.6],
     [0.4, 0.6, 0.3, 0.9]]                            [0.3, 0.1, 0.7],
                                                       [0.6, 0.2, 0.4]]

Output = X × W  (matrix multiplication)
Shape: (3×4) × (4×3) = (3×3)  ← 3 samples, 3 output values each

This single matrix multiply = an entire layer of a neural network!
```

### Eigenvalues & Eigenvectors

Used in PCA (dimensionality reduction), covariance analysis.

```
For matrix A, eigenvector v satisfies:  A × v = λ × v

Where λ (lambda) = eigenvalue (scalar)

Interpretation:
  - Eigenvector = direction that doesn't change under transformation
  - Eigenvalue  = how much it stretches/shrinks in that direction

PCA uses eigenvalues to find which directions have most variance
(most "information") in your data.
```

---

## 1.2 Calculus

Calculus is how neural networks **learn**. Gradient descent relies entirely on derivatives.

### Derivatives

The derivative tells you the **rate of change** of a function.

```
f(x) = x²
f'(x) = 2x

At x = 3: f'(3) = 6  → function is increasing at rate 6
At x = -2: f'(-2) = -4 → function is decreasing at rate 4

Visual:
    f(x) = x²
    |      /
    |     /  ← slope at x=3 is 6
    |    /
    |   /
    |  /
    | /
    |/_________ x
   0  1  2  3
```

### Partial Derivatives (Multivariable Calculus)

Neural networks have many parameters, so we use **partial derivatives**.

```
f(x, y) = x² + 3xy + y²

∂f/∂x = 2x + 3y    (derivative with respect to x, treating y as constant)
∂f/∂y = 3x + 2y    (derivative with respect to y, treating x as constant)

At point (1, 2):
∂f/∂x = 2(1) + 3(2) = 8    → f increases by 8 per unit increase in x
∂f/∂y = 3(1) + 2(2) = 7    → f increases by 7 per unit increase in y
```

### Gradient Descent (The Learning Algorithm)

```
Goal: Find the parameter θ that minimizes Loss(θ)

Algorithm:
1. Start with random θ
2. Compute gradient: ∇Loss = ∂Loss/∂θ
3. Update: θ_new = θ_old - learning_rate × gradient
4. Repeat until convergence

Visual:
    Loss
    |  \
    |   \  ← gradient points uphill
    |    \
    |     \_____  ← we step DOWNHILL (opposite of gradient)
    |           \____
    |                \_______  ← minimum (where we want to be)
    |__________________________ θ (parameter)

Learning Rate:
  - Too large: overshoot, oscillate, diverge
  - Too small: converge very slowly
  - Just right: smooth convergence to minimum
```

### Chain Rule (Backpropagation)

```
If y = f(g(x)), then dy/dx = f'(g(x)) × g'(x)

Neural Network Example:
  Input x → Linear(z = wx + b) → Activation(a = relu(z)) → Loss

  ∂Loss/∂w = ∂Loss/∂a × ∂a/∂z × ∂z/∂w

  This "chaining" of derivatives backwards through the network = BACKPROPAGATION

  Layer 4 (Loss) → Layer 3 → Layer 2 → Layer 1 → Input
       ←←←←←←←←←←← gradients flow backwards ←←←←←←←←←←←←
```

---

## 1.3 Probability & Statistics

### Probability Basics

```
P(A) = number of favorable outcomes / total outcomes

P(heads) = 1/2 = 0.5
P(rolling a 6 on a die) = 1/6

Joint Probability:
P(A and B) = P(A) × P(B)     [if A and B are independent]

Conditional Probability:
P(A|B) = P(A and B) / P(B)   [probability of A given B happened]
```

### Bayes' Theorem (Foundation of Naive Bayes, Bayesian ML)

```
P(A|B) = P(B|A) × P(A) / P(B)

Example - Medical Test:
  P(Disease) = 0.01          (1% prevalence)
  P(Positive|Disease) = 0.99 (test sensitivity)
  P(Positive|No Disease) = 0.05 (false positive rate)

  P(Disease|Positive) = 0.99 × 0.01 / P(Positive)

  P(Positive) = 0.99×0.01 + 0.05×0.99 = 0.0594

  P(Disease|Positive) = 0.0099 / 0.0594 = 0.167 = 16.7%

  Even with a positive test, only 16.7% chance of disease!
  This is why prior probability matters enormously.
```

### Distributions

```
Normal (Gaussian) Distribution:
          ┌──────────┐
         /            \
        /              \
       /                \
      /                  \
─────/────────────────────\─────
   -3σ  -2σ  -1σ  μ  1σ  2σ  3σ

68% of data within ±1σ
95% of data within ±2σ
99.7% of data within ±3σ

Why important:
  - Weights initialized from normal distribution
  - Assume errors are normally distributed
  - Batch normalization uses this
```

### Key Statistical Concepts

```
Mean (Average):      μ = (1/n) × Σ xᵢ
Variance:            σ² = (1/n) × Σ (xᵢ - μ)²
Standard Deviation:  σ = √variance

Covariance: How two variables change together
  Cov(X,Y) = E[(X-μₓ)(Y-μᵧ)]

Correlation: Normalized covariance (-1 to 1)
  ρ = Cov(X,Y) / (σₓ × σᵧ)

  ρ = 1  → perfect positive correlation
  ρ = 0  → no correlation
  ρ = -1 → perfect negative correlation
```

---

## 1.4 Optimization

### Gradient Descent Variants

```
Batch Gradient Descent:
  - Uses ALL training data per update
  - Stable but slow
  - Memory intensive

Stochastic Gradient Descent (SGD):
  - Uses ONE random sample per update
  - Fast but noisy
  - Can escape local minima

Mini-Batch Gradient Descent (most common):
  - Uses a batch (32, 64, 128 samples) per update
  - Balance of speed and stability
  - GPU-friendly (parallel computation)

┌──────────────────────────────────────────────────┐
│           Batch Size Comparison                   │
├──────────────────────────────────────────────────┤
│ Batch=1    │ ■■■■■■■■■■■■■■■■■■■ (noisy path)   │
│ Batch=32   │ ■■■■■■■■■■■■■ (smooth path)         │
│ Batch=ALL  │ ■■■■■■■■■ (very smooth, slow step)  │
└──────────────────────────────────────────────────┘
```

### Learning Rate Scheduling

```
Constant:        lr = 0.01 (always)
Step Decay:      lr = lr × 0.1 every N epochs
Cosine Annealing: lr follows cosine curve → smooth decay
Warmup:          lr starts low, increases, then decreases

Loss
|\
| \         ← good: loss decreases smoothly
|  \___
|      \___
|          \____
|_______________ Epochs

Bad - lr too high:       Bad - lr too low:
|\/\  /\  /\            |\
|  \/  \/  \/            | \___
|                        |     \___
| (oscillating)          |         (barely moving)
```

---

# 2. Programming & Tools

## 2.1 Python for ML

### Essential Libraries

```
NumPy:       Numerical computing, arrays, linear algebra
Pandas:      Data manipulation, DataFrames
Matplotlib:  Plotting, visualization
Scikit-learn: Classical ML algorithms
PyTorch:     Deep learning framework (research & production)
TensorFlow:  Deep learning framework (production-focused)
Hugging Face: Pre-trained models, datasets, tokenizers
```

### NumPy Basics

```python
import numpy as np

# Arrays
a = np.array([1, 2, 3])        # 1D array (vector)
b = np.array([[1,2],[3,4]])     # 2D array (matrix)

# Operations (vectorized - fast!)
a + 2          # [3, 4, 5]
a * 3          # [3, 6, 9]
np.dot(a, a)   # 14 (dot product)

# Matrix multiply
A = np.random.randn(3, 4)   # 3×4 random normal
B = np.random.randn(4, 2)   # 4×2 random normal
C = A @ B                    # (3×4) × (4×2) = (3×2)
```

### Pandas Basics

```python
import pandas as pd

# DataFrame
df = pd.DataFrame({
    'age': [25, 30, 35],
    'salary': [50000, 60000, 70000],
    'department': ['eng', 'sales', 'eng']
})

df.describe()          # summary statistics
df[df['age'] > 28]     # filter rows
df.groupby('department').mean()  # group and aggregate
```

## 2.2 Git & Version Control

```bash
git init                    # initialize repo
git add .                   # stage all changes
git commit -m "message"     # commit
git branch feature-x        # create branch
git checkout feature-x      # switch branch
git merge feature-x         # merge branch
git push origin main        # push to remote
```

## 2.3 Docker

```dockerfile
# Dockerfile for ML model
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t my-model .
docker run -p 8000:8000 my-model
```

---

# 3. Machine Learning Fundamentals

## 3.1 What is Machine Learning?

```
Traditional Programming:
  Input (data) + Rules → Output
  Example: if temperature > 30: "Hot"

Machine Learning:
  Input (data) + Output → Rules (model)
  Example: learn from examples what "Hot" looks like

  ┌─────────┐      ┌──────────────┐      ┌─────────┐
  │  Data   │─────→│   ML Model   │─────→│Predictions│
  └─────────┘      └──────────────┘      └─────────┘
                        ↑
                   Training on
                    labeled data
```

## 3.2 Supervised Learning

The model learns from **labeled data** (input → correct output pairs).

### Classification (predict categories)

```
Input: Features (X)    Output: Label (y)

Email → [word frequencies] → Spam / Not Spam
Image → [pixel values]   → Cat / Dog / Bird
Text  → [tokens]         → Positive / Negative

Logistic Regression (binary classification):
  z = w₁x₁ + w₂x₂ + ... + b     (linear combination)
  p = sigmoid(z) = 1/(1 + e⁻ᶻ)  (squash to 0-1)

  sigmoid:
  1 |            ___________
    |          /
  0.5| · · · ·/· · · · · · ·
    |       /
  0 |_____/
    ───────────────────── z
   -5    0    5

  p > 0.5 → class 1
  p ≤ 0.5 → class 0

Decision Tree:
  ┌──────────────┐
  │   Age > 30?  │
  └──────┬───────┘
     Yes │      No
  ┌──────▼──────┐  ┌──────▼──────┐
  │Income>50k? │  │  Student?   │
  └──────┬──────┘  └──────┬──────┘
   Yes   │  No    Yes    │   No
  ┌──────▼───┐ ┌────▼────┐ ┌──▼───┐
  │ BUY      │ │ MAYBE   │ │ DON'T│
  └──────────┘ └─────────┘ └──────┘

Random Forest: Many decision trees + voting
  Tree 1: BUY     ─┐
  Tree 2: DON'T    ├──→ Majority Vote → BUY
  Tree 3: BUY     ─┘
```

### Regression (predict continuous values)

```
Input: Features (X)    Output: Value (y)

House → [sqft, beds, location] → Price ($350,000)
Stock → [past prices, volume]  → Future price

Linear Regression:
  ŷ = w₁x₁ + w₂x₂ + ... + wₙxₙ + b

  Minimize: MSE = (1/n) × Σ(yᵢ - ŷᵢ)²

  Visual (best fit line):
  Price ($)
  |        *  *
  |      *  /
  |    *  /
  |   * /
  |  */
  | /
  |/_______________ Sqft
```

## 3.3 Unsupervised Learning

The model finds **patterns in unlabeled data**.

### Clustering

```
K-Means Clustering:
  1. Choose K clusters
  2. Randomly place K centroids
  3. Assign each point to nearest centroid
  4. Recompute centroids as mean of assigned points
  5. Repeat until convergence

  Step 1:           Step 2:           Step 3:
  *  *             *  *             *  *
    *    *           *    *           *  *
  *    *           *    *           *    *
  *  *             *  *             *  *
  (random)        (assigned)     (centroids updated)

DBSCAN: Density-based clustering (finds arbitrary shapes)
  - Good for: clusters of different shapes
  - Handles outliers as "noise"
```

### Dimensionality Reduction

```
PCA (Principal Component Analysis):
  - Find directions of maximum variance
  - Project data onto those directions

  3D Data → PCA → 2D (keeps most variance)

  Original (3D):          After PCA (2D):
    *  *                   *  *
      *    *      →         *    *
    *    *                 *    *
    *  *                   *  *

  Why: reduces computation, removes noise, enables visualization
```

## 3.4 Reinforcement Learning

An agent learns by **interacting with an environment** and receiving rewards.

```
┌─────────────────────────────────────────────┐
│                ENVIRONMENT                   │
│                                              │
│    State → Agent → Action → Reward          │
│                                              │
│    ┌─────┐    ┌─────────┐                   │
│    │Agent│───→│ Action  │───→ New State     │
│    │     │←───│  Space  │←─── Reward        │
│    └─────┘    └─────────┘                   │
│                                              │
│    Goal: Learn policy π that maximizes       │
│          cumulative reward                   │
└─────────────────────────────────────────────┘

Example: Game Playing
  State:   Screen pixels
  Action:  Move left/right/jump
  Reward:  +10 points, -1 falling, +100 winning

  Agent learns: "When I see gap, jump" (through trial and error)
```

## 3.5 Model Evaluation

### Classification Metrics

```
Confusion Matrix:
                    Predicted
                  Pos      Neg
  Actual  Pos  [ TP    │  FN  ]   True Pos / False Neg
          Neg  [ FP    │  TN  ]   False Pos / True Neg

Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)    → "Of predicted positives, how many correct?"
Recall    = TP / (TP + FN)    → "Of actual positives, how many found?"
F1 Score  = 2 × (Precision × Recall) / (Precision + Recall)

When to use what:
  Balanced classes    → Accuracy
  Imbalanced classes  → F1, Precision, Recall
  Medical diagnosis   → High Recall (don't miss sick patients)
  Spam detection      → High Precision (don't mark good email as spam)
```

### Regression Metrics

```
MSE  = (1/n) × Σ(yᵢ - ŷᵢ)²     (Mean Squared Error)
RMSE = √MSE                       (Root MSE, same units as y)
MAE  = (1/n) × Σ|yᵢ - ŷᵢ|       (Mean Absolute Error)
R²   = 1 - (SS_res / SS_tot)      (how much variance explained)

R² = 1.0  → perfect predictions
R² = 0.0  → model is as good as predicting the mean
R² < 0    → model is worse than predicting the mean
```

### Cross-Validation

```
K-Fold Cross-Validation (K=5):

Dataset split into 5 folds:

Fold 1: [TEST] [Train] [Train] [Train] [Train]  → Score₁
Fold 2: [Train] [TEST] [Train] [Train] [Train]  → Score₂
Fold 3: [Train] [Train] [TEST] [Train] [Train]  → Score₃
Fold 4: [Train] [Train] [Train] [TEST] [Train]  → Score₄
Fold 5: [Train] [Train] [Train] [Train] [TEST]  → Score₅

Final Score = avg(Score₁, Score₂, Score₃, Score₄, Score₅)

Why: Every sample gets to be in test set once
     More reliable estimate of model performance
```

## 3.6 Feature Engineering & Data Preprocessing

```
Feature Scaling:
  Min-Max:   x_scaled = (x - min) / (max - min)   → [0, 1]
  Standard:  x_scaled = (x - mean) / std            → mean=0, std=1

Why scale? Features on different scales confuse gradient descent
  Age: 20-70      vs  Salary: 30000-200000
  Without scaling, salary dominates

Handling Missing Data:
  - Drop rows (loses data)
  - Fill with mean/median/mode
  - Use model to predict missing values (imputation)

Encoding Categorical Data:
  One-Hot Encoding:
    Color: Red    → [1, 0, 0]
    Color: Blue   → [0, 1, 0]
    Color: Green  → [0, 0, 1]

Feature Selection:
  - Correlation analysis (remove highly correlated features)
  - Feature importance (from tree models)
  - Recursive feature elimination
```

---

# 4. Deep Learning

## 4.1 Neural Networks

### The Perceptron (Building Block)

```
          Inputs        Weights       Sum      Activation     Output
         ┌─────┐
  x₁ ───┤w₁   │───┐
         └─────┘   │
         ┌─────┐   │    ┌───────┐    ┌─────┐    ┌─────┐
  x₂ ───┤w₂   ├───┼───→│ Σ wᵢxᵢ│───→│ f(z)│───→│  y  │
         └─────┘   │    └───────┘    └─────┘    └─────┘
         ┌─────┐   │         ↑
  x₃ ───┤w₃   ├───┘         +
         └─────┘           bias (b)

  z = w₁x₁ + w₂x₂ + w₃x₃ + b
  y = activation(z)
```

### Activation Functions

```
Sigmoid:              ReLU:                Tanh:
  1 │    _____         │  /                1 │    _____
    │   /              │ /                   │   /
0.5 │··/······         │/                    │··/······
    │ /               0│                    0│
  0 │/                 │                    -1│\_______
    ─────────          ──────                 ──────────
   -5  0  5           -5  0  5              -5  0  5

Range: (0,1)        Range: [0, ∞)        Range: (-1, 1)

GELU (used in Transformers):
  1 │      ___________
    │     /
    │    /
    │   /
  0 │__/
    ──────────────
   -3  0  3

Modern default: ReLU (hidden layers), Softmax/Sigmoid (output)
Transformer default: GELU
```

### Multi-Layer Neural Network

```
Input Layer     Hidden Layers      Output Layer

  x₁ ─────┬─── h₁₁ ───┬─── h₂₁ ───┬─── y₁
           │           │           │
  x₂ ─────┼─── h₁₂ ───┼─── h₂₂ ───┼─── y₂
           │           │           │
  x₃ ─────┼─── h₁₃ ───┼─── h₂₃ ───┼─── y₃
           │           │           │
  x₄ ─────┴─── h₁₄ ───┴─── h₂₄ ───┴─── y₄
               ↑           ↑
           Layer 1      Layer 2
           (64 neurons) (32 neurons)

Each arrow = weight (learned parameter)
Total params = (4×64 + 64) + (64×32 + 32) + (32×4 + 4)
             = 320 + 2080 + 132 = 2532 parameters
```

## 4.2 Training a Neural Network

```
┌─────────────────────────────────────────────────────┐
│                TRAINING LOOP                         │
│                                                      │
│  For each epoch:                                     │
│    For each batch:                                   │
│                                                      │
│    1. Forward Pass                                   │
│       input → [layers] → prediction                  │
│                                                      │
│    2. Compute Loss                                  │
│       loss = LossFunction(prediction, true_label)    │
│                                                      │
│    3. Backward Pass (Backpropagation)               │
│       Compute gradients of loss w.r.t. each weight   │
│       ∂Loss/∂w for every weight w                    │
│                                                      │
│    4. Update Weights                                │
│       w = w - learning_rate × ∂Loss/∂w              │
│                                                      │
│    5. Zero Gradients                                │
│       optimizer.zero_grad()                          │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### Loss Functions

```
Classification:
  Binary Cross-Entropy:  L = -[y×log(ŷ) + (1-y)×log(1-ŷ)]
  Cross-Entropy:         L = -Σ yᵢ × log(ŷᵢ)  (multi-class)

Regression:
  MSE:  L = (1/n) × Σ(y - ŷ)²
  MAE:  L = (1/n) × Σ|y - ŷ|

Why Cross-Entropy for classification?
  If true label = 1 and prediction = 0.99 → L ≈ 0.01 (good)
  If true label = 1 and prediction = 0.01 → L ≈ 4.6  (penalized heavily)
```

### Optimizers

```
SGD:           w = w - lr × gradient
  Problem: oscillates in ravines

SGD + Momentum: v = β×v + gradient; w = w - lr × v
  Benefit: accumulates velocity, smooths updates

Adam (Adaptive Moment Estimation):  ← MOST COMMON
  Combines Momentum + RMSProp
  Adapts learning rate per parameter
  m = β₁×m + (1-β₁)×grad      (first moment - mean)
  v = β₂×v + (1-β₂)×grad²     (second moment - variance)
  w = w - lr × m̂ / (√v̂ + ε)

  Default: β₁=0.9, β₂=0.999, ε=1e-8
```

## 4.3 Convolutional Neural Networks (CNNs)

CNNs process **grid-structured data** (images, audio spectrograms).

```
Convolution Operation:
  Input (5×5):         Filter (3×3):        Output (3×3):
  ┌─────────────┐      ┌──────────┐         ┌──────────┐
  │ 1  0  1  0  1│      │ 1  0  1  │         │ 4  3  4  │
  │ 0  1  0  1  0│  *   │ 0  1  0  │    =    │ 2  4  3  │
  │ 1  0  1  0  1│      │ 1  0  1  │         │ 4  3  4  │
  │ 0  1  0  1  0│      └──────────┘         └──────────┘
  │ 1  0  1  0  1│
  └─────────────┘

  Filter slides across input, computing dot product at each position.
  Different filters detect different features (edges, textures, shapes).
```

### CNN Architecture

```
Image → [Conv → ReLU → Pool] × N → [Flatten] → [FC] → Output

Conv Layer:    Extracts features (edges, textures, patterns)
ReLU:          Adds non-linearity
Pooling:       Reduces spatial dimensions (max/average)

Typical CNN:
┌────────┐   ┌────────┐   ┌────────┐   ┌───────┐   ┌────┐
│ Input  │──→│ Conv2D │──→│  Max   │──→│ Conv2D│──→│Max │──→ ...
│224×224 │   │ 32     │   │ Pool   │   │ 64    │   │Pool│
│  ×3    │   │ filters│   │2×2     │   │filters│   │2×2 │
└────────┘   └────────┘   └────────┘   └───────┘   └────┘
                                                      │
                                                  ┌───▼───┐
                                                  │FC +   │
                                                  │Softmax│
                                                  └───────┘

Filter Visualization:
  Layer 1: detects edges (horizontal, vertical, diagonal)
  Layer 2: detects textures (grids, waves, corners)
  Layer 3: detects parts (eyes, wheels, windows)
  Layer 4: detects objects (faces, cars, buildings)
```

### Famous CNN Architectures

```
LeNet-5 (1998):     First practical CNN. Digit recognition.
AlexNet (2012):      Won ImageNet. ReLU, Dropout, GPU training.
VGG-16 (2014):       16 layers, 3×3 filters everywhere.
GoogLeNet (2014):    Inception modules (parallel filter sizes).
ResNet (2015):       Skip connections. Enabled 152+ layer networks.
  Skip Connection:   output = F(x) + x  (residual learning)
                     Allows gradients to flow through deep networks.
EfficientNet (2019): Compound scaling (depth × width × resolution).
```

## 4.4 Recurrent Neural Networks (RNNs)

RNNs process **sequential data** (text, time series, audio).

```
RNN Unrolled Through Time:

  x₁       x₂       x₃       x₄
  │        │        │        │
  ▼        ▼        ▼        ▼
┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐
│ RNN │→ │ RNN │→ │ RNN │→ │ RNN │→ Output
│     │  │     │  │     │  │     │
└─────┘  └─────┘  └─────┘  └─────┘
  h₁       h₂       h₃       h₄

At each timestep:
  hₜ = tanh(Wₓₕ × xₜ + Wₕₕ × hₜ₋₁ + bₕ)
  yₜ = Wₕᵧ × hₜ + bᵧ

  hₜ = "memory" that carries information from past to future
```

### LSTM (Long Short-Term Memory)

```
LSTM Cell:
                    ┌──────────────────────────────────┐
  hₜ₋₁ ────────────→│         Cell State (Cₜ)          │────────→ Cₜ
                    │    ══════════════════════════     │
  xₜ ──→ ┌────────┐│         │                    │    │
         │ Forget │─┼──→ × ──┤                    │    │
         │ Gate   │ │         │                    │    │
         └────────┘ │    ┌────▼────┐               │    │
         ┌────────┐ │    │ tanh    │  ← candidate  │    │
         │ Input  │─┼──→ │ new    │   values       │    │
         │ Gate   │ │    │ Cₜ~    │               │    │
         └────────┘ │    └────┬────┘               │    │
                    │         │                    │    │
         ┌────────┐ │    ┌────▼────┐               │    │
         │ Output │─┼──→ │ tanh    │──→ × ────────┼──→ hₜ
         │ Gate   │ │    │ Cₜ      │    │         │
         └────────┘ │    └─────────┘    │         │
                    └────────────────────┼─────────┘
                                         └──→ output

Forget Gate: what to throw away from cell state
Input Gate:  what new information to store
Output Gate: what to output based on cell state

Why LSTM over vanilla RNN?
  - Solves vanishing gradient problem
  - Can remember patterns over 100+ timesteps
  - RNN forgets after ~10-20 steps
```

### GRU (Gated Recurrent Unit)

Simplified LSTM — fewer parameters, similar performance.

```
GRU combines forget + input gates into "update gate"
  zₜ = sigmoid(Wz × [hₜ₋₁, xₜ])           # update gate
  rₜ = sigmoid(Wr × [hₜ₋₁, xₜ])           # reset gate
  h̃ₜ = tanh(W × [rₜ × hₜ₋₁, xₜ])          # candidate
  hₜ = (1 - zₜ) × hₜ₋₁ + zₜ × h̃ₜ         # interpolation
```

---

# 5. Natural Language Processing (NLP)

## 5.1 Text Preprocessing

```
Raw Text: "The quick brown fox jumps over the lazy dog!"

1. Tokenization (split into tokens):
   ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "!"]

2. Lowercasing:
   ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "!"]

3. Remove stop words:
   ["quick", "brown", "fox", "jumps", "lazy", "dog"]

4. Stemming/Lemmatization:
   Stemming:  "jumps" → "jump", "running" → "run"
   Lemmatization: "better" → "good", "ran" → "run" (knows word meaning)

5. Vectorization (convert to numbers):
   Bag of Words, TF-IDF, or Word Embeddings
```

## 5.2 Word Embeddings

### One-Hot Encoding (Naive approach)

```
Vocabulary: [cat, dog, fish, bird]

cat = [1, 0, 0, 0]
dog = [0, 1, 0, 0]
fish = [0, 0, 1, 0]

Problems:
  - Sparse (mostly zeros)
  - No semantic meaning (cat and dog equally similar to fish)
  - Fixed vocabulary size (can't handle new words)
```

### Word2Vec (2013, Google)

```
Idea: Words appearing in similar contexts have similar meanings

Two architectures:
  CBOW:    Predict center word from context
  Skip-gram: Predict context words from center word

"the cat sits on the mat"

Skip-gram training example:
  Center word: "sits"
  Context window = 2:
    Target pairs: (sits, cat), (sits, on), (sits, the), (sits, mat)

After training, similar words cluster in vector space:

         cat
        /
       /      dog
      /      /
  ───*──────────── * ← word2vec space
      \      \
       \      horse
        \
         mat

cat - dog ≈ horse - pony  (analogy preserved!)
king - man + woman ≈ queen  (famous analogy)
```

### GloVe (2014, Stanford)

```
Uses global word co-occurrence statistics

Build co-occurrence matrix X:
           the  cat  dog  mat
    the  [  0   5    3    8  ]
    cat  [  5   0    7    2  ]
    dog  [  3   7    0    1  ]
    mat  [  8   2    1    0  ]

Optimize: wᵢ · wⱼ + bᵢ + bⱼ = log(Xᵢⱼ)

Result: dense vectors capturing global statistics
```

### FastText (2016, Facebook)

```
Represents words as bags of character n-grams

"where" → <wh, whe, her, ere, re> + <where>

Advantages:
  - Handles out-of-vocabulary words
  - Handles misspellings
  - Works for morphologically rich languages
```

## 5.3 Text Classification

```
Sentiment Analysis Pipeline:

Text → Preprocess → Vectorize → Model → Label
"I love this movie!" → tokens → embeddings → classifier → Positive

Approaches:
1. Bag of Words + Logistic Regression (baseline)
2. TF-IDF + SVM (classic)
3. Word2Vec + LSTM (deep learning)
4. BERT fine-tuned (state-of-the-art)
```

## 5.4 Sequence-to-Sequence Tasks

```
Machine Translation:
  Input:  "I love you" (English)
  Output: "Je t'aime" (French)

  ┌──────────┐     ┌──────────┐
  │ Encoder  │────→│ Decoder  │
  │  (LSTM)  │     │  (LSTM)  │
  └──────────┘     └──────────┘
  "I love you"     "Je t'aime"
                   ↑ generated word by word

Text Summarization:
  Long article → Encoder → Summary

Question Answering:
  Context + Question → Model → Answer
```

---

# 6. Transformers & Attention

## 6.1 The Problem with RNNs

```
RNN Limitations:
  1. Sequential processing (can't parallelize)
  2. Long-range dependencies fade (vanishing gradients)
  3. Slow for long sequences

  RNN processes tokens one by one:
  "The" → "cat" → "sat" → "on" → "the" → "mat"
   t=1    t=2    t=3    t=4    t=5    t=6
   ↑ must wait for previous step

  Transformer processes all tokens simultaneously:
  "The cat sat on the mat"
   t=1  t=2  t=3  t=4  t=5  t=6
   ↑ all processed in parallel!
```

## 6.2 Self-Attention Mechanism

```
Core Idea: For each word, compute how much attention to pay to every other word.

"The cat sat on the mat because it was tired"
                                      ↑
                           "it" refers to "cat" (not "mat"!)

Attention(Q, K, V) = softmax(QKᵀ / √dₖ) × V

Where:
  Q (Query)  = "What am I looking for?"
  K (Key)    = "What do I contain?"
  V (Value)  = "What information do I provide?"

Step-by-step:
1. Create Q, K, V matrices from input embeddings
2. Compute attention scores: Q × Kᵀ
3. Scale by √dₖ (prevent large values)
4. Apply softmax (normalize to probabilities)
5. Multiply by V (weighted sum of values)

Visual:
          Q       K         Attention Scores
  "it"  [0.2]  [0.1,0.3,0.5,0.1,0.0,0.0,0.0,0.0]
         ↓
  Score: [0.1, 0.3, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0]
                              ↑ highest attention to "cat"
```

## 6.3 Multi-Head Attention

```
Instead of one attention, use multiple "heads" (parallel attention computations):

Head 1: focuses on syntax (subject-verb agreement)
Head 2: focuses on semantics (pronoun references)
Head 3: focuses on position (adjacent words)
...

MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) × Wᴼ

headᵢ = Attention(QWᵢᴷ, KWᵢᴷ, VWᵢⱽ)

Visual:
                    Input
                      │
         ┌────────┬───┴───┬────────┐
         ▼        ▼       ▼        ▼
      Head 1   Head 2  Head 3   Head 4
      (syntax) (semantics)(position)(rare)
         │        │       │        │
         └────────┴───┬───┴────────┘
                      ▼
                  Concatenate
                      │
                      ▼
                   Linear
                      │
                      ▼
                   Output

Each head learns different "perspectives" on the input.
```

## 6.4 Transformer Architecture

```
Full Transformer (Encoder-Decoder):

  ┌─────────────────────────────────────────────────┐
  │                   ENCODER                        │
  │  ┌───────────────────────────────────────────┐  │
  │  │  Input Embedding + Positional Encoding    │  │
  │  └─────────────────────┬─────────────────────┘  │
  │                        ↓                         │
  │  ┌───────────────────────────────────────────┐  │
  │  │       Multi-Head Self-Attention           │  │
  │  └─────────────────────┬─────────────────────┘  │
  │                        ↓                         │
  │  │              Add & Norm (residual)          │  │
  │                        ↓                         │
  │  ┌───────────────────────────────────────────┐  │
  │  │          Feed-Forward Network             │  │
  │  └─────────────────────┬─────────────────────┘  │
  │                        ↓                         │
  │  │              Add & Norm (residual)          │  │
  │                        ↓                         │
  │              × N layers (6 in original)          │
  └──────────────────────┬──────────────────────────┘
                         │
  ┌──────────────────────▼──────────────────────────┐
  │                   DECODER                        │
  │  ┌───────────────────────────────────────────┐  │
  │  │   Output Embedding + Positional Encoding  │  │
  │  └─────────────────────┬─────────────────────┘  │
  │                        ↓                         │
  │  ┌───────────────────────────────────────────┐  │
  │  │  Masked Multi-Head Self-Attention         │  │
  │  │  (can't look at future tokens!)           │  │
  │  └─────────────────────┬─────────────────────┘  │
  │                        ↓                         │
  │  ┌───────────────────────────────────────────┐  │
  │  │  Cross-Attention (over encoder output)    │  │
  │  └─────────────────────┬─────────────────────┘  │
  │                        ↓                         │
  │  ┌───────────────────────────────────────────┐  │
  │  │          Feed-Forward Network             │  │
  │  └─────────────────────┬─────────────────────┘  │
  │              × N layers (6 in original)          │
  └──────────────────────┬──────────────────────────┘
                         │
                         ▼
                    Linear + Softmax → Output Probabilities
```

### Positional Encoding

```
Transformers have no inherent notion of order.
Positional encoding adds position information to embeddings.

PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

Where:
  pos = position in sequence
  i   = dimension index
  d   = embedding dimension

Visual (positional encoding values):
Dim 0: ─/──/──/──/──/── (sine wave, low frequency)
Dim 1: /──/──/──/──/──\ (cosine wave, low frequency)
Dim 2: ╱╲╱╲╱╲╱╲╱╲╱╲╱╲ (sine wave, high frequency)
Dim 3: ╲╱╲╱╲╱╲╱╲╱╲╱╲╱ (cosine wave, high frequency)

Each position gets a UNIQUE encoding vector.
Model can learn to attend to relative positions.
```

---

# 7. Large Language Models (LLMs)

## 7.1 From Transformers to LLMs

```
Evolution:
  2017: Transformer (original paper, 65M params)
  2018: GPT-1 (117M params, decoder-only)
  2018: BERT (340M params, encoder-only)
  2019: GPT-2 (1.5B params)
  2020: GPT-3 (175B params) ← "Few-shot learning" breakthrough
  2022: ChatGPT (GPT-3.5 + RLHF)
  2023: GPT-4, Llama 2, Mistral
  2024: GPT-4o, Llama 3, Claude 3, Gemini
  2025: GPT-4.5, Claude 4, Gemini 2.5

Key insight: scale everything up
  More data + More parameters + More compute = emergent abilities
```

## 7.2 GPT Architecture (Decoder-Only Transformer)

```
GPT uses only the decoder part of the transformer:

  "The cat" → [Token Embeddings + Positional Encodings]
                    │
                    ▼
              ┌──────────────┐
              │ Masked       │  ← can only attend to
              │ Self-Attention│    PREVIOUS tokens
              └──────┬───────┘
                     ↓
              ┌──────────────┐
              │ Feed-Forward │
              └──────┬───────┘
                     ↓
              × N layers (GPT-3: 96 layers!)
                     │
                     ↓
              Linear → Softmax → next token probability

  "The cat" → P(sat)=0.4, P(jumps)=0.3, P(is)=0.15, ...
```

## 7.3 How LLMs Generate Text

```
Autoregressive Generation:

Step 1: "The cat"
  P(sat|The cat) = 0.4    → sample "sat"

Step 2: "The cat sat"
  P(on|The cat sat) = 0.6  → sample "on"

Step 3: "The cat sat on"
  P(the|The cat sat on) = 0.8 → sample "the"

Step 4: "The cat sat on the"
  P(mat|The cat sat on the) = 0.5 → sample "mat"

Step 5: "The cat sat on the mat"
  P(<EOS>) > 0.9 → stop

Temperature controls randomness:
  T = 0.0: always pick highest probability (deterministic)
  T = 0.7: balanced (recommended for most tasks)
  T = 1.0: sample from raw probabilities
  T = 2.0: very random (creative but incoherent)

Top-k sampling: only consider top k most likely tokens
Top-p (nucleus) sampling: consider smallest set with cumulative prob ≥ p
```

## 7.4 RLHF (Reinforcement Learning from Human Feedback)

```
How ChatGPT became "helpful" instead of just "text completion":

Step 1: Supervised Fine-Tuning (SFT)
  Collect human-written conversations
  Fine-tune GPT on these examples
  Result: model that tries to be helpful

Step 2: Reward Model Training
  Generate multiple responses to same prompt
  Humans rank responses from best to worst
  Train a reward model to predict human preferences

Step 3: PPO (Proximal Policy Optimization)
  Use reward model to optimize the LLM
  Model learns to generate responses that humans prefer

  ┌──────────┐    Response    ┌──────────────┐
  │   LLM    │───────────────→│ Reward Model │──→ Score
  └──────────┘                └──────────────┘
       ↑                            │
       │          ┌─────────┐       │
       └──────────┤   PPO   │←──────┘
                  │ Optimizer│
                  └─────────┘

DPO (Direct Preference Optimization):
  Newer, simpler alternative to RLHF
  Directly optimizes LLM on human preferences
  No separate reward model needed
```

## 7.5 Key LLM Concepts

### Context Window

```
Maximum number of tokens a model can process at once.

Model           Context Window    ~Word Count
GPT-3.5         4,096 tokens     ~3,000 words
GPT-4           8,192/32K/128K   ~6K-100K words
Claude 3        200K tokens      ~150K words
Gemini 1.5 Pro  1M tokens        ~750K words
Llama 3         8K/128K tokens

1 token ≈ 0.75 words (English)
1 token ≈ 3-4 characters
"The cat sat" → 3 tokens
```

### Tokenization

```
BPE (Byte Pair Encoding) - used by GPT models:

Raw text: "unbelievable"
→ Tokens: ["un", "believ", "able"]

Vocabulary built from training data:
  Start with individual characters
  Merge most frequent pairs iteratively

Common tokens have short IDs:
  "the" → token 262
  " and" → token 290

Rare tokens split into subwords:
  "antidisestablishmentarianism"
  → ["ant", "id", "is", "establish", "ment", "arian", "ism"]
```

### Emergent Abilities

Abilities that appear suddenly at certain model scales but not at smaller scales:

```
Model Size     │ Few-shot | CoT Reasoning | Translation
───────────────┼──────────┼───────────────┼─────────────
1B params      │ Poor     │ No            │ Poor
10B params     │ Fair     │ Barely        │ Fair
100B params    │ Good     │ Yes           │ Good
175B+ params   │ Excellent│ Strong        │ Excellent

These abilities are NOT present in smaller models, regardless of training.
```

---

# 8. Prompt Engineering

## 8.1 What is Prompt Engineering?

```
Designing inputs to LLMs to get desired outputs.

Bad prompt:  "Write about dogs"
Good prompt: "Write a 200-word informative article about the history 
              of Golden Retrievers, targeting first-time dog owners, 
              in a friendly but professional tone"

The difference in output quality is enormous.
```

## 8.2 Core Techniques

### Zero-Shot Prompting

```
No examples provided. Model relies on pre-trained knowledge.

Prompt: "Classify the sentiment: 'This movie was absolutely fantastic!'"
Output: Positive

When to use: Simple, well-defined tasks
```

### Few-Shot Prompting

```
Provide examples before the actual task.

Prompt:
Classify the sentiment of these reviews:

Review: "The food was terrible" → Negative
Review: "I loved every minute" → Positive
Review: "It was okay, nothing special" → Neutral

Review: "Best purchase I've ever made" → ???

Output: Positive

Why it works: Model learns the pattern from examples
Best practice: 3-5 diverse examples
```

### Chain-of-Thought (CoT) Prompting

```
Ask the model to show its reasoning step by step.

Without CoT:
  Q: "Roger has 5 tennis balls. He buys 2 more cans of 3. 
      How many does he have now?"
  A: 11 ✓ (but often fails harder problems)

With CoT:
  Q: Same question + "Let's think step by step."
  A: 
    "Roger starts with 5 balls.
     2 cans × 3 balls = 6 balls.
     5 + 6 = 11.
     Answer: 11"

Improves accuracy on math/logic from ~20% to ~90% on some benchmarks!
```

### Self-Consistency

```
Generate multiple reasoning paths, then take majority vote.

Prompt: "Solve this, showing your work." (run 5 times)

Path 1: "Step 1... Answer: 42"
Path 2: "Step 1... Answer: 45"  (wrong reasoning)
Path 3: "Step 1... Answer: 42"
Path 4: "Step 1... Answer: 42"
Path 5: "Step 1... Answer: 45"  (wrong reasoning)

Majority vote: 42 (3/5 agree) → Final answer: 42
```

## 8.3 Advanced Techniques

### ReAct (Reasoning + Acting)

```
Combine reasoning with tool use:

Thought: I need to find the population of France
Action: search("population of France 2024")
Observation: France population is 68.4 million (2024)
Thought: Now I need the population of Germany
Action: search("population of Germany 2024")
Observation: Germany population is 84.5 million (2024)
Thought: I can now calculate: 84.5 - 68.4 = 16.1 million
Answer: Germany has approximately 16.1 million more people than France
```

### Tree of Thoughts (ToT)

```
Explore multiple reasoning branches, evaluate, and backtrack:

                Start
              /       \
         Path A       Path B
         /    \          |
      A1      A2       B1 (dead end - backtrack)
      |        |
     A1a      A2a
      |        |
   [Best]    [Good]
   
Model evaluates each branch and chooses the most promising.
```

### Prompt Chaining

```
Break complex tasks into sequential simpler prompts:

Step 1: "Extract key facts from this article"
Step 2: "Based on these facts, create an outline"
Step 3: "Write each section of the outline"
Step 4: "Review and polish the complete document"
```

## 8.4 Prompt Engineering Best Practices

```
1. Be specific and detailed
   Bad:  "Write about AI"
   Good: "Write a 500-word technical blog post about transformer 
         architecture, explaining self-attention to software engineers 
         with no ML background"

2. Use delimiters to separate instructions from content
   """text to analyze"""
   --- 
   <doc>document</doc>

3. Specify output format
   "Respond in JSON format with keys: title, summary, keywords"
   "Return as a markdown table"

4. Provide context and role
   "You are an expert Python developer. Review this code for bugs:"

5. Iterate and refine
   Start simple → test → add constraints → test → refine

6. Temperature settings for different tasks:
   Factual/precise: T = 0.0 - 0.3
   Balanced:        T = 0.5 - 0.7
   Creative:        T = 0.8 - 1.2
```

---

# 9. RAG (Retrieval-Augmented Generation)

## 9.1 What is RAG?

```
RAG = Retrieve relevant documents from a knowledge base, 
      then use them as context for LLM generation.

Without RAG:                     With RAG:
  LLM relies on training data     LLM + retrieved knowledge
  (outdated, hallucinations)      (current, grounded, verifiable)

  "What's in the company           "Based on your internal docs,
   policy handbook?"               here's the policy on..."
  → LLM guesses                    → LLM cites actual policy
```

## 9.2 RAG Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG PIPELINE                               │
│                                                               │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │   User   │───→│ Query        │───→│ Vector Search     │  │
│  │  Query   │    │ Embedding    │    │ (find relevant    │  │
│  └──────────┘    └──────────────┘    │  documents)       │  │
│                                      └─────────┬─────────┘  │
│                                                │             │
│                                        ┌───────▼─────────┐  │
│                                        │ Top K Documents  │  │
│                                        │ (retrieved)      │  │
│                                        └───────┬─────────┘  │
│                                                │             │
│  ┌─────────────────────────────────────────────▼──────────┐ │
│  │                    LLM Generation                       │ │
│  │                                                         │ │
│  │  System: "Answer based on the provided context."        │ │
│  │                                                         │ │
│  │  Context: [Retrieved Document 1]                        │ │
│  │           [Retrieved Document 2]                        │ │
│  │           [Retrieved Document 3]                        │ │
│  │                                                         │ │
│  │  User Question: "What is the return policy?"            │ │
│  │                                                         │ │
│  │  → LLM generates answer grounded in retrieved docs      │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 9.3 Document Ingestion Pipeline

```
Step 1: Load Documents
  Sources: PDFs, Word docs, HTML, databases, APIs
  Tools: LangChain document loaders, Unstructured, PyPDF

Step 2: Split into Chunks
  Why: LLMs have context limits; small chunks improve retrieval precision
  
  Chunking strategies:
  ┌──────────────────────────────────────────┐
  │ Fixed-size: every 500 tokens             │
  │ Sliding window: 500 tokens, 50 overlap   │
  │ Recursive: split by paragraphs/sections  │
  │ Semantic: split at topic boundaries      │
  │ Document-aware: split at headings        │
  └──────────────────────────────────────────┘

  Document:
  ┌─────────────────────────────────┐
  │  Chapter 1: Introduction        │
  │  This is the introduction...    │
  │  It covers the basics...        │
  ├─────────────────────────────────┤ ← natural split point
  │  Chapter 2: Methods             │
  │  The methods used were...       │
  │  Including...                   │
  └─────────────────────────────────┘

Step 3: Create Embeddings
  Each chunk → vector (e.g., 1536 dimensions for OpenAI embeddings)
  
  "Return policy: Items can be returned within 30 days..."
  → [0.023, -0.156, 0.089, ..., 0.234]  (dense vector)

Step 4: Store in Vector Database
  Vectors + metadata → vector DB
  Metadata: source, page, timestamp, category
```

## 9.4 Retrieval Strategies

```
1. Similarity Search (basic)
   - Compute cosine similarity between query and all chunks
   - Return top K most similar
   
2. Maximum Marginal Relevance (MMR)
   - Balance relevance AND diversity
   - Avoids returning near-duplicate chunks
   - MMR = λ × relevance - (1-λ) × max_similarity_to_selected

3. Hybrid Search
   - Combine keyword search (BM25) + semantic search (embeddings)
   - Best of both worlds: exact matches + meaning-based matches

4. Multi-Query Retrieval
   - Generate multiple query variations
   - Retrieve for each, merge results
   - "What is the return policy?" 
   → "return policy", "refund rules", "how to return items"

5. Parent Document Retrieval
   - Retrieve small chunks for precision
   - Return the parent (larger) chunk for context
```

## 9.5 Advanced RAG Patterns

```
Query Transformation:
  User query → [Rewrite] → Better query → Retrieval
  "that thing from last week" → "meeting notes from July 12, 2024"

Query Routing:
  User query → [Classifier] → Route to appropriate knowledge base
  "What's our Q3 revenue?" → Financial DB
  "How do I reset my password?" → Support docs

Self-RAG:
  Model decides: Do I need to retrieve? → Retrieve → Evaluate quality → Generate

Corrective RAG (CRAG):
  Retrieve → Evaluate relevance → If irrelevant, search web instead

Graph RAG:
  Build knowledge graph from documents
  Retrieve via graph relationships, not just similarity
  
  [Company] ← employs ← [Employee] ← works_on ← [Project]
      │                        │                      │
  [Department]            [Skill]               [Deadline]
```

---

# 10. Vector Databases

## 10.1 What is a Vector Database?

```
A database designed to store, index, and query high-dimensional vectors.

Traditional DB:                    Vector DB:
  SELECT * FROM users              Find vectors similar
  WHERE age = 30                   to query vector
  → exact match                     → approximate nearest neighbors

  SQL: WHERE = exact              Vector: WHERE similarity > 0.8
```

## 10.2 How Vector Search Works

```
Indexing: Build efficient search structure

  Brute Force (exact):
    Compare query to ALL vectors → O(n) — slow for millions of vectors

  Approximate Nearest Neighbors (ANN):
    Use index structure → O(log n) — fast, slight accuracy tradeoff

ANN Algorithms:

1. HNSW (Hierarchical Navigable Small World)
   ┌─────────────────────────────────────────┐
   │ Layer 2:  A ────────── D               │
   │           │           / │               │
   │ Layer 1:  A ── B ── D ── E             │
   │           │ / │     │     │             │
   │ Layer 0:  A ── B ── C ── D ── E ── F   │
   └─────────────────────────────────────────┘
   Start at top layer (sparse, long edges)
   Navigate down, getting more precise at each layer
   Like GPS: highway → main roads → local streets

2. IVF (Inverted File Index)
   - Partition vectors into clusters (Voronoi cells)
   - Query only searches nearest clusters
   - Like organizing books by genre

3. PQ (Product Quantization)
   - Compress vectors to save memory
   - Split vector into subvectors, quantize each
   - 128-dim float32 → 128 bytes (4x compression)
```

## 10.3 Popular Vector Databases

```
┌───────────────┬──────────────┬───────────┬──────────────┐
│ Database      │ Type         │ Scale     │ Best For     │
├───────────────┼──────────────┼───────────┼──────────────┤
│ Pinecone      │ Managed      │ Very High │ Production   │
│ Weaviate      │ Open Source  │ High      │ Multi-modal  │
│ Milvus        │ Open Source  │ Very High │ Large scale  │
│ Qdrant        │ Open Source  │ High      │ Performance  │
│ ChromaDB      │ Open Source  │ Low       │ Prototyping  │
│ FAISS         │ Library      │ High      │ Research     │
│ pgvector      │ Extension    │ Medium    │ PostgreSQL   │
│ Redis         │ Extension    │ High      │ Low latency  │
└───────────────┴──────────────┴───────────┴──────────────┘

Choosing:
  Just starting / prototyping → ChromaDB (simple, local)
  PostgreSQL already in use   → pgvector (no new infra)
  Production, managed service → Pinecone
  Self-hosted, large scale   → Milvus or Weaviate
```

## 10.4 Embeddings Models

```
Provider              Dimensions  Cost        Performance
─────────────────────────────────────────────────────────
OpenAI text-embedding-3-small  1536  $0.02/1M    Good
OpenAI text-embedding-3-large  3072  $0.13/1M    Better
Cohere embed-v3              1024  $0.10/1M    Good
BGE-large-en (open)          1024  Free (self)  Good
E5-large (open)              1024  Free (self)  Good
GTE-large (open)             1024  Free (self)  Good

OpenAI embeddings are easiest to start with.
Open-source models avoid vendor lock-in and API costs.
```

---

# 11. LangChain & LangGraph

## 11.1 LangChain Overview

```
LangChain = framework for building LLM-powered applications.

Core Components:

┌──────────────────────────────────────────────────────────┐
│                    LangChain Stack                         │
│                                                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  LangChain   │  │  LangGraph   │  │  LangSmith   │   │
│  │  (core libs) │  │  (agents &   │  │  (monitoring │   │
│  │              │  │   workflows) │  │   & tracing) │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  LangChain Libraries                                 │ │
│  │  - langchain-core (base abstractions)                │ │
│  │  - langchain (chains, retrieval, agents)             │ │
│  │  - langchain-community (third-party integrations)    │ │
│  └──────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

## 11.2 Core Concepts

### Models

```python
# LLMs (text-in, text-out)
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Chat Models (message-based)
from langchain_core.messages import HumanMessage, SystemMessage
response = llm.invoke([
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="What is RAG?")
])

# Embeddings (text-in, vector-out)
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
vector = embeddings.embed_query("What is RAG?")
```

### Prompts

```python
from langchain_core.prompts import ChatPromptTemplate

# Simple prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}"),
    ("human", "{question}")
])

# Few-shot prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify sentiment"),
    ("human", "Great product!"), ("ai", "Positive"),
    ("human", "Terrible service"), ("ai", "Negative"),
    ("human", "{text}")
])
```

### Chains

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Simple chain: prompt → LLM → output parser
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} to a {audience}"
)
llm = ChatOpenAI(model="gpt-4")
parser = StrOutputParser()

chain = prompt | llm | parser
result = chain.invoke({"topic": "RAG", "audience": "5 year old"})

# Sequential chain (pipe output of one into next)
chain1 = prompt1 | llm | parser
chain2 = prompt2 | llm | parser
full_chain = chain1 | chain2
```

### Retrievers

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Create vector store
vectorstore = Chroma.from_documents(documents, OpenAIEmbeddings())

# Create retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",  # Maximum Marginal Relevance
    search_kwargs={"k": 4}
)

# Use in chain
retriever_chain = retriever | format_docs
```

### Memory

```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# Simple in-memory
history = InMemoryChatMessageHistory()
history.add_user_message("Hi!")
history.add_ai_message("Hello! How can I help?")

# Conversation buffer (keeps last N messages)
from langchain.memory import ConversationBufferWindowMemory
memory = ConversationBufferWindowMemory(k=10)
```

## 11.3 RAG with LangChain

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Load and split documents
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("knowledge_base.pdf")
docs = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = splitter.split_documents(docs)

# 2. Create vector store
vectorstore = Chroma.from_documents(splits, OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 3. Create RAG chain
prompt = ChatPromptTemplate.from_template("""
Answer the question based on the context below.
If you can't find the answer, say "I don't have that information."

Context: {context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-4")
    | StrOutputParser()
)

# 4. Use it
answer = rag_chain.invoke("What is the return policy?")
```

## 11.4 LangGraph Overview

```
LangGraph = stateful, multi-actor applications with LLMs.

Think of it as: "How to build complex agent workflows with cycles"

LangChain Chains:  A → B → C → D  (linear)
LangGraph:         A → B → C      (cycles, branching, conditional)
                      ↑   ↓
                      D ←─┘

Key Concepts:
  - State: shared data passed between nodes
  - Nodes: functions that process state
  - Edges: connections between nodes (can be conditional)
  - Conditional edges: route to different nodes based on state
```

## 11.5 LangGraph Architecture

```
┌──────────────────────────────────────────────────────┐
│                  LangGraph App                        │
│                                                       │
│  ┌──────────┐                                        │
│  │  START   │                                        │
│  └────┬─────┘                                        │
│       ↓                                               │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐       │
│  │  Node A  │───→│  Node B  │───→│  Node C  │       │
│  │(process) │    │(decide)  │    │(respond) │       │
│  └──────────┘    └────┬─────┘    └────┬─────┘       │
│                       │               │              │
│                       ↓               ↓              │
│                  ┌──────────┐    ┌──────────┐       │
│                  │  Node D  │    │   END    │       │
│                  │(fallback)│    └──────────┘       │
│                  └────┬─────┘                        │
│                       │                              │
│                       └────→ back to Node A          │
│                                                       │
│  STATE = { messages: [], context: "", decision: "" } │
└──────────────────────────────────────────────────────┘
```

## 11.6 Building a LangGraph Agent

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")

# Define nodes (functions)
def assistant(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def tool_caller(state: MessagesState):
    # Route to appropriate tool based on last message
    last_msg = state["messages"][-1]
    # ... tool logic ...
    return {"messages": [tool_result]}

def should_continue(state: MessagesState):
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "tools"
    return END

# Build the graph
graph = StateGraph(MessagesState)
graph.add_node("assistant", assistant)
graph.add_node("tools", tool_caller)

graph.add_edge(START, "assistant")
graph.add_conditional_edges("assistant", should_continue, {
    "tools": "tools",
    END: END
})
graph.add_edge("tools", "assistant")  # tools → back to assistant

app = graph.compile()

# Run it
result = app.invoke({"messages": [("user", "What's the weather?")]})
```

---

# 12. MCP (Model Context Protocol)

## 12.1 What is MCP?

```
MCP = A standardized protocol for connecting AI models to external tools and data.

Think of it as: "USB-C for AI"
  - One universal standard
  - Any model can connect to any tool
  - No custom integration needed for each tool

Before MCP:                    With MCP:
  Each AI app builds custom     One standard protocol
  integrations for each tool    
  
  App A → Tool 1 (custom)      App A ─┐
  App A → Tool 2 (custom)            ├→ MCP Server → Tool 1
  App B → Tool 1 (custom)      App B ─┤           → Tool 2
  App B → Tool 2 (custom)            └→ MCP Server → Tool 3
  (lots of duplicated work)     (write once, use everywhere)
```

## 12.2 MCP Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP Architecture                          │
│                                                               │
│  ┌───────────────┐         ┌───────────────┐                │
│  │  MCP Host     │◄───────►│  MCP Server   │                │
│  │  (AI App)     │  JSON-RPC│  (Tool Provider)│              │
│  │               │  over    │               │                │
│  │  - Claude     │  stdio/  │  - GitHub     │                │
│  │  - Cursor     │  SSE     │  - Slack      │                │
│  │  - Custom     │         │  - Database    │                │
│  └───────────────┘         └───────────────┘                │
│         │                          │                         │
│         ▼                          ▼                         │
│  ┌───────────────┐         ┌───────────────┐                │
│  │  Client       │         │  Resources    │                │
│  │  (manages     │         │  (data/tools) │                │
│  │   connection) │         │               │                │
│  └───────────────┘         └───────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

## 12.3 MCP Protocol Details

```
Transport Options:
  1. stdio (standard I/O) — local processes, most common
  2. SSE (Server-Sent Events) — remote/networked

Message Types (JSON-RPC 2.0):

  Request:
  {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "read_file",
      "arguments": {"path": "/docs/report.pdf"}
    }
  }

  Response:
  {
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
      "content": [{"type": "text", "text": "File contents..."}]
    }
  }

Core Capabilities:

  ┌─────────────────────────────────────────────────┐
  │ 1. Resources    — expose data (like GET endpoint)│
  │    "Here's data you can read"                    │
  │                                                  │
  │ 2. Tools        — expose actions (like POST)     │
  │    "Here's what you can do"                      │
  │                                                  │
  │ 3. Prompts      — expose prompt templates        │
  │    "Here's how to use me"                        │
  │                                                  │
  │ 4. Sampling    — server can request LLM completion│
  │    "Help me think about this"                    │
  └─────────────────────────────────────────────────┘
```

## 12.4 Example MCP Server

```python
# server.py — MCP Server for file operations
from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio

server = Server("file-server")

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="read_file",
            description="Read contents of a file",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"}
                },
                "required": ["path"]
            }
        ),
        Tool(
            name="write_file",
            description="Write content to a file",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "read_file":
        with open(arguments["path"], "r") as f:
            content = f.read()
        return [TextContent(type="text", text=content)]
    
    elif name == "write_file":
        with open(arguments["path"], "w") as f:
            f.write(arguments["content"])
        return [TextContent(type="text", text="File written successfully")]

async def main():
    async with mcp.server.stdio.stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())
```

## 12.5 MCP in Practice

```
Configuration (in Claude Desktop, Cursor, etc.):

{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": { "GITHUB_TOKEN": "your-token" }
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres", 
               "postgresql://localhost/mydb"]
    },
    "slack": {
      "command": "npx", 
      "args": ["-y", "@modelcontextprotocol/server-slack"],
      "env": { "SLACK_TOKEN": "your-token" }
    }
  }
}

Available MCP Servers (community):
  - GitHub (repos, issues, PRs)
  - PostgreSQL / SQLite (databases)
  - Slack (messaging)
  - Google Drive / Filesystem
  - Web search (Brave, Bing)
  - Puppeteer (browser automation)
  - Memory (knowledge graph)
```

---

# 13. Fine-Tuning & PEFT

## 13.1 When to Fine-Tune?

```
                    Use Prompt    Fine-tune     Train from
                    Engineering                Scratch
───────────────────────────────────────────────────────────
Cost                Low           Medium        Very High
Time                Minutes       Hours         Weeks-Months
Data needed         None          100-10K       100K+
Expertise           Low           Medium        High
Performance gain    Good          Better        Best
───────────────────────────────────────────────────────────

Start with prompting → try RAG → fine-tune only if needed

Fine-tune when:
  ✓ Need consistent output format/style
  ✓ Domain-specific knowledge not in training data
  ✓ Need smaller/faster model for production
  ✓ Prompt engineering isn't achieving desired results
```

## 13.2 Full Fine-Tuning

```
Update ALL model parameters on your specific dataset.

Original Model (175B params)    Fine-tuned Model (175B params)
┌──────────────────────┐        ┌──────────────────────┐
│  All weights updated  │        │  Weights adapted to   │
│  on your data         │───────→│  your specific task   │
└──────────────────────┘        └──────────────────────┘

Requirements:
  - GPU: Multiple A100/H100 GPUs (80GB+ each)
  - RAM: Hundreds of GB
  - Storage: TBs for model + dataset
  - Cost: $100s-$1000s per training run

  ┌───────────────────────────────────────┐
  │   GPU Memory for Full Fine-tuning     │
  │                                       │
  │   Model (float16):     14 GB/1B params│
  │   Optimizer (AdamW):   8 GB/1B params │
  │   Gradients:           4 GB/1B params │
  │   Activations:         varies         │
  │                                       │
  │   7B model needs ~60GB VRAM           │
  │   13B model needs ~120GB VRAM         │
  │   70B model needs ~600GB+ VRAM        │
  └───────────────────────────────────────┘
```

## 13.3 LoRA (Low-Rank Adaptation)

```
Instead of updating all weights, update a SMALL pair of matrices.

Original weight matrix W (d × d):      Frozen (not trained)
Low-rank matrices A (d × r) and        Trained (small)
B (r × d):

  W (4096 × 4096)        A (4096 × 16)    B (16 × 4096)
  ┌──────────────┐       ┌──────────┐     ┌──────────┐
  │ 16M params   │       │ 65K      │     │ 65K      │
  │ (frozen)     │       │ params   │     │ params   │
  └──────────────┘       └──────────┘     └──────────┘
  
  Output = W·x + A·B·x

  Train A and B instead of W!
  
  Memory savings: 16M → 130K params (99.2% reduction!)

Visual:
  Input ──→ ┌──────────┐ ──→ ┌──────────┐ ──→ Output
            │Original  │     │  LoRA    │
            │  Weight  │ +   │ (A × B)  │
            │ (frozen) │     │ (trained)│
            └──────────┘     └──────────┘
```

## 13.4 QLoRA (Quantized LoRA)

```
LoRA + Quantization = Even less memory needed

Quantization: reduce precision of model weights
  float32 (32-bit) → float16 (16-bit) → int8 (8-bit) → int4 (4-bit)

  float32: 0.123456789 (full precision)
  int4:    0.125       (quantized, approximate)

QLoRA steps:
1. Quantize base model to 4-bit (NF4 format)
2. Add LoRA adapters (trainable, in float16)
3. Train only LoRA adapters

Memory comparison:
  Full fine-tuning (7B):  ~60 GB VRAM
  LoRA (7B):              ~16 GB VRAM
  QLoRA (7B):             ~6 GB VRAM ← fits on consumer GPU!

  ┌─────────────────────────────────────────────────┐
  │  QLoRA Memory Usage (7B model)                   │
  │                                                   │
  │  4-bit quantized model:     3.5 GB               │
  │  LoRA adapters:             0.1 GB               │
  │  Optimizer states:          0.2 GB               │
  │  Activations:               ~2 GB               │
  │  ─────────────────────────────────               │
  │  Total:                     ~6 GB  ✓ Fits!       │
  │                                                   │
  │  (vs 60 GB for full fine-tuning)                 │
  └─────────────────────────────────────────────────┘
```

## 13.5 Fine-Tuning Process

```
Step 1: Prepare Dataset

  Format (for instruction tuning):
  {
    "instruction": "Summarize the following text",
    "input": "Long article about climate change...",
    "output": "Climate change is causing global temperatures..."
  }

  Dataset size: 100-10,000 examples (sweet spot: 1,000-5,000)
  Quality > Quantity: 1,000 high-quality examples > 100,000 noisy ones

Step 2: Choose Base Model
  - Llama 3 8B (open, good performance)
  - Mistral 7B (open, efficient)
  - Phi-3 (small, surprisingly capable)
  - Qwen 2.5 (multilingual)

Step 3: Configure Training
  hyperparameters:
    learning_rate: 2e-4 (for LoRA)
    epochs: 2-5 (watch for overfitting)
    batch_size: 4-16
    lora_r: 8-64 (rank, higher = more capacity)
    lora_alpha: 16-128 (scaling factor)
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

Step 4: Train
  Tools: Hugging Face TRL, Axolotl, LLaMA-Factory
  
Step 5: Evaluate
  - Run on held-out test set
  - Compare with base model
  - Human evaluation of output quality
  - Check for overfitting (train loss ↓ but eval loss ↑)

Step 6: Deploy
  - Merge LoRA weights into base model (optional)
  - Quantize for inference (GPTQ, AWQ, GGUF)
  - Serve with vLLM, TGI, or llama.cpp
```

## 13.6 Transfer Learning

```
Use knowledge from one task to improve performance on another.

General → Specific:
  GPT trained on internet text → Fine-tuned on medical texts → Medical AI

Computer Vision Transfer Learning:
  ImageNet pre-trained ResNet → Fine-tuned on medical images → X-ray classifier

  ┌─────────────────┐     ┌──────────────────┐     ┌──────────────┐
  │ ImageNet (1M    │────→│ Remove last layer │────→│ Add new      │
  │ images, 1000    │     │ (pretrained       │     │ classifier   │
  │ classes)        │     │  features)        │     │ for X-rays   │
  └─────────────────┘     └──────────────────┘     └──────────────┘
  
  Feature extraction (freeze base): train only new layer (fast, less data)
  Full fine-tuning: train all layers (slower, needs more data)
```

---

# 14. MLOps & Deployment

## 14.1 MLOps Overview

```
MLOps = DevOps for Machine Learning

┌──────────────────────────────────────────────────────────┐
│                    MLOps Lifecycle                         │
│                                                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │  Data    │→ │  Model   │→ │  Model   │→ │  Model   │ │
│  │  Pipeline│  │ Training │  │ Registry │  │ Deployment│ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │
│       ↑                                           │       │
│       │           ┌──────────┐                    │       │
│       └───────────│Monitoring│←───────────────────┘       │
│                   └──────────┘                             │
└──────────────────────────────────────────────────────────┘

Key Principles:
  1. Reproducibility: same code + same data = same model
  2. Versioning: version data, code, models, configs
  3. Automation: automate training, testing, deployment
  4. Monitoring: track model performance in production
  5. Collaboration: data scientists + engineers working together
```

## 14.2 CI/CD for ML

```
Traditional CI/CD:               ML CI/CD:
  Code → Build → Test → Deploy   Code+Data → Train → Evaluate → Deploy
                                        ↓
                                  Model Registry
                                        ↓
                                  Monitor → Retrain (if needed)

┌─────────────────────────────────────────────────────┐
│  ML Pipeline CI/CD                                    │
│                                                       │
│  1. Data Validation                                   │
│     - Check schema (features, types, ranges)         │
│     - Check data quality (nulls, duplicates)         │
│     - Check data drift (distribution shifts)         │
│                                                       │
│  2. Model Training                                    │
│     - Triggered by: new data, schedule, code change  │
│     - Tracked by: MLflow, W&B, TensorBoard           │
│                                                       │
│  3. Model Evaluation                                  │
│     - Automated tests: performance thresholds        │
│     - A/B testing against current production model   │
│     - Fairness/bias checks                           │
│                                                       │
│  4. Model Deployment                                  │
│     - Canary deployment (gradual rollout)            │
│     - Shadow deployment (run alongside, don't serve) │
│     - Blue/green deployment (instant switch)         │
│                                                       │
│  5. Monitoring & Alerting                             │
│     - Performance metrics (accuracy, latency)        │
│     - Data drift detection                           │
│     - Error rate monitoring                          │
└─────────────────────────────────────────────────────┘
```

## 14.3 Model Serving

```
Serving Options:

1. REST API (FastAPI / Flask)
   - Most common for production
   - Simple to implement
   
   from fastapi import FastAPI
   app = FastAPI()
   
   @app.post("/predict")
   async def predict(text: str):
       result = model.predict(text)
       return {"prediction": result}

2. Dedicated ML Serving (vLLM, TGI, Triton)
   - Optimized for ML inference
   - Batching, GPU optimization
   
   vLLM:   Optimized for LLMs (continuous batching)
   TGI:    Hugging Face's serving solution
   Triton: NVIDIA's inference server

3. Managed Services
   - AWS SageMaker
   - Google Vertex AI
   - Azure ML

Deployment Configurations:
┌──────────────┬────────────┬──────────────┐
│ Config       │ Pros       │ Cons         │
├──────────────┼────────────┼──────────────┤
│ Serverless   │ Auto-scale │ Cold starts  │
│ (Lambda)     │ Pay per    │ Time limits  │
│              │ request    │              │
├──────────────┼────────────┼──────────────┤
│ Container    │ Full       │ Manage infra │
│ (ECS/EKS)    │ control    │              │
├──────────────┼────────────┼──────────────┤
│ GPU Instance │ Best for   │ Expensive    │
│ (g5.xlarge)  │ LLMs       │ idle cost    │
└──────────────┴────────────┴──────────────┘
```

## 14.4 Docker for ML

```dockerfile
# Dockerfile for ML API
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download model during build (not runtime)
RUN python download_model.py

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - MODEL_PATH=/app/models/my_model.pt
      - API_KEY=${API_KEY}
```

## 14.5 Infrastructure as Code (Terraform)

```hcl
# main.tf — Deploy ML API to AWS

# ECS Cluster
resource "aws_ecs_cluster" "ml_api" {
  name = "ml-api-cluster"
}

# Task Definition (what to run)
resource "aws_ecs_task_definition" "ml_api" {
  family                   = "ml-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 2048
  memory                   = 4096
  
  container_definitions = jsonencode([{
    name  = "ml-api"
    image = "${aws_ecr_repository.ml_api.repository_url}:latest"
    
    portMappings = [{
      containerPort = 8000
      hostPort      = 8000
    }]
    
    environment = [
      { name = "MODEL_PATH", value = "/models/model.pt" }
    ]
    
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = "/ecs/ml-api"
        "awslogs-region"        = "us-east-1"
        "awslogs-stream-prefix" = "ecs"
      }
    }
  }])
}

# Service (keeps tasks running)
resource "aws_ecs_service" "ml_api" {
  name            = "ml-api-service"
  cluster         = aws_ecs_cluster.ml_api.id
  task_definition = aws_ecs_task_definition.ml_api.arn
  desired_count   = 2
  launch_type     = "FARGATE"
  
  network_configuration {
    subnets         = [aws_subnet.private.id]
    security_groups = [aws_security_group.ml_api.id]
  }
  
  load_balancer {
    target_group_arn = aws_lb_target_group.ml_api.arn
    container_name   = "ml-api"
    container_port   = 8000
  }
}

# Auto Scaling
resource "aws_appautoscaling_target" "ml_api" {
  max_capacity       = 10
  min_capacity       = 2
  resource_id        = "service/${aws_ecs_cluster.ml_api.name}/${aws_ecs_service.ml_api.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}
```

---

# 15. Production Best Practices

## 15.1 Model Monitoring

```
What to Monitor:

1. Performance Metrics
   - Accuracy, F1, precision, recall (classification)
   - MSE, MAE, R² (regression)
   - Perplexity (language models)
   - Log these per hour/day/week

2. Data Drift
   - Statistical properties of input data change over time
   - Example: model trained on 2023 data, serving 2024 data
   
   Detecting drift:
   ┌───────────────────────────────────────────────┐
   │ Feature    │ Train Mean │ Prod Mean │ Drift?  │
   ├───────────┼────────────┼───────────┼─────────┤
   │ age        │ 35.2       │ 36.1      │ No      │
   │ income     │ 55000      │ 72000     │ YES ⚠️  │
   │ tenure     │ 2.3        │ 2.1       │ No      │
   └───────────┴────────────┴───────────┴─────────┘
   
   Methods: PSI, KS test, Jensen-Shannon divergence

3. Prediction Drift
   - Distribution of model outputs changes
   - Example: suddenly predicting more "positive" than usual

4. System Metrics
   - Latency (p50, p95, p99)
   - Throughput (requests/second)
   - Error rate
   - GPU/CPU utilization
   - Memory usage

Alert Thresholds:
  Latency p99 > 500ms     → Warning
  Error rate > 1%          → Alert
  F1 score drops > 5%     → Retrain trigger
  Data drift PSI > 0.2    → Investigation
```

## 15.2 A/B Testing

```
Compare two model versions in production:

┌──────────────┐
│    Users     │
│              │
│  50% → Model A (current)  → Metric A
│  50% → Model B (new)      → Metric B
└──────────────┘

Statistical Significance:
  - Run for enough time (days/weeks)
  - Need sufficient sample size
  - Use proper statistical tests (t-test, chi-squared)
  - Don't peek and stop early (p-hacking)

  Results:
  Model A: accuracy = 87.2% (n=10000)
  Model B: accuracy = 88.1% (n=10000)
  p-value = 0.03 (< 0.05) → Statistically significant!
  → Deploy Model B

Multi-Armed Bandit (more advanced):
  - Dynamically shift traffic to better-performing model
  - Faster than traditional A/B testing
  - Explore vs exploit tradeoff
```

## 15.3 Observability

```
Three Pillars of Observability:

1. Logs
   - Structured logs for every prediction
   - Include: input, output, latency, model version, user_id
   
   {
     "timestamp": "2024-01-15T10:30:00Z",
     "model_version": "v2.3.1",
     "input_tokens": 156,
     "output_tokens": 42,
     "latency_ms": 234,
     "user_id": "user_123",
     "prediction": "positive",
     "confidence": 0.92
   }

2. Metrics
   - Quantitative measurements over time
   - Use Prometheus, Datadog, CloudWatch
   - Dashboard with key metrics

3. Traces
   - Track request through entire pipeline
   - Identify bottlenecks
   
   Request Flow:
   [API Gateway: 5ms]
       ↓
   [Preprocessing: 12ms]
       ↓
   [Vector Search: 45ms]  ← bottleneck!
       ↓
   [LLM Inference: 180ms]
       ↓
   [Postprocessing: 3ms]
       ↓
   Total: 245ms
```

## 15.4 Cost Optimization

```
1. Model Optimization
   - Quantization (int8/int4) → faster inference, less memory
   - Model distillation → smaller model mimics larger one
   - Pruning → remove unnecessary weights
   
2. Infrastructure Optimization
   - Right-size instances (don't over-provision)
   - Use spot instances for training (70% cheaper)
   - Auto-scaling based on traffic
   - GPU sharing (time-slicing)

3. Caching
   - Cache frequent queries (same prompt → same response)
   - Semantic cache: similar prompts → cached response
   - Reduces API costs by 30-50% in many cases

4. Batching
   - Batch multiple requests together (GPU utilization)
   - vLLM continuous batching: dynamic batch sizes
   
Cost Estimation (GPT-4 API):
  Input:  $0.03 / 1M tokens
  Output: $0.06 / 1M tokens
  
  If 1M requests/month, avg 1000 input + 500 output tokens:
  Cost = (1M × 1000 × $0.03/1M) + (1M × 500 × $0.06/1M)
       = $30 + $30 = $60/month (API only)
  
  Self-hosted 7B model: ~$200-400/month (GPU instance)
  → API cheaper for low traffic, self-hosted for high traffic
```

## 15.5 Security Best Practices

```
1. API Key Management
   - Never hardcode keys
   - Use environment variables or secret managers
   - Rotate keys regularly
   - Use least-privilege access

2. Input Validation
   - Sanitize user inputs
   - Rate limiting (prevent abuse)
   - Content filtering (prevent harmful outputs)
   - Max token limits

3. Data Privacy
   - Don't log sensitive data (PII)
   - Anonymize data in logs
   - Comply with GDPR/CCPA
   - Data encryption at rest and in transit

4. Model Security
   - Guard against prompt injection
     Malicious: "Ignore all previous instructions and..."
     Defense: input sanitization, instruction hierarchy
   - Don't expose model internals
   - Model watermarking (detect AI-generated content)

5. Infrastructure Security
   - Network isolation (VPC, private subnets)
   - Container scanning for vulnerabilities
   - Regular security audits
```

---

# 16. Capstone Projects

## Beginner Projects

```
Project 1: Sentiment Analyzer
  Stack: Scikit-learn, Pandas, NLTK
  Data: IMDB movie reviews
  Model: Logistic Regression / Naive Bayes
  Deploy: Flask API

Project 2: Image Classifier
  Stack: PyTorch, ResNet (pretrained)
  Data: Custom dataset (cats/dogs, flowers)
  Deploy: FastAPI + Docker

Project 3: RAG Chatbot (Basic)
  Stack: LangChain, OpenAI, ChromaDB
  Data: PDF documents
  Features: Document upload, Q&A, source citations
```

## Intermediate Projects

```
Project 4: Fine-tuned LLM for Domain
  Stack: Hugging Face, LoRA/QLoRA
  Data: Domain-specific dataset
  Base: Llama 3 8B or Mistral 7B
  Deploy: vLLM + FastAPI

Project 5: Multi-Agent System
  Stack: LangGraph, LangChain
  Features: Agent with tools (web search, code execution, file ops)
  Pattern: ReAct agent with tool routing

Project 6: Production RAG Pipeline
  Stack: LangChain, Pinecone/Weaviate, LangSmith
  Features:
    - Document ingestion pipeline
    - Hybrid search (keyword + semantic)
    - Query routing
    - Evaluation metrics
    - Monitoring dashboard
```

## Advanced Projects

```
Project 7: End-to-End ML Platform
  Stack: MLflow, Airflow, Docker, Kubernetes, Terraform
  Features:
    - Automated retraining pipeline
    - Model registry
    - A/B testing framework
    - Monitoring and alerting
    - Infrastructure as code

Project 8: Production AI SaaS
  Stack: Full MLOps stack
  Features:
    - Multi-tenant architecture
    - Usage-based billing
    - Model marketplace
    - Real-time analytics
    - SOC2 compliance

Project 9: Custom LLM from Scratch
  Stack: PyTorch, custom tokenizer, distributed training
  Features:
    - Train a small GPT model (100M-1B params)
    - Custom BPE tokenizer
    - Distributed training (multi-GPU)
    - Evaluation benchmarks
```

## Project Roadmap

```
Month 1-2:   Projects 1-3 (foundations)
Month 3-4:   Projects 4-6 (intermediate)
Month 5-6:   Projects 7-9 (advanced)

Portfolio展示:
  GitHub repos with:
    ✓ Clean, documented code
    ✓ README with architecture diagrams
    ✓ Tests
    ✓ CI/CD pipeline
    ✓ Live demo (Hugging Face Spaces, Streamlit)
```

---

# Appendix: Learning Resources

## Books
- "Deep Learning" by Ian Goodfellow (theory)
- "Hands-On Machine Learning" by Aurélien Géron (practice)
- "Designing Machine Learning Systems" by Chip Huyen (MLOps)

## Online Courses
- fast.ai (free, practical deep learning)
- Stanford CS229 (ML theory)
- Stanford CS231n (computer vision)
- Stanford CS224n (NLP)
- DeepLearning.AI (various specializations)

## Practice
- Kaggle competitions
- Papers With Code
- Hugging Face tutorials

---

*Last updated: July 2026*
*This guide is a living document — update as the field evolves.*
