# Project Specifications: Neural Network Library

## 1. Technical Constraints
* **Language:** Python 3.10
* **Allowed Libraries:** `numpy` (for logic), `matplotlib/pandas` (viz), `sklearn` (SVM/Data), `tensorflow/keras` (for comparison only).

---

## PART 1: Core Library & Validation (The Foundation)
[cite_start]**Goal:** Build the engine and prove it works on simple logic gates.

### 1.1 Mathematical Engine (The Library)
* [cite_start]**Layer Abstraction:** Base class with `forward` / `backward`[cite: 20].
* [cite_start]**Dense Layer:** Handles Weights ($W$), Biases ($b$), and gradients ($dW, db, dX$) [cite: 21-22].
* **Activations:**
    * **ReLU:** $f(x) = \max(0, x)$
    * **Sigmoid:** $f(x) = \frac{1}{1 + e^{-x}}$
    * **Tanh:** $f(x) = \tanh(x)$
    * [cite_start]**Softmax:** (Optional for Part 1, required for general library completeness)[cite: 26].
* [cite_start]**Loss Function:** Mean Squared Error (MSE)[cite: 32].
* [cite_start]**Optimizer:** SGD (Stochastic Gradient Descent)[cite: 35].

### 1.2 Required Validation (XOR)
* **Gradient Checking:** **CRITICAL**. [cite_start]You must mathematically prove your derivatives match numerical approximations [cite: 59-62].
* **XOR Task:**
    * **Input:** 2 neurons.
    * [cite_start]**Architecture:** 2 -> 4 (Tanh/Sigmoid) -> 1 (Sigmoid)[cite: 40].
    * [cite_start]**Goal:** 100% accuracy on inputs `[[0,0], [0,1], [1,0], [1,1]]`[cite: 42].

---

## PART 2: Advanced Applications (The "Real" AI)
[cite_start]**Goal:** Apply the library to images (MNIST), perform unsupervised learning, and compare with industry tools.

### 2.1 Autoencoder (Unsupervised)
* **Data:** MNIST Digits.
* [cite_start]**Architecture:** 784 (Input) -> 32/64 (Latent) -> 784 (Output)[cite: 46].
* [cite_start]**Loss:** MSE between Input and Output[cite: 48].
* [cite_start]**Viz:** Plot original vs. reconstructed images[cite: 64].

### 2.2 Latent Space Classification
* [cite_start]**Feature Extraction:** Use the trained Encoder to turn images into 32/64-size vectors[cite: 50].
* [cite_start]**SVM:** Train `sklearn.svm.SVC` on these vectors[cite: 51].
* [cite_start]**Metrics:** Confusion Matrix & Accuracy[cite: 52].

### 2.3 Benchmark (TensorFlow/Keras)
* [cite_start]**Task:** Re-build the XOR and Autoencoder in Keras[cite: 66].
* [cite_start]**Compare:** Training time, implementation difficulty, and final loss [cite: 67-70].

---

## Deliverables Structure
* [cite_start]**`lib/`**: All core classes (`layers.py`, `activations.py`, `losses.py`, `optimizer.py`, `network.py`) [cite: 80-89].
* **`notebooks/project_demo.ipynb`**:
    * **Part 1 Sections:** 1. Gradient Check, 2. XOR Training.
    * [cite_start]**Part 2 Sections:** 3. Autoencoder, 4. SVM, 5. TF Comparison [cite: 101-110].
* [cite_start]**`report/project_report.pdf`**: Final analysis of both parts[cite: 111].