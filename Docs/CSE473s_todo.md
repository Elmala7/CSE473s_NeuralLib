# Project Roadmap

## ---------------- PART 1 ----------------

### Step 1: Repo & Setup
- [x] [cite_start]**Init:** Create Git repo with `lib/`, `notebooks/`, `report/` [cite: 73-80].
- [x] **Files:** Create empty python files: `layers.py`, `activations.py`, `losses.py`, `optimizer.py`, `network.py`.

### Step 2: The Core Logic (Forward Pass)
- [x] **Base:** Implement `class Layer` (abstract).
- [x] **Activations:** Implement `ReLU`, `Sigmoid`, `Tanh` (Forward math).
- [x] **Dense:** Implement $Z = X \cdot W + b$. [cite_start]Initialize $W$ randomly, $b$ as zeros[cite: 21].

### Step 3: The "Brain" (Backward Pass)
- [x] **Activations:** Implement derivatives ($f'(x)$) in `backward()` methods.
- [x] **Dense:** Implement Chain Rule for $dW$, $db$, and $dX$.
- [x] [cite_start]**Loss:** Implement `MSE` class (loss calculation + gradient)[cite: 32].
- [x] [cite_start]**Optimizer:** Implement `SGD` in separate `lib/optimizer.py` to update weights ($W = W - \eta \cdot dW$)[cite: 35].
- [x] **Gradient Accumulation:** Properly accumulate gradients across batch samples in each epoch.

### Step 4: Verification (The "Must-Haves")
- [x] **Gradient Check:** Write a script in the notebook to compare your manual gradient vs. numerical gradient $(f(x+\epsilon) - f(x-\epsilon))/2\epsilon$. [cite_start]**This proves your library works** [cite: 60-61].
- [x] **XOR Script:** Train a 2-4-1 network on XOR data.
- [x] **Debug:** Ensure loss goes down and predictions are correct (e.g., `[0,1]` -> `~1`).

## ---------------- PART 2 ----------------

### Step 5: Autoencoder (MNIST)
- [x] **Data:** Load and normalize MNIST (using sklearn or keras just for loading).
- [x] **Train:** Build 784->Latent->784 net. [cite_start]Train with MSE (Input = Target)[cite: 48].
- [x] **Viz:** Show "Input Image" vs "Reconstructed Image" in notebook.

### Step 6: SVM Classification
- [x] **Extract:** Run test set through Encoder -> Get Latent Vectors.
- [x] [cite_start]**Classify:** Train SVM on Latent Vectors -> Get Accuracy[cite: 51].
- [x] **Analyze:** Generate Confusion Matrix.

### Step 7: Keras Comparison & Report
- [x] **Benchmark:** Build XOR/Autoencoder in Keras. [cite_start]Compare speed/loss[cite: 66].
- [x] [cite_start]**Report:** Write PDF covering Design, XOR results, Autoencoder quality, SVM results, and Comparison[cite: 111].

---

✅ **Project Complete & Verified** (Completed: 2025-12-25)
- All 7 steps implemented and tested
- Part 1: Gradient check passed; XOR solved (100% accuracy)
- Part 2: Autoencoder trained on MNIST (784 -> 128 -> 784). Reconstruction MSE (test): **0.007915**
- SVM classification on latent features: **96.89%** accuracy (confusion matrix in `notebooks/project_demo.ipynb`)
- Keras comparison: Reconstruction MSE (Keras): **0.002647**
- Reports and notebooks are included in `report/` and `notebooks/`

---