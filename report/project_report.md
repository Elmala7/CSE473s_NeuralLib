# CSE473s: Project Report (Part 1)

## 1. Library Design and Architecture Choices

The library was built with a modular, Object-Oriented architecture to ensure flexibility and ease of debugging. The core design principles are Separation of Concerns and Extensibility.

### 1.1 Core Components
* **`Layer` (Abstract Base Class):** Located in `lib/layers.py`. Defines the contract (`forward` and `backward`) that all layers must follow. This allows the network to treat Dense layers and Activation functions identically during the training loop.
* **`Dense` Layer:** Handles the linear transformation $Z = X \cdot W + b$.
    * **Initialization:** Implements **He Initialization** (`sqrt(2/n)`) to ensure variance is maintained through deep networks, preventing vanishing gradients during the initial epochs.
    * **Gradient Storage:** Unlike simple implementations, this layer does not update its own weights. It calculates gradients (`grad_weights`, `grad_bias`) and stores them, delegating the update step to the Optimizer.
* **`Sequential` Network:** Located in `lib/network.py`. Acts as a container that orchestrates the forward pass (inference) and the training loop. It manages the flow of data through the stack of layers.

### 1.2 The Optimizer (`optimizer.py`)
To satisfy the mandatory requirement for a modular optimizer, an `SGD` (Stochastic Gradient Descent) class was implemented.
* **Decoupled Logic:** The weight update rule ($W_{new} = W_{old} - \eta \cdot \nabla W$) is encapsulated here.
* **Advantage:** This design allows for easy swapping of optimization algorithms (e.g., adding Adam or RMSProp in the future) without modifying the layer code.

### 1.3 Mathematical Implementation
* **Activations:** `Sigmoid`, `Tanh`, and `ReLU` are implemented as independent layers in `lib/activations.py`, calculating their specific derivatives in the backward pass.
* **Loss Function:** `MSE` (Mean Squared Error) in `lib/losses.py` calculates the error and the initial gradient vector $2(Y_{pred} - Y_{true})/N$.

---

## 2. Results from the XOR Test

The library was validated using the classic XOR problem, which requires a non-linear decision boundary that a single perceptron cannot solve.

### 2.1 Experiment Setup
* **Architecture:** `Dense(2, 16)` $\rightarrow$ `Tanh` $\rightarrow$ `Dense(16, 1)` $\rightarrow$ `Sigmoid`.
* **Hyperparameters:** Learning Rate = 1.0, Epochs = 10,000.
* **Seed:** Fixed (15) for reproducibility.

### 2.2 Numerical Results
The model successfully converged to a near-zero loss, correctly classifying all four logical states.

| Input A | Input B | Target | Predicted Probability | Rounded Prediction |
| :--- | :--- | :--- | :--- | :--- |
| 0 | 0 | **0** | 0.0016 | **0** |
| 0 | 1 | **1** | 0.9972 | **1** |
| 1 | 0 | **1** | 0.9974 | **1** |
| 1 | 1 | **0** | 0.0031 | **0** |

* **Final Loss:** ~0.000007
* **Gradient Check:** Passed with a difference of $1.13 \times 10^{-12}$, verifying the correctness of the backpropagation engine.

### 2.3 Conclusion
The library correctly handles forward propagation, backpropagation, and parameter updates, solving the non-linear XOR task with 100% accuracy.
