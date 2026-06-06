"""MultiLayer Perceptron (MLP) with Backpropagation from scratch using NumPy."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class MLPClassifier:
    """MultiLayer Perceptron Classifier trained with Backpropagation."""

    def __init__(self, layer_sizes: list[int], learning_rate: float = 0.1, epochs: int = 1000):
        self.layer_sizes = layer_sizes
        self.lr = learning_rate
        self.epochs = epochs
        
        # Initialize weights and biases
        # W[l] has shape (n_neurons_at_l, n_neurons_at_l-1)
        # b[l] has shape (n_neurons_at_l, 1)
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        
        for i in range(1, len(layer_sizes)):
            # He initialization (scaled normal distribution)
            limit = np.sqrt(2.0 / layer_sizes[i - 1])
            self.W.append(np.random.randn(layer_sizes[i], layer_sizes[i - 1]) * limit)
            self.b.append(np.zeros((layer_sizes[i], 1)))
            
        self.loss_history: list[float] = []
        self.accuracy_history: list[float] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    @staticmethod
    def _sigmoid_derivative(a: np.ndarray) -> np.ndarray:
        # Expects 'a' to be the sigmoid output (a = sigmoid(z))
        return a * (1.0 - a)

    def _forward(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Perform Forward Propagation.
        
        X: input matrix of shape (n_features, n_samples)
        Returns:
            a_values: list of activation outputs at each layer (including input layer)
            z_values: list of pre-activation outputs at each layer (from layer 1 onwards)
        """
        a_values = [X]
        z_values = []
        
        a = X
        for W_l, b_l in zip(self.W, self.b):
            z = W_l @ a + b_l
            a = self._sigmoid(z)
            z_values.append(z)
            a_values.append(a)
            
        return a_values, z_values

    def _backward(self, a_values: list[np.ndarray], y: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Perform Backward Propagation.
        
        a_values: activation outputs at each layer
        y: target labels of shape (n_output, n_samples)
        Returns:
            dW: gradients of loss w.r.t weights at each layer
            db: gradients of loss w.r.t biases at each layer
        """
        n_samples = y.shape[1]
        dW = []
        db = []
        
        # Output layer error
        a_out = a_values[-1]
        # Loss: 1/2 * mean(sum((y - y_hat)^2))
        # dL/da = a_out - y
        delta = (a_out - y) * self._sigmoid_derivative(a_out)
        
        # Iterate backwards through layers
        # L - 1 down to 0
        num_layers = len(self.W)
        for l in reversed(range(num_layers)):
            a_prev = a_values[l]
            
            # Gradients for layer l
            dW_l = (1.0 / n_samples) * (delta @ a_prev.T)
            db_l = (1.0 / n_samples) * np.sum(delta, axis=1, keepdims=True)
            
            dW.insert(0, dW_l)
            db.insert(0, db_l)
            
            if l > 0:
                # Backpropagate error to previous layer
                delta = (self.W[l].T @ delta) * self._sigmoid_derivative(a_prev)
                
        return dW, db

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPClassifier":
        """Train MLP model using Batch Gradient Descent.
        
        X: input data shape (n_samples, n_features)
        y: target labels shape (n_samples,) or (n_samples, n_output)
        """
        # Transpose input to match mathematical notations:
        # X_input: (n_features, n_samples)
        # y_input: (n_output, n_samples)
        X_input = X.T
        if len(y.shape) == 1:
            y_input = y.reshape(1, -1)
        else:
            y_input = y.T
            
        self.loss_history = []
        self.accuracy_history = []
        
        for epoch in range(self.epochs):
            # Forward Pass
            a_values, _ = self._forward(X_input)
            y_hat = a_values[-1]
            
            # Compute Loss (MSE)
            loss = 0.5 * np.mean(np.sum((y_hat - y_input) ** 2, axis=0))
            self.loss_history.append(loss)
            
            # Compute Accuracy (threshold at 0.5 for binary)
            preds = (y_hat >= 0.5).astype(int)
            acc = float(np.mean(preds == y_input))
            self.accuracy_history.append(acc)
            
            # Backward Pass
            dW, db = self._backward(a_values, y_input)
            
            # Update parameters
            for l in range(len(self.W)):
                self.W[l] -= self.lr * dW[l]
                self.b[l] -= self.lr * db[l]
                
            if (epoch + 1) % (self.epochs // 10 or 1) == 0 or epoch == 0:
                print(f"Epoch {epoch+1:4d}/{self.epochs} - Loss: {loss:.6f} - Acc: {acc:.2%}")
                
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities of classes."""
        X_input = X.T
        a_values, _ = self._forward(X_input)
        return a_values[-1].T

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels (0 or 1)."""
        probs = self.predict_proba(X)
        if probs.shape[1] == 1:
            return (probs >= threshold).astype(int).ravel()
        return (probs >= threshold).astype(int)


def run_xor_demo() -> None:
    print("=" * 50)
    print("--- Trực quan hóa toán học: Giải bài toán XOR ---")
    print("=" * 50)
    
    # XOR Inputs and Outputs
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    
    # MLP with 2 input, 4 hidden, 1 output neurons
    mlp = MLPClassifier(layer_sizes=[2, 4, 1], learning_rate=0.5, epochs=3000)
    mlp.fit(X, y)
    
    print("\nKết quả dự đoán bài toán XOR:")
    probs = mlp.predict_proba(X).ravel()
    preds = mlp.predict(X)
    for i in range(len(X)):
        print(f"Input: {X[i]} | Target: {y[i]} | Prob: {probs[i]:.4f} | Pred: {preds[i]}")


def main() -> None:
    # Set seed for reproducibility
    np.random.seed(42)
    
    # 1. Run XOR demo
    run_xor_demo()
    
    # 2. Run Moons dataset classification
    print("\n" + "=" * 50)
    print("--- Huấn luyện MLP trên tập dữ liệu Moons (Phi tuyến) ---")
    print("=" * 50)
    
    X, y = make_moons(n_samples=400, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Setup MLP: 2 Inputs -> 8 Hidden -> 4 Hidden -> 1 Output
    mlp = MLPClassifier(layer_sizes=[2, 8, 4, 1], learning_rate=0.2, epochs=2000)
    mlp.fit(X_train, y_train)
    
    test_preds = mlp.predict(X_test)
    test_acc = float(np.mean(test_preds == y_test))
    print(f"\n[MLP Moon Classifier] Test Accuracy: {test_acc:.2%}")
    
    # Plotting Loss & Accuracy curves and Decision Boundary
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss & Accuracy Curves
    color = "tab:red"
    axes[0].plot(mlp.loss_history, color=color, label="Loss (MSE)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss", color=color)
    axes[0].tick_params(axis="y", labelcolor=color)
    
    ax2 = axes[0].twinx()
    color = "tab:blue"
    ax2.plot(mlp.accuracy_history, color=color, label="Accuracy")
    ax2.set_ylabel("Accuracy", color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    
    axes[0].set_title("MLP — Lịch sử Huấn luyện (Loss & Accuracy)")
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Decision Boundary
    x_min, x_max = X_test[:, 0].min() - 0.5, X_test[:, 0].max() + 0.5
    y_min, y_max = X_test[:, 1].min() - 0.5, X_test[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = mlp.predict_proba(grid).reshape(xx.shape)
    
    axes[1].contourf(xx, yy, probs, levels=50, cmap="RdYlBu", alpha=0.6)
    scatter = axes[1].scatter(
        X_test[:, 0], X_test[:, 1], c=y_test,
        cmap="RdYlBu", edgecolors="black", linewidth=0.5, s=40
    )
    axes[1].set_xlabel("Feature 1 (scaled)")
    axes[1].set_ylabel("Feature 2 (scaled)")
    axes[1].set_title(f"MLP — Ranh giới Quyết định (Test Acc: {test_acc:.2%})")
    plt.colorbar(scatter, ax=axes[1], label="Xác suất đầu ra")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("mlp_backpropagation_results.png", dpi=150)
    plt.close()
    print("Đã lưu đồ thị kết quả vào file: mlp_backpropagation_results.png")


if __name__ == "__main__":
    main()
